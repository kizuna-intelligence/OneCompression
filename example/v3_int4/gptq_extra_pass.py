"""GPTQ 4-bit pass for the encoder Linears, using REAL captured activations.

Why this exists
---------------
``quantize_gptq.py`` only quantizes ``model.blocks`` (the QEP scope).  The
encoders (text_encoder / speaker_encoder) are left fp16 because *blind* RTN
4-bit destroys them on v3.  But the DiT blocks proved that **real-activation
GPTQ** (not blind rounding) recovers quality — so this script applies the same
idea to the encoders to shrink the checkpoint toward the v2 size.

How
---
For every target encoder ``nn.Linear`` we:
  1. hook it and accumulate the GPTQ Hessian ``H = (2/N) * sum(x^T x)`` over
     real syntheses (the same utterances as ``capture_calibration.py``);
  2. run OneCompression ``run_gptq(H, layer, wbits=4, groupsize=32)``;
  3. pack the result with ``GPTQ.create_inference_layer`` (same GPTQLinear
     format the DiT blocks use, so the Irodori-TTS-Lite fused loader handles
     them identically);
  4. merge the packed buffers + a ``quant_layers_json`` entry into an existing
     packed checkpoint (the output of ``quantize_gptq.py``), dropping the fp16
     weight.

Layers whose ``in_features`` is not divisible by the groupsize are left fp16.
``--actorder`` is intentionally unsupported (the Lite loader rejects ``.perm``).

Run (inside an irodori_tts-capable venv, onecomp on PYTHONPATH)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=/path/to/OneCompression:/path/to/Irodori-streaming \\
    python example/v3_int4/gptq_extra_pass.py \\
        /path/to/v3/model.safetensors ./v3_int4/model.safetensors \\
        /path/to/reference.wav --groupsize 32

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch import nn


DEFAULT_TEXTS = [
    "こんにちは、メラだよ。テスト中なの。今日もいい天気だね。",
    "新しい機能を追加したよ。確認してみてね。",
    "ありがとうございます。とても助かりました。",
    "明日の会議は午後三時からの予定です。",
    "雨が降りそうだから傘を持っていったほうがいいよ。",
    "このプロジェクトはもうすぐ完成するの。",
    "おはよう。今日は何から始めようか。",
    "音声合成のテストをしています。聞こえますか。",
]

DEFAULT_PREFIXES = ("text_encoder.", "speaker_encoder.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="original fp32 v3 model.safetensors (fp source)")
    ap.add_argument("packed", help="packed checkpoint to merge into (in place)")
    ap.add_argument("ref_wav")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--prefixes", nargs="*", default=list(DEFAULT_PREFIXES))
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = "cuda:0"

    from irodori_tts.inference_runtime import (
        InferenceRuntime, RuntimeKey, SamplingRequest,
    )
    from onecomp.quantizer.gptq import GPTQ
    from onecomp.quantizer.gptq._gptq import run_gptq, GPTQResult

    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=args.checkpoint, model_device=device,
        codec_device=device, codec_precision="bf16", model_precision="fp32",
    ))
    model = runtime.model

    # Collect target Linears under the requested prefixes.
    targets: dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name.startswith(p) for p in args.prefixes):
            continue
        if mod.in_features % args.groupsize != 0:
            continue  # leave fp16
        if mod.out_features % 8 != 0:
            continue  # AutoGPTQ zero-packing needs out_features divisible by 8
        targets[name] = mod
    print(f"targeting {len(targets)} encoder Linears "
          f"(prefixes={args.prefixes}, gs={args.groupsize})")

    # Accumulate raw S = sum(x^T x) and token count N per layer.
    acc: dict[str, dict] = {n: {"S": None, "N": 0} for n in targets}

    def make_hook(name):
        in_f = targets[name].in_features

        def hook(module, inp, out):
            x = (inp[0] if isinstance(inp, tuple) else inp)
            x = x.reshape(-1, in_f).float()
            a = acc[name]
            s = x.t() @ x
            a["S"] = s if a["S"] is None else a["S"] + s
            a["N"] += x.shape[0]
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in targets.items()]

    for i, text in enumerate(DEFAULT_TEXTS):
        runtime.synthesize(SamplingRequest(
            text=text, num_steps=args.num_steps, seconds=args.seconds,
            seed=args.seed + i, ref_wav=args.ref_wav, no_ref=False,
        ))
        print(f"  [{i+1}/{len(DEFAULT_TEXTS)}] captured: {text[:16]}...")
    for h in handles:
        h.remove()

    gptq = GPTQ(wbits=4, groupsize=args.groupsize, actorder=False, sym=True)

    # Quantize each layer and collect packed buffers + metadata entries.
    new_state: dict[str, torch.Tensor] = {}
    new_entries: list[dict] = []
    skipped: list[str] = []
    for name, module in targets.items():
        a = acc[name]
        if a["S"] is None or a["N"] == 0:
            skipped.append(f"{name} (no activations captured)")
            continue
        H = (2.0 / a["N"]) * a["S"].to(device)
        result_dict = run_gptq(
            H, module, wbits=4, groupsize=args.groupsize,
            actorder=False, sym=True,
        )
        result = GPTQResult(
            wbits=4, groupsize=args.groupsize, actorder=False, sym=True,
            qweight=result_dict["qweight"], scales=result_dict["scales"],
            qzeros=result_dict["qzeros"], perm=result_dict["perm"],
        )
        layer = gptq.create_inference_layer(result, module, pack_weights=True,
                                            use_gemlite=False)
        for k, v in layer.state_dict().items():
            new_state[f"{name}.{k}"] = v.contiguous().cpu()
        new_entries.append({
            "name": name,
            "wbits": 4,
            "groupsize": args.groupsize,
            "actorder": False,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
        })
        print(f"  quantized {name}  (in={module.in_features}, "
              f"out={module.out_features}, N={a['N']})")

    # Merge into the packed checkpoint.
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(args.packed, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    quant_layers = json.loads(meta.get("quant_layers_json", "[]"))
    existing = {e["name"] for e in quant_layers}

    added = 0
    for entry in new_entries:
        name = entry["name"]
        if name in existing:
            continue
        # Drop the fp16 weight (and old bias — re-added from packed layer).
        tensors.pop(f"{name}.weight", None)
        for k in [k for k in tensors if k.startswith(f"{name}.") and k != f"{name}.bias"]:
            if k.endswith(".weight"):
                tensors.pop(k, None)
        quant_layers.append(entry)
        added += 1

    for k, v in new_state.items():
        tensors[k] = v

    meta["quant_layers_json"] = json.dumps(quant_layers, ensure_ascii=False)
    meta.setdefault("quant_method", "autobit")
    meta.setdefault("checkpoint_format", "gptq")

    save_file(tensors, args.packed, metadata=meta)
    size_mb = os.path.getsize(args.packed) / 1024**2
    print(f"\nmerged {added} GPTQ encoder Linears into {args.packed}")
    print(f"total packed Linears now: {len(quant_layers)}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped[:10]:
            print(f"  {s}")
    print(f"output size: {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

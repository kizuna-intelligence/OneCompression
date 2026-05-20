"""RTN 4-bit pass for ``text_encoder.text_embedding`` (the 97 MB vocab table).

Why this exists
---------------
After ``quantize_gptq.py`` (DiT blocks) and ``gptq_extra_pass.py`` (encoder
attention Linears), the single biggest remaining fp16 tensor is the text
embedding ``nn.Embedding(99574, 512)`` — ~100 MB. Unlike a Linear it is never
matmul'd: a forward only *gathers* the rows for the input token ids. So we can
group-wise RTN-pack it to 4-bit on disk and dequant only the gathered rows at
runtime (see ``PackedEmbedding`` in Irodori-TTS-Lite) — saving both disk and
VRAM with negligible compute.

How
---
The embedding weight ``(vocab, dim)`` is treated exactly like a Linear weight
``(out=vocab, in=dim)`` and packed along the ``dim`` axis with
``rtn_pack_4bit_groupwise`` (groupsize divides ``dim``). The packed buffers go
under ``_embed.{name}.{qweight_u8,scales,zeros}`` and a metadata entry is added
to ``embed_quant_layers_json``; the fp16 ``{name}.weight`` is dropped.

Embeddings are *not* GPTQ-able (no activations to build a Hessian over per
row), but blind RTN is fine here: each row is its own token's vector and
group-wise min/max quantization on 512 values per row is accurate. Quality is
verified downstream with ``eval_metrics.py``.

Run (onecomp on PYTHONPATH)::

    python example/v3_int4/quantize_embedding.py \\
        /tmp/v3_int4/model.safetensors --groupsize 32

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Reuse the exact packer the encoder RTN pass uses, so the runtime's existing
# dequant_extra_u8_to_weight inverse applies unchanged.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtn_extra_pass import rtn_pack_4bit_groupwise  # noqa: E402


DEFAULT_NAMES = ("text_encoder.text_embedding",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packed", help="packed checkpoint to merge into (in place)")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--names", nargs="*", default=list(DEFAULT_NAMES),
                    help="embedding module names (without .weight)")
    args = ap.parse_args()

    with safe_open(args.packed, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Drop stale _embed.* from a previous run.
    stale = [k for k in tensors if k.startswith("_embed.")]
    for k in stale:
        del tensors[k]
    if stale:
        print(f"removed {len(stale)} stale _embed.* tensors")

    embed_entries = json.loads(meta.get("embed_quant_layers_json", "[]"))
    already = {e["name"] for e in embed_entries}

    added = 0
    for name in args.names:
        if name in already:
            print(f"  skip {name} (already packed)")
            continue
        wkey = f"{name}.weight"
        if wkey not in tensors:
            print(f"  skip {name}: {wkey} not in checkpoint")
            continue
        w = tensors[wkey]  # (vocab, dim)
        vocab, dim = int(w.shape[0]), int(w.shape[1])
        if dim % args.groupsize != 0:
            print(f"  skip {name}: dim={dim} not divisible by groupsize={args.groupsize}")
            continue

        qweight_u8, scales, zeros = rtn_pack_4bit_groupwise(w, args.groupsize)
        tensors[f"_embed.{name}.qweight_u8"] = qweight_u8
        tensors[f"_embed.{name}.scales"] = scales
        tensors[f"_embed.{name}.zeros"] = zeros
        tensors.pop(wkey, None)
        embed_entries.append({
            "name": name,
            "num_embeddings": vocab,
            "embedding_dim": dim,
            "num_groups": dim // args.groupsize,
        })
        added += 1
        fp16_mb = vocab * dim * 2 / 1024**2
        packed_mb = (qweight_u8.numel() + scales.numel() * 2 + zeros.numel() * 2) / 1024**2
        print(f"  packed {name}  ({vocab}x{dim})  {fp16_mb:.1f} MB -> {packed_mb:.1f} MB")

    meta["embed_quant_layers_json"] = json.dumps(embed_entries, ensure_ascii=False)
    meta.setdefault("quant_method", "autobit")
    meta.setdefault("checkpoint_format", "gptq")

    save_file(tensors, args.packed, metadata=meta)
    size_mb = os.path.getsize(args.packed) / 1024**2
    print(f"\nmerged {added} embeddings into {args.packed}")
    print(f"output size: {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

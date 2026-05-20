"""Generate (with reference voice), time and write wavs for eager vs packed RTN.

Loads the quantized v3 checkpoint via the Irodori-TTS-Lite runtime and
synthesizes a fixed sentence, once with eager dequant and once with the packed
RTN-extras path, writing ``<prefix>_eager.wav`` / ``<prefix>_packed.wav`` plus
median latency and peak VRAM for each.

Usage::

    REF_WAV=/home/yusuke/.claude/mera3.wav MODE=both \\
    python example/v3_int4/generate_compare.py <ckpt.safetensors> <out_prefix>

``ckpt`` must be the safetensors FILE, not its directory. The audio length is
chosen by the model's duration predictor (no manual ``seconds``); for a
checkpoint without one (v2/moespeech) set ``DURATION_DONOR`` to graft v3's.
Env: MODE=eager|packed|both, REF_WAV=path/to/wav (omit for no_ref synthesis),
DURATION_DONOR=path/to/v3.safetensors (optional duration-predictor donor).

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import os
import sys
import time

import torch


def _mem(label: str) -> None:
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1024**2
    pa = torch.cuda.max_memory_allocated() / 1024**2
    pr = torch.cuda.max_memory_reserved() / 1024**2
    print(f"  {label:30s}  alloc={a:7.1f}  peak_alloc={pa:7.1f}  peak_res={pr:7.1f}")


def run_one(packed: bool, ckpt: str, out_wav: str, ref_wav: str | None) -> None:
    label = "PACKED" if packed else "EAGER "
    print(f"\n========== {label} ==========")
    import irodori_tts_lite
    donor = os.environ.get("DURATION_DONOR") or None
    irodori_tts_lite.configure(
        use_fused=True, force_fp16=True, pack_rtn_extras=packed,
        duration_donor=donor,
    )
    irodori_tts_lite.patch()

    from irodori_tts.inference_runtime import (
        InferenceRuntime, RuntimeKey, SamplingRequest,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _mem("baseline (CUDA ctx)")

    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt, model_device="cuda:0",
        codec_device="cuda:0", codec_precision="bf16",
        model_precision="fp32",
    ))
    _mem("after load")

    # No manual `seconds`: let the (v3) duration predictor decide the length.
    req_kwargs = dict(num_steps=6, seed=42)
    if ref_wav:
        req_kwargs["ref_wav"] = ref_wav
        req_kwargs["no_ref"] = False
    else:
        req_kwargs["no_ref"] = True

    # warmup
    runtime.synthesize(SamplingRequest(text="メラ", **req_kwargs))

    # measure 3 inferences, take median latency
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runtime.synthesize(SamplingRequest(
            text="こんにちは、メラだよ。テスト中なの。", **req_kwargs,
        ))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    _mem("after 3 synth")
    print(f"  median latency 3s synth: {times[1]:.0f} ms  "
          f"(times: {[f'{t:.0f}' for t in times]})")

    # final big run -> write wav
    res = runtime.synthesize(SamplingRequest(
        text="こんにちは、メラだよ。テスト中なの。今日もいい天気だね。",
        num_steps=6, seed=12345,
        **({"ref_wav": ref_wav, "no_ref": False} if ref_wav else {"no_ref": True}),
    ))
    audio = res.audio.detach().cpu()
    sr = int(res.sample_rate) if hasattr(res, "sample_rate") else 48000
    import torchaudio
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 3:
        audio = audio[0]
    torchaudio.save(out_wav, audio, sr)
    print(f"  wrote {out_wav}  ({audio.shape[-1]/sr:.2f} s @ {sr} Hz)")


def main() -> int:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ckpt = sys.argv[1]
    prefix = sys.argv[2]
    print(f"checkpoint: {ckpt}")
    ref_wav = os.environ.get("REF_WAV") or None
    if ref_wav:
        print(f"reference: {ref_wav}")
    else:
        print("reference: (none, no_ref=True)")
    mode = os.environ.get("MODE", "both")
    if mode in ("eager", "both"):
        run_one(False, ckpt, f"{prefix}_eager.wav", ref_wav)
    if mode in ("packed", "both"):
        run_one(True, ckpt, f"{prefix}_packed.wav", ref_wav)
    return 0


if __name__ == "__main__":
    sys.exit(main())

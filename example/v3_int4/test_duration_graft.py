"""Test: graft the v3 duration predictor onto the v2 (moespeech) checkpoint.

v2/moespeech has no duration predictor (``use_duration_predictor=False``), so
synthesis must either be given a manual ``seconds`` or falls back to 30 s. A
fixed manual duration garbles pacing (CER ~25 % when 4 s is forced on a longer
utterance). v3 ships a trained duration predictor and — crucially — v2 and v3
share identical encoder dims (text_dim=512, speaker_dim=768, same tokenizer), so
v3's predictor can be grafted onto v2 via ``configure(duration_donor=...)``.

This script synthesizes the same sentence with the grafted predictor (no manual
``seconds``) and writes the wav so CER/similarity can be measured. Compare
against the manual-duration baseline (``out_v2_check_packed.wav``).

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=<onecomp>:<lite>:<irodori-streaming> \\
    python example/v3_int4/test_duration_graft.py \\
        /tmp/moespeech_ft_5000_int4.safetensors \\
        /tmp/v3_int4/model.safetensors \\
        /tmp/out_v2_graft.wav

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    ckpt = sys.argv[1]          # checkpoint lacking a duration predictor (v2)
    donor = sys.argv[2]         # donor checkpoint with one (v3)
    out_wav = sys.argv[3]
    ref_wav = os.environ.get("REF_WAV", "/home/yusuke/.claude/mera3.wav")
    text = os.environ.get(
        "TEXT", "こんにちは、メラだよ。テスト中なの。今日もいい天気だね。"
    )

    import irodori_tts_lite
    irodori_tts_lite.configure(
        use_fused=True, force_fp16=True, pack_rtn_extras=True,
        duration_donor=donor,
    )
    irodori_tts_lite.patch()

    from irodori_tts.inference_runtime import (
        InferenceRuntime, RuntimeKey, SamplingRequest,
    )

    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=ckpt, model_device="cuda:0",
        codec_device="cuda:0", codec_precision="bf16", model_precision="fp32",
    ))
    print(f"use_duration_predictor = {runtime.model_cfg.use_duration_predictor}")
    print(f"model.duration_predictor = {type(runtime.model.duration_predictor).__name__}")

    # NOTE: no `seconds=` -> the (grafted) duration predictor decides the length.
    res = runtime.synthesize(SamplingRequest(
        text=text, num_steps=8, seed=12345,
        ref_wav=ref_wav, no_ref=False,
    ))
    audio = res.audio.detach().cpu()
    sr = int(getattr(res, "sample_rate", 48000))
    import torchaudio
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 3:
        audio = audio[0]
    torchaudio.save(out_wav, audio, sr)
    print(f"wrote {out_wav}  ({audio.shape[-1]/sr:.2f} s @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

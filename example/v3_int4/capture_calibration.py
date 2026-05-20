"""Capture REAL DiT calibration inputs from genuine Irodori-TTS syntheses.

Why this exists
---------------
``DiTAdapter.prepare_calibration_inputs`` defaults to *random Gaussian*
tensors.  GPTQ estimates each Linear's Hessian from those activations, so
random calibration points the error-correction at a distribution the model
never sees.  On Irodori-TTS-500M-v3 that left speech badly degraded
(CER ~33% vs FP32 ~8%).

This script runs real syntheses and records the exact tensors that flow into
``TextToLatentRFDiT.forward_with_encoded_conditions`` at every rectified-flow
step: the noisy latent ``x_t``, the timestep ``t``, and the (already-encoded,
un-quantised) ``text_state`` / ``speaker_state`` conditioning.  Feeding these
back as calibration (via ``DiTAdapter(calibration_inputs_path=...)``) makes the
DiT blocks see exactly the activation statistics of real generation.

Run (inside an irodori_tts-capable venv, e.g. Irodori-streaming/.venv)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python example/v3_int4/capture_calibration.py \\
        /path/to/v3/model.safetensors \\
        /path/to/reference.wav \\
        ./v3_calib.pt

All captured ``x_t`` share one sequence length (fixed ``--seconds``) so the
QEP block-input catcher can ``torch.cat`` them.  ``text_state`` /
``speaker_state`` are zero-padded to a common length (padding is masked and,
in the block loop, only sample-0's conditioning kwargs are reused anyway).

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


# A spread of short/medium Japanese utterances so the captured latent
# trajectory covers varied phonetic content.
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


def _pad_states(states: list[torch.Tensor], masks: list[torch.Tensor]):
    """Zero-pad a list of (1, L_i, D) states to common max L; extend masks."""
    if states[0] is None:
        return None, None
    max_len = max(s.shape[1] for s in states)
    dim = states[0].shape[2]
    out_s, out_m = [], []
    for s, m in zip(states, masks):
        L = s.shape[1]
        if L < max_len:
            pad_s = torch.zeros(s.shape[0], max_len - L, dim, dtype=s.dtype)
            s = torch.cat([s, pad_s], dim=1)
            pad_m = torch.zeros(m.shape[0], max_len - L, dtype=m.dtype)
            m = torch.cat([m, pad_m], dim=1)
        out_s.append(s)
        out_m.append(m)
    return torch.cat(out_s, 0), torch.cat(out_m, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("ref_wav")
    ap.add_argument("output", help="path to write the .pt calibration payload")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--texts-file", default=None,
                    help="optional file, one utterance per line")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from irodori_tts.inference_runtime import (
        InferenceRuntime, RuntimeKey, SamplingRequest,
    )

    if args.texts_file:
        with open(args.texts_file, encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    else:
        texts = DEFAULT_TEXTS

    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=args.checkpoint, model_device="cuda:0",
        codec_device="cuda:0", codec_precision="bf16", model_precision="fp32",
    ))
    print(f"loaded; capturing from {len(texts)} utterances "
          f"x {args.num_steps} steps")

    captured: list[dict] = []
    orig_fwd = runtime.model.forward_with_encoded_conditions

    def hook(x_t, t, text_state, text_mask, speaker_state, speaker_mask,
             caption_state=None, caption_mask=None, **kw):
        # Keep only batch row 0 (drops the CFG-uncond duplicate if present).
        def r0(v):
            return None if v is None else v[:1].detach().to("cpu", torch.float32)

        def r0b(v):
            return None if v is None else v[:1].detach().cpu()

        captured.append(dict(
            x_t=r0(x_t), t=r0(t),
            text_state=r0(text_state), text_mask=r0b(text_mask),
            speaker_state=r0(speaker_state), speaker_mask=r0b(speaker_mask),
        ))
        return orig_fwd(x_t, t, text_state, text_mask, speaker_state,
                        speaker_mask, caption_state, caption_mask, **kw)

    runtime.model.forward_with_encoded_conditions = hook

    for i, text in enumerate(texts):
        runtime.synthesize(SamplingRequest(
            text=text, num_steps=args.num_steps, seconds=args.seconds,
            seed=args.seed + i, ref_wav=args.ref_wav, no_ref=False,
        ))
        print(f"  [{i+1}/{len(texts)}] captured "
              f"{len(captured)} steps so far: {text[:18]}...")

    runtime.model.forward_with_encoded_conditions = orig_fwd

    # All x_t share the same (1, S, D) — fixed seconds. Stack on dim 0.
    x_t = torch.cat([c["x_t"] for c in captured], 0)
    t = torch.cat([c["t"] for c in captured], 0)
    text_state, text_mask = _pad_states(
        [c["text_state"] for c in captured],
        [c["text_mask"] for c in captured],
    )
    speaker_state, speaker_mask = _pad_states(
        [c["speaker_state"] for c in captured],
        [c["speaker_mask"] for c in captured],
    )

    payload = {
        "x_t": x_t, "t": t,
        "text_state": text_state, "text_mask": text_mask,
        "speaker_state": speaker_state, "speaker_mask": speaker_mask,
    }
    torch.save(payload, args.output)
    print(f"wrote {args.output}: x_t={tuple(x_t.shape)}, "
          f"text_state={None if text_state is None else tuple(text_state.shape)}, "
          f"speaker_state={None if speaker_state is None else tuple(speaker_state.shape)}, "
          f"samples={x_t.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

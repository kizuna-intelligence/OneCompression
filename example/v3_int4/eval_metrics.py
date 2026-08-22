"""Measure CER (faster-whisper large-v3) and speaker similarity (OpenVoice v2).

Usage::

    python example/v3_int4/eval_metrics.py <ref_wav> <gen_wav> <expected_text>

Prints:
  - cosine similarity between OpenVoice-v2 speaker embeddings of ref vs gen
    (higher is better; >0.7 typically OK)
  - CER of the generated audio against ``expected_text`` (lower is better)

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import os
import sys

import soundfile as sf
import torch
import librosa


OPENVOICE_DIR = "/home/yusuke/gitrepos/seed-vc/modules/openvoice"
OPENVOICE_CKPT = f"{OPENVOICE_DIR}/checkpoints_v2/converter/checkpoint.pth"
OPENVOICE_CFG = f"{OPENVOICE_DIR}/checkpoints_v2/converter/config.json"

# Make ``from openvoice.api import ...`` resolve to seed-vc's local copy.
sys.path.insert(0, "/home/yusuke/gitrepos/seed-vc/modules")
from openvoice.api import ToneColorConverter


def _load_wav_mono(path: str, sr: int = 22050) -> torch.Tensor:
    audio, native_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if native_sr != sr:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
    return torch.tensor(audio)


def extract_se(converter: ToneColorConverter, wav_path: str) -> torch.Tensor:
    target_sr = int(converter.hps.data.sampling_rate)
    wav = _load_wav_mono(wav_path, sr=target_sr)
    lengths = torch.tensor([wav.shape[0]], dtype=torch.long)
    se = converter.extract_se([wav], lengths)
    return se[0].squeeze().detach().cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item())


def main() -> int:
    ref_wav = sys.argv[1]
    gen_wav = sys.argv[2]
    expected_text = sys.argv[3]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("== speaker similarity (OpenVoice v2 ref_enc) ==")
    converter = ToneColorConverter(OPENVOICE_CFG, device=device)
    converter.load_ckpt(OPENVOICE_CKPT)
    ref_se = extract_se(converter, ref_wav)
    gen_se = extract_se(converter, gen_wav)
    sim = cosine_sim(ref_se, gen_se)
    print(f"  ref:  {ref_wav}")
    print(f"  gen:  {gen_wav}")
    print(f"  cosine similarity = {sim:.4f}  (higher is better; >0.7 typically OK)")

    print()
    print("== CER (faster-whisper large-v3, ja) ==")
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device="cuda" if device.startswith("cuda") else "cpu",
                       compute_type="float16")
    segments, _info = asr.transcribe(gen_wav, language="ja", beam_size=5)
    text = "".join(s.text for s in segments).strip()
    print(f"  expected:    {expected_text}")
    print(f"  transcribed: {text}")
    import jiwer
    ref_norm = expected_text.replace(" ", "").replace("、", "").replace("。", "")
    hyp_norm = text.replace(" ", "").replace("、", "").replace("。", "")
    cer = jiwer.cer(ref_norm, hyp_norm)
    print(f"  CER = {cer:.4f}  ({cer*100:.2f}%; lower is better)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

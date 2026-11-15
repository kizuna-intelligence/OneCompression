"""Decode MioTTS codec tokens -> 24kHz wav via MioCodec.

Reads the JSON produced by ``vllm_generate.py`` (or the bf16 baseline) and
writes one wav per utterance using a speaker preset's global embedding.

Run with the interpreter that has miocodec installed (.venv-flux).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from miocodec import MioCodecModel

CODEC_ID = "Aratako/MioCodec-25Hz-24kHz"
PRESET_BY_LANG = {"ja": "jp_female.pt", "en": "en_female.pt"}


def _prepare_embedding(emb: torch.Tensor, device) -> torch.Tensor:
    emb = emb.squeeze()
    if emb.dim() != 1:
        emb = emb.flatten()
    return emb.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="JSON from vllm_generate.py")
    ap.add_argument("--presets", required=True, help="dir with *.pt presets")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="int4")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codec = MioCodecModel.from_pretrained(CODEC_ID).eval().to(args.device)
    sr = int(codec.config.sample_rate)
    print(f"[decode] MioCodec loaded, sample_rate={sr}", flush=True)

    import soundfile as sf
    data = json.loads(Path(args.gen).read_text())
    presets_dir = Path(args.presets)
    emb_cache: dict[str, torch.Tensor] = {}

    for i, r in enumerate(data["results"]):
        lang = r["lang"]
        toks = r["tokens"]
        if not toks:
            print(f"[decode] {i} {lang}: no tokens, skip", flush=True)
            continue
        preset = PRESET_BY_LANG.get(lang, "jp_female.pt")
        if preset not in emb_cache:
            raw = torch.load(presets_dir / preset, map_location="cpu",
                             weights_only=True)
            emb_cache[preset] = _prepare_embedding(raw, args.device)
        emb = emb_cache[preset]
        tok_t = torch.tensor(toks, dtype=torch.long, device=args.device)
        with torch.no_grad():
            wav = codec.decode(global_embedding=emb,
                               content_token_indices=tok_t)
        wav = wav.squeeze().float().cpu().numpy()
        path = out_dir / f"{args.tag}_{i:02d}_{lang}.wav"
        sf.write(str(path), wav, sr)
        dur = len(wav) / sr
        print(f"[decode] {i} {lang}: {len(toks)} toks -> {path.name} "
              f"({dur:.2f}s)", flush=True)

    print(f"[decode] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

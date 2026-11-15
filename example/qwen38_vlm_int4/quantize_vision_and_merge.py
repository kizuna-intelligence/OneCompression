#!/usr/bin/env python3
"""Add an RTN-int4 vision encoder to the existing Qwen3.8 text checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from onecomp.vlm_checkpoint import build_qwen38_vlm_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--text-model", default="kizuna-intelligence/Qwen3.8-27B-int4-qep")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--vision-bits", type=int, default=4)
    parser.add_argument("--maximum-group-size", type=int, default=128)
    args = parser.parse_args()

    text_snapshot = snapshot_download(
        args.text_model,
        cache_dir=args.cache_dir,
        allow_patterns=[
            "model.safetensors",
            "config.json",
            "*.jinja",
            "tokenizer*",
            "preprocessor_config.json",
            "generation_config.json",
        ],
    )
    base_snapshot = snapshot_download(
        args.base_model,
        cache_dir=args.cache_dir,
        allow_patterns=[
            "model-00001-of-00018.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "*.jinja",
            "tokenizer*",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
            "LICENSE",
        ],
    )
    api = HfApi()
    provenance = {
        "base_model": args.base_model,
        "base_revision": api.model_info(args.base_model).sha,
        "text_model": args.text_model,
        "text_revision": api.model_info(args.text_model).sha,
    }
    output = build_qwen38_vlm_checkpoint(
        base_snapshot=base_snapshot,
        text_snapshot=text_snapshot,
        output_dir=args.output,
        vision_bits=args.vision_bits,
        maximum_group_size=args.maximum_group_size,
        provenance=provenance,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit and run Japanese generation from a saved fixed-int4 checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from onecomp import load_quantized_model

from example.gemma4_e2b_int4.common import audit_fixed_int4_config


PROMPTS = (
    "今日ちょっと疲れたから、優しく一言だけ励まして。",
    "雨の日に家で楽しめることを二つ、短く教えて。",
    "おはよう。眠そうだね、と言われたときの自然な返事をして。",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    config_path = args.checkpoint / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        bit_counts = audit_fixed_int4_config(json.load(handle))
    print("fixed_int4_audit=" + json.dumps(bit_counts, sort_keys=True))

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model, tokenizer = load_quantized_model(
        str(args.checkpoint),
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    load_seconds = time.perf_counter() - load_started
    device = next(model.parameters()).device

    results = []
    for prompt in PROMPTS:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        new_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        result = {
            "prompt": prompt,
            "response": text.strip(),
            "new_tokens": new_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": new_tokens / elapsed,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    summary = {
        "load_seconds": load_seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "results": results,
    }
    print("verification_summary=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

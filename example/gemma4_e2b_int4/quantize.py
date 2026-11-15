#!/usr/bin/env python3
"""Quantize the Gemma 4 E2B language decoder to fixed GPTQ 4-bit + QEP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner, setup_logger
from onecomp.qep import QEPConfig

from example.gemma4_e2b_int4.common import (
    CALIBRATION_DATASET_ID,
    GROUP_SIZE,
    MODEL_ID,
    WBITS,
    Gemma4TextAdapter,
    audit_fixed_int4_config,
    prepare_japanese_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path)
    parser.add_argument("--calibration-dataset", default=CALIBRATION_DATASET_ID)
    parser.add_argument("--prepare-rows", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1 or args.max_length < 1 or args.prepare_rows < 1:
        raise ValueError("sample counts and max length must be positive")
    if args.prepare_rows < args.num_samples:
        raise ValueError("--prepare-rows must be >= --num-samples")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output = args.output.resolve()
    calibration_path = (
        args.calibration_jsonl.resolve()
        if args.calibration_jsonl
        else output.parent / "gemma4_e2b_japanese_calibration.jsonl"
    )
    run_config = {
        "model_id": args.model_id,
        "output": str(output),
        "calibration_jsonl": str(calibration_path),
        "calibration_dataset": args.calibration_dataset,
        "prepare_rows": args.prepare_rows,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": args.device,
        "wbits": WBITS,
        "group_size": GROUP_SIZE,
        "qep": True,
    }
    print(json.dumps(run_config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    setup_logger()
    # An implicit path belongs to this run configuration, so rebuild it on
    # every run.  Otherwise a preceding smoke run can silently leave a smaller
    # corpus that fails or changes a later production calibration.
    if args.calibration_jsonl is None or not calibration_path.is_file():
        prepare_japanese_calibration(
            calibration_path,
            model_id=args.model_id,
            dataset_id=args.calibration_dataset,
            num_rows=args.prepare_rows,
            seed=args.seed,
        )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter = Gemma4TextAdapter(
        model_id=args.model_id,
        dtype="bfloat16",
        device=args.device,
    )
    model_config = ModelConfig(adapter=adapter)
    quantizer = GPTQ(wbits=WBITS, groupsize=GROUP_SIZE)
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        qep=True,
        qep_config=QEPConfig(general=False, device=args.device),
        calibration_config=CalibrationConfig(
            calibration_dataset=str(calibration_path),
            text_key="text",
            max_length=args.max_length,
            num_calibration_samples=args.num_samples,
            strategy="drop_rand",
            seed=args.seed,
        ),
    )
    runner.run()
    runner.save_quantized_model(str(output))

    config_path = output / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        bit_counts = audit_fixed_int4_config(json.load(handle))
    summary = {
        **run_config,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "quantized_layer_bits": bit_counts,
        "quantized_result_count": len(quantizer.results),
    }
    with (output / "onecomp_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

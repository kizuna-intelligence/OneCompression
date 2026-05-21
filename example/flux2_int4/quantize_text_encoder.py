"""GPTQ 4-bit quantisation of the FLUX.2-klein Qwen3 text encoder.

The FLUX.2-klein-4B pipeline uses a Qwen3ForCausalLM (~8B, hidden 2560,
36 layers) as its prompt encoder. It is the single largest component of the
pipeline in bf16 (~8 GB). Quantising it to int4 (groupsize 32, symmetric — the
GemLite-compatible packing the DiT runtime already uses) brings it to ~2.5 GB.

The text encoder folder ships no tokenizer, so point ``--src`` at a directory
that combines the text_encoder weights with the pipeline tokenizer files (see
flux2_te_src created alongside this script).

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    .venv-flux/bin/python example/flux2_int4/quantize_text_encoder.py \\
        ./flux2_te_src ./flux2_te_int4 --groupsize 32 --num-samples 128

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner, setup_logger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="dir with Qwen3 text-encoder weights + tokenizer")
    ap.add_argument("save_dir")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="calibration forward chunk size (avoids lm_head OOM)")
    ap.add_argument("--asym", action="store_true")
    args = ap.parse_args()

    setup_logger()

    model_config = ModelConfig(model_id=args.src, device="cuda:0")
    gptq = GPTQ(wbits=4, groupsize=args.groupsize, sym=not args.asym)
    calib = CalibrationConfig(
        max_length=args.max_length,
        num_calibration_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    runner = Runner(
        model_config=model_config,
        quantizer=gptq,
        calibration_config=calib,
        qep=False,
    )
    runner.run()
    runner.save_quantized_model(args.save_dir)
    print(f"saved quantized Qwen3 text encoder to {args.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

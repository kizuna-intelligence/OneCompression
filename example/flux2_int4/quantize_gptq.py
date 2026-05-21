"""GPTQ 4-bit quantisation of the FLUX.2-klein-4B single-stream DiT blocks.

FLUX.2 is a two-stream flow-matching image transformer.  Only the
``single_transformer_blocks`` (single residual stream) fit OneCompression's
block-input catcher, so this script quantises those — the bulk of the model
(20 of 25 blocks here).  The double-stream ``transformer_blocks`` and the
embedders / output projection are left in their loaded dtype.

Synthetic calibration draws the timestep from a logit-normal prior (the
flow-matching default); pass ``--calib`` to use real captured activations
instead, which generally calibrate GPTQ's Hessian far better.

Run (inside the FLUX venv, e.g. ``.venv-flux``)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    .venv-flux/bin/python example/flux2_int4/quantize_gptq.py \\
        black-forest-labs/FLUX.2-klein-4B ./flux2_int4 \\
        --groupsize 32 --num-samples 64

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.adapters import Flux2DiTAdapter
from onecomp.quantizer.gptq import GPTQ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="HF id or local path with a transformer/ subfolder")
    ap.add_argument("save_dir")
    ap.add_argument("--calib", default=None,
                    help="real-activation calibration .pt (optional)")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--actorder", action="store_true")
    ap.add_argument("--mse", action="store_true")
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--quant-all", action="store_true",
                    help="quantize every Linear (empty exclude set), including "
                         "embedders / modulation / proj_out")
    ap.add_argument("--image-grid", type=int, default=32,
                    help="synthetic latent side length (grid*grid image tokens)")
    ap.add_argument("--text-seq-len", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    setup_logger()

    adapter = Flux2DiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device="cuda:0",
        calibration_inputs_path=args.calib,
        image_grid=args.image_grid,
        text_seq_len=args.text_seq_len,
    )
    model_config = ModelConfig(adapter=adapter)

    exclude = [] if args.quant_all else Flux2DiTAdapter.default_exclude_layer_keywords()
    gptq = GPTQ(
        wbits=4,
        groupsize=args.groupsize,
        actorder=args.actorder,
        mse=args.mse,
        sym=not args.asym,
        exclude_layer_keywords=exclude,
    )

    calib = CalibrationConfig(
        num_calibration_samples=args.num_samples,
        max_length=256,
    )
    runner = Runner(
        model_config=model_config,
        quantizer=gptq,
        calibration_config=calib,
        qep=False,
    )
    runner.run()
    adapter.save_quantized_model(runner, args.save_dir)
    print(f"saved quantized FLUX.2 transformer to {args.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

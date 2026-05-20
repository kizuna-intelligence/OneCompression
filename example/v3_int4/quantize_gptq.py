"""GPTQ 4-bit quantisation of the Irodori-TTS-500M-v3 DiT blocks.

Quantises only ``model.blocks`` (the rectified-flow DiT transformer blocks).
Encoders / AdaLN / duration_predictor are left in fp16 — on v3 they do not
survive blind low-bit quantisation (see ``rtn_extra_pass.py`` notes).

Key knob: ``--calib`` points at a real-activation calibration file produced by
``capture_calibration.py``.  Without it, ``DiTAdapter`` falls back to random
Gaussian calibration, which on v3 degrades speech intelligibility
(CER ~33% vs FP32 ~8%).  ``--actorder`` enables activation-order reordering,
which only helps when the calibration activations are realistic.

Run (inside an irodori_tts-capable venv)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python example/v3_int4/quantize_gptq.py \\
        /path/to/v3/model.safetensors ./v3_int4 \\
        --calib ./v3_calib.pt --actorder --groupsize 32

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.adapters import DiTAdapter
from onecomp.quantizer.gptq import GPTQ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("save_dir")
    ap.add_argument("--calib", default=None,
                    help="real-activation calibration .pt (capture_calibration.py)")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--actorder", action="store_true")
    ap.add_argument("--mse", action="store_true",
                    help="MSE grid-search for scale/zero (slower, more accurate)")
    ap.add_argument("--asym", action="store_true",
                    help="asymmetric quantization (sym=False)")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    setup_logger()

    adapter = DiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype="float32",
        device="cuda:0",
        calibration_inputs_path=args.calib,
    )
    model_config = ModelConfig(adapter=adapter)

    exclude = DiTAdapter.default_exclude_layer_keywords() + ["duration_predictor"]
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
        max_length=args.max_length,
    )
    runner = Runner(
        model_config=model_config,
        quantizer=gptq,
        calibration_config=calib,
        qep=True,
    )
    runner.run()
    adapter.save_quantized_model(runner, args.save_dir)
    print(f"saved quantized DiT to {args.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

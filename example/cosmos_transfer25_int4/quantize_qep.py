"""QEP/GPTQ quantisation for NVIDIA Cosmos-Transfer2.5-2B.

This targets the base ``CosmosTransformer3DModel`` from the diffusers
``diffusers/general`` revision of ``nvidia/Cosmos-Transfer2.5-2B``.  The model
is kept on CPU and QEP streams one transformer block at a time to the GPU, so
calibration avoids holding the full bf16 denoiser in VRAM.

The HF repository is gated; accept the NVIDIA license on Hugging Face and log in
before running.

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    .venv-flux/bin/python example/cosmos_transfer25_int4/quantize_qep.py \\
        nvidia/Cosmos-Transfer2.5-2B /path/to/cosmos_transfer25_qep_int4 \\
        --wbits 4 --groupsize 32 --num-samples 8

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.adapters import CosmosTransferDiTAdapter
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="HF repo/local dir, or a local single-file checkpoint")
    ap.add_argument("save_dir")
    ap.add_argument("--revision", default="diffusers/general")
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--single-file", action="store_true",
                    help="load CHECKPOINT through CosmosTransformer3DModel.from_single_file")
    ap.add_argument("--wbits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--latent-frames", type=int, default=3)
    ap.add_argument("--latent-height", type=int, default=16)
    ap.add_argument("--latent-width", type=int, default=16)
    ap.add_argument("--text-seq-len", type=int, default=512)
    ap.add_argument("--calibration-inputs", default=None,
                    help="torch file captured by capture_calibration.py; "
                         "uses real pipeline transformer inputs instead of synthetic")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keep-adaln-fp", action="store_true",
                    help="leave per-block norm1/norm2/norm3 AdaLN projection "
                         "Linears in bf16; lower risk, larger checkpoint")
    args = ap.parse_args()

    setup_logger()

    adapter = CosmosTransferDiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
        revision=args.revision,
        subfolder=args.subfolder,
        single_file=args.single_file,
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        text_seq_len=args.text_seq_len,
        calibration_inputs_path=args.calibration_inputs,
    )
    model_config = ModelConfig(adapter=adapter)

    exclude = ["norm1", "norm2", "norm3"] if args.keep_adaln_fp else []
    gptq = GPTQ(
        wbits=args.wbits,
        groupsize=args.groupsize,
        sym=True,
        include_layer_keywords=["transformer_blocks"],
        exclude_layer_keywords=exclude,
    )
    calib = CalibrationConfig(num_calibration_samples=args.num_samples, max_length=256)
    qep_config = QEPConfig(general=False, device=args.device, exclude_layer_keywords=[])

    runner = Runner(
        model_config=model_config,
        quantizer=gptq,
        calibration_config=calib,
        qep=True,
        qep_config=qep_config,
    )
    runner.run()

    out = adapter.save_quantized_model(runner, args.save_dir)
    print(f"saved QEP-quantized Cosmos Transfer2.5 transformer to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

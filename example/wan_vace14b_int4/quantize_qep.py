"""QEP/GPTQ int4 quantisation for Wan2.1 VACE 14B.

Main transformer blocks are quantized with Fujitsu QEP + GPTQ.  VACE hint
blocks are packed with int4 RTN during save so the whole 14B transformer can be
loaded by ``onecomp-runtime`` without expanding bf16 weights.

Run::

    CUDA_VISIBLE_DEVICES=0 .venv-flux/bin/python \
      example/wan_vace14b_int4/quantize_qep.py \
      Wan-AI/Wan2.1-VACE-14B-diffusers ./wan_vace14b_qep_int4 \
      --groupsize 32 --num-samples 4

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.adapters import WanVACE14BDiTAdapter
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="HF repo/local dir containing the Wan VACE transformer")
    ap.add_argument("save_dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--latent-frames", type=int, default=3)
    ap.add_argument("--latent-height", type=int, default=16)
    ap.add_argument("--latent-width", type=int, default=16)
    ap.add_argument("--text-seq-len", type=int, default=512)
    ap.add_argument("--calibration-inputs", default=None,
                    help="torch file with real transformer inputs captured from WanVACEPipeline")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-vace-rtn", action="store_true",
                    help="leave vace_blocks in bf16 instead of packed RTN int4")
    args = ap.parse_args()

    setup_logger()
    adapter = WanVACE14BDiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
        revision=args.revision,
        subfolder=args.subfolder,
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        text_seq_len=args.text_seq_len,
        calibration_inputs_path=args.calibration_inputs,
        quantize_vace_rtn=not args.no_vace_rtn,
    )
    model_config = ModelConfig(adapter=adapter)
    gptq = GPTQ(
        wbits=4,
        groupsize=args.groupsize,
        sym=True,
        include_layer_keywords=["blocks"],
        exclude_layer_keywords=["vace_blocks"],
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
    print(f"saved QEP-quantized Wan VACE transformer to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

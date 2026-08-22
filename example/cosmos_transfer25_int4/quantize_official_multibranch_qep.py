"""QEP/GPTQ quantization for NVIDIA official Cosmos Transfer2.5 multibranch.

This is for the reference ``cosmos-transfer2.5`` multibranch model, not the
diffusers ``CosmosTransformer3DModel`` checkpoint.  Main DiT blocks are
quantized with OneCompression QEP/GPTQ; control-side Linears are packed with
OneCompression RTN during save.
"""
from __future__ import annotations

import argparse
import os
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.adapters import CosmosOfficialMultibranchAdapter
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save_dir")
    ap.add_argument("--cosmos-repo", default="/home/yusuke/gitrepos/cosmos-transfer2.5")
    ap.add_argument("--tmp-output-dir", default="/mnt/hojo/cosmos_transfer25_official_multibranch_qep_tmp")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wbits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--latent-frames", type=int, default=5)
    ap.add_argument("--latent-height", type=int, default=16)
    ap.add_argument("--latent-width", type=int, default=16)
    ap.add_argument("--text-seq-len", type=int, default=512)
    ap.add_argument("--calibration-inputs", default=None)
    ap.add_argument("--no-control-rtn", action="store_true")
    ap.add_argument("--rtn-groupsize", type=int, default=32)
    ap.add_argument(
        "--include-layer-keyword",
        action="append",
        dest="include_layer_keywords",
        default=None,
        help="Layer keyword to include for GPTQ. Repeatable; default is blocks.",
    )
    ap.add_argument(
        "--exclude-layer-keyword",
        action="append",
        dest="exclude_layer_keywords",
        default=None,
        help="Layer keyword to exclude from GPTQ. Repeatable.",
    )
    ap.add_argument("--qep-general", action="store_true", help="Use generic QEP capture; slower fallback.")
    args = ap.parse_args()

    setup_logger()
    os.environ.setdefault("COSMOS_TEXT_ENCODER_DEVICE", "cpu")
    os.environ.setdefault("COSMOS_OFFLOAD_NET_DURING_CONDITION", "0")

    adapter = CosmosOfficialMultibranchAdapter(
        dtype=args.dtype,
        device=args.device,
        cosmos_repo=args.cosmos_repo,
        output_dir=args.tmp_output_dir,
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        text_seq_len=args.text_seq_len,
        calibration_inputs_path=args.calibration_inputs,
        quantize_control_rtn=not args.no_control_rtn,
        rtn_groupsize=args.rtn_groupsize,
    )
    model_config = ModelConfig(adapter=adapter)
    gptq = GPTQ(
        wbits=args.wbits,
        groupsize=args.groupsize,
        sym=True,
        include_layer_keywords=args.include_layer_keywords or ["blocks."],
        exclude_layer_keywords=args.exclude_layer_keywords or [],
    )
    calib = CalibrationConfig(num_calibration_samples=args.num_samples, max_length=256)
    qep_config = QEPConfig(
        general=args.qep_general,
        device=args.device,
        exclude_layer_keywords=args.exclude_layer_keywords or [],
    )

    runner = Runner(
        model_config=model_config,
        quantizer=gptq,
        calibration_config=calib,
        qep=True,
        qep_config=qep_config,
    )
    runner.run()
    out = adapter.save_quantized_model(runner, args.save_dir)
    print(f"saved official multibranch QEP/GPTQ checkpoint to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

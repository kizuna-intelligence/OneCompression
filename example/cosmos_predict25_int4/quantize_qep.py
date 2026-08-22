"""QEP/GPTQ quantization for NVIDIA official Cosmos-Predict2.5 2B DiT.

This quantizes the official Predict2.5 ``model.net`` used by the local
``cosmos-transfer2.5`` checkout.  The output is a OneCompression packed-int4
``model.safetensors`` intended to be loaded into the official inference stack
by ``onecomp_runtime.models.cosmos_predict25``.
"""
from __future__ import annotations

import argparse
import os
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.adapters import CosmosPredictDiTAdapter
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ


DEFAULT_EXPERIMENT = (
    "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-"
    "Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Cosmos Predict checkpoint dir/UUID/local path")
    ap.add_argument("save_dir")
    ap.add_argument(
        "--cosmos-repo",
        default=os.environ.get("COSMOS_REPO", os.path.expanduser("~/gitrepos/cosmos-transfer2.5")),
    )
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--config-file", default="cosmos_transfer2/_src/predict2/configs/video2world/config.py")
    ap.add_argument("--experiment-opt", action="append", dest="experiment_opts", default=None)
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

    adapter = CosmosPredictDiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
        cosmos_repo=args.cosmos_repo,
        experiment_name=args.experiment,
        config_file=args.config_file,
        experiment_opts=args.experiment_opts or [],
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        text_seq_len=args.text_seq_len,
        calibration_inputs_path=args.calibration_inputs,
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
    print(f"saved official Cosmos Predict2.5 QEP/GPTQ checkpoint to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

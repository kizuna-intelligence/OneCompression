"""Example: AutoBit + ILP mixed-precision quantisation of a Faster-Irodori-TTS2 DiT.

Demonstrates the full Phase-1 OneCompression pipeline on a non-HuggingFace
Rectified-Flow text-to-latent diffusion transformer:

    1. Wrap the DiT checkpoint in ``IrodoriDiTAdapter``.
    2. Hand the adapter to ``ModelConfig`` and call ``Runner.auto_run`` —
       VRAM is auto-detected, target bpw is solved from quantizable
       parameter count, and AutoBit's activation-aware ILP assigns each
       Linear in every ``DiffusionBlock`` to a candidate GPTQ bitwidth.
    3. The resulting safetensors is written back in the same flat-config
       schema used by ``irodori_tts.inference_runtime`` so it loads
       directly into the existing TTS pipeline.

Run::

    python example/example_dit_autobit.py \\
        /home/yusuke/gitrepos/Faster-Irodori-TTS2/checkpoints/moespeech_ft_5000.safetensors \\
        --save-dir ./moespeech_ft_5000-autobit

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner
from onecomp.adapters import IrodoriDiTAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        help="Path to the DiT checkpoint (.safetensors or .pt).",
    )
    parser.add_argument(
        "--wbits",
        type=float,
        default=None,
        help="Target effective bpw.  When omitted, derived from VRAM.",
    )
    parser.add_argument(
        "--total-vram-gb",
        type=float,
        default=None,
        help="Override detected GPU VRAM (used only when --wbits is omitted).",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=32,
        help=(
            "GPTQ group size.  Default 32 because the DiT's SwiGLU MLP "
            "hidden dim (3680) is not divisible by 64/128."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for the model during quantisation.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Working dtype: float32 / bfloat16 / float16.",
    )
    parser.add_argument(
        "--save-dir",
        default="auto",
        help="Output directory.  'auto' derives a name from the checkpoint.",
    )
    parser.add_argument(
        "--no-qep",
        action="store_true",
        help="Skip the QEP refinement pass.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=16,
        help=(
            "AutoBit calibration samples.  Default 16 keeps activation "
            "stats within ~30 GB VRAM for the Faster-Irodori-TTS2 DiT "
            "(144 quantizable Linears × Gram + curvature stats)."
        ),
    )
    parser.add_argument(
        "--calibration-max-length",
        type=int,
        default=128,
        help="Calibration sequence length (default: 128).",
    )
    args = parser.parse_args()

    adapter = IrodoriDiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
    )
    model_config = ModelConfig(adapter=adapter)

    calibration_config = CalibrationConfig(
        num_calibration_samples=args.num_calibration_samples,
        max_length=args.calibration_max_length,
    )

    Runner.auto_run(
        model_config=model_config,
        wbits=args.wbits,
        total_vram_gb=args.total_vram_gb,
        groupsize=args.groupsize,
        device=args.device,
        qep=not args.no_qep,
        evaluate=False,
        save_dir=args.save_dir,
        calibration_config=calibration_config,
        exclude_layer_keywords=IrodoriDiTAdapter.default_exclude_layer_keywords(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

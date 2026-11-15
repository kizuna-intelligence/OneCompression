"""End-to-end QEP-vs-plain-GPTQ comparison for FLUX.2 single-stream blocks.

Both checkpoints quantize the *same* 40 ``single_transformer_blocks`` Linears
(plain GPTQ vs Fujitsu QEP); everything else stays bf16.  This isolates QEP's
contribution.  We feed one identical seeded calibration batch through:

  * the original bf16 transformer            -> reference output
  * the plain-GPTQ int4 checkpoint           -> out_gptq
  * the QEP int4 checkpoint                  -> out_qep

and report the relative output error  ||out - ref||^2 / ||ref||^2  (mean over
batch).  Per-layer reconstruction MSE is *not* the right metric for QEP — QEP
deliberately trades per-layer weight error for lower final-output error — so we
measure the model output directly.  Eager backend is used so the comparison
reflects the dequantized-weight math, not the GEMM kernel choice.

Run (inside .venv-flux)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    .venv-flux/bin/python example/flux2_int4/compare_qep.py \\
        black-forest-labs/FLUX.2-klein-4B \\
        --gptq /home/yusuke/flux_gptq_single \\
        --qep  /home/yusuke/flux_qep_on \\
        --num-samples 8

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from onecomp import CalibrationConfig, setup_logger
from onecomp.adapters import Flux2DiTAdapter
from onecomp_runtime.diffusion import load_int4_model


def build_flux2(cfg):
    from diffusers import Flux2Transformer2DModel

    return Flux2Transformer2DModel.from_config(cfg)


def rel_sq_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    out = out.float()
    ref = ref.float()
    num = (out - ref).pow(2).flatten(1).sum(dim=1)
    den = ref.pow(2).flatten(1).sum(dim=1).clamp_min(1e-12)
    return (num / den).mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="original HF FLUX.2 (bf16 reference)")
    ap.add_argument("--gptq", required=True, help="plain-GPTQ int4 save dir")
    ap.add_argument("--qep", required=True, help="QEP int4 save dir")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--image-grid", type=int, default=32)
    ap.add_argument("--text-seq-len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    setup_logger()
    device = torch.device(args.device)

    adapter = Flux2DiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
        image_grid=args.image_grid,
        text_seq_len=args.text_seq_len,
    )
    calib = CalibrationConfig(num_calibration_samples=args.num_samples, max_length=256)

    # 1) Reference forward on the original bf16 transformer.
    orig = adapter.load_model(device_map=args.device)
    inputs = adapter.prepare_calibration_inputs(orig, calib, device)
    with torch.no_grad():
        ref = adapter.run_calibration_forward(orig, inputs).detach().to("cpu")
    del orig
    torch.cuda.empty_cache()

    # 2) Each quantized checkpoint, same inputs, eager backend (dtype match).
    results = {}
    for tag, path in (("gptq", args.gptq), ("qep", args.qep)):
        if os.path.isdir(path):
            path = os.path.join(path, "model.safetensors")
        model = load_int4_model(
            path,
            build_flux2,
            device=args.device,
            dtype=args.dtype,
            backend="eager",
            warmup=False,
            label=f"compare-{tag}",
        )
        with torch.no_grad():
            out = adapter.run_calibration_forward(model, inputs).detach().to("cpu")
        results[tag] = rel_sq_error(out, ref)
        del model
        torch.cuda.empty_cache()

    g, q = results["gptq"], results["qep"]
    print("\n================ FLUX.2 QEP vs plain GPTQ (single-stream, 40 layers) ================")
    print(f"  reference     : original bf16 FLUX.2-klein-4B, {args.num_samples} seeded samples")
    print(f"  plain GPTQ    : rel_output_sq_error = {g:.6e}")
    print(f"  QEP (Fujitsu) : rel_output_sq_error = {q:.6e}")
    if g > 0:
        print(f"  QEP / GPTQ    : {q / g:.4f}  ({(1 - q / g) * 100:+.2f}% vs plain GPTQ)")
    print("====================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

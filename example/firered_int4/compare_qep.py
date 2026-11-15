"""End-to-end QEP-vs-RTN comparison for FireRed / Qwen-Image dual-stream blocks.

Both checkpoints quantize the *same* ``transformer_blocks`` Linears (minus the
AdaLN ``img_mod`` / ``txt_mod`` projections) at the same groupsize/symmetry; one
via calibration-free **RTN**, one via Fujitsu **QEP** (block error propagation
driven over the dual-stream blocks through the adapter's stream-packing hooks).
Everything else stays bf16, isolating the quantizer's contribution.

Plain whole-model GPTQ is not a practical baseline here: the bf16 transformer is
~41GB and will not fit a 24GB GPU for a single resident calibration forward, so
RTN is the realistic alternative QEP must beat.

We feed one identical seeded calibration batch through:

  * the original bf16 transformer            -> reference output
  * the RTN int4 checkpoint                  -> out_rtn
  * the QEP int4 checkpoint                  -> out_qep

and report the relative output error  ||out - ref||^2 / ||ref||^2  (mean over
batch).  Per-layer reconstruction MSE is *not* the right metric for QEP — QEP
deliberately trades per-layer weight error for lower final-output error — so we
measure the model output directly.  Eager backend is used so the comparison
reflects the dequantized-weight math, not the GEMM kernel choice.

Run (inside .venv-flux)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    HF_HOME=/media/yusuke/Curry/hf_archive \\
    .venv-flux/bin/python example/firered_int4/compare_qep.py \\
        /media/yusuke/Curry/firered_work/bf16 \\
        --rtn /home/yusuke/firered_int4_weights/model_nomod.safetensors \\
        --qep /home/yusuke/firered_qep \\
        --num-samples 8

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from onecomp import CalibrationConfig, setup_logger
from onecomp.adapters import QwenImageDiTAdapter
from onecomp_runtime.diffusion import load_int4_model


def build_qwenimage(cfg):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_config(cfg)


def _materialize_rope_tables(model: torch.nn.Module) -> None:
    """Recompute QwenEmbedRope ``pos_freqs`` / ``neg_freqs`` after a meta build.

    Those are plain (complex) tensor attributes built in ``__init__`` — *not*
    registered buffers (registering would drop the imaginary part), so they are
    absent from the checkpoint and stay on the ``meta`` device after the
    int4 loader's meta construction.  Rebuild them on CPU with the module's own
    ``rope_params`` so the RoPE forward has real data.
    """
    for mod in model.modules():
        if hasattr(mod, "rope_params") and getattr(mod, "pos_freqs", None) is not None:
            if not mod.pos_freqs.is_meta:
                continue
            pos_index = torch.arange(4096)
            neg_index = torch.arange(4096).flip(0) * -1 - 1
            mod.pos_freqs = torch.cat(
                [mod.rope_params(pos_index, d, mod.theta) for d in mod.axes_dim], dim=1
            )
            mod.neg_freqs = torch.cat(
                [mod.rope_params(neg_index, d, mod.theta) for d in mod.axes_dim], dim=1
            )


def rel_sq_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    out = out.float()
    ref = ref.float()
    num = (out - ref).pow(2).flatten(1).sum(dim=1)
    den = ref.pow(2).flatten(1).sum(dim=1).clamp_min(1e-12)
    return (num / den).mean().item()


def _resolve(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "model.safetensors")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="original FireRed bf16 (transformer/ subfolder)")
    ap.add_argument("--rtn", required=True, help="RTN int4 save dir or .safetensors")
    ap.add_argument("--qep", required=True, help="QEP int4 save dir or .safetensors")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--image-grid", type=int, default=32)
    ap.add_argument("--text-seq-len", type=int, default=128)
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--calib-path", default=None,
                    help="captured real calibration .pt; must be set so the bf16 "
                         "reference forward stays finite (synthetic noise NaNs).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    setup_logger()
    device = torch.device(args.device)

    adapter = QwenImageDiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
        image_grid=args.image_grid,
        text_seq_len=args.text_seq_len,
        subfolder=args.subfolder,
        calibration_inputs_path=args.calib_path,
    )
    calib = CalibrationConfig(num_calibration_samples=args.num_samples, max_length=256)

    # 1) Reference forward on the original bf16 transformer.  The 20B bf16
    #    transformer is ~41GB and will not fit a 24GB GPU, so run the reference
    #    forward on CPU (slow but only once, num-samples small).
    ref_device = "cpu"
    orig = adapter.load_model(device_map=ref_device)
    inputs = adapter.prepare_calibration_inputs(orig, calib, torch.device(ref_device))
    with torch.no_grad():
        ref = adapter.run_calibration_forward(orig, inputs).detach().to("cpu")
    del orig
    torch.cuda.empty_cache()

    # Move inputs to the GPU for the (small, int4-resident) quantized forwards.
    gpu_inputs = adapter.slice_calibration_inputs(inputs, 0, args.num_samples)
    gpu_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in gpu_inputs.items()
    }

    # 2) Each quantized checkpoint, same inputs, eager backend (dtype match).
    results = {}
    for tag, path in (("rtn", args.rtn), ("qep", args.qep)):
        model = load_int4_model(
            _resolve(path),
            build_qwenimage,
            device=args.device,
            dtype=args.dtype,
            backend="eager",
            warmup=False,
            post_load=_materialize_rope_tables,
            label=f"compare-{tag}",
        )
        with torch.no_grad():
            out = adapter.run_calibration_forward(model, gpu_inputs).detach().to("cpu")
        results[tag] = rel_sq_error(out, ref)
        del model
        torch.cuda.empty_cache()

    r, q = results["rtn"], results["qep"]
    print("\n========== FireRed / Qwen-Image QEP vs RTN (dual-stream, transformer_blocks) ==========")
    print(f"  reference     : original bf16 FireRed DiT, {args.num_samples} seeded samples")
    print(f"  RTN           : rel_output_sq_error = {r:.6e}")
    print(f"  QEP (Fujitsu) : rel_output_sq_error = {q:.6e}")
    if r > 0:
        print(f"  QEP / RTN     : {q / r:.4f}  ({(1 - q / r) * 100:+.2f}% vs RTN)")
    print("=======================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

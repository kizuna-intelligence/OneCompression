"""Capture *real* top-level DiT forward inputs from the FireRed pipeline.

Synthetic N(0,1) calibration works for small single-stream DiTs but **diverges**
on FireRed's 60-block dual-stream MMDiT: a random text stream
(``encoder_hidden_states``) is wildly out-of-distribution, so joint attention
amplifies it block-over-block until the activations overflow to NaN (≈block 22),
which corrupts the QEP Hessians.  Real text-encoder embeddings are bounded, so
real calibration stays finite — and matches the v3 finding that real-activation
calibration is what makes blind 4-bit survive.

This runs the *already int4-quantized* FireRed pipeline (≈12GB, fits a 24GB GPU
with ``--offload``) and hooks the **top-level** ``transformer.forward`` — the
same input keys :meth:`QwenImageDiTAdapter.prepare_calibration_inputs` builds
synthetically (``hidden_states`` raw latent, ``encoder_hidden_states`` real text
embeds, ``timestep``, ``img_shapes`` ...).  The block-0 inputs the QEP catcher
needs are produced by the bf16 embedders downstream, so quantizing the *blocks*
does not change these captured inputs.  Capturing across every denoising step
(and both CFG branches) yields a calibration set spanning the flow-matching
trajectory: ``hidden_states`` / ``timestep`` vary per call, giving real per-step
modulation conditioning for free.

Run (inside the FireRed int4 runtime venv)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    python example/firered_int4/capture_calibration.py \\
        --repo /media/yusuke/Curry/firered_work/bf16-repo \\
        --dit /home/yusuke/firered_int4_weights/model.safetensors \\
        --image /home/yusuke/firered_int4_weights/pug_input.jpg \\
        --steps 16 --out /home/yusuke/firered_calib.pt

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

import torch


_CAPTURE_KEYS = (
    "hidden_states",
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "timestep",
    "img_shapes",
    "txt_seq_lens",
    "guidance",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="FireRed repo dir (tokenizer/TE/VAE/scheduler)")
    ap.add_argument("--dit", required=True, help="packed int4 DiT safetensors")
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="Make it a snowy winter night with a glowing moon.")
    ap.add_argument("--negative-prompt", default=" ")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .pt with captured inputs")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image
    from firered_image_edit_int4 import load_int4_transformer

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # bf16 + fused backend: bf16 is numerically stable for Qwen-Image (fp16 NaNs)
    # and matches the QEP quantization dtype.
    dit = load_int4_transformer(args.dit, device="cpu", dtype=args.dtype,
                                backend="fused", warmup=False)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.repo, transformer=dit, torch_dtype=torch_dtype)
    pipe.enable_model_cpu_offload(device="cuda:0")

    captured: list[dict] = []

    def _pre_hook(module, args_, kwargs_):
        # diffusers calls the transformer with keyword args; be defensive.
        rec = {}
        for k in _CAPTURE_KEYS:
            v = kwargs_.get(k, None)
            if isinstance(v, torch.Tensor):
                rec[k] = v.detach().to("cpu")
            else:
                rec[k] = v  # img_shapes / txt_seq_lens (lists) or None
        captured.append(rec)
        return None

    handle = pipe.transformer.register_forward_pre_hook(_pre_hook, with_kwargs=True)

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        image = Image.new("RGB", (args.size, args.size), (135, 206, 235))

    print(f"[capture] running {args.steps} steps to capture top-level DiT inputs ...", flush=True)
    with torch.no_grad():
        pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.size,
            width=args.size,
            true_cfg_scale=args.true_cfg,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
    handle.remove()

    print(f"[capture] captured {len(captured)} transformer calls", flush=True)

    # ``true_cfg`` runs the positive and negative prompts through the
    # transformer separately, and their text streams tokenize to *different*
    # sequence lengths (e.g. a rich edit prompt vs. a blank negative), so
    # ``encoder_hidden_states`` / its mask cannot be stacked across both
    # branches.  The image stream is identical across calls.  Group calls by
    # their full tensor-shape signature and keep the largest uniform group:
    # that is the conditional (positive-prompt) branch repeated across every
    # denoising step — the real signal QEP should be calibrated on.
    def _sig(rec):
        return tuple(
            tuple(rec[k].shape) if isinstance(rec[k], torch.Tensor) else None
            for k in _CAPTURE_KEYS
        )

    groups: dict = {}
    for rec in captured:
        groups.setdefault(_sig(rec), []).append(rec)
    best = max(groups.values(), key=len)
    print(f"[capture] {len(groups)} shape group(s); keeping largest "
          f"({len(best)}/{len(captured)} calls)", flush=True)

    out: dict = {"num_calls": len(best)}
    ref = best[0]
    for k in _CAPTURE_KEYS:
        vals = [c[k] for c in best]
        if isinstance(ref[k], torch.Tensor):
            out[k] = torch.cat([v if v.dim() else v.unsqueeze(0) for v in vals], dim=0)
        else:
            out[k] = ref[k]  # img_shapes / txt_seq_lens describe a single sample
    # Real image-token count, so the dual-stream pack wrapper splits the packed
    # tensor at the *real* boundary (not the synthetic image_grid**2 guess).
    out["img_seq"] = int(out["hidden_states"].shape[1])
    print(f"  img_seq (real image tokens): {out['img_seq']}", flush=True)
    for k in _CAPTURE_KEYS:
        v = out[k]
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else v
        print(f"  {k}: {shape}")
    torch.save(out, args.out)
    print(f"[capture] saved real calibration to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

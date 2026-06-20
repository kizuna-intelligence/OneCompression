"""Capture real Cosmos Transfer2.5 transformer inputs for QEP/PTQ.

The synthetic fallback is useful for smoke tests, but Cosmos Transfer quality
depends strongly on real text embeddings, denoising timesteps, condition masks,
and ControlNet residuals.  This script hooks the top-level transformer forward
inside a real ``Cosmos2_5_TransferPipeline`` run and saves a uniform calibration
set that :class:`CosmosTransferDiTAdapter` can load via ``--calibration-inputs``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch

from generate import _load_controls, load_int4_cosmos_transformer


_CAPTURE_KEYS = (
    "hidden_states",
    "timestep",
    "encoder_hidden_states",
    "block_controlnet_hidden_states",
    "attention_mask",
    "fps",
    "condition_mask",
    "padding_mask",
)


def _detach_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_cpu(v) for v in value)
    if isinstance(value, list):
        return [_detach_cpu(v) for v in value]
    return value


def _shape_sig(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, tuple):
        return tuple(_shape_sig(v) for v in value)
    if isinstance(value, list):
        return tuple(_shape_sig(v) for v in value)
    return type(value).__name__


def _cat_values(values: list[Any]) -> Any:
    ref = values[0]
    if isinstance(ref, torch.Tensor):
        prepared = [v if v.dim() > 0 else v.unsqueeze(0) for v in values]
        return torch.cat(prepared, dim=0)
    if isinstance(ref, tuple):
        return tuple(_cat_values([v[i] for v in values]) for i in range(len(ref)))
    if isinstance(ref, list) and ref and isinstance(ref[0], torch.Tensor):
        return [_cat_values([v[i] for v in values]) for i in range(len(ref))]
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/Cosmos-Transfer2.5-2B")
    ap.add_argument("--revision", default="diffusers/general")
    ap.add_argument("--controlnet-revision", default="diffusers/controlnet/general/edge")
    ap.add_argument("--dit", required=True, help="int4 transformer checkpoint")
    ap.add_argument("--controls", default=None)
    ap.add_argument("--prompt", default="A realistic robotics laboratory video.")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--num-frames-per-chunk", type=int, default=17)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--backend", default="fused", choices=["auto", "gemlite", "fused", "eager"])
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")

    transformer = load_int4_cosmos_transformer(
        args.dit,
        device="cpu",
        dtype=args.dtype,
        backend=args.backend,
        warmup=False,
    )

    from diffusers import AutoModel, Cosmos2_5_TransferPipeline

    controlnet = AutoModel.from_pretrained(
        args.repo,
        revision=args.controlnet_revision,
        torch_dtype=torch_dtype,
    )
    pipe = Cosmos2_5_TransferPipeline.from_pretrained(
        args.repo,
        revision=args.revision,
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    if args.sequential:
        pipe.enable_sequential_cpu_offload(device="cuda:0")
    else:
        pipe.enable_model_cpu_offload(device="cuda:0")

    captured: list[dict[str, Any]] = []

    def _pre_hook(_module, _args, kwargs):
        rec = {k: _detach_cpu(kwargs.get(k)) for k in _CAPTURE_KEYS}
        captured.append(rec)
        return None

    handle = pipe.transformer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    controls = _load_controls(args.controls, args.num_frames, args.height, args.width)
    print(f"[capture] running {args.steps} steps...", flush=True)
    with torch.no_grad():
        pipe(
            controls=controls,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_frames_per_chunk=args.num_frames_per_chunk,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
    handle.remove()

    print(f"[capture] captured {len(captured)} transformer calls", flush=True)
    if not captured:
        raise RuntimeError("No transformer calls were captured")

    groups: dict[Any, list[dict[str, Any]]] = {}
    for rec in captured:
        sig = tuple(_shape_sig(rec[k]) for k in _CAPTURE_KEYS)
        groups.setdefault(sig, []).append(rec)
    best = max(groups.values(), key=len)
    print(f"[capture] {len(groups)} shape group(s); keeping {len(best)} calls", flush=True)

    out: dict[str, Any] = {"num_calls": len(best)}
    for key in _CAPTURE_KEYS:
        vals = [rec[key] for rec in best]
        out[key] = _cat_values(vals) if vals[0] is not None else None

    for key in _CAPTURE_KEYS:
        value = out[key]
        print(f"  {key}: {_shape_sig(value)}", flush=True)

    torch.save(out, args.out)
    print(f"[capture] saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

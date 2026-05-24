"""Load an OneCompression-packed INT4 checkpoint with the shared inference runtime.

OneCompression *produces* packed-int4 ``safetensors`` (via the adapters'
``save_quantized_model``). Production inference does **not** need the full
OneCompression install — it uses the lightweight, standalone
`onecomp-runtime` package, which carries only the int4 GEMM kernels
(GemLite / fused Triton / eager) and a generic loader.

    pip install "onecomp-runtime @ git+https://github.com/kizuna-intelligence/onecompression-runtime@main"
    # GemLite backend (preferred): onecomp-runtime[gemlite]

The single per-model seam is ``build_meta_model(cfg) -> nn.Module`` (the model
is constructed under ``torch.device("meta")`` from the embedded ``config_json``)
plus an optional ``post_load(model)`` hook for buffer fixups (e.g. rope freqs).

Run (FireRed / Qwen-Image example)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    python example/onecomp_runtime_inference.py \\
        /home/yusuke/firered_int4_weights/model_nomod.safetensors \\
        --model qwenimage --device cuda:0 --dtype bfloat16

This is the same path Flux2-klein-Lite and Irodori-TTS-Lite now use internally
(each is a ~15-line adapter over ``onecomp_runtime.diffusion.load_int4_model``).

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

import torch

from onecomp_runtime.diffusion import load_int4_model
from onecomp_runtime.layers import FusedInt4Linear, GemLiteInt4Linear


def build_qwenimage(cfg):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_config(cfg)


def build_flux2(cfg):
    from diffusers import Flux2Transformer2DModel

    return Flux2Transformer2DModel.from_config(cfg)


def recompute_qwen_rope(model):
    """QwenEmbedRope keeps pos/neg_freqs as plain complex tensors built under
    meta; recompute them on a real device so forward-time ``.to()`` works."""
    n = 0
    for module in model.modules():
        pf = getattr(module, "pos_freqs", None)
        if (
            isinstance(pf, torch.Tensor)
            and pf.is_meta
            and hasattr(module, "rope_params")
            and hasattr(module, "axes_dim")
        ):
            pos_index = torch.arange(4096)
            neg_index = torch.arange(4096).flip(0) * -1 - 1
            module.pos_freqs = torch.cat(
                [module.rope_params(pos_index, module.axes_dim[i], module.theta)
                 for i in range(len(module.axes_dim))], dim=1)
            module.neg_freqs = torch.cat(
                [module.rope_params(neg_index, module.axes_dim[i], module.theta)
                 for i in range(len(module.axes_dim))], dim=1)
            n += 1
    if n:
        print(f"[post_load] recomputed rope freqs on {n} module(s)")


BUILDERS = {
    "qwenimage": (build_qwenimage, recompute_qwen_rope),
    "flux2": (build_flux2, None),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--model", choices=sorted(BUILDERS), default="qwenimage")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--backend", default="auto", choices=["auto", "gemlite", "fused", "eager"])
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    build, post_load = BUILDERS[args.model]
    model = load_int4_model(
        args.checkpoint,
        build,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        post_load=post_load,
        warmup=not args.no_warmup,
        label="onecomp_runtime_inference",
    )

    n_fused = sum(isinstance(m, FusedInt4Linear) for m in model.modules())
    n_gemlite = sum(isinstance(m, GemLiteInt4Linear) for m in model.modules())
    n_meta = sum(p.is_meta for p in model.parameters())
    print(f"loaded: fused={n_fused} gemlite={n_gemlite}; meta params left={n_meta}")
    if n_meta:
        raise SystemExit(f"ERROR: {n_meta} params left on meta")
    print("OK — checkpoint loaded via shared onecomp-runtime")
    return 0


if __name__ == "__main__":
    sys.exit(main())

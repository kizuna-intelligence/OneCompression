"""Reload a packed FLUX.2 INT4 checkpoint and run one forward pass.

Confirms that ``Flux2DiTAdapter.load_quantized_model`` rebuilds the
``GPTQLinear`` modules from the saved metadata and that the model runs.

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    .venv-flux/bin/python example/flux2_int4/load_check.py ./flux2_int4/model.safetensors

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

import torch

from onecomp.adapters import Flux2DiTAdapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--text-seq-len", type=int, default=64)
    args = ap.parse_args()

    model = Flux2DiTAdapter.load_quantized_model(
        args.checkpoint, device=args.device, dtype="bfloat16"
    )
    n_gptq = sum(1 for m in model.modules() if type(m).__name__ == "GPTQLinear")
    print(f"loaded; GPTQLinear modules: {n_gptq}")

    cfg = dict(model.config)
    grid = args.grid
    img_seq = grid * grid
    txt_seq = args.text_seq_len
    dev = torch.device(args.device)

    img_ids = torch.zeros((img_seq, 4), device=dev)
    img_ids[:, 1] = torch.arange(grid, device=dev).repeat_interleave(grid)[:img_seq]
    img_ids[:, 2] = torch.arange(grid, device=dev).repeat(grid)[:img_seq]
    txt_ids = torch.zeros((txt_seq, 4), device=dev)

    with torch.no_grad():
        out = model(
            hidden_states=torch.randn(1, img_seq, int(cfg["in_channels"]), device=dev, dtype=torch.bfloat16),
            encoder_hidden_states=torch.randn(1, txt_seq, int(cfg["joint_attention_dim"]), device=dev, dtype=torch.bfloat16),
            timestep=torch.rand(1, device=dev, dtype=torch.bfloat16),
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
    print("forward OK, output shape:", tuple(out.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())

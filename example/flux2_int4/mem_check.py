"""Measure VRAM peak of the packed FLUX.2 INT4 checkpoint (load + one forward).

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

import torch

from onecomp.adapters import Flux2DiTAdapter


def _mb(x: int) -> float:
    return x / (1024 * 1024)


def _report(tag: str, dev: torch.device) -> None:
    alloc = _mb(torch.cuda.memory_allocated(dev))
    peak_a = _mb(torch.cuda.max_memory_allocated(dev))
    peak_r = _mb(torch.cuda.max_memory_reserved(dev))
    print(f"  {tag:<28} alloc={alloc:8.1f}  peak_alloc={peak_a:8.1f}  peak_res={peak_r:8.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--text-seq-len", type=int, default=64)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(dev)
    _report("baseline", dev)

    model = Flux2DiTAdapter.load_quantized_model(
        args.checkpoint, device=args.device, dtype="bfloat16"
    )
    n_gptq = sum(1 for m in model.modules() if type(m).__name__ == "GPTQLinear")
    _report("after load", dev)

    cfg = dict(model.config)
    grid = args.grid
    img_seq = grid * grid
    txt_seq = args.text_seq_len

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
    torch.cuda.synchronize(dev)
    _report("after forward", dev)

    print(f"GPTQLinear modules: {n_gptq}; output shape: {tuple(out.shape)}")
    print(f"PEAK VRAM (allocated): {_mb(torch.cuda.max_memory_allocated(dev)):.1f} MB")
    print(f"PEAK VRAM (reserved):  {_mb(torch.cuda.max_memory_reserved(dev)):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

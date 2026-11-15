"""Shard-streaming weight reader for DeepSeek-V4-Flash.

The checkpoint is 46 safetensors shards / ~160GB of packed FP8 (dense) + FP4
(experts) weights. A bf16 dequant of the full 284B-param model is ~280GB, far
over the host RAM budget, so the model can never be fully resident. This reader
opens shards lazily and dequantizes **one tensor / one layer at a time** on
demand, which is what the QEP block-streaming loop (move each block to GPU,
quantize, free) needs.

See ``onecomp.utils.dsv4_dequant`` for the validated FP8/FP4 dequant kernels and
the ``project_dsv4_flash_qep`` memory for the layout provenance.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import torch
from safetensors import safe_open

from onecomp.utils.dsv4_dequant import dequant_fp8_dense, dequant_fp4_expert


class Dsv4ShardReader:
    """Lazily reads + dequantizes DeepSeek-V4-Flash checkpoint tensors.

    Args:
        ckpt_dir: directory holding ``model.safetensors.index.json`` and shards.
        max_open_shards: LRU cache size for open ``safe_open`` handles.
    """

    def __init__(self, ckpt_dir: str, max_open_shards: int = 4):
        self.ckpt_dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        # bound the open-handle cache
        self._open = lru_cache(maxsize=max_open_shards)(self._open_shard)

    def _open_shard(self, shard: str):
        return safe_open(os.path.join(self.ckpt_dir, shard), framework="pt", device="cpu")

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def raw(self, name: str) -> torch.Tensor:
        """Return the raw (still-packed) tensor for ``name``."""
        return self._open(self.weight_map[name]).get_tensor(name)

    def dequant_linear(
        self, prefix: str, out_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Dequantize a Linear weight stored as ``{prefix}.weight`` + ``{prefix}.scale``.

        Dispatches FP4 (int8/float4 packed, per-32 scale) vs FP8 (e4m3, per-128
        block scale) by inspecting the weight dtype and scale shape.
        """
        w = self.raw(f"{prefix}.weight")
        s = self.raw(f"{prefix}.scale")
        if w.dtype == torch.float4_e2m1fn_x2 or (
            w.dtype == torch.int8 and s.shape[1] * 32 == w.shape[1] * 2
        ):
            return dequant_fp4_expert(w, s, out_dtype)
        if w.dtype == torch.float8_e4m3fn:
            return dequant_fp8_dense(w, s, out_dtype)
        # already-unpacked (bf16 gate/embed/head/norms)
        return w.to(out_dtype)

    def plain(self, name: str, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return a non-quantized tensor (norms, gate, embed, head, hc params)."""
        t = self.raw(name)
        return t.to(out_dtype) if out_dtype is not None else t

    def layer_linear_prefixes(self, layer_idx: int, n_experts: int = 256) -> list[str]:
        """All quantizable Linear prefixes in one transformer layer.

        Dense MLA attention + shared expert + routed experts. Excludes norms,
        gate, attn_sink, and the hc_* manifold-connection params.
        """
        L = f"layers.{layer_idx}"
        dense = [
            f"{L}.attn.wq_a", f"{L}.attn.wq_b", f"{L}.attn.wkv",
            f"{L}.attn.wo_a", f"{L}.attn.wo_b",
            f"{L}.ffn.shared_experts.w1", f"{L}.ffn.shared_experts.w2",
            f"{L}.ffn.shared_experts.w3",
        ]
        experts = [
            f"{L}.ffn.experts.{e}.{w}"
            for e in range(n_experts)
            for w in ("w1", "w2", "w3")
        ]
        return [p for p in dense + experts if self.has(f"{p}.weight")]

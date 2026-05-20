"""Shared helpers for model adapters.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import torch


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(dtype) -> torch.dtype:
    """Coerce a dtype string / ``torch.dtype`` / ``None`` to a ``torch.dtype``."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        return torch.float32
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in _DTYPE_MAP:
            return _DTYPE_MAP[key]
        raise ValueError(f"Unrecognised dtype string: {dtype!r}")
    raise TypeError(f"Unsupported dtype: {dtype!r}")

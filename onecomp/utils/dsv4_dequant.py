"""DeepSeek-V4-Flash on-disk weight dequantization.

The shipped checkpoint stores weights in two packed low-bit formats:

* **Dense attention Linears** (``wq_a/wq_b/wkv/wo_b`` etc.) in
  ``float8_e4m3fn`` with a ``float8_e8m0fnu`` per-128x128-block scale.
* **MoE expert matmuls** (``experts.*.w1/w2/w3``) in FP4 ``e2m1`` packed two
  values per ``int8`` container ([N, K//2]) with a ``float8_e8m0fnu``
  per-32 (along K) block scale.

Both layouts are transcribed from the authoritative ``inference/convert.py``
that produced the tensors (``FP4_TABLE``, low-nibble=even-column interleave,
e8m0 block scales). Validated against the shipped tilelang ``fp8_gemm`` /
``fp4_gemm`` kernels on real checkpoint tensors (see
``project_dsv4_flash_qep`` memory). This module gives full-precision (bf16/
fp32) reconstructions for GPTQ/QEP calibration — it does NOT need to match the
runtime kernel's storage order, only the *values*.
"""

from __future__ import annotations

import torch

# E2M1 FP4 code -> value lookup, identical to inference/convert.py FP4_TABLE.
# Index by the 4-bit code; sign bit is the high bit of the nibble.
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

FP8_BLOCK = 128  # dense fp8 block (both dims)
FP4_BLOCK = 32   # fp4 expert block (along K)


def dequant_fp8_dense(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an fp8-e4m3 dense weight with a per-128x128 e8m0 block scale.

    Args:
        weight: [N, K] in ``float8_e4m3fn``.
        scale:  [ceil(N/128), ceil(K/128)] in ``float8_e8m0fnu`` (or any float).
        out_dtype: reconstruction dtype.

    Returns:
        [N, K] dense weight in ``out_dtype``.
    """
    n, k = weight.shape
    s = scale.float()
    s = s.repeat_interleave(FP8_BLOCK, dim=0)[:n]
    s = s.repeat_interleave(FP8_BLOCK, dim=1)[:, :k]
    return (weight.float() * s).to(out_dtype)


def dequant_fp4_expert(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP4-e2m1 expert weight (2-per-int8) with per-32 e8m0 scale.

    Layout (from convert.py ``cast_e2m1fn_to_e4m3fn``): the low nibble of each
    byte is the even output column, the high nibble the odd column.

    Args:
        packed: [N, K//2] in ``int8`` (or ``uint8`` / ``float4_e2m1fn_x2`` view).
        scale:  [N, K//32] in ``float8_e8m0fnu`` (or any float), per-32 along K.
        out_dtype: reconstruction dtype.

    Returns:
        [N, K] dense weight in ``out_dtype``.
    """
    b = packed.view(torch.uint8)
    n, k_half = b.shape
    k = k_half * 2
    lut = FP4_TABLE.to(b.device)
    lo = lut[(b & 0x0F).long()]
    hi = lut[((b >> 4) & 0x0F).long()]
    w = torch.empty(n, k, dtype=torch.float32, device=b.device)
    w[:, 0::2] = lo
    w[:, 1::2] = hi
    s = scale.float().repeat_interleave(FP4_BLOCK, dim=1)
    assert s.shape[1] == k, f"scale K mismatch: {s.shape[1]} vs {k}"
    return (w * s).to(out_dtype)


def dequant_tensor(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dispatch on storage dtype: fp4-packed -> expert path, fp8 -> dense path."""
    if weight.dtype == torch.float4_e2m1fn_x2 or (
        weight.dtype == torch.int8 and scale.shape[1] * FP4_BLOCK == weight.shape[1] * 2
    ):
        return dequant_fp4_expert(weight, scale, out_dtype)
    if weight.dtype == torch.float8_e4m3fn:
        return dequant_fp8_dense(weight, scale, out_dtype)
    raise ValueError(f"unhandled weight dtype {weight.dtype} for dsv4 dequant")

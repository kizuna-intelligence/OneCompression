"""OneCompression Runtime: native int4 inference kernels.

Phase 2 — DiT-shape-tuned fused dequant+GEMM Triton kernel for 4-bit GPTQ-packed
weights.  Optimised for medium-batch (M~64-128) inference shapes where existing
LLM-decode kernels (M=1) lose to fp32 cuBLAS.
"""

from .fused_int4_linear import FusedInt4Linear, fused_int4_gemm

__all__ = ["FusedInt4Linear", "fused_int4_gemm"]

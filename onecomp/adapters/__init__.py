"""Model adapters for plugging non-HuggingFace architectures into OneComp.

The default :class:`HFLLMAdapter` reproduces OneCompression's historical
behavior for HuggingFace causal LMs.  Custom adapters (e.g.
:class:`DiTAdapter` for diffusion transformers) are imported lazily so
that their dependencies (``irodori_tts``, etc.) stay optional.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .base import ModelAdapter
from .hf_llm import HFLLMAdapter


def __getattr__(name):  # PEP 562 lazy import
    if name == "DiTAdapter":
        from .dit import DiTAdapter

        return DiTAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelAdapter", "HFLLMAdapter", "DiTAdapter"]

"""Model adapters for plugging non-HuggingFace architectures into OneComp.

The default :class:`HFLLMAdapter` reproduces OneCompression's historical
behavior for HuggingFace causal LMs.  Diffusion-transformer adapters live
under :mod:`onecomp.adapters.diffusion`; the architecture-agnostic
:class:`DiffusionTransformerAdapter` base and concrete subclasses such as
:class:`IrodoriDiTAdapter` are imported lazily so their optional
dependencies (``irodori_tts``, ``diffusers``…) stay optional.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .base import ModelAdapter
from .hf_llm import HFLLMAdapter


def __getattr__(name):  # PEP 562 lazy import
    if name == "DiffusionTransformerAdapter":
        from .diffusion import DiffusionTransformerAdapter

        return DiffusionTransformerAdapter
    if name == "IrodoriDiTAdapter":
        from .diffusion import IrodoriDiTAdapter

        return IrodoriDiTAdapter
    if name == "Flux2DiTAdapter":
        from .diffusion import Flux2DiTAdapter

        return Flux2DiTAdapter
    if name == "QwenImageDiTAdapter":
        from .diffusion import QwenImageDiTAdapter

        return QwenImageDiTAdapter
    if name == "CosmosTransferDiTAdapter":
        from .diffusion import CosmosTransferDiTAdapter

        return CosmosTransferDiTAdapter
    if name == "CosmosOfficialMultibranchAdapter":
        from .diffusion import CosmosOfficialMultibranchAdapter

        return CosmosOfficialMultibranchAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelAdapter",
    "HFLLMAdapter",
    "DiffusionTransformerAdapter",
    "IrodoriDiTAdapter",
    "Flux2DiTAdapter",
    "QwenImageDiTAdapter",
    "CosmosTransferDiTAdapter",
    "CosmosOfficialMultibranchAdapter",
]

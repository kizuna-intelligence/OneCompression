"""Diffusion / flow-matching transformer adapters.

``DiffusionTransformerAdapter`` is the architecture-agnostic base; concrete
adapters (e.g. :class:`IrodoriDiTAdapter`) are imported lazily so their
heavy, optional dependencies (``irodori_tts``, ``diffusers``…) stay
optional.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .base import DiffusionTransformerAdapter
from .distributions import (
    GaussianNoise,
    LogitNormalTimestep,
    Sampler,
    UniformTimestep,
)


def __getattr__(name):  # PEP 562 lazy import
    if name == "IrodoriDiTAdapter":
        from .irodori import IrodoriDiTAdapter

        return IrodoriDiTAdapter
    if name == "Flux2DiTAdapter":
        from .flux2 import Flux2DiTAdapter

        return Flux2DiTAdapter
    if name == "QwenImageDiTAdapter":
        from .qwenimage import QwenImageDiTAdapter

        return QwenImageDiTAdapter
    if name == "CosmosTransferDiTAdapter":
        from .cosmos import CosmosTransferDiTAdapter

        return CosmosTransferDiTAdapter
    if name == "CosmosOfficialMultibranchAdapter":
        from .cosmos_official_multibranch import CosmosOfficialMultibranchAdapter

        return CosmosOfficialMultibranchAdapter
    if name == "CosmosPredictDiTAdapter":
        from .cosmos_predict import CosmosPredictDiTAdapter

        return CosmosPredictDiTAdapter
    if name == "WanVACE14BDiTAdapter":
        from .wan_vace import WanVACE14BDiTAdapter

        return WanVACE14BDiTAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiffusionTransformerAdapter",
    "IrodoriDiTAdapter",
    "Flux2DiTAdapter",
    "QwenImageDiTAdapter",
    "CosmosTransferDiTAdapter",
    "CosmosOfficialMultibranchAdapter",
    "CosmosPredictDiTAdapter",
    "WanVACE14BDiTAdapter",
    "Sampler",
    "GaussianNoise",
    "UniformTimestep",
    "LogitNormalTimestep",
]

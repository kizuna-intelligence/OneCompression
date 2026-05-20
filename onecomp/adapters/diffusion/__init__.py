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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiffusionTransformerAdapter",
    "IrodoriDiTAdapter",
    "Flux2DiTAdapter",
    "Sampler",
    "GaussianNoise",
    "UniformTimestep",
    "LogitNormalTimestep",
]

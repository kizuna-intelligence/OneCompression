"""Pluggable sampling distributions for synthetic DiT calibration.

When no *real* captured activations are available, the diffusion adapters
fall back to synthesising calibration inputs.  The noised latent and the
timestep should not be hard-coded to "Gaussian latent + uniform t" —
different architectures and schedulers use different distributions.  In
particular, flow-matching models (SD3 / FLUX) sample the timestep from a
**logit-normal**, not a uniform.

These samplers are deliberately tiny ``nn``-free callables that respect an
explicit ``torch.Generator`` so synthetic calibration stays reproducible
(``torch.distributions`` does not accept a generator).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    """A reproducible distribution over tensors of a requested shape."""

    @abstractmethod
    def sample(
        self,
        shape: tuple[int, ...],
        generator: torch.Generator,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ...


class GaussianNoise(Sampler):
    """Standard-normal latent noise (the usual diffusion prior)."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, shape, generator, dtype=torch.float32) -> torch.Tensor:
        x = torch.randn(shape, generator=generator, dtype=torch.float32)
        if self.mean != 0.0 or self.std != 1.0:
            x = x * self.std + self.mean
        return x.to(dtype)


class UniformTimestep(Sampler):
    """Uniform timestep in ``[low, high]`` (classic DDPM-style sampling)."""

    def __init__(self, low: float = 0.001, high: float = 0.999):
        self.low = float(low)
        self.high = float(high)

    def sample(self, shape, generator, dtype=torch.float32) -> torch.Tensor:
        u = torch.rand(shape, generator=generator, dtype=torch.float32)
        return (u * (self.high - self.low) + self.low).to(dtype)


class LogitNormalTimestep(Sampler):
    """Logit-normal timestep ``sigmoid(mu + sigma * N(0,1))`` in (0, 1).

    This is the flow-matching timestep distribution used by SD3 / FLUX,
    concentrating samples near the middle of the trajectory.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample(self, shape, generator, dtype=torch.float32) -> torch.Tensor:
        z = torch.randn(shape, generator=generator, dtype=torch.float32)
        return torch.sigmoid(self.mu + self.sigma * z).to(dtype)

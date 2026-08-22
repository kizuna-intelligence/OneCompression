"""Model adapter abstract base class.

Adapters decouple OneCompression's quantization machinery from
HuggingFace-specific assumptions so that custom model architectures
(diffusion transformers, encoder-decoder TTS, etc.) can be quantized
through the same Runner / AutoBit / QEP pipeline.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ModelAdapter(ABC):
    """Bridge between a concrete model and OneCompression's pipeline.

    A concrete adapter owns:

    - **Model lifecycle**: load weights, save quantized output.
    - **Structural queries**: which ``nn.ModuleList`` are the quantization
      blocks, which modules form the output head, the final norm, etc.
    - **Calibration data**: produces a dict of inputs that can be fed
      via ``run_calibration_forward`` to drive a single forward pass
      through every block.
    - **Loss for AutoBit curvature**: the per-sample scalar loss used to
      derive ``b`` (output-gradient) statistics.  For LLMs this is
      cross-entropy on shifted labels; for a velocity-prediction
      diffusion model this is MSE between predicted and target velocity.

    Adapters do **not** own the quantizer — they describe *how* to
    feed data through the model so the quantizer can hook into it.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def load_model(self, device_map: str | None = None) -> nn.Module:
        """Return a freshly loaded model on the requested device."""

    def load_tokenizer(self):
        """Return a tokenizer or ``None`` if the model has none.

        Default implementation returns ``None``; HF adapter overrides.
        """
        return None

    @abstractmethod
    def save_quantized_model(self, runner, save_directory: str) -> str:
        """Persist the quantized model to ``save_directory``.

        ``runner`` provides ``runner.create_quantized_model()`` for
        adapters that want to reuse OneCompression's quantized-Linear
        replacement; adapters are free to write the model in their
        native format (safetensors with custom metadata, etc.).
        """

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------
    @abstractmethod
    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        """Return the ``nn.ModuleList`` of quantization blocks.

        These are the units through which calibration activations flow.
        For an LLM this is ``model.model.layers``; for a DiT it is
        ``model.blocks``.
        """

    def get_head_modules(
        self, model: nn.Module
    ) -> tuple[nn.Module | None, nn.Module | None]:
        """Return ``(final_norm, output_head)`` modules.

        Used by AutoBit's curvature path to backprop loss gradients
        from the head into the last block's output.  Adapters that
        cannot supply these must return ``(None, None)`` and accept
        that AutoBit will be forced to ``use_curvature_b=False``.
        """
        return None, None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    @abstractmethod
    def prepare_calibration_inputs(
        self,
        model: nn.Module,
        calibration_config,
        device: torch.device,
        logger=None,
    ) -> dict[str, Any]:
        """Return a dict of model inputs.

        The dict's tensors share a leading batch dimension equal to
        ``calibration_config.num_calibration_samples``.  Non-tensor
        entries are passed through verbatim and re-attached after each
        slice via :meth:`slice_calibration_inputs`.
        """

    @abstractmethod
    def num_calibration_samples(self, inputs: dict[str, Any]) -> int:
        """Return the number of calibration samples in ``inputs``."""

    @abstractmethod
    def slice_calibration_inputs(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        """Return a sub-dict where every batched tensor is sliced [start:end]."""

    @abstractmethod
    def run_calibration_forward(
        self, model: nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor | None:
        """Run a single forward pass with ``inputs``.

        OneCompression typically wraps ``blocks[0]`` with a Catcher that
        raises ``StopForward`` after capturing the first block's input,
        so this method should let that exception propagate (or catch
        it) without aborting the surrounding pipeline.

        For full forwards used by curvature loss, return the model output.
        """

    # ------------------------------------------------------------------
    # AutoBit curvature loss
    # ------------------------------------------------------------------
    def compute_curvature_loss(
        self,
        final_hidden: torch.Tensor,
        norm: nn.Module,
        head: nn.Module,
        sample_inputs: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute scalar loss for one sample given the last-block output.

        Used to derive ``b`` (output-gradient curvature) for
        activation-aware AutoBit.  Default raises ``NotImplementedError``
        — subclasses opt in.
        """
        raise NotImplementedError(
            "compute_curvature_loss is not implemented for this adapter; "
            "set AutoBitQuantizer(use_curvature_b=False) to skip the "
            "curvature term."
        )

    # ------------------------------------------------------------------
    # Optional VRAM estimation
    # ------------------------------------------------------------------
    def get_quantizable_param_count(self, model: nn.Module) -> int:
        """Return total parameter count of quantizable Linear weights.

        Used by ``estimate_wbits_from_vram`` for VRAM-aware target
        bitwidth estimation.  Default counts every Linear weight.
        """
        total = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total += module.weight.numel()
        return total

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def has_additional_data(self) -> bool:
        """Return ``True`` if this adapter registers extra hooks/state.

        Mirrors :meth:`ModelConfig.has_additional_data` so that callers
        like the quantized-save path can warn the user.
        """
        return False

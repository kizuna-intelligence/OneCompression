"""HuggingFace causal-LM adapter (default behavior).

Wraps the historical OneCompression flow: AutoModelForCausalLM /
AutoTokenizer loading, ``model.model.layers`` block discovery, text-only
calibration via ``prepare_calibration_dataset``, and cross-entropy
loss for AutoBit curvature.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base import ModelAdapter


class HFLLMAdapter(ModelAdapter):
    """Adapter for HuggingFace causal LMs (Llama, Qwen3, Gemma, ...).

    Reproduces the behavior that OneCompression had before adapters
    existed.  Constructed automatically by :class:`ModelConfig` when
    no explicit adapter is supplied.
    """

    def __init__(
        self,
        model_id: str | None = None,
        path: str | None = None,
        dtype: str = "float16",
        device: str = "auto",
    ):
        if model_id is None and path is None:
            raise ValueError("Either model_id or path must be provided")
        self.model_id = model_id
        self.path = path
        self.dtype = dtype
        self.device = device
        self.logger = getLogger(__name__)
        self._cached_config = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def get_model_id_or_path(self) -> str:
        return self.model_id if self.model_id is not None else self.path

    def load_config(self):
        if self._cached_config is None:
            from transformers import AutoConfig

            self._cached_config = AutoConfig.from_pretrained(
                self.get_model_id_or_path(), trust_remote_code=True
            )
        return self._cached_config

    def load_model(self, device_map: str | None = None) -> nn.Module:
        from transformers import AutoModelForCausalLM

        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM

            has_vlm = True
        except ImportError:
            has_vlm = False

        effective_device = device_map if device_map is not None else self.device
        kwargs = dict(
            dtype=self.dtype if self.dtype == "auto" else getattr(torch, self.dtype),
            device_map=effective_device,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.get_model_id_or_path(), **kwargs
            )
        except ValueError as exc:
            hints = (
                "Unrecognized configuration class",
                "Unrecognized model",
                "is not supported",
            )
            if not has_vlm or not any(h in str(exc) for h in hints):
                raise
            self.logger.info(
                "AutoModelForCausalLM failed; trying AutoModelForImageTextToText."
            )
            model = _AutoVLM.from_pretrained(self.get_model_id_or_path(), **kwargs)
        model.eval()
        self.logger.info(
            "Model loaded with dtype=%s", next(model.parameters()).dtype
        )
        return model

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.get_model_id_or_path())
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info("pad_token is not set. Using eos_token as pad_token.")
        return tokenizer

    def save_quantized_model(self, runner, save_directory: str) -> str:
        # HF causal LMs are saved by ``Runner.save_quantized_model``
        # directly (the runner skips adapter dispatch for HFLLMAdapter
        # so this method is effectively a no-op safety hatch — invoked
        # only if a caller bypasses the runner shortcut).
        return runner.save_quantized_model(save_directory)

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------
    _VLM_TEXT_SUFFIXES = ("language_model", "text_model")

    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        from transformers.modeling_layers import GradientCheckpointingLayer

        search_root = model
        for name, mod in model.named_modules():
            if any(name.endswith(s) for s in self._VLM_TEXT_SUFFIXES):
                search_root = mod
                self.logger.info(
                    "Using text submodel: %s (%s)", name, type(mod).__name__
                )
                break

        for module in search_root.modules():
            if isinstance(module, nn.ModuleList):
                if len(module) > 0 and isinstance(
                    module[0], GradientCheckpointingLayer
                ):
                    return module

        raise RuntimeError("Transformer blocks not found.")

    def get_head_modules(
        self, model: nn.Module
    ) -> tuple[nn.Module | None, nn.Module | None]:
        blocks = self.get_blocks(model)
        parent = None
        for name, module in model.named_modules():
            if module is blocks:
                parent_name = name.rpartition(".")[0]
                parent = (
                    model.get_submodule(parent_name) if parent_name else model
                )
                break

        norm = None
        for attr in ("norm", "final_layer_norm", "ln_f"):
            norm = getattr(parent, attr, None)
            if norm is not None:
                break

        head = None
        for attr in ("lm_head", "embed_out", "output"):
            head = getattr(model, attr, None)
            if head is not None:
                break

        return norm, head

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def prepare_calibration_inputs(
        self,
        model: nn.Module,
        calibration_config,
        device: torch.device,
        logger=None,
    ) -> dict[str, Any]:
        from ..calibration import prepare_calibration_dataset

        return prepare_calibration_dataset(
            tokenizer=self.load_tokenizer(),
            device=device,
            calibration_config=calibration_config,
            logger=logger or self.logger,
            model=model,
        )

    def num_calibration_samples(self, inputs: dict[str, Any]) -> int:
        return inputs["input_ids"].shape[0]

    def slice_calibration_inputs(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        sliced = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                sliced[k] = v[start:end]
            else:
                sliced[k] = v
        return sliced

    def run_calibration_forward(
        self, model: nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor | None:
        # The legacy flow always splits on input_ids and passes the
        # remaining keys as kwargs; replicate that here.
        input_ids = inputs["input_ids"]
        kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
        kwargs.setdefault("use_cache", False)
        return model(input_ids, **kwargs)

    # ------------------------------------------------------------------
    # AutoBit curvature loss (cross-entropy on shifted labels)
    # ------------------------------------------------------------------
    def compute_curvature_loss(
        self,
        final_hidden: torch.Tensor,
        norm: nn.Module,
        head: nn.Module,
        sample_inputs: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        normed = norm(final_hidden)
        logits = head(normed)
        ids = sample_inputs["input_ids"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def has_additional_data(self) -> bool:
        return False

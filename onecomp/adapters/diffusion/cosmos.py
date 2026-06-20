"""Cosmos Transfer2.5 ``CosmosTransformer3DModel`` adapter.

The NVIDIA Cosmos Transfer2.5 base denoiser is a single-stream 3D DiT:
``transformer_blocks`` carry one latent token stream, while text/image
conditioning, timestep embeddings and RoPE tensors are fixed block kwargs.
Diffusers calls each block with positional auxiliary arguments, so this adapter
wraps blocks in a tiny kwargs-compatible shim for OneCompression's block-wise
QEP loop.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch import nn

from .base import DiffusionTransformerAdapter
from .distributions import LogitNormalTimestep, Sampler


class _CosmosBlockKwargsWrapper(nn.Module):
    """Expose a Cosmos block through a kwargs-friendly forward signature."""

    _ARG_NAMES = (
        "encoder_hidden_states",
        "embedded_timestep",
        "temb",
        "image_rotary_emb",
        "extra_pos_emb",
        "attention_mask",
        "controlnet_residual",
    )

    def __init__(self, block: nn.Module, block_idx: int):
        super().__init__()
        self.block = block
        self._onecomp_block_idx = int(block_idx)

    @property
    def _onecomp_wrapped_block(self) -> nn.Module:
        return self.block

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if args:
            kwargs = dict(kwargs)
            for name, value in zip(self._ARG_NAMES, args):
                kwargs.setdefault(name, value)
        return self.block(hidden_states, **kwargs)


class CosmosTransferDiTAdapter(DiffusionTransformerAdapter):
    """Plug diffusers ``CosmosTransformer3DModel`` into OneCompression."""

    PRIMARY_LATENT_KEY = "hidden_states"

    # The quantization script scopes GPTQ to ``transformer_blocks``.  Within
    # those blocks we default to full int4 coverage, including AdaLN LoRA
    # projections, because the requested target is an aggressive 24GB setup.
    _EXCLUDE_KEYWORDS: tuple[str, ...] = ()

    _MODEL_FORWARD_KEYS = (
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "block_controlnet_hidden_states",
        "attention_mask",
        "fps",
        "condition_mask",
        "padding_mask",
    )

    _BATCH_INDEPENDENT_KEYS = ("padding_mask", "fps")

    def __init__(
        self,
        checkpoint_path: str,
        dtype: str | torch.dtype = "bfloat16",
        device: str = "cpu",
        seed: int = 0,
        calibration_inputs_path: str | None = None,
        noise_sampler: Sampler | None = None,
        timestep_sampler: Sampler | None = None,
        revision: str | None = "diffusers/general",
        subfolder: str = "transformer",
        single_file: bool = False,
        latent_frames: int = 3,
        latent_height: int = 16,
        latent_width: int = 16,
        text_seq_len: int = 512,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            dtype=dtype,
            device=device,
            seed=seed,
            calibration_inputs_path=calibration_inputs_path,
            noise_sampler=noise_sampler,
            timestep_sampler=timestep_sampler or LogitNormalTimestep(),
        )
        self.revision = revision
        self.subfolder = subfolder
        self.single_file = bool(single_file)
        self.latent_frames = int(latent_frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.text_seq_len = int(text_seq_len)
        self._config: dict[str, Any] | None = None
        self._wrap_block_counter = 0
        self._current_block_controlnet_hidden_states: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self, device_map: str | None = None) -> nn.Module:
        from diffusers import CosmosTransformer3DModel

        target_device = device_map if device_map is not None else self.device
        if target_device in (None, "auto"):
            target_device = "cpu"

        is_file = os.path.isfile(os.path.expanduser(self.checkpoint_path))
        if self.single_file or is_file:
            model = CosmosTransformer3DModel.from_single_file(
                os.path.expanduser(self.checkpoint_path),
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
        else:
            model = CosmosTransformer3DModel.from_pretrained(
                self.checkpoint_path,
                subfolder=self.subfolder,
                revision=self.revision,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
        self._config = dict(model.config)
        model = model.to(device=target_device)
        model.eval()
        self.logger.info(
            "Cosmos transformer loaded from %s (revision=%s, dtype=%s, "
            "device=%s, %d blocks)",
            self.checkpoint_path,
            self.revision,
            self.dtype,
            target_device,
            len(model.transformer_blocks),
        )
        return model

    # ------------------------------------------------------------------
    # Structural queries and positional-arg block support.
    # ------------------------------------------------------------------
    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        blocks = getattr(model, "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError(
                "CosmosTransferDiTAdapter expected model.transformer_blocks "
                f"to be an nn.ModuleList; got {type(model).__name__}"
            )
        self._wrap_block_counter = 0
        return blocks

    def wrap_block(self, block: nn.Module) -> nn.Module:
        idx = self._wrap_block_counter
        self._wrap_block_counter += 1
        return _CosmosBlockKwargsWrapper(block, idx)

    def pack_catcher_input(
        self, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not args:
            raise RuntimeError("Cosmos block catcher expected positional args")
        hidden_states = args[0]
        packed_kwargs = dict(kwargs)
        for name, value in zip(_CosmosBlockKwargsWrapper._ARG_NAMES, args[1:]):
            packed_kwargs.setdefault(name, value)
        per_block = self._build_controlnet_residual_map()
        if per_block:
            packed_kwargs.pop("controlnet_residual", None)
            packed_kwargs["_per_block_kwargs"] = per_block
        return hidden_states, packed_kwargs

    def _build_controlnet_residual_map(self) -> dict[int, dict[str, Any]]:
        residuals = self._current_block_controlnet_hidden_states
        if not residuals:
            return {}
        if self._config is None:
            raise RuntimeError("Cosmos config is not available")
        every_n = int(self._config.get("controlnet_block_every_n", 1))
        n_blocks = int(self._config.get("num_layers", 0)) or len(residuals) * every_n
        return {
            block_idx: {"controlnet_residual": residual}
            for residual, block_idx in zip(residuals, range(0, n_blocks, every_n))
        }

    # ------------------------------------------------------------------
    # Calibration.
    # ------------------------------------------------------------------
    def prepare_calibration_inputs(
        self,
        model: nn.Module,
        calibration_config,
        device: torch.device,
        logger=None,
    ) -> dict[str, Any]:
        if self._config is None:
            self._config = dict(model.config)
        cfg = self._config
        log = logger or self.logger

        if self.calibration_inputs_path is not None:
            return self._load_real_calibration_inputs(calibration_config, device, log)

        n = int(calibration_config.num_calibration_samples)
        latent_c = int(cfg["in_channels"]) - 1
        if cfg.get("use_crossattn_projection"):
            text_dim = int(
                cfg.get(
                    "crossattn_proj_in_channels",
                    cfg.get("encoder_hidden_states_channels", cfg.get("text_embed_dim", 1024)),
                )
            )
        else:
            text_dim = int(cfg.get("text_embed_dim", cfg.get("encoder_hidden_states_channels", 1024)))
        img_context_dim = cfg.get("img_context_dim_in")

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype = self.dtype
        hidden_states = self.noise_sampler.sample(
            (
                n,
                latent_c,
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            ),
            gen,
            dtype,
        )
        timestep = self.timestep_sampler.sample(
            (n, 1, self.latent_frames, 1, 1), gen, dtype
        )
        text_context = self.noise_sampler.sample(
            (n, self.text_seq_len, text_dim), gen, dtype
        )
        if img_context_dim:
            img_context = torch.zeros(
                n,
                int(cfg.get("img_context_num_tokens", 256)),
                int(img_context_dim),
                dtype=dtype,
            )
            encoder_hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = (
                text_context,
                img_context,
            )
        else:
            encoder_hidden_states = text_context

        condition_mask = torch.zeros(
            n,
            1,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            dtype=dtype,
        )
        # Cosmos' pipeline passes a batch-independent image-space padding mask.
        # A latent-size zero mask keeps synthetic calibration small and is
        # resized to the same latent size by the model.
        padding_mask = torch.zeros(
            1,
            1,
            self.latent_height,
            self.latent_width,
            dtype=dtype,
        )

        inputs: dict[str, Any] = {
            "hidden_states": hidden_states.to(device),
            "timestep": timestep.to(device),
            "encoder_hidden_states": self._move_value(encoder_hidden_states, device),
            "condition_mask": condition_mask.to(device),
            "padding_mask": padding_mask.to(device),
            # Keep QEP block kwargs uniform across blocks.  Real Transfer
            # inference injects per-block ControlNet residuals, but the current
            # block-wise QEP loop has no per-block kwarg dispatch.
            "block_controlnet_hidden_states": None,
        }
        log.info(
            "Cosmos calibration: %d samples, latent=(%d,%d,%d,%d), "
            "text_seq=%d, text_dim=%d, img_context=%s",
            n,
            latent_c,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            self.text_seq_len,
            text_dim,
            bool(img_context_dim),
        )
        return inputs

    def _load_real_calibration_inputs(
        self, calibration_config, device: torch.device, log
    ) -> dict[str, Any]:
        blob = torch.load(self.calibration_inputs_path, map_location="cpu")
        n_avail = int(blob["hidden_states"].shape[0])
        n = min(int(calibration_config.num_calibration_samples), n_avail)

        def take(key):
            value = blob.get(key)
            return self._slice_value(value, 0, n, n_avail, dtype=self.dtype, device=device)

        inputs = {
            "hidden_states": take("hidden_states"),
            "timestep": take("timestep"),
            "encoder_hidden_states": take("encoder_hidden_states"),
            "attention_mask": take("attention_mask"),
            "condition_mask": take("condition_mask"),
            "padding_mask": blob.get("padding_mask"),
            "fps": take("fps"),
            "block_controlnet_hidden_states": blob.get("block_controlnet_hidden_states"),
        }
        if isinstance(inputs["padding_mask"], torch.Tensor):
            if inputs["padding_mask"].dim() >= 1:
                inputs["padding_mask"] = inputs["padding_mask"][:1]
            inputs["padding_mask"] = inputs["padding_mask"].to(device=device, dtype=self.dtype)
        if isinstance(inputs["block_controlnet_hidden_states"], list):
            inputs["block_controlnet_hidden_states"] = [
                self._slice_value(v, 0, n, n_avail, dtype=self.dtype, device=device)
                for v in inputs["block_controlnet_hidden_states"]
            ]
        log.info(
            "Cosmos REAL calibration: %d samples from %s (available=%d)",
            n,
            self.calibration_inputs_path,
            n_avail,
        )
        return inputs

    def slice_calibration_inputs(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        n_total = self.num_calibration_samples(inputs)

        def slice_or_keep(k: str, v: Any) -> Any:
            if k in self._BATCH_INDEPENDENT_KEYS:
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == n_total:
                    return self._slice_value(v, start, end, n_total)
                return v
            if k == "block_controlnet_hidden_states":
                return v
            return self._slice_value(v, start, end, n_total)

        return {
            k: slice_or_keep(k, v)
            for k, v in inputs.items()
        }

    @classmethod
    def _slice_value(
        cls,
        value: Any,
        start: int,
        end: int,
        n_total: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            out = value[start:end] if value.dim() >= 1 and value.shape[0] == n_total else value
            if dtype is not None and out.dtype.is_floating_point:
                out = out.to(dtype=dtype)
            return out.to(device) if device is not None else out
        if isinstance(value, tuple):
            return tuple(cls._slice_value(v, start, end, n_total, dtype, device) for v in value)
        if isinstance(value, list):
            return [cls._slice_value(v, start, end, n_total, dtype, device) for v in value]
        return value

    @classmethod
    def _move_value(cls, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(cls._move_value(v, device) for v in value)
        if isinstance(value, list):
            return [cls._move_value(v, device) for v in value]
        return value

    def run_calibration_forward(
        self, model: nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor | None:
        fwd = {k: inputs[k] for k in self._MODEL_FORWARD_KEYS if k in inputs}
        fwd = self._align_forward_batch(fwd)
        self._current_block_controlnet_hidden_states = fwd.get("block_controlnet_hidden_states")
        try:
            return model(**fwd, return_dict=False)[0]
        finally:
            self._current_block_controlnet_hidden_states = None

    @classmethod
    def _align_forward_batch(cls, fwd: dict[str, Any]) -> dict[str, Any]:
        hidden = fwd.get("hidden_states")
        if not isinstance(hidden, torch.Tensor):
            return fwd
        batch = int(hidden.shape[0])

        def align(value: Any) -> Any:
            if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] > batch:
                return value[:batch]
            if isinstance(value, tuple):
                return tuple(align(v) for v in value)
            return value

        out = dict(fwd)
        for key in (
            "timestep",
            "encoder_hidden_states",
            "attention_mask",
            "condition_mask",
            "padding_mask",
            "fps",
        ):
            if key in out:
                out[key] = align(out[key])
        return out

    # ------------------------------------------------------------------
    # Save / load metadata.
    # ------------------------------------------------------------------
    def build_save_metadata(
        self, quant_layers: list[dict[str, Any]]
    ) -> dict[str, str]:
        if self._config is None:
            raise RuntimeError(
                "CosmosTransferDiTAdapter.build_save_metadata called before "
                "load_model; the diffusers config has not been captured."
            )
        clean_cfg = {k: v for k, v in self._config.items() if not k.startswith("_")}
        return {
            "config_json": json.dumps(clean_cfg, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "qep_gptq",
            "checkpoint_format": "gptq",
            "base_model": "nvidia/Cosmos-Transfer2.5-2B",
        }

    @classmethod
    def _instantiate_model(
        cls, cfg_dict: dict[str, Any], dtype: torch.dtype
    ) -> nn.Module:
        from diffusers import CosmosTransformer3DModel

        clean_cfg = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
        model = CosmosTransformer3DModel.from_config(clean_cfg)
        return model.to(dtype=dtype)

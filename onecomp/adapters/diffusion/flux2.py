"""FLUX.2 (``Flux2Transformer2DModel``) diffusion-transformer adapter.

A concrete :class:`DiffusionTransformerAdapter` for Black Forest Labs'
FLUX.2 flow-matching image DiT, loaded through ``diffusers``.

FLUX.2 is a **two-stream** architecture, which matters for how much of it
OneCompression's block-wise quantizer can reach:

- ``transformer_blocks`` (double-stream) consume *and return* two evolving
  residual streams ``(encoder_hidden_states, hidden_states)``.  The
  block-input catcher in :mod:`onecomp.utils.blockwise` propagates a
  **single** stream (it keeps ``out[0]`` and freezes block kwargs), so the
  double-stream blocks do not fit that contract and are left unquantized
  for now.
- ``single_transformer_blocks`` (single-stream) take the concatenated
  ``[text, image]`` sequence as one ``hidden_states`` tensor and return it
  unchanged in shape — a clean single residual stream.  These are the bulk
  of the model (20 vs 5 here) and are what this adapter quantizes.

The timestep is sampled from a **logit-normal** prior (the SD3 / FLUX
flow-matching default) for synthetic calibration; real captured
activations remain the better calibration source when available.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import nn

from .base import DiffusionTransformerAdapter
from .distributions import LogitNormalTimestep, Sampler


class Flux2DiTAdapter(DiffusionTransformerAdapter):
    """Plug ``Flux2Transformer2DModel`` (single-stream blocks) into OneComp."""

    #: FLUX names its primary latent ``hidden_states``, not ``x_t``.
    PRIMARY_LATENT_KEY = "hidden_states"

    #: Input/output projections, modulation and timestep MLPs are tiny and
    #: quality-sensitive — keep them in their loaded dtype.  (Plain GPTQ
    #: quantizes every non-excluded Linear in the model, including the
    #: double-stream blocks; ``get_blocks`` only scopes the QEP/AutoBit
    #: block-propagation path to the single-stream blocks.)
    _EXCLUDE_KEYWORDS = (
        "x_embedder",
        "context_embedder",
        "time_guidance_embed",
        "modulation",
        "norm_out",
        "proj_out",
    )

    def __init__(
        self,
        checkpoint_path: str,
        dtype: str | torch.dtype = "bfloat16",
        device: str = "cpu",
        seed: int = 0,
        calibration_inputs_path: str | None = None,
        noise_sampler: Sampler | None = None,
        timestep_sampler: Sampler | None = None,
        image_grid: int = 32,
        text_seq_len: int = 128,
    ):
        # Default the timestep prior to flow-matching's logit-normal.
        super().__init__(
            checkpoint_path=checkpoint_path,
            dtype=dtype,
            device=device,
            seed=seed,
            calibration_inputs_path=calibration_inputs_path,
            noise_sampler=noise_sampler,
            timestep_sampler=timestep_sampler or LogitNormalTimestep(),
        )
        self.image_grid = int(image_grid)
        self.text_seq_len = int(text_seq_len)
        self._config: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self, device_map: str | None = None) -> nn.Module:
        from diffusers import Flux2Transformer2DModel

        target_device = device_map if device_map is not None else self.device
        if target_device in (None, "auto"):
            target_device = "cpu"

        model = Flux2Transformer2DModel.from_pretrained(
            self.checkpoint_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        self._config = dict(model.config)
        model = model.to(device=target_device)
        model.eval()
        self.logger.info(
            "FLUX.2 transformer loaded from %s (dtype=%s, device=%s, "
            "%d double + %d single blocks)",
            self.checkpoint_path,
            self.dtype,
            target_device,
            len(model.transformer_blocks),
            len(model.single_transformer_blocks),
        )
        return model

    # ------------------------------------------------------------------
    # Structural queries: quantize the single-stream blocks only.
    # ------------------------------------------------------------------
    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        blocks = getattr(model, "single_transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError(
                "Flux2DiTAdapter expected model.single_transformer_blocks to "
                "be an nn.ModuleList; got %r" % (type(model).__name__,)
            )
        return blocks

    #: ``img_ids`` / ``txt_ids`` are shared RoPE position grids of shape
    #: ``[seq, 4]`` with no batch dimension; slicing them on dim 0 (as the
    #: generic base does for batched tensors) would corrupt the position
    #: grid, so they must be passed through untouched.
    _BATCH_INDEPENDENT_KEYS = ("img_ids", "txt_ids")

    def slice_calibration_inputs(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for k, v in inputs.items():
            if k in self._BATCH_INDEPENDENT_KEYS:
                sliced[k] = v
            elif isinstance(v, torch.Tensor) and v.dim() >= 1:
                sliced[k] = v[start:end]
            else:
                sliced[k] = v
        return sliced

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _build_position_ids(
        self, img_seq: int, txt_seq: int, grid: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct FLUX RoPE position ids of shape ``[seq, 4]``.

        Text tokens sit at the origin (all-zero ids, as in the diffusers
        FLUX pipelines); image tokens carry their (row, col) on axes 1 and 2
        of the 4-axis rope layout.
        """
        txt_ids = torch.zeros((txt_seq, 4), dtype=torch.float32)
        rows = torch.arange(grid).repeat_interleave(grid)[:img_seq]
        cols = torch.arange(grid).repeat(grid)[:img_seq]
        img_ids = torch.zeros((img_seq, 4), dtype=torch.float32)
        img_ids[:, 1] = rows
        img_ids[:, 2] = cols
        return img_ids, txt_ids

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
            return self._load_real_calibration_inputs(
                calibration_config, device, log
            )

        n = int(calibration_config.num_calibration_samples)
        grid = self.image_grid
        img_seq = grid * grid
        txt_seq = self.text_seq_len

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype = self.dtype

        hidden_states = self.noise_sampler.sample(
            (n, img_seq, int(cfg["in_channels"])), gen, dtype
        )
        encoder_hidden_states = self.noise_sampler.sample(
            (n, txt_seq, int(cfg["joint_attention_dim"])), gen, dtype
        )
        timestep = self.timestep_sampler.sample((n,), gen, dtype)
        img_ids, txt_ids = self._build_position_ids(img_seq, txt_seq, grid)

        inputs: dict[str, Any] = {
            "hidden_states": hidden_states.to(device),
            "encoder_hidden_states": encoder_hidden_states.to(device),
            "timestep": timestep.to(device),
            "img_ids": img_ids.to(device),
            "txt_ids": txt_ids.to(device),
        }
        if cfg.get("guidance_embeds"):
            inputs["guidance"] = torch.full(
                (n,), 4.0, dtype=dtype, device=device
            )

        log.info(
            "FLUX.2 calibration: %d samples, img_seq=%d (%dx%d), txt_seq=%d, "
            "guidance=%s",
            n,
            img_seq,
            grid,
            grid,
            txt_seq,
            bool(cfg.get("guidance_embeds")),
        )
        return inputs

    def _load_real_calibration_inputs(
        self, calibration_config, device: torch.device, log
    ) -> dict[str, Any]:
        """Load real FLUX.2 forward inputs captured from genuine generations.

        Expects a ``torch.save`` dict with ``hidden_states``,
        ``encoder_hidden_states``, ``timestep`` stacked on dim 0, plus the
        shared ``img_ids`` / ``txt_ids`` (and optional ``guidance``).
        """
        payload = torch.load(self.calibration_inputs_path, map_location="cpu")
        n_avail = int(payload["hidden_states"].shape[0])
        n = min(int(calibration_config.num_calibration_samples), n_avail)

        def _f(key):
            v = payload.get(key)
            return None if v is None else v[:n].to(self.dtype).to(device)

        inputs: dict[str, Any] = {
            "hidden_states": _f("hidden_states"),
            "encoder_hidden_states": _f("encoder_hidden_states"),
            "timestep": _f("timestep"),
            "img_ids": payload["img_ids"].to(device),
            "txt_ids": payload["txt_ids"].to(device),
        }
        if payload.get("guidance") is not None:
            inputs["guidance"] = _f("guidance")

        log.info(
            "FLUX.2 calibration: %d real samples from %s (available=%d)",
            n,
            self.calibration_inputs_path,
            n_avail,
        )
        return inputs

    _MODEL_FORWARD_KEYS = (
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "img_ids",
        "txt_ids",
        "guidance",
    )

    def run_calibration_forward(
        self, model: nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor | None:
        fwd = {k: inputs[k] for k in self._MODEL_FORWARD_KEYS if k in inputs}
        return model(**fwd, return_dict=False)[0]

    # ------------------------------------------------------------------
    # Save / load metadata: embed the diffusers config so the loader can
    # rebuild a fresh ``Flux2Transformer2DModel``.
    # ------------------------------------------------------------------
    def build_save_metadata(
        self, quant_layers: list[dict[str, Any]]
    ) -> dict[str, str]:
        if self._config is None:
            raise RuntimeError(
                "Flux2DiTAdapter.build_save_metadata called before load_model; "
                "the diffusers config has not been captured."
            )
        clean_cfg = {
            k: v for k, v in self._config.items() if not k.startswith("_")
        }
        return {
            "config_json": json.dumps(clean_cfg, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "autobit",
            "checkpoint_format": "gptq",
        }

    @classmethod
    def _instantiate_model(
        cls, cfg_dict: dict[str, Any], dtype: torch.dtype
    ) -> nn.Module:
        from diffusers import Flux2Transformer2DModel

        clean_cfg = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
        model = Flux2Transformer2DModel.from_config(clean_cfg)
        return model.to(dtype=dtype)

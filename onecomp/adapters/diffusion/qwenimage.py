"""Qwen-Image (``QwenImageTransformer2DModel``) diffusion-transformer adapter.

A concrete :class:`DiffusionTransformerAdapter` for the Qwen-Image MMDiT, the
backbone of Alibaba's Qwen-Image / Qwen-Image-Edit family and derivatives such
as FireRedTeam's ``FireRed-Image-Edit-1.0`` (``QwenImageEditPlusPipeline``).
Loaded through ``diffusers``.

Architecture
------------
Qwen-Image is a **dual-stream MMDiT**: every ``QwenImageTransformerBlock``
consumes *and returns* two evolving residual streams — an image stream
(``hidden_states``) and a text stream (``encoder_hidden_states``).  This is the
same shape contract as FLUX.2's double-stream blocks, so OneCompression's
single-stream block-input catcher (which keeps ``out[0]`` and freezes block
kwargs) cannot drive a QEP / block-propagation forward over these blocks.

Two quantization paths remain available:

* **RTN (calibration-free)** — ``Runner.quantize_without_calibration`` iterates
  the target ``nn.Linear`` modules and quantizes their weights directly from
  weight statistics, with no forward at all.  This is the path used for the
  20B transformer on a single 24GB GPU (the bf16 model is ~41GB, so it is kept
  on CPU and quantized per-layer there).
* **Plain GPTQ (single full forward)** — ``Runner.quantize_with_calibration``
  registers hooks on *all* Linears and runs one whole-model calibration forward
  (it does not use the block catcher), so the dual-stream blocks calibrate
  fine.  This needs the full model resident wherever the forward runs, so it is
  only practical when the model fits on the calibration device.  The synthetic
  calibration inputs below support it; ``timestep`` is drawn from the
  flow-matching prior.

Scope: quantize the ``transformer_blocks`` Linears only.  Drive scoping from
the quantizer's ``include_layer_keywords=["transformer_blocks"]`` plus this
adapter's :attr:`_EXCLUDE_KEYWORDS` for the AdaLN modulation projections
(``img_mod`` / ``txt_mod``), which emit scale/shift/gate tables and are both
cheap and quality-sensitive.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import nn

from .base import DiffusionTransformerAdapter
from .distributions import LogitNormalTimestep, Sampler


class QwenImageDiTAdapter(DiffusionTransformerAdapter):
    """Plug ``QwenImageTransformer2DModel`` into OneCompression."""

    #: Qwen-Image names its primary (image) latent ``hidden_states``.
    PRIMARY_LATENT_KEY = "hidden_states"

    #: In-block modulation (AdaLN) projections — emit scale/shift/gate tables,
    #: cheap and quality-sensitive, so left in their loaded dtype.  Top-level
    #: embedders / projections (``img_in``, ``txt_in``, ``time_text_embed``,
    #: ``norm_out``, ``proj_out``) sit outside ``transformer_blocks`` and are
    #: already excluded by ``include_layer_keywords=["transformer_blocks"]``.
    _EXCLUDE_KEYWORDS = (
        "img_mod",
        "txt_mod",
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
        subfolder: str = "transformer",
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
        self.image_grid = int(image_grid)
        self.text_seq_len = int(text_seq_len)
        self.subfolder = subfolder
        self._config: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self, device_map: str | None = None) -> nn.Module:
        from diffusers import QwenImageTransformer2DModel

        target_device = device_map if device_map is not None else self.device
        if target_device in (None, "auto"):
            target_device = "cpu"

        model = QwenImageTransformer2DModel.from_pretrained(
            self.checkpoint_path,
            subfolder=self.subfolder,
            torch_dtype=self.dtype,
        )
        self._config = dict(model.config)
        model = model.to(device=target_device)
        model.eval()
        self.logger.info(
            "Qwen-Image transformer loaded from %s (dtype=%s, device=%s, "
            "%d blocks)",
            self.checkpoint_path,
            self.dtype,
            target_device,
            len(model.transformer_blocks),
        )
        return model

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------
    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        blocks = getattr(model, "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError(
                "QwenImageDiTAdapter expected model.transformer_blocks to be "
                f"an nn.ModuleList; got {type(model).__name__}"
            )
        return blocks

    #: ``img_shapes`` (per-sample RoPE grid) and ``txt_seq_lens`` are Python
    #: lists with one entry per batch item; slice them like batched tensors.
    _BATCH_LIST_KEYS = ("img_shapes", "txt_seq_lens")

    def slice_calibration_inputs(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for k, v in inputs.items():
            if k in self._BATCH_LIST_KEYS and isinstance(v, list):
                sliced[k] = v[start:end]
            elif isinstance(v, torch.Tensor) and v.dim() >= 1:
                sliced[k] = v[start:end]
            else:
                sliced[k] = v
        return sliced

    # ------------------------------------------------------------------
    # Calibration (only reached by a calibration quantizer, e.g. GPTQ;
    # RTN runs no forward).
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
        encoder_hidden_states_mask = torch.ones(
            (n, txt_seq), dtype=torch.long
        )
        # Flow-matching timestep prior, scaled to the [0, 1000] range the
        # Qwen-Image timestep embedder expects.
        timestep = self.timestep_sampler.sample((n,), gen, dtype) * 1000.0
        img_shapes = [(1, grid, grid)] * n
        txt_seq_lens = [txt_seq] * n

        inputs: dict[str, Any] = {
            "hidden_states": hidden_states.to(device),
            "encoder_hidden_states": encoder_hidden_states.to(device),
            "encoder_hidden_states_mask": encoder_hidden_states_mask.to(device),
            "timestep": timestep.to(device),
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
        }
        if cfg.get("guidance_embeds"):
            inputs["guidance"] = torch.full((n,), 4.0, dtype=dtype, device=device)

        log.info(
            "Qwen-Image calibration: %d samples, img_seq=%d (%dx%d), txt_seq=%d, "
            "guidance=%s",
            n, img_seq, grid, grid, txt_seq, bool(cfg.get("guidance_embeds")),
        )
        return inputs

    _MODEL_FORWARD_KEYS = (
        "hidden_states",
        "encoder_hidden_states",
        "encoder_hidden_states_mask",
        "timestep",
        "img_shapes",
        "txt_seq_lens",
        "guidance",
    )

    def run_calibration_forward(
        self, model: nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor | None:
        fwd = {k: inputs[k] for k in self._MODEL_FORWARD_KEYS if k in inputs}
        return model(**fwd, return_dict=False)[0]

    # ------------------------------------------------------------------
    # Save / load metadata: embed the diffusers config so the loader can
    # rebuild a fresh ``QwenImageTransformer2DModel``.
    # ------------------------------------------------------------------
    def build_save_metadata(
        self, quant_layers: list[dict[str, Any]]
    ) -> dict[str, str]:
        if self._config is None:
            raise RuntimeError(
                "QwenImageDiTAdapter.build_save_metadata called before "
                "load_model; the diffusers config has not been captured."
            )
        clean_cfg = {
            k: v for k, v in self._config.items() if not k.startswith("_")
        }
        return {
            "config_json": json.dumps(clean_cfg, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "rtn",
            "checkpoint_format": "gptq",
        }

    @classmethod
    def _instantiate_model(
        cls, cfg_dict: dict[str, Any], dtype: torch.dtype
    ) -> nn.Module:
        from diffusers import QwenImageTransformer2DModel

        clean_cfg = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
        model = QwenImageTransformer2DModel.from_config(clean_cfg)
        return model.to(dtype=dtype)

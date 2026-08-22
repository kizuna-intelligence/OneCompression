"""Irodori-TTS :class:`TextToLatentRFDiT` diffusion-transformer adapter.

A concrete :class:`DiffusionTransformerAdapter` for the Irodori-TTS
rectified-flow text-to-latent DiT.  Everything architecture-agnostic
(curvature loss, calibration slicing, packed GPTQ save/load mechanics)
lives in the base; this subclass only fills in the Irodori specifics:

- **checkpoint loading** via ``irodori_tts.inference_runtime``,
- **block / head discovery** (``model.blocks`` and ``out_norm`` /
  ``out_proj``),
- **the forward signature** (text ids, ref latent, caption…),
- **the on-disk metadata schema** consumed by inference_runtime.

The heavy ``irodori_tts`` import is performed lazily inside the methods
that need it so the package stays an optional dependency.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import DiffusionTransformerAdapter
from .distributions import Sampler


class IrodoriDiTAdapter(DiffusionTransformerAdapter):
    """Plug ``TextToLatentRFDiT`` into the OneCompression pipeline."""

    #: AdaLN, cond/timestep MLPs, in/out projections and the encoders are
    #: tiny FP16 layers that must not be quantized.
    _EXCLUDE_KEYWORDS = (
        "adaln",
        "cond_module",
        "in_proj",
        "out_proj",
        "text_encoder",
        "caption_encoder",
        "speaker_encoder",
        "text_norm",
        "speaker_norm",
        "caption_norm",
        "out_norm",
    )

    def __init__(
        self,
        checkpoint_path: str,
        dtype: str | torch.dtype = "float32",
        device: str = "cpu",
        seed: int = 0,
        calibration_inputs_path: str | None = None,
        noise_sampler: Sampler | None = None,
        timestep_sampler: Sampler | None = None,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            dtype=dtype,
            device=device,
            seed=seed,
            calibration_inputs_path=calibration_inputs_path,
            noise_sampler=noise_sampler,
            timestep_sampler=timestep_sampler,
        )
        self._model_cfg = None
        self._train_cfg = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _load_state_and_cfg(self):
        from irodori_tts.inference_runtime import _load_checkpoint_for_inference
        from irodori_tts.config import ModelConfig as _DiTModelConfig

        state, cfg_dict, train_cfg = _load_checkpoint_for_inference(
            Path(self.checkpoint_path)
        )
        model_cfg = _DiTModelConfig(**cfg_dict)
        self._model_cfg = model_cfg
        self._train_cfg = train_cfg
        return state, model_cfg

    def load_model(self, device_map: str | None = None) -> nn.Module:
        from irodori_tts.model import TextToLatentRFDiT

        state, model_cfg = self._load_state_and_cfg()
        target_device = device_map if device_map is not None else self.device
        if target_device in (None, "auto"):
            target_device = "cpu"

        model = TextToLatentRFDiT(model_cfg)
        model.load_state_dict(state)
        model = model.to(device=target_device, dtype=self.dtype)
        model.eval()
        self.logger.info(
            "DiT model loaded from %s (dtype=%s, device=%s, %d blocks)",
            self.checkpoint_path,
            self.dtype,
            target_device,
            len(model.blocks),
        )
        return model

    def load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from irodori_tts.tokenizer import PretrainedTextTokenizer

        if self._model_cfg is None:
            self._load_state_and_cfg()
        self._tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=self._model_cfg.text_tokenizer_repo,
            add_bos=bool(self._model_cfg.text_add_bos),
        )
        return self._tokenizer

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------
    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
            raise RuntimeError(
                "IrodoriDiTAdapter expected model.blocks to be an nn.ModuleList "
                "of DiffusionBlock; got %r" % (type(model).__name__,)
            )
        return model.blocks

    def get_head_modules(
        self, model: nn.Module
    ) -> tuple[nn.Module | None, nn.Module | None]:
        norm = getattr(model, "out_norm", None)
        head = getattr(model, "out_proj", None)
        return norm, head

    # ------------------------------------------------------------------
    # Calibration: synthesise model inputs via the pluggable samplers, or
    # load real captured activations when ``calibration_inputs_path`` is set.
    # ------------------------------------------------------------------
    def prepare_calibration_inputs(
        self,
        model: nn.Module,
        calibration_config,
        device: torch.device,
        logger=None,
    ) -> dict[str, Any]:
        if self._model_cfg is None:
            self._load_state_and_cfg()
        cfg = self._model_cfg
        log = logger or self.logger

        if self.calibration_inputs_path is not None:
            return self._load_real_calibration_inputs(
                calibration_config, device, log
            )

        n = int(calibration_config.num_calibration_samples)
        seq_len = int(getattr(calibration_config, "max_length", 256))
        text_len = int(min(getattr(calibration_config, "max_length", 64), 64))

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype = self.dtype

        x_t = self.noise_sampler.sample(
            (n, seq_len, cfg.patched_latent_dim), gen, dtype
        )
        t = self.timestep_sampler.sample((n,), gen, dtype)

        text_input_ids = torch.randint(
            low=0,
            high=int(cfg.text_vocab_size),
            size=(n, text_len),
            generator=gen,
            dtype=torch.long,
        )
        text_mask = torch.ones((n, text_len), dtype=torch.bool)

        ref_latent = None
        ref_mask = None
        if cfg.use_speaker_condition:
            ref_seq = max(8, seq_len // 4)
            ref_latent = self.noise_sampler.sample(
                (n, ref_seq, cfg.speaker_patched_latent_dim), gen, dtype
            )
            ref_mask = torch.ones((n, ref_seq), dtype=torch.bool)

        caption_input_ids = None
        caption_mask = None
        if cfg.use_caption_condition:
            cap_len = text_len
            caption_input_ids = torch.randint(
                low=0,
                high=int(cfg.caption_vocab_size_resolved),
                size=(n, cap_len),
                generator=gen,
                dtype=torch.long,
            )
            caption_mask = torch.ones((n, cap_len), dtype=torch.bool)

        target_velocity = self.noise_sampler.sample(
            (n, seq_len, cfg.patched_latent_dim), gen, dtype
        )

        inputs: dict[str, Any] = {
            "x_t": x_t.to(device),
            "t": t.to(device),
            "text_input_ids": text_input_ids.to(device),
            "text_mask": text_mask.to(device),
            "ref_latent": ref_latent.to(device) if ref_latent is not None else None,
            "ref_mask": ref_mask.to(device) if ref_mask is not None else None,
            self.TARGET_VELOCITY_KEY: target_velocity.to(device),
        }
        if caption_input_ids is not None:
            inputs["caption_input_ids"] = caption_input_ids.to(device)
            inputs["caption_mask"] = caption_mask.to(device)

        log.info(
            "DiT calibration: %d samples, seq=%d, text=%d, "
            "use_speaker=%s, use_caption=%s",
            n,
            seq_len,
            text_len,
            cfg.use_speaker_condition,
            cfg.use_caption_condition,
        )
        return inputs

    def _load_real_calibration_inputs(
        self, calibration_config, device: torch.device, log
    ) -> dict[str, Any]:
        """Load real DiT forward inputs captured from genuine syntheses.

        The file is a ``torch.save`` dict produced by
        ``example/v3_int4/capture_calibration.py``.  All tensors are stacked
        on dim 0 (one row per captured RF step) and share identical sequence
        dimensions so the block-input catcher can ``torch.cat`` them.

        Two payload shapes are supported:

        * **encoded-conditions** (preferred) — carries already-encoded
          ``text_state`` / ``speaker_state``; calibration dispatches to
          ``model.forward_with_encoded_conditions`` so the (un-quantised)
          encoders are skipped and the blocks see exactly the activations
          that real synthesis produces.  Marked by the ``_encoded`` key.
        * **raw** — carries ``text_input_ids`` / ``ref_latent`` and runs the
          full ``model.forward``.
        """
        payload = torch.load(self.calibration_inputs_path, map_location="cpu")
        n_avail = int(payload["x_t"].shape[0])
        n = min(int(calibration_config.num_calibration_samples), n_avail)

        def _f(key):
            v = payload.get(key)
            return None if v is None else v[:n].to(self.dtype).to(device)

        def _raw(key):
            v = payload.get(key)
            return None if v is None else v[:n].to(device)

        if "text_state" in payload:
            inputs: dict[str, Any] = {
                "_encoded": True,
                "x_t": _f("x_t"),
                "t": _f("t"),
                "text_state": _f("text_state"),
                "text_mask": _raw("text_mask"),
                "speaker_state": _f("speaker_state"),
                "speaker_mask": _raw("speaker_mask"),
                "caption_state": _f("caption_state"),
                "caption_mask": _raw("caption_mask"),
            }
            mode = "encoded-conditions"
        else:
            inputs = {
                "x_t": _f("x_t"),
                "t": _f("t"),
                "text_input_ids": _raw("text_input_ids"),
                "text_mask": _raw("text_mask"),
                "ref_latent": _f("ref_latent"),
                "ref_mask": _raw("ref_mask"),
            }
            if payload.get("caption_input_ids") is not None:
                inputs["caption_input_ids"] = _raw("caption_input_ids")
                inputs["caption_mask"] = _raw("caption_mask")
            mode = "raw-forward"

        log.info(
            "DiT calibration: %d real samples (%s) from %s (available=%d, x_t=%s)",
            n,
            mode,
            self.calibration_inputs_path,
            n_avail,
            tuple(inputs["x_t"].shape),
        )
        return inputs

    # The keys passed to ``model.forward``.  ``_target_velocity`` is
    # adapter-private and only consumed by ``compute_curvature_loss``.
    _MODEL_FORWARD_KEYS = (
        "x_t",
        "t",
        "text_input_ids",
        "text_mask",
        "ref_latent",
        "ref_mask",
        "caption_input_ids",
        "caption_mask",
    )

    def _split_forward_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {k: inputs[k] for k in self._MODEL_FORWARD_KEYS if k in inputs}

    def run_calibration_forward(
        self, model: nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor | None:
        if inputs.get("_encoded"):
            return model.forward_with_encoded_conditions(
                x_t=inputs["x_t"],
                t=inputs["t"],
                text_state=inputs["text_state"],
                text_mask=inputs["text_mask"],
                speaker_state=inputs.get("speaker_state"),
                speaker_mask=inputs.get("speaker_mask"),
                caption_state=inputs.get("caption_state"),
                caption_mask=inputs.get("caption_mask"),
            )
        fwd_inputs = self._split_forward_inputs(inputs)
        return model(**fwd_inputs)

    # ------------------------------------------------------------------
    # Save / load metadata: embed the irodori_tts ModelConfig (plus a few
    # inference-only keys lifted from the training config) so the matching
    # loader can rebuild a ``TextToLatentRFDiT``.
    # ------------------------------------------------------------------
    _INFERENCE_CONFIG_KEYS = (
        "max_text_len",
        "max_caption_len",
        "fixed_target_latent_steps",
    )

    def build_save_metadata(
        self, quant_layers: list[dict[str, Any]]
    ) -> dict[str, str]:
        from dataclasses import asdict

        if self._model_cfg is None:
            self._load_state_and_cfg()

        flat_config = dict(asdict(self._model_cfg))
        if isinstance(self._train_cfg, dict):
            for key in self._INFERENCE_CONFIG_KEYS:
                value = self._train_cfg.get(key)
                if isinstance(value, int):
                    flat_config[key] = int(value)
        return {
            "config_json": json.dumps(flat_config, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "autobit",
            "checkpoint_format": "gptq",
        }

    @classmethod
    def _instantiate_model(
        cls, cfg_dict: dict[str, Any], dtype: torch.dtype
    ) -> nn.Module:
        from irodori_tts.config import ModelConfig as _DiTModelConfig
        from irodori_tts.model import TextToLatentRFDiT

        model_cfg = _DiTModelConfig(
            **{
                k: v
                for k, v in cfg_dict.items()
                if k in _DiTModelConfig.__dataclass_fields__
            }
        )
        return TextToLatentRFDiT(model_cfg).to(dtype=dtype)

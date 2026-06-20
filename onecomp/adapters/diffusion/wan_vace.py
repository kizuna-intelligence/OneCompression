"""Wan2.1 VACE ``WanVACETransformer3DModel`` adapter.

The Wan VACE denoiser runs eight VACE hint blocks before the main 40
``blocks``.  The main blocks receive those hints at ``config.vace_layers``.
This adapter lets OneCompression's architecture-aware QEP run over the main
block chain while injecting the precomputed VACE hints into the block-wise
forward.  The VACE hint blocks are quantized during save with packed RTN int4,
so the full 14B transformer is stored in OneCompression's packed GPTQ-compatible
format without keeping an extra full-size model in host memory.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import DiffusionTransformerAdapter
from .distributions import LogitNormalTimestep, Sampler


class _WanBlockKwargsWrapper(nn.Module):
    """Expose Wan main blocks through kwargs and apply per-block VACE hints."""

    _ARG_NAMES = ("encoder_hidden_states", "temb", "rotary_emb")

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
        control_hint = kwargs.pop("control_hint", None)
        control_hint_scale = kwargs.pop("control_hint_scale", None)
        hidden_states = self.block(hidden_states, **kwargs)
        if control_hint is not None:
            scale = 1.0 if control_hint_scale is None else control_hint_scale
            hidden_states = hidden_states + control_hint * scale
        return hidden_states


class WanVACE14BDiTAdapter(DiffusionTransformerAdapter):
    """Plug diffusers ``WanVACETransformer3DModel`` into OneCompression."""

    PRIMARY_LATENT_KEY = "hidden_states"

    _MODEL_FORWARD_KEYS = (
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "encoder_hidden_states_image",
        "control_hidden_states",
        "control_hidden_states_scale",
        "attention_kwargs",
    )

    _EXCLUDE_KEYWORDS: tuple[str, ...] = ()

    def __init__(
        self,
        checkpoint_path: str,
        dtype: str | torch.dtype = "bfloat16",
        device: str = "cpu",
        seed: int = 0,
        calibration_inputs_path: str | None = None,
        noise_sampler: Sampler | None = None,
        timestep_sampler: Sampler | None = None,
        revision: str | None = None,
        subfolder: str = "transformer",
        latent_frames: int = 3,
        latent_height: int = 16,
        latent_width: int = 16,
        text_seq_len: int = 512,
        quantize_vace_rtn: bool = True,
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
        self.latent_frames = int(latent_frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.text_seq_len = int(text_seq_len)
        self.quantize_vace_rtn = bool(quantize_vace_rtn)
        self._config: dict[str, Any] | None = None
        self._wrap_block_counter = 0
        self._current_vace_hints: list[torch.Tensor] | None = None
        self._current_vace_scales: tuple[torch.Tensor, ...] | None = None
        self._calibration_vace_hints: list[torch.Tensor] | None = None
        self._calibration_vace_scales: tuple[torch.Tensor, ...] | None = None

    def load_model(self, device_map: str | None = None) -> nn.Module:
        from diffusers import WanVACETransformer3DModel

        target_device = device_map if device_map is not None else self.device
        if target_device in (None, "auto"):
            target_device = "cpu"
        model = WanVACETransformer3DModel.from_pretrained(
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
            "Wan VACE transformer loaded from %s (revision=%s, dtype=%s, "
            "device=%s, main_blocks=%d, vace_blocks=%d)",
            self.checkpoint_path,
            self.revision,
            self.dtype,
            target_device,
            len(model.blocks),
            len(model.vace_blocks),
        )
        return model

    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError(
                "WanVACE14BDiTAdapter expected model.blocks to be an nn.ModuleList; "
                f"got {type(model).__name__}"
            )
        self._wrap_block_counter = 0
        return blocks

    def wrap_block(self, block: nn.Module) -> nn.Module:
        idx = self._wrap_block_counter
        self._wrap_block_counter += 1
        return _WanBlockKwargsWrapper(block, idx)

    def pack_catcher_input(
        self, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not args:
            raise RuntimeError("Wan block catcher expected positional args")
        hidden_states = args[0]
        packed_kwargs = dict(kwargs)
        for name, value in zip(_WanBlockKwargsWrapper._ARG_NAMES, args[1:]):
            packed_kwargs.setdefault(name, value)
        per_block = self._build_vace_hint_map()
        if per_block:
            packed_kwargs["_per_block_kwargs"] = per_block
        return hidden_states, packed_kwargs

    def _build_vace_hint_map(self) -> dict[int, dict[str, Any]]:
        hints = self._calibration_vace_hints or self._current_vace_hints
        if not hints:
            return {}
        if self._config is None:
            raise RuntimeError("Wan VACE config is not available")
        scales = self._calibration_vace_scales or self._current_vace_scales or ()
        mapping: dict[int, dict[str, Any]] = {}
        for hint_idx, block_idx in enumerate(self._config.get("vace_layers", [])):
            if hint_idx >= len(hints):
                break
            mapping[int(block_idx)] = {
                "control_hint": hints[hint_idx],
                "control_hint_scale": scales[hint_idx] if hint_idx < len(scales) else None,
            }
        return mapping

    def prepare_calibration_inputs(
        self,
        model: nn.Module,
        calibration_config,
        device: torch.device,
        logger=None,
    ) -> dict[str, Any]:
        if self._config is None:
            self._config = dict(model.config)
        log = logger or self.logger
        if self.calibration_inputs_path is not None:
            return self._load_real_calibration_inputs(calibration_config, device, log)

        cfg = self._config
        n = int(calibration_config.num_calibration_samples)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype = self.dtype
        hidden_states = self.noise_sampler.sample(
            (
                n,
                int(cfg.get("in_channels", 16)),
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            ),
            gen,
            dtype,
        )
        control_hidden_states = self.noise_sampler.sample(
            (
                n,
                int(cfg.get("vace_in_channels", 96)),
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            ),
            gen,
            dtype,
        )
        encoder_hidden_states = self.noise_sampler.sample(
            (n, self.text_seq_len, int(cfg.get("text_dim", 4096))),
            gen,
            dtype,
        )
        timestep = (self.timestep_sampler.sample((n,), gen, dtype) * 1000.0).to(torch.long)
        control_scale = torch.ones(len(cfg.get("vace_layers", [])), dtype=dtype)

        log.info(
            "Wan VACE calibration: %d samples, latent=(%d,%d,%d,%d), "
            "control_channels=%d, text_seq=%d",
            n,
            int(cfg.get("in_channels", 16)),
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            int(cfg.get("vace_in_channels", 96)),
            self.text_seq_len,
        )
        inputs = {
            "hidden_states": hidden_states.to(device),
            "timestep": timestep.to(device),
            "encoder_hidden_states": encoder_hidden_states.to(device),
            "control_hidden_states": control_hidden_states.to(device),
            "control_hidden_states_scale": control_scale.to(device),
        }
        self._calibration_vace_hints, self._calibration_vace_scales = self._compute_vace_hints(
            model, inputs
        )
        return inputs

    def _load_real_calibration_inputs(
        self, calibration_config, device: torch.device, log
    ) -> dict[str, Any]:
        blob = torch.load(self.calibration_inputs_path, map_location="cpu")
        n_avail = int(blob["hidden_states"].shape[0])
        n = min(int(calibration_config.num_calibration_samples), n_avail)

        def take(key: str):
            value = blob.get(key)
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                out = value[:n]
                if out.dtype.is_floating_point:
                    out = out.to(dtype=self.dtype)
                return out.to(device)
            return value

        inputs = {
            "hidden_states": take("hidden_states"),
            "timestep": take("timestep"),
            "encoder_hidden_states": take("encoder_hidden_states"),
            "encoder_hidden_states_image": take("encoder_hidden_states_image"),
            "control_hidden_states": take("control_hidden_states"),
            "control_hidden_states_scale": blob.get("control_hidden_states_scale"),
            "attention_kwargs": blob.get("attention_kwargs"),
        }
        if isinstance(inputs["control_hidden_states_scale"], torch.Tensor):
            inputs["control_hidden_states_scale"] = inputs["control_hidden_states_scale"].to(device)
        log.info(
            "Wan VACE REAL calibration: %d samples from %s (available=%d)",
            n,
            self.calibration_inputs_path,
            n_avail,
        )
        inputs = {k: v for k, v in inputs.items() if v is not None}
        self._calibration_vace_hints, self._calibration_vace_scales = self._compute_vace_hints(
            model, inputs
        )
        return inputs

    def slice_calibration_inputs(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        n_total = self.num_calibration_samples(inputs)
        sliced: dict[str, Any] = {}
        for key, value in inputs.items():
            if key in ("control_hidden_states_scale", "attention_kwargs"):
                sliced[key] = value
            elif isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == n_total:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def run_calibration_forward(
        self, model: nn.Module, inputs: dict[str, Any]
    ) -> torch.Tensor | None:
        fwd = {k: inputs[k] for k in self._MODEL_FORWARD_KEYS if k in inputs}
        if self._calibration_vace_hints is None:
            self._current_vace_hints, self._current_vace_scales = self._compute_vace_hints(
                model, fwd
            )
        try:
            return model(**fwd, return_dict=False)[0]
        finally:
            self._current_vace_hints = None
            self._current_vace_scales = None

    @torch.no_grad()
    def _compute_vace_hints(
        self, model: nn.Module, fwd: dict[str, Any]
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, ...]]:
        hidden_states = fwd["hidden_states"]
        control_hidden_states = fwd["control_hidden_states"]
        timestep = fwd["timestep"]
        encoder_hidden_states = fwd["encoder_hidden_states"]
        encoder_hidden_states_image = fwd.get("encoder_hidden_states_image")
        control_scale = fwd.get("control_hidden_states_scale")
        num_vace_layers = len(model.config.vace_layers)
        if control_scale is None:
            control_scale = control_hidden_states.new_ones(num_vace_layers)
        elif not isinstance(control_scale, torch.Tensor):
            control_scale = torch.as_tensor(
                control_scale,
                dtype=control_hidden_states.dtype,
                device=control_hidden_states.device,
            )
        else:
            if control_scale.dtype.is_floating_point:
                control_scale = control_scale.to(
                    device=control_hidden_states.device,
                    dtype=control_hidden_states.dtype,
                )
            else:
                control_scale = control_scale.to(device=control_hidden_states.device)
        if control_scale.dim() == 0:
            control_scale = control_scale.repeat(num_vace_layers)
        scales = torch.unbind(control_scale)

        rotary_emb = model.rope(hidden_states)
        hidden_states = model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        control_hidden_states = model.vace_patch_embedding(control_hidden_states)
        control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
        seq_delta = hidden_states.size(1) - control_hidden_states.size(1)
        if seq_delta > 0:
            pad = control_hidden_states.new_zeros(
                hidden_states.shape[0],
                seq_delta,
                control_hidden_states.size(2),
            )
            control_hidden_states = torch.cat([control_hidden_states, pad], dim=1)
        elif seq_delta < 0:
            control_hidden_states = control_hidden_states[:, : hidden_states.size(1)]
        _, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = model.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        hints: list[torch.Tensor] = []
        for block in model.vace_blocks:
            conditioning_states, control_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                control_hidden_states,
                timestep_proj,
                rotary_emb,
            )
            if conditioning_states is None:
                conditioning_states = torch.zeros_like(hidden_states)
            hints.append(conditioning_states.detach().cpu())
        return hints, tuple(s.detach().cpu() for s in scales)

    def build_save_metadata(
        self, quant_layers: list[dict[str, Any]]
    ) -> dict[str, str]:
        if self._config is None:
            raise RuntimeError(
                "WanVACE14BDiTAdapter.build_save_metadata called before load_model"
            )
        clean_cfg = {k: v for k, v in self._config.items() if not k.startswith("_")}
        return {
            "config_json": json.dumps(clean_cfg, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "qep_gptq_main_rtn_vace",
            "checkpoint_format": "gptq",
            "base_model": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        }

    @classmethod
    def _instantiate_model(
        cls, cfg_dict: dict[str, Any], dtype: torch.dtype
    ) -> nn.Module:
        from diffusers import WanVACETransformer3DModel

        clean_cfg = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
        model = WanVACETransformer3DModel.from_config(clean_cfg)
        return model.to(dtype=dtype)

    def save_quantized_model(self, runner, save_directory: str) -> str:
        from safetensors.torch import save_file

        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
        from onecomp.quantizer.rtn import RTN

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        model, _ = runner.create_quantized_model(pack_weights=True, use_gemlite=False)

        extra_layers: list[dict[str, Any]] = []
        if self.quantize_vace_rtn:
            rtn = RTN(wbits=4, groupsize=32, sym=True)
            rtn.logger = runner.logger
            for name, module in list(model.named_modules()):
                if not name.startswith("vace_blocks.") or not isinstance(module, nn.Linear):
                    continue
                result = rtn.quantize_layer(module)
                packed = rtn.create_inference_layer(result, module, pack_weights=True, use_gemlite=False)
                parent = model
                *parents, child = name.split(".")
                for part in parents:
                    parent = getattr(parent, part)
                setattr(parent, child, packed)
                extra_layers.append(
                    {
                        "name": name,
                        "wbits": 4,
                        "groupsize": 32,
                        "actorder": False,
                        "in_features": int(module.in_features),
                        "out_features": int(module.out_features),
                    }
                )
                del result, packed, module
                gc.collect()

        state_dict: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            t = v.contiguous().cpu()
            if t.dtype == torch.float32 and not any(
                k.endswith(s) for s in self._PACKED_INTEGER_SUFFIXES
            ):
                t = t.to(torch.float16)
            state_dict[k] = t

        quant_layers = self._collect_quant_layers(runner, model) + extra_layers
        quant_names = {entry["name"] for entry in quant_layers}
        for name, module in model.named_modules():
            if name in quant_names or not isinstance(module, GPTQLinear):
                continue
            quant_layers.append(
                {
                    "name": name,
                    "wbits": int(module.wbits),
                    "groupsize": int(module.groupsize),
                    "actorder": bool(module.actorder),
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                }
            )

        metadata = self.build_save_metadata(quant_layers)
        out_path = save_path / self.output_filename
        save_file(state_dict, str(out_path), metadata=metadata)
        runner.logger.info(
            "Quantized Wan VACE model saved to %s (%d tensors, %d packed Linears)",
            out_path,
            len(state_dict),
            len(quant_layers),
        )
        return str(out_path)

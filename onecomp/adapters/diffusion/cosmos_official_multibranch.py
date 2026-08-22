"""NVIDIA official Cosmos-Transfer2.5 multibranch adapter.

This targets the reference ``MinimalV4LVGControlVaceDiT`` used by
``cosmos-transfer2.5`` for true multi-control inference.  The main
``blocks`` are quantized by OneCompression QEP/GPTQ.  The control branch
Linears are packed with OneCompression RTN during save, following the Wan VACE
adapter pattern.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import DiffusionTransformerAdapter
from .distributions import LogitNormalTimestep, Sampler


class _OfficialCosmosBlockWrapper(nn.Module):
    """Expose official ControlAwareDiTBlock through a kwargs-only QEP call."""

    _ARG_NAMES = ("hints",)

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
        hints = kwargs.pop("hints")
        return self.block(hidden_states, hints, **kwargs)


class CosmosOfficialMultibranchAdapter(DiffusionTransformerAdapter):
    """Plug NVIDIA's official multibranch Cosmos DiT into OneCompression."""

    PRIMARY_LATENT_KEY = "x_B_C_T_H_W"

    _MODEL_FORWARD_KEYS = (
        "x_B_C_T_H_W",
        "timesteps_B_T",
        "crossattn_emb",
        "latent_control_input",
        "condition_video_input_mask_B_C_T_H_W",
        "fps",
        "padding_mask",
        "img_context_emb",
        "control_context_scale",
    )

    _BATCH_INDEPENDENT_KEYS = ("padding_mask",)
    _EXCLUDE_KEYWORDS: tuple[str, ...] = ()

    def __init__(
        self,
        checkpoint_path: str = "nvidia/Cosmos-Transfer2.5-2B",
        dtype: str | torch.dtype = "bfloat16",
        device: str = "cuda:0",
        seed: int = 0,
        calibration_inputs_path: str | None = None,
        noise_sampler: Sampler | None = None,
        timestep_sampler: Sampler | None = None,
        cosmos_repo: str | None = "/home/yusuke/gitrepos/cosmos-transfer2.5",
        output_dir: str = "/mnt/hojo/cosmos_transfer25_official_multibranch_qep_tmp",
        batch_hint_keys: tuple[str, ...] = ("depth", "seg"),
        latent_frames: int = 5,
        latent_height: int = 16,
        latent_width: int = 16,
        text_seq_len: int = 512,
        quantize_control_rtn: bool = True,
        rtn_groupsize: int = 32,
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
        self.cosmos_repo = cosmos_repo
        self.output_dir = output_dir
        self.batch_hint_keys = tuple(batch_hint_keys)
        self.latent_frames = int(latent_frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.text_seq_len = int(text_seq_len)
        self.quantize_control_rtn = bool(quantize_control_rtn)
        self.rtn_groupsize = int(rtn_groupsize)
        self._model_config_summary: dict[str, Any] | None = None
        self._wrap_block_counter = 0

    def _ensure_cosmos_on_path(self) -> None:
        if self.cosmos_repo:
            import sys

            repo = str(Path(self.cosmos_repo).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

    def load_model(self, device_map: str | None = None) -> nn.Module:
        self._ensure_cosmos_on_path()
        from cosmos_transfer2.config import SetupArguments
        from cosmos_transfer2.inference import Control2WorldInference

        # The official loader internally expects CUDA for this model family.
        target_device = self.device if device_map in (None, "cpu", "auto") else device_map
        setup = SetupArguments.model_validate(
            {
                "output_dir": Path(self.output_dir),
                "model": "edge",
                "disable_guardrails": True,
                "offload_guardrail_models": False,
                "keep_going": False,
                "profile": False,
                "benchmark": False,
            }
        )
        inference = Control2WorldInference(setup, batch_hint_keys=list(self.batch_hint_keys))
        net = inference.inference_pipeline.model.net
        net.eval()
        net = net.to(device=target_device, dtype=self.dtype)
        self._model_config_summary = self._summarize_net(net)
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info(
            "Official Cosmos multibranch net loaded (device=%s, dtype=%s, blocks=%d, control_branches=%d)",
            target_device,
            self.dtype,
            len(net.blocks),
            int(getattr(net, "num_control_branches", 1)),
        )
        return net

    @staticmethod
    def _summarize_net(net: nn.Module) -> dict[str, Any]:
        return {
            "class": type(net).__name__,
            "num_blocks": int(getattr(net, "num_blocks", len(getattr(net, "blocks", [])))),
            "num_control_branches": int(getattr(net, "num_control_branches", 1)),
            "control_layers": list(getattr(net, "control_layers", [])),
            "in_channels": int(getattr(net, "in_channels", 0)),
            "vace_in_channels": int(getattr(net, "vace_in_channels", 0)),
            "model_channels": int(getattr(net, "model_channels", 0)),
            "crossattn_emb_channels": int(getattr(net, "crossattn_emb_channels", 0)),
            "crossattn_proj_in_channels": int(getattr(net, "crossattn_proj_in_channels", 0)),
            "use_crossattn_projection": bool(getattr(net, "use_crossattn_projection", False)),
            "extra_image_context_dim": getattr(net, "extra_image_context_dim", None),
        }

    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError(
                "CosmosOfficialMultibranchAdapter expected model.blocks to be an nn.ModuleList; "
                f"got {type(model).__name__}"
            )
        self._wrap_block_counter = 0
        return blocks

    def wrap_block(self, block: nn.Module) -> nn.Module:
        idx = self._wrap_block_counter
        self._wrap_block_counter += 1
        return _OfficialCosmosBlockWrapper(block, idx)

    def pack_catcher_input(self, args: tuple, kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if not args:
            raise RuntimeError("Official Cosmos block catcher expected positional args")
        hidden_states = args[0]
        packed_kwargs = dict(kwargs)
        for name, value in zip(_OfficialCosmosBlockWrapper._ARG_NAMES, args[1:]):
            packed_kwargs.setdefault(name, value)
        return hidden_states, packed_kwargs

    def prepare_calibration_inputs(
        self,
        model: nn.Module,
        calibration_config,
        device: torch.device,
        logger=None,
    ) -> dict[str, Any]:
        log = logger or self.logger
        if self.calibration_inputs_path is not None:
            return self._load_real_calibration_inputs(calibration_config, device, log)

        n = int(calibration_config.num_calibration_samples)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype = self.dtype
        latent_c = int(getattr(model, "in_channels")) - 1
        control_branches = int(getattr(model, "num_control_branches", 1))
        text_dim = (
            int(getattr(model, "crossattn_proj_in_channels"))
            if bool(getattr(model, "use_crossattn_projection", False))
            else int(getattr(model, "crossattn_emb_channels"))
        )
        x = self.noise_sampler.sample(
            (n, latent_c, self.latent_frames, self.latent_height, self.latent_width),
            gen,
            dtype,
        )
        controls = self.noise_sampler.sample(
            (
                n,
                latent_c * control_branches,
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            ),
            gen,
            dtype,
        )
        timesteps = self.timestep_sampler.sample((n, self.latent_frames), gen, dtype)
        text = self.noise_sampler.sample((n, self.text_seq_len, text_dim), gen, dtype)
        cond_mask = torch.zeros(n, 1, self.latent_frames, self.latent_height, self.latent_width, dtype=dtype)
        padding_mask = torch.zeros(1, 1, self.latent_height, self.latent_width, dtype=dtype)
        control_scale = torch.ones(control_branches, dtype=dtype)

        log.info(
            "Official Cosmos calibration: n=%d latent=(%d,%d,%d,%d), controls=%d, text=(%d,%d)",
            n,
            latent_c,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            control_branches,
            self.text_seq_len,
            text_dim,
        )
        return {
            "x_B_C_T_H_W": x.to(device),
            "timesteps_B_T": timesteps.to(device),
            "crossattn_emb": text.to(device),
            "latent_control_input": controls.to(device),
            "condition_video_input_mask_B_C_T_H_W": cond_mask.to(device),
            "padding_mask": padding_mask.to(device),
            "control_context_scale": control_scale.to(device),
        }

    def _load_real_calibration_inputs(self, calibration_config, device: torch.device, log) -> dict[str, Any]:
        blob = torch.load(self.calibration_inputs_path, map_location="cpu")
        n_avail = int(blob[self.PRIMARY_LATENT_KEY].shape[0])
        n = min(int(calibration_config.num_calibration_samples), n_avail)

        def take(key: str):
            value = blob.get(key)
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                out = value[:n] if value.dim() >= 1 and value.shape[0] == n_avail else value
                if out.dtype.is_floating_point:
                    out = out.to(dtype=self.dtype)
                return out.to(device)
            return value

        inputs = {k: take(k) for k in self._MODEL_FORWARD_KEYS}
        inputs = {k: v for k, v in inputs.items() if v is not None}
        log.info("Official Cosmos REAL calibration: %d samples from %s", n, self.calibration_inputs_path)
        return inputs

    def slice_calibration_inputs(self, inputs: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        n_total = self.num_calibration_samples(inputs)
        out: dict[str, Any] = {}
        for key, value in inputs.items():
            if key in self._BATCH_INDEPENDENT_KEYS:
                out[key] = value
            elif isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == n_total:
                out[key] = value[start:end]
            else:
                out[key] = value
        return out

    def run_calibration_forward(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor | None:
        device = next(model.parameters()).device
        fwd = {k: inputs[k] for k in self._MODEL_FORWARD_KEYS if k in inputs}
        fwd = self._move_value(fwd, device)
        return model(**fwd)

    @classmethod
    def _move_value(cls, value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: cls._move_value(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._move_value(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._move_value(v, device) for v in value)
        return value

    def build_save_metadata(self, quant_layers: list[dict[str, Any]]) -> dict[str, str]:
        return {
            "config_json": json.dumps(self._model_config_summary or {}, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "qep_gptq_main_rtn_control",
            "checkpoint_format": "gptq",
            "base_model": "nvidia/Cosmos-Transfer2.5-2B",
            "official_multibranch": "true",
        }

    @classmethod
    def _instantiate_model(cls, cfg_dict: dict[str, Any], dtype: torch.dtype) -> nn.Module:
        raise NotImplementedError(
            "Official multibranch checkpoints must be loaded by instantiating "
            "the NVIDIA official pipeline, then replacing packed layers."
        )

    def save_quantized_model(self, runner, save_directory: str) -> str:
        from safetensors.torch import save_file

        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
        from onecomp.quantizer.rtn import RTN

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        model, _ = runner.create_quantized_model(pack_weights=True, use_gemlite=False)

        extra_layers: list[dict[str, Any]] = []
        if self.quantize_control_rtn:
            for name, module in list(model.named_modules()):
                if not isinstance(module, nn.Linear):
                    continue
                if name.startswith("blocks."):
                    continue
                if not self._is_control_side_linear(name):
                    continue
                groupsize = self._effective_groupsize(module.in_features, self.rtn_groupsize)
                rtn = RTN(wbits=4, groupsize=groupsize, sym=True)
                rtn.logger = runner.logger
                result = rtn.quantize_layer(module)
                packed = rtn.create_inference_layer(result, module, pack_weights=True, use_gemlite=False)
                self._set_module(model, name, packed)
                extra_layers.append(
                    {
                        "name": name,
                        "wbits": 4,
                        "groupsize": int(groupsize),
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
            if t.dtype == torch.float32 and not any(k.endswith(s) for s in self._PACKED_INTEGER_SUFFIXES):
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
            "Quantized official Cosmos multibranch model saved to %s (%d tensors, %d packed Linears)",
            out_path,
            len(state_dict),
            len(quant_layers),
        )
        return str(out_path)

    @staticmethod
    def _is_control_side_linear(name: str) -> bool:
        return (
            name.startswith("control_blocks")
            or name.startswith("control_embedder")
            or name.startswith("after_proj")
            or name.startswith("input_hint_block")
            or name.startswith("t_embedder_for_control_branch")
            or name.startswith("x_embedder_for_control_branch")
        )

    @staticmethod
    def _effective_groupsize(in_features: int, requested_groupsize: int) -> int:
        if requested_groupsize <= 0:
            return -1
        if in_features % requested_groupsize == 0:
            return requested_groupsize
        for candidate in range(min(requested_groupsize, in_features), 0, -1):
            if in_features % candidate == 0:
                return candidate
        return math.gcd(in_features, requested_groupsize) or 1

    @staticmethod
    def _set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
        parent = root
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

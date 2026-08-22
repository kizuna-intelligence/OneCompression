"""NVIDIA official Cosmos-Predict2.5 DiT adapter.

This targets the reference ``cosmos-transfer2.5`` Predict2.5 video2world /
text2world stack.  The adapter loads the official checkpoint, returns the
bare ``model.net`` for OneCompression QEP/GPTQ, and writes a packed safetensors
checkpoint that can be loaded back into an already-built official inference
pipeline.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .base import DiffusionTransformerAdapter
from .distributions import LogitNormalTimestep, Sampler


class _PredictBlockWrapper(nn.Module):
    """Expose Predict ``Block`` through a kwargs-friendly QEP signature."""

    _ARG_NAMES = (
        "emb_B_T_D",
        "crossattn_emb",
        "rope_emb_L_1_1_D",
        "adaln_lora_B_T_3D",
        "extra_per_block_pos_emb",
        "kv_cache_cfg",
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


class CosmosPredictDiTAdapter(DiffusionTransformerAdapter):
    """Plug NVIDIA's official Cosmos-Predict2.5 DiT into OneCompression."""

    PRIMARY_LATENT_KEY = "x_B_C_T_H_W"

    _MODEL_FORWARD_KEYS = (
        "x_B_C_T_H_W",
        "timesteps_B_T",
        "crossattn_emb",
        "condition_video_input_mask_B_C_T_H_W",
        "fps",
        "padding_mask",
        "img_context_emb",
    )
    _BATCH_INDEPENDENT_KEYS = ("padding_mask",)
    _EXCLUDE_KEYWORDS: tuple[str, ...] = ()

    def __init__(
        self,
        checkpoint_path: str,
        dtype: str | torch.dtype = "bfloat16",
        device: str = "cuda:0",
        seed: int = 0,
        calibration_inputs_path: str | None = None,
        noise_sampler: Sampler | None = None,
        timestep_sampler: Sampler | None = None,
        cosmos_repo: str | None = "/home/yusuke/gitrepos/cosmos-transfer2.5",
        experiment_name: str = (
            "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-"
            "Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only"
        ),
        config_file: str = "cosmos_transfer2/_src/predict2/configs/video2world/config.py",
        experiment_opts: list[str] | None = None,
        latent_frames: int = 5,
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
        self.cosmos_repo = cosmos_repo
        self.experiment_name = experiment_name
        self.config_file = config_file
        self.experiment_opts = list(experiment_opts or [])
        self.latent_frames = int(latent_frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.text_seq_len = int(text_seq_len)
        self._model_config_summary: dict[str, Any] | None = None
        self._wrap_block_counter = 0
        self._loaded_model_ref: nn.Module | None = None

    def _ensure_cosmos_on_path(self) -> None:
        if self.cosmos_repo:
            repo = str(Path(self.cosmos_repo).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

    def load_model(self, device_map: str | None = None) -> nn.Module:
        self._ensure_cosmos_on_path()
        from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint

        os.environ.setdefault("COSMOS_TEXT_ENCODER_DEVICE", "cpu")
        os.environ.setdefault("COSMOS_PREDICT2_OFFLOAD_DIT", "0")

        target_device = self.device if device_map in (None, "cpu", "auto") else device_map
        opts = list(self.experiment_opts)
        if "~data_train" not in opts:
            opts.append("~data_train")
        opts.extend(
            [
                "model.config.fsdp_shard_size=1",
                "model.config.ema.enabled=False",
            ]
        )

        model, _config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.checkpoint_path,
            config_file=self.config_file,
            load_ema_to_reg=True,
            experiment_opts=opts,
            to_device=target_device,
        )
        net = model.net.to(device=target_device, dtype=self.dtype)
        net.eval()
        self._loaded_model_ref = model
        self._model_config_summary = self._summarize_net(net)
        self.logger.info(
            "Official Cosmos Predict net loaded (experiment=%s, ckpt=%s, device=%s, dtype=%s, blocks=%d)",
            self.experiment_name,
            self.checkpoint_path,
            target_device,
            self.dtype,
            len(getattr(net, "blocks", [])),
        )
        return net

    @staticmethod
    def _summarize_net(net: nn.Module) -> dict[str, Any]:
        keys = (
            "max_img_h",
            "max_img_w",
            "max_frames",
            "in_channels",
            "out_channels",
            "patch_spatial",
            "patch_temporal",
            "model_channels",
            "num_blocks",
            "num_heads",
            "crossattn_emb_channels",
            "use_crossattn_projection",
            "crossattn_proj_in_channels",
            "extra_image_context_dim",
            "concat_padding_mask",
            "use_adaln_lora",
            "adaln_lora_dim",
        )
        out = {"class": type(net).__name__}
        for key in keys:
            if hasattr(net, key):
                value = getattr(net, key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[key] = value
        out["num_blocks"] = int(getattr(net, "num_blocks", len(getattr(net, "blocks", []))))
        return out

    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError(
                "CosmosPredictDiTAdapter expected model.blocks to be an nn.ModuleList; "
                f"got {type(model).__name__}"
            )
        self._wrap_block_counter = 0
        return blocks

    def wrap_block(self, block: nn.Module) -> nn.Module:
        idx = self._wrap_block_counter
        self._wrap_block_counter += 1
        return _PredictBlockWrapper(block, idx)

    def pack_catcher_input(self, args: tuple, kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if not args:
            raise RuntimeError("Cosmos Predict block catcher expected positional args")
        hidden_states = args[0]
        packed_kwargs = dict(kwargs)
        for name, value in zip(_PredictBlockWrapper._ARG_NAMES, args[1:]):
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
        latent_c = int(getattr(model, "out_channels", 0)) or max(1, int(getattr(model, "in_channels")) - 1)
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
        timesteps = self.timestep_sampler.sample((n, self.latent_frames), gen, dtype)
        text = self.noise_sampler.sample((n, self.text_seq_len, text_dim), gen, dtype)
        condition_mask = torch.zeros(n, 1, self.latent_frames, self.latent_height, self.latent_width, dtype=dtype)
        padding_mask = torch.zeros(1, 1, self.latent_height, self.latent_width, dtype=dtype)

        log.info(
            "Official Cosmos Predict calibration: n=%d latent=(%d,%d,%d,%d), text=(%d,%d)",
            n,
            latent_c,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            self.text_seq_len,
            text_dim,
        )
        return {
            "x_B_C_T_H_W": x.to(device),
            "timesteps_B_T": timesteps.to(device),
            "crossattn_emb": text.to(device),
            "condition_video_input_mask_B_C_T_H_W": condition_mask.to(device),
            "padding_mask": padding_mask.to(device),
        }

    def _load_real_calibration_inputs(self, calibration_config, device: torch.device, log) -> dict[str, Any]:
        blob = torch.load(self.calibration_inputs_path, map_location="cpu")
        n_avail = int(blob[self.PRIMARY_LATENT_KEY].shape[0])
        n = min(int(calibration_config.num_calibration_samples), n_avail)

        def take(key: str):
            value = blob.get(key)
            if value is None:
                return None
            return self._slice_value(value, 0, n, n_avail, dtype=self.dtype, device=device)

        inputs = {k: take(k) for k in self._MODEL_FORWARD_KEYS}
        inputs = {k: v for k, v in inputs.items() if v is not None}
        log.info("Official Cosmos Predict REAL calibration: %d samples from %s", n, self.calibration_inputs_path)
        return inputs

    def slice_calibration_inputs(self, inputs: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        n_total = self.num_calibration_samples(inputs)
        out: dict[str, Any] = {}
        for key, value in inputs.items():
            if key in self._BATCH_INDEPENDENT_KEYS:
                out[key] = value
            else:
                out[key] = self._slice_value(value, start, end, n_total)
        return out

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
        if isinstance(value, dict):
            return {k: cls._slice_value(v, start, end, n_total, dtype, device) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._slice_value(v, start, end, n_total, dtype, device) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._slice_value(v, start, end, n_total, dtype, device) for v in value)
        return value

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

    def run_calibration_forward(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor | None:
        device = next(model.parameters()).device
        accepted = set(inspect.signature(model.forward).parameters)
        fwd = {k: inputs[k] for k in self._MODEL_FORWARD_KEYS if k in inputs and k in accepted}
        fwd = self._move_value(fwd, device)
        return model(**fwd)

    def build_save_metadata(self, quant_layers: list[dict[str, Any]]) -> dict[str, str]:
        return {
            "config_json": json.dumps(self._model_config_summary or {}, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "qep_gptq_predict",
            "checkpoint_format": "gptq",
            "base_model": "nvidia/Cosmos-Predict2.5-2B",
            "official_predict": "true",
            "experiment_name": self.experiment_name,
        }

    @classmethod
    def _instantiate_model(cls, cfg_dict: dict[str, Any], dtype: torch.dtype) -> nn.Module:
        raise NotImplementedError(
            "Official Cosmos Predict checkpoints must be loaded by instantiating "
            "the NVIDIA official pipeline, then replacing packed layers."
        )

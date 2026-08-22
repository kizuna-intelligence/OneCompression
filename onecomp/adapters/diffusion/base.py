"""Abstract adapter for diffusion / flow-matching transformers.

A diffusion transformer (DiT) is, from OneCompression's point of view, a
stack of transformer ``blocks`` driven by a noised latent ``x_t`` and a
timestep ``t`` plus some conditioning, trained to predict a velocity
(rectified-flow / flow-matching).  Concrete architectures differ only in:

- **how the checkpoint is loaded** and the model instantiated,
- **which attribute(s) hold the quantization blocks**,
- **the forward signature** (what conditioning the blocks consume),
- **the on-disk metadata schema**.

:class:`DiffusionTransformerAdapter` captures everything else — the
velocity-MSE curvature loss, calibration slicing, exclude-keyword
parameter counting, and the packed GPTQ save/load mechanics — so a new
architecture is a small subclass that fills in the arch-specific hooks.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..base import ModelAdapter
from ..utils import resolve_dtype
from .distributions import GaussianNoise, Sampler, UniformTimestep


class DiffusionTransformerAdapter(ModelAdapter):
    """Shared machinery for rectified-flow / flow-matching DiT adapters."""

    #: Key in the calibration-inputs dict holding the primary noised latent.
    #: Its leading dimension is the calibration batch size.
    PRIMARY_LATENT_KEY: str = "x_t"

    #: Adapter-private key carrying the target velocity for the curvature loss.
    TARGET_VELOCITY_KEY: str = "_target_velocity"

    #: Substrings of module names that must NOT be quantized (AdaLN, cond
    #: MLPs, in/out projections, encoders…).  Subclasses override.
    _EXCLUDE_KEYWORDS: tuple[str, ...] = ()

    #: Filename written by :meth:`save_quantized_model`.
    output_filename: str = "model.safetensors"

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
        self.checkpoint_path = str(checkpoint_path)
        self.dtype = resolve_dtype(dtype)
        self.device = device
        self.seed = int(seed)
        # Pluggable distributions for the *synthetic* calibration fallback.
        # Defaults reproduce the classic "Gaussian latent + uniform t"; a
        # flow-matching subclass can pass a LogitNormalTimestep instead.
        self.noise_sampler = noise_sampler or GaussianNoise()
        self.timestep_sampler = timestep_sampler or UniformTimestep()
        # When set, ``prepare_calibration_inputs`` should load *real* forward
        # inputs captured from genuine generations instead of random tensors.
        # Random calibration points GPTQ's Hessian estimate at a distribution
        # the model never sees, which badly degrades quality on real models.
        self.calibration_inputs_path = (
            str(calibration_inputs_path) if calibration_inputs_path else None
        )
        self.logger = getLogger(type(self).__module__)

    def get_model_id_or_path(self) -> str:
        return self.checkpoint_path

    # ------------------------------------------------------------------
    # Exclude keywords / quantizable param count
    # ------------------------------------------------------------------
    @classmethod
    def default_exclude_layer_keywords(cls) -> list[str]:
        return list(cls._EXCLUDE_KEYWORDS)

    def get_quantizable_param_count(self, model: nn.Module) -> int:
        total = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(kw in name for kw in self._EXCLUDE_KEYWORDS):
                continue
            total += module.weight.numel()
        return total

    # ------------------------------------------------------------------
    # Calibration helpers (generic)
    # ------------------------------------------------------------------
    def num_calibration_samples(self, inputs: dict[str, Any]) -> int:
        return int(inputs[self.PRIMARY_LATENT_KEY].shape[0])

    def slice_calibration_inputs(
        self, inputs: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                sliced[k] = v[start:end]
            else:
                sliced[k] = v
        return sliced

    # ------------------------------------------------------------------
    # AutoBit curvature loss: MSE on velocity prediction (flow matching).
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
        v_pred = head(normed)
        target = sample_inputs.get(self.TARGET_VELOCITY_KEY)
        if target is None:
            target = torch.zeros_like(v_pred)
        target = target.to(device=v_pred.device, dtype=v_pred.dtype)
        return F.mse_loss(v_pred, target)

    # ------------------------------------------------------------------
    # Packed GPTQ save (generic): replace quantized nn.Linear with packed
    # GPTQLinear, cast the remaining fp32 tensors to fp16, and record
    # per-layer quant metadata.  Subclasses only describe the model config
    # via :meth:`build_save_metadata`.
    # ------------------------------------------------------------------
    _PACKED_INTEGER_SUFFIXES = (".qweight", ".qzeros", ".g_idx", ".scales")

    def save_quantized_model(self, runner, save_directory: str) -> str:
        from safetensors.torch import save_file

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        model, _ = runner.create_quantized_model(
            pack_weights=True, use_gemlite=False
        )

        state_dict: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            t = v.contiguous().cpu()
            if t.dtype == torch.float32 and not any(
                k.endswith(s) for s in self._PACKED_INTEGER_SUFFIXES
            ):
                t = t.to(torch.float16)
            state_dict[k] = t

        quant_layers = self._collect_quant_layers(runner, model)
        metadata = self.build_save_metadata(quant_layers)

        out_path = save_path / self.output_filename
        save_file(state_dict, str(out_path), metadata=metadata)
        runner.logger.info(
            "Quantized model saved to %s (%d tensors, %d packed Linears)",
            out_path,
            len(state_dict),
            len(quant_layers),
        )
        return str(out_path)

    @staticmethod
    def _collect_quant_layers(runner, model: nn.Module) -> list[dict[str, Any]]:
        quantizer = runner.quantizer
        modules = dict(model.named_modules())
        name_to_q = getattr(quantizer, "_name_to_quantizer", None) or {}
        quant_layers: list[dict[str, Any]] = []
        for layer_name in sorted(quantizer.results.keys()):
            child_q = name_to_q.get(layer_name) or quantizer
            module = modules.get(layer_name)
            if module is None:
                continue
            wbits = getattr(child_q, "wbits", None)
            quant_layers.append(
                {
                    "name": layer_name,
                    "wbits": int(wbits) if wbits is not None else 0,
                    "groupsize": int(getattr(child_q, "groupsize", -1)),
                    "actorder": bool(getattr(child_q, "actorder", False)),
                    "in_features": int(getattr(module, "in_features", 0)),
                    "out_features": int(getattr(module, "out_features", 0)),
                }
            )
        return quant_layers

    @abstractmethod
    def build_save_metadata(
        self, quant_layers: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Return the safetensors metadata dict for the quantized checkpoint.

        Must embed ``quant_layers`` (typically as ``quant_layers_json``) plus
        whatever config the matching :meth:`load_quantized_model` needs to
        rebuild the model.
        """

    # ------------------------------------------------------------------
    # Packed GPTQ load (generic): rebuild GPTQLinear modules in place and
    # load the remaining (fp16) tensors.  Subclasses only build the bare
    # model from the embedded config via :meth:`_instantiate_model`.
    # ------------------------------------------------------------------
    @classmethod
    def load_quantized_model(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        dtype: str | torch.dtype = "float32",
    ) -> nn.Module:
        from safetensors import safe_open

        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

        resolved_dtype = resolve_dtype(dtype)

        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
            cfg_json = md.get("config_json")
            quant_layers_json = md.get("quant_layers_json")
            if cfg_json is None or quant_layers_json is None:
                raise ValueError(
                    f"{checkpoint_path} is missing 'config_json' or "
                    "'quant_layers_json' metadata; not a packed checkpoint "
                    f"produced by {cls.__name__}.save_quantized_model"
                )
            cfg_dict = json.loads(cfg_json)
            quant_layers = json.loads(quant_layers_json)
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        model = cls._instantiate_model(cfg_dict, resolved_dtype)

        modules = dict(model.named_modules())
        for entry in quant_layers:
            name = entry["name"]
            parent_name, _, child_name = name.rpartition(".")
            parent = modules.get(parent_name) if parent_name else model
            if parent is None:
                raise KeyError(f"Quantized layer parent not found: {parent_name!r}")
            layer_state = {
                "qweight": tensors[f"{name}.qweight"],
                "scales": tensors[f"{name}.scales"],
                "qzeros": tensors[f"{name}.qzeros"],
            }
            g_idx_key = f"{name}.g_idx"
            if g_idx_key in tensors:
                layer_state["g_idx"] = tensors[g_idx_key]
            bias_key = f"{name}.bias"
            if bias_key in tensors:
                layer_state["bias"] = tensors[bias_key]
            quant_layer = GPTQLinear.from_saved_state(
                layer_state_dict=layer_state,
                in_features=int(entry["in_features"]),
                out_features=int(entry["out_features"]),
                wbits=int(entry["wbits"]),
                groupsize=int(entry["groupsize"]),
                actorder=bool(entry.get("actorder", False)),
                checkpoint_format=md.get("checkpoint_format", "gptq"),
            )
            setattr(parent, child_name, quant_layer)

        quant_suffixes = ("qweight", "scales", "qzeros", "g_idx", "bias")
        quant_tensor_keys = {
            f"{e['name']}.{suffix}" for e in quant_layers for suffix in quant_suffixes
        }
        non_quant = {k: v for k, v in tensors.items() if k not in quant_tensor_keys}
        missing, unexpected = model.load_state_dict(non_quant, strict=False)
        # GPTQLinear buffers show up as "missing" on the freshly-built model.
        quant_keys = quant_tensor_keys | {f"{e['name']}.weight" for e in quant_layers}
        real_missing = [k for k in missing if k not in quant_keys]
        if real_missing:
            raise RuntimeError(
                f"Missing keys in quantized checkpoint: {real_missing[:8]} ..."
            )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in quantized checkpoint: {unexpected[:8]} ..."
            )

        model = model.to(device=device)
        model.eval()
        return model

    @classmethod
    @abstractmethod
    def _instantiate_model(cls, cfg_dict: dict[str, Any], dtype: torch.dtype) -> nn.Module:
        """Build a fresh (un-quantized) model from the embedded config dict."""

    # ------------------------------------------------------------------
    def has_additional_data(self) -> bool:
        return False

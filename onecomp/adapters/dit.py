"""Adapter for the Faster-Irodori-TTS2 :class:`TextToLatentRFDiT` model.

The DiT is a Rectified-Flow text-to-latent diffusion transformer.  It
has no HuggingFace ``AutoModelForCausalLM`` interface, no causal LM
tokenizer in the OneCompression sense, and no shifted-CE loss.  This
adapter bridges those gaps:

- ``load_model`` deserialises a ``.safetensors`` (or ``.pt``)
  checkpoint via ``irodori_tts.inference_runtime`` and constructs a
  fresh ``TextToLatentRFDiT``.
- ``get_blocks`` returns ``model.blocks`` (the ``nn.ModuleList`` of
  ``DiffusionBlock``).
- ``get_head_modules`` returns ``(out_norm, out_proj)`` so AutoBit can
  backprop the curvature loss into the last block's output.
- ``prepare_calibration_inputs`` synthesises random tensors that match
  the model's forward signature (``x_t``, ``t``, text ids, ref latent…).
- ``compute_curvature_loss`` is MSE between predicted and target
  velocity — the natural per-sample loss for Flow Matching / RF.
- ``save_quantized_model`` persists a single safetensors file with the
  same metadata schema used by inference_runtime so that downstream
  loaders work unchanged.

The adapter intentionally pins ``exclude_layer_keywords`` defaults to
skip AdaLN, the cond MLP, the timestep MLP, and the in/out projections
— these are tiny FP16 layers that shouldn't be quantised.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base import ModelAdapter


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        return torch.float32
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in _DTYPE_MAP:
            return _DTYPE_MAP[key]
        raise ValueError(f"Unrecognised dtype string: {dtype!r}")
    raise TypeError(f"Unsupported dtype: {dtype!r}")


class DiTAdapter(ModelAdapter):
    """Plug ``TextToLatentRFDiT`` into the OneCompression pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: str | torch.dtype = "float32",
        device: str = "cpu",
        seed: int = 0,
        calibration_inputs_path: str | None = None,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.dtype = _resolve_dtype(dtype)
        self.device = device
        self.seed = int(seed)
        # When set, ``prepare_calibration_inputs`` loads real DiT forward
        # inputs captured from genuine syntheses (see
        # ``example/v3_int4/capture_calibration.py``) instead of synthesising
        # random Gaussian tensors.  Random calibration makes GPTQ estimate the
        # Hessian from a distribution the model never sees, which on v3 left
        # speech intelligibility badly degraded (CER ~33% vs FP32 ~8%).
        self.calibration_inputs_path = (
            str(calibration_inputs_path) if calibration_inputs_path else None
        )
        self.logger = getLogger(__name__)
        self._model_cfg = None
        self._train_cfg = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def get_model_id_or_path(self) -> str:
        return self.checkpoint_path

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
                "DiTAdapter expected model.blocks to be an nn.ModuleList "
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
    # Quantizable param count (excludes AdaLN / cond MLP / in_proj /
    # out_proj — the same modules excluded from quantisation).
    # ------------------------------------------------------------------
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
    # Calibration: synthesise random model inputs.  AutoBit only needs
    # gradient *direction* through each block, so randomised inputs
    # (with deterministic seed) are sufficient and avoid pulling in a
    # real audio dataset.
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

        x_t = torch.randn(
            (n, seq_len, cfg.patched_latent_dim),
            generator=gen,
            dtype=torch.float32,
        ).to(dtype)
        t = torch.rand((n,), generator=gen, dtype=torch.float32).clamp_(0.001, 0.999)

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
            ref_latent = torch.randn(
                (n, ref_seq, cfg.speaker_patched_latent_dim),
                generator=gen,
                dtype=torch.float32,
            ).to(dtype)
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

        target_velocity = torch.randn(
            (n, seq_len, cfg.patched_latent_dim),
            generator=gen,
            dtype=torch.float32,
        ).to(dtype)

        inputs: dict[str, Any] = {
            "x_t": x_t.to(device),
            "t": t.to(device),
            "text_input_ids": text_input_ids.to(device),
            "text_mask": text_mask.to(device),
            "ref_latent": ref_latent.to(device) if ref_latent is not None else None,
            "ref_mask": ref_mask.to(device) if ref_mask is not None else None,
            "_target_velocity": target_velocity.to(device),
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

    def num_calibration_samples(self, inputs: dict[str, Any]) -> int:
        return int(inputs["x_t"].shape[0])

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
        return {
            k: inputs[k]
            for k in self._MODEL_FORWARD_KEYS
            if k in inputs
        }

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
    # AutoBit curvature loss: MSE on velocity prediction.
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
        target = sample_inputs.get("_target_velocity")
        if target is None:
            target = torch.zeros_like(v_pred)
        target = target.to(device=v_pred.device, dtype=v_pred.dtype)
        return F.mse_loss(v_pred, target)

    # ------------------------------------------------------------------
    # Save: write a single safetensors file with the same metadata
    # schema used by ``irodori_tts.inference_runtime`` so downstream
    # loaders work unchanged.
    # ------------------------------------------------------------------
    def save_quantized_model(self, runner, save_directory: str) -> str:
        from dataclasses import asdict
        from safetensors.torch import save_file

        if self._model_cfg is None:
            self._load_state_and_cfg()

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Build a model where each quantized nn.Linear has been replaced
        # by a packed GPTQLinear (qweight + scales + qzeros + g_idx).
        # This is the form that yields actual disk-size reduction.
        # The companion ``DiTAdapter.load_quantized_model`` reverses the
        # swap on load.
        model, _ = runner.create_quantized_model(
            pack_weights=True, use_gemlite=False
        )

        # Cast non-quantized tensors to fp16 for additional disk-size
        # reduction.  GPTQ packed buffers (qweight/qzeros/g_idx) are
        # already integer types and ``scales`` is already fp16; leave
        # those untouched.  Non-Linear modules (text encoder, AdaLN,
        # caption encoder, in/out projections) are the bulk of what
        # remains and tolerate fp16 cleanly at inference time.
        _PACKED_INTEGER_SUFFIXES = (".qweight", ".qzeros", ".g_idx", ".scales")
        state_dict: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            t = v.contiguous().cpu()
            if t.dtype == torch.float32 and not any(k.endswith(s) for s in _PACKED_INTEGER_SUFFIXES):
                t = t.to(torch.float16)
            state_dict[k] = t

        # Per-layer quant metadata so the loader can reconstruct
        # GPTQLinear modules with the right shape/bits.
        quantizer = runner.quantizer
        quant_layers: list[dict[str, Any]] = []
        name_to_q = getattr(quantizer, "_name_to_quantizer", None) or {}
        for layer_name in sorted(quantizer.results.keys()):
            child_q = name_to_q.get(layer_name) or quantizer
            wbits = getattr(child_q, "wbits", None)
            groupsize = getattr(child_q, "groupsize", -1)
            actorder = getattr(child_q, "actorder", False)
            module = dict(model.named_modules()).get(layer_name)
            if module is None:
                continue
            in_features = int(getattr(module, "in_features", 0))
            out_features = int(getattr(module, "out_features", 0))
            quant_layers.append(
                {
                    "name": layer_name,
                    "wbits": int(wbits) if wbits is not None else 0,
                    "groupsize": int(groupsize),
                    "actorder": bool(actorder),
                    "in_features": in_features,
                    "out_features": out_features,
                }
            )

        flat_config = dict(asdict(self._model_cfg))
        _INFERENCE_CONFIG_KEYS = (
            "max_text_len",
            "max_caption_len",
            "fixed_target_latent_steps",
        )
        if isinstance(self._train_cfg, dict):
            for key in _INFERENCE_CONFIG_KEYS:
                value = self._train_cfg.get(key)
                if isinstance(value, int):
                    flat_config[key] = int(value)
        metadata = {
            "config_json": json.dumps(flat_config, ensure_ascii=False),
            "quant_layers_json": json.dumps(quant_layers, ensure_ascii=False),
            "quant_method": "autobit",
            "checkpoint_format": "gptq",
        }

        out_path = save_path / "model.safetensors"
        save_file(state_dict, str(out_path), metadata=metadata)
        runner.logger.info(
            "DiT quantized model saved to %s (%d tensors, %d packed Linears)",
            out_path,
            len(state_dict),
            len(quant_layers),
        )
        return str(out_path)

    # ------------------------------------------------------------------
    # Load a packed AutoBit/GPTQ checkpoint produced by
    # ``DiTAdapter.save_quantized_model`` and return a ready-to-run
    # ``TextToLatentRFDiT`` with ``GPTQLinear`` modules in place of the
    # quantized ``nn.Linear`` modules.
    # ------------------------------------------------------------------
    @staticmethod
    def load_quantized_model(
        checkpoint_path: str,
        device: str = "cpu",
        dtype: str | torch.dtype = "float32",
    ) -> nn.Module:
        from safetensors import safe_open
        from irodori_tts.config import ModelConfig as _DiTModelConfig
        from irodori_tts.model import TextToLatentRFDiT
        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

        resolved_dtype = _resolve_dtype(dtype)

        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
            cfg_json = md.get("config_json")
            quant_layers_json = md.get("quant_layers_json")
            if cfg_json is None or quant_layers_json is None:
                raise ValueError(
                    f"{checkpoint_path} is missing 'config_json' or "
                    "'quant_layers_json' metadata; not a packed DiT "
                    "checkpoint produced by DiTAdapter.save_quantized_model"
                )
            cfg_dict = json.loads(cfg_json)
            quant_layers = json.loads(quant_layers_json)
            keys = list(f.keys())
            tensors = {k: f.get_tensor(k) for k in keys}

        model_cfg = _DiTModelConfig(**{
            k: v for k, v in cfg_dict.items() if k in _DiTModelConfig.__dataclass_fields__
        })
        model = TextToLatentRFDiT(model_cfg).to(dtype=resolved_dtype)

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

        # Refresh module map after substitutions and load remaining
        # (non-quantized) tensors with strict=False.
        non_quant = {k: v for k, v in tensors.items() if k not in {
            f"{e['name']}.{suffix}" for e in quant_layers
            for suffix in ("qweight", "scales", "qzeros", "g_idx", "bias")
        }}
        missing, unexpected = model.load_state_dict(non_quant, strict=False)
        # Quantized buffer keys appear as "missing" because they live on
        # the freshly-built GPTQLinear; filter those out.
        quant_keys = {
            f"{e['name']}.{suffix}" for e in quant_layers
            for suffix in ("qweight", "scales", "qzeros", "g_idx", "bias", "weight")
        }
        real_missing = [k for k in missing if k not in quant_keys]
        if real_missing:
            raise RuntimeError(f"Missing keys in quantized DiT checkpoint: {real_missing[:8]} ...")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in quantized DiT checkpoint: {unexpected[:8]} ...")

        model = model.to(device=device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    def has_additional_data(self) -> bool:
        return False

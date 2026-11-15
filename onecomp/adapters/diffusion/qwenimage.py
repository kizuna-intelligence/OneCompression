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


class _DualStreamPackWrapper(nn.Module):
    """Collapse Qwen-Image's two evolving residual streams into one tensor.

    A ``QwenImageTransformerBlock`` consumes and returns an image stream
    (``hidden_states``) and a text stream (``encoder_hidden_states``), which
    share the inner hidden dim.  OneCompression's block-propagation loop carries
    a single tensor between blocks, so this wrapper concatenates the two streams
    along the sequence axis (``packed = cat([image, text], dim=1)``).  On
    forward it splits them back, calls the real block — which returns
    ``(encoder_out, hidden_out)`` — and re-packs in the same order.  All other
    block inputs (``temb``, ``image_rotary_emb``, masks) are batch-independent
    and pass through ``kwargs`` unchanged.

    The wrapper reuses the real block (and therefore its real ``nn.Linear``
    submodules) by reference, so the quantizer's pre-built ``module_to_name``
    map and the saved checkpoint's layer names are unaffected once unwrapped.
    """

    def __init__(self, block: nn.Module, img_seq: int):
        super().__init__()
        self.block = block
        self.img_seq = int(img_seq)

    @property
    def _onecomp_wrapped_block(self) -> nn.Module:
        # A property (not an attribute) so the real block is not registered as
        # a second submodule.  The QEP driver reads this to unwrap before save.
        return self.block

    def forward(self, packed: torch.Tensor, **kwargs):
        hidden_states = packed[:, : self.img_seq]
        encoder_hidden_states = packed[:, self.img_seq :]
        encoder_out, hidden_out = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        return torch.cat([hidden_out, encoder_out], dim=1)


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
        #: Real image-token count from captured calibration, set when real
        #: inputs are loaded; the dual-stream pack wrapper splits the packed
        #: tensor here instead of at the synthetic ``image_grid**2``.
        self._real_img_seq: int | None = None

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

    # ------------------------------------------------------------------
    # Dual-stream support: let the QEP block-propagation loop carry both
    # residual streams.  These opt-in hooks are detected by
    # ``get_blocks_and_inputs``; single-stream adapters omit them.
    # ------------------------------------------------------------------
    def wrap_block(self, block: nn.Module) -> nn.Module:
        """Wrap a dual-stream block so it reads/writes one packed tensor.

        ``prepare_calibration_inputs`` runs before the blocks are wrapped, so
        when real calibration was loaded ``self._real_img_seq`` already holds
        the true image-token count; otherwise fall back to ``image_grid**2``.
        """
        img_seq = self._real_img_seq or (self.image_grid * self.image_grid)
        return _DualStreamPackWrapper(block, img_seq=img_seq)

    def pack_catcher_input(
        self, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Pack the captured block input the same way :class:`_DualStreamPackWrapper`
        packs its output: ``cat([image_stream, text_stream], dim=1)``.  The
        remaining kwargs (temb, rotary, masks) are returned for reuse."""
        kwargs = dict(kwargs)
        hidden_states = args[0] if args else kwargs.pop("hidden_states")
        encoder_hidden_states = kwargs.pop("encoder_hidden_states")
        packed = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        return packed, kwargs

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

        # Real-activation calibration (preferred for this dual-stream MMDiT).
        # Synthetic N(0,1) text embeddings are out-of-distribution and make the
        # 60-block joint attention diverge to NaN; real captured inputs stay
        # finite.  ``capture_calibration.py`` saves the top-level transformer
        # forward inputs across denoising steps.
        if self.calibration_inputs_path:
            return self._load_real_calibration_inputs(n, cfg, device, log)

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

    def _load_real_calibration_inputs(
        self, n: int, cfg: dict[str, Any], device: torch.device, log
    ) -> dict[str, Any]:
        """Load captured real top-level DiT inputs and slice to ``n`` samples.

        The saved file (see ``example/firered_int4/capture_calibration.py``)
        holds batched ``hidden_states`` / ``encoder_hidden_states`` /
        ``encoder_hidden_states_mask`` / ``timestep`` (+ optional ``guidance``)
        plus a single-sample ``img_shapes`` / ``txt_seq_lens`` descriptor.  The
        per-sample list fields are replicated to the chosen batch size.
        """
        blob = torch.load(self.calibration_inputs_path, map_location="cpu")
        avail = int(blob.get("num_calls", blob["hidden_states"].shape[0]))
        b = min(n, avail)
        if b < n:
            log.warning(
                "Real calibration has only %d samples; using %d (requested %d)",
                avail, b, n,
            )
        dtype = self.dtype

        def _take(key):
            t = blob[key]
            return t[:b].to(device=device, dtype=dtype)

        mask = blob.get("encoder_hidden_states_mask")
        inputs: dict[str, Any] = {
            "hidden_states": _take("hidden_states"),
            "encoder_hidden_states": _take("encoder_hidden_states"),
            # The real pipeline calls the transformer with mask=None (no text
            # padding to mask); pass it through unchanged so the forward matches.
            "encoder_hidden_states_mask": (
                mask[:b].to(device) if isinstance(mask, torch.Tensor) else None
            ),
            "timestep": _take("timestep"),
        }
        # ``img_shapes`` is the RoPE grid spec.  QwenEmbedRope reads only
        # ``img_shapes[0]`` (a shared spec for the whole batch — it does *not*
        # index per batch item), and for QwenImageEditPlus that entry is itself
        # a *list of grids* (target + reference image), e.g. the captured
        # ``[[(1,64,64),(1,64,64)]]``.  Replicate the inner grid-*list* to the
        # batch size so per-chunk slicing in the catcher stays non-empty; it
        # must stay a list (a tuple breaks the per-grid ``(f,h,w)`` unpack).
        grids = blob["img_shapes"]
        if isinstance(grids, list) and len(grids) == 1:
            grids = grids[0]
        grids = list(grids)
        inputs["img_shapes"] = [grids] * b
        txt0 = blob.get("txt_seq_lens")
        if isinstance(txt0, list) and txt0:
            txt0 = txt0[0]
        # The pipeline passes txt_seq_lens=None (the model derives it from the
        # encoder mask / shape); preserve that, else replicate the scalar.
        inputs["txt_seq_lens"] = None if txt0 is None else [int(txt0)] * b
        if cfg.get("guidance_embeds") and blob.get("guidance") is not None:
            inputs["guidance"] = blob["guidance"][:b].to(device=device, dtype=dtype)
        elif cfg.get("guidance_embeds"):
            inputs["guidance"] = torch.full((b,), 4.0, dtype=dtype, device=device)

        # The dual-stream pack wrapper must split at the *real* image-token
        # boundary, not the synthetic ``image_grid**2``.  Prefer the value the
        # capture recorded; fall back to the loaded tensor's seq dim.
        self._real_img_seq = int(blob.get("img_seq", inputs["hidden_states"].shape[1]))

        log.info(
            "Qwen-Image REAL calibration: %d samples from %s (img=%s, txt=%s, img_seq=%d)",
            b, self.calibration_inputs_path,
            tuple(inputs["hidden_states"].shape),
            tuple(inputs["encoder_hidden_states"].shape),
            self._real_img_seq,
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

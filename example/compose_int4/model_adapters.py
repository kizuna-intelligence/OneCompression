"""HF adapters for the local YuE2 and SongComposer int4 experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from onecomp.adapters.hf_llm import HFLLMAdapter


class ComposeModelAdapter(HFLLMAdapter):
    """Load a remote-code causal LM and expose its decoder blocks to OneComp."""

    def __init__(self, model_kind: str, model_path: str, *, dtype: str, device: str):
        if model_kind not in {"yue2", "songcomposer"}:
            raise ValueError(f"unsupported model_kind: {model_kind}")
        self.model_kind = model_kind
        super().__init__(path=model_path, dtype=dtype, device=device)

    def load_config(self):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            self.get_model_id_or_path(), trust_remote_code=True, local_files_only=True
        )

    def load_model(self, device_map: str | None = None) -> nn.Module:
        from transformers import AutoModelForCausalLM

        selected_map = self.device if device_map is None else device_map
        dtype = getattr(torch, self.dtype)
        config = self.load_config()
        if self.model_kind == "songcomposer" and not hasattr(config, "max_length"):
            # Transformers 5 no longer exposes this checkpoint field on the
            # custom config class, while the saved remote model code reads it
            # during __init__. Preserve the value recorded by the checkpoint.
            raw_config = json.loads(
                (Path(self.get_model_id_or_path()) / "config.json").read_text(encoding="utf-8")
            )
            max_length = raw_config.get(
                "max_length", getattr(config, "max_position_embeddings", None)
            )
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("SongComposer config must provide a positive max_length")
            config.max_length = max_length
        model = AutoModelForCausalLM.from_pretrained(
            self.get_model_id_or_path(),
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
            device_map=selected_map,
        )
        model.eval()
        return model

    def load_tokenizer(self):
        if self.model_kind == "yue2":
            from yue2.tokenization_yue2 import YuE2TextTokenizer

            return YuE2TextTokenizer(Path(self.get_model_id_or_path()) / "qwen.tiktoken")

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.get_model_id_or_path(), trust_remote_code=True, local_files_only=True
        )

    def get_blocks(self, model: nn.Module) -> nn.ModuleList:
        core = getattr(model, "model", None)
        blocks = getattr(core, "layers", None)
        if not isinstance(blocks, nn.ModuleList) or not len(blocks):
            raise RuntimeError(f"{type(model).__name__} has no model.layers ModuleList")
        return blocks

    def get_head_modules(
        self, model: nn.Module
    ) -> tuple[nn.Module | None, nn.Module | None]:
        core = getattr(model, "model", None)
        norm = getattr(core, "norm", None)
        head = None
        getter = getattr(model, "get_output_embeddings", None)
        if callable(getter):
            head = getter()
        if head is None:
            head = getattr(model, "lm_head", None) or getattr(model, "output", None)
        return norm, head

    def prepare_calibration_inputs(
        self,
        model: nn.Module,
        calibration_config,
        device: torch.device,
        logger=None,
    ) -> dict[str, Any]:
        path = Path(calibration_config.calibration_dataset)
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        count = int(calibration_config.num_calibration_samples)
        if len(records) < count:
            raise ValueError(
                f"need {count} {self.model_kind} calibration records, found {len(records)}"
            )
        length = int(calibration_config.max_length)
        if self.model_kind == "yue2":
            rows = [record["input_ids"] for record in records[:count]]
            pad_id = int(getattr(model.config, "eos_token_id", 0) or 0)
        else:
            tokenizer = None
            rows = []
            for index, record in enumerate(records[:count]):
                token_ids = record.get("input_ids")
                if token_ids is None:
                    text = record.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"empty Canto calibration prompt at row {index}")
                    # Match modeling_internlm2.py::inference, which wraps the
                    # prompt in the same user turn before tokenization.
                    wrapped = f"[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n"
                    if tokenizer is None:
                        tokenizer = self.load_tokenizer()
                    token_ids = tokenizer(wrapped, add_special_tokens=True)["input_ids"]
                    if token_ids and isinstance(token_ids[0], list):
                        token_ids = token_ids[0]
                if not isinstance(token_ids, (list, tuple)) or any(
                    not isinstance(token, int) for token in token_ids
                ):
                    raise ValueError(f"invalid Canto calibration token IDs at row {index}")
                if len(token_ids) > length:
                    raise ValueError(
                        f"Canto calibration prompt {index} has {len(token_ids)} tokens, "
                        f"exceeding configured max_length={length}"
                    )
                rows.append(token_ids)
            pad_id = next(
                (
                    value
                    for value in (
                        getattr(tokenizer, "pad_token_id", None),
                        getattr(tokenizer, "eos_token_id", None),
                        getattr(tokenizer, "bos_token_id", None),
                        0,
                    )
                    if value is not None
                ),
                0,
            )
        input_ids = torch.full((count, length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((count, length), dtype=torch.long)
        for index, row in enumerate(rows[:count]):
            tokens = [int(token) for token in row[:length]]
            if not tokens:
                raise ValueError(f"empty YuE2 plan sequence at row {index}")
            input_ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[index, : len(tokens)] = 1
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }

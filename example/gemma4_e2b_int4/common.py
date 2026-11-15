"""Shared Gemma 4 E2B text-only quantization helpers.

The upstream checkpoint is multimodal.  Voice chat only consumes the language
decoder, so this adapter drops the vision and audio towers before calibration
and saves a standalone ``Gemma4ForCausalLM`` checkpoint.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from onecomp.adapters.hf_llm import HFLLMAdapter

from .contracts import (
    CALIBRATION_DATASET_ID,
    GROUP_SIZE,
    MODEL_ID,
    WBITS,
    audit_fixed_int4_config,
)


class Gemma4TextAdapter(HFLLMAdapter):
    """Load only the Gemma 4 language decoder from the multimodal checkpoint."""

    def load_config(self):
        if self._cached_config is None:
            from transformers import AutoConfig

            outer_config = AutoConfig.from_pretrained(
                self.get_model_id_or_path(), trust_remote_code=True
            )
            if getattr(outer_config, "model_type", None) != "gemma4":
                raise ValueError(
                    "Gemma4TextAdapter requires a Gemma 4 checkpoint; "
                    f"got model_type={getattr(outer_config, 'model_type', None)!r}"
                )
            self._cached_config = outer_config.text_config
        return self._cached_config

    def load_model(self, device_map: str | None = None):
        from transformers import AutoModelForCausalLM
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        effective_device = device_map if device_map is not None else self.device
        wrapper = AutoModelForCausalLM.from_pretrained(
            self.get_model_id_or_path(),
            dtype=torch.bfloat16,
            device_map=effective_device,
        )
        if not hasattr(wrapper.model, "language_model"):
            raise RuntimeError("Gemma 4 checkpoint has no model.language_model")

        # Construct the shell on meta so extracting the decoder never duplicates
        # its roughly multi-gigabyte parameters in host memory.
        with torch.device("meta"):
            text_model = Gemma4ForCausalLM(wrapper.config.text_config)
        text_model.model = wrapper.model.language_model
        text_model.lm_head = wrapper.lm_head
        text_model.generation_config = wrapper.generation_config
        text_model.eval()

        # Keep only references owned by text_model before releasing the wrapper
        # and its unused image/audio modules.
        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text_model


def render_dolly_dialogue(tokenizer, row: dict[str, Any]) -> str:
    """Render one Japanese Dolly row with the model's own chat template."""

    instruction = str(row.get("instruction") or "").strip()
    context = str(row.get("context") or "").strip()
    response = str(row.get("response") or "").strip()
    if context:
        instruction = f"{instruction}\n\n参考情報:\n{context}"
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def prepare_japanese_calibration(
    output_path: Path,
    *,
    model_id: str = MODEL_ID,
    dataset_id: str = CALIBRATION_DATASET_ID,
    num_rows: int = 1024,
    seed: int = 0,
) -> Path:
    """Create deterministic Japanese dialogue JSONL for calibration."""

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset(dataset_id, split="train").shuffle(seed=seed)
    if num_rows > len(dataset):
        raise ValueError(f"requested {num_rows} rows, dataset has only {len(dataset)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset.select(range(num_rows)):
            text = render_dolly_dialogue(tokenizer, row)
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return output_path

"""Dependency-free constants and checkpoint contracts for the Gemma 4 example."""

from __future__ import annotations

from typing import Any


MODEL_ID = "google/gemma-4-E2B-it"
CALIBRATION_DATASET_ID = "llm-jp/databricks-dolly-15k-ja"
WBITS = 4
GROUP_SIZE = 128


def audit_fixed_int4_config(config: dict[str, Any]) -> dict[str, int]:
    """Validate that every declared GPTQ layer uses exactly four bits."""

    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        raise ValueError("config has no quantization_config object")
    if int(quant.get("bits", -1)) != WBITS:
        raise ValueError(f"global GPTQ bits must be {WBITS}: {quant.get('bits')!r}")
    if int(quant.get("group_size", quant.get("groupsize", -1))) != GROUP_SIZE:
        raise ValueError("global GPTQ group size must be 128")

    counts: dict[str, int] = {}
    for layer in quant.get("quantization_bits") or []:
        for module in layer.values():
            bits = int(module["bits"] if isinstance(module, dict) else module)
            counts[str(bits)] = counts.get(str(bits), 0) + 1
    if not counts:
        raise ValueError("quantization_bits contains no quantized modules")
    if set(counts) != {str(WBITS)}:
        raise ValueError(f"checkpoint is mixed-bit, found layer bits: {counts}")
    return counts

"""Build a full Qwen3.8 VLM checkpoint from an int4 text checkpoint.

The text decoder is already expensive to quantize with QEP.  This module keeps
that immutable packed shard and applies OneCompression RTN to only the missing
vision-encoder Linear weights from the upstream conditional-generation model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .quantizer.rtn import RTN


def choose_group_size(in_features: int, maximum: int = 128) -> int:
    """Use the largest power-of-two group no larger than ``maximum``.

    Qwen3.8's vision MLP width (4304) is not divisible by 128.  Selecting the
    largest exact divisor keeps all vision Linears at 4-bit without padding or
    silently leaving the large ``linear_fc2`` matrices in bf16.
    """

    if in_features <= 0 or maximum <= 0:
        raise ValueError("in_features and maximum must be positive")
    candidate = 1 << int(math.floor(math.log2(maximum)))
    while candidate > 1 and in_features % candidate:
        candidate //= 2
    return candidate


def _vlm_text_name(name: str) -> str:
    prefix = "model.layers."
    if name.startswith(prefix):
        return "model.language_model.layers." + name[len(prefix) :]
    return name


def merged_vlm_config(
    base_config: dict[str, Any],
    text_config: dict[str, Any],
    vision_groupsizes: dict[str, int],
    *,
    vision_bits: int = 4,
) -> dict[str, Any]:
    """Restore the upstream VLM config and extend its quantization metadata."""

    merged = json.loads(json.dumps(base_config))
    quant = json.loads(json.dumps(text_config["quantization_config"]))
    for key in ("modules_in_block_to_quantize", "quantized_layer_names"):
        text_names = [_vlm_text_name(name) for name in quant.get(key, [])]
        quant[key] = sorted(set(text_names) | set(vision_groupsizes))
    quant["vision_quantization"] = {
        "bits": int(vision_bits),
        "groupsizes": dict(sorted(vision_groupsizes.items())),
    }
    merged["quantization_config"] = quant
    return merged


@dataclass(frozen=True)
class LayerQuantizationMetric:
    name: str
    in_features: int
    out_features: int
    bits: int
    group_size: int
    normalized_mse: float
    weight_cosine: float


def _quantize_linear(
    name: str,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
) -> tuple[dict[str, torch.Tensor], LayerQuantizationMetric]:
    out_features, in_features = weight.shape
    linear = torch.nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)
    quantizer = RTN(wbits=bits, groupsize=group_size, sym=False, mse=False)
    quantizer.validate_params()
    result = quantizer.quantize_layer(linear)
    reconstructed = result.dequantized_weight.float()
    original = weight.float().cpu()
    error = torch.mean((original - reconstructed) ** 2)
    energy = torch.mean(original**2).clamp_min(1e-20)
    cosine = torch.nn.functional.cosine_similarity(
        original.flatten(), reconstructed.flatten(), dim=0
    )
    packed = quantizer.create_inference_layer(
        result,
        linear,
        pack_weights=True,
        use_gemlite=False,
    )
    tensors = {key: value.detach().cpu() for key, value in packed.state_dict().items()}
    metric = LayerQuantizationMetric(
        name=name,
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        normalized_mse=float((error / energy).item()),
        weight_cosine=float(cosine.item()),
    )
    return tensors, metric


def quantize_vision_shard(
    source_shard: str | Path,
    output_shard: str | Path,
    *,
    bits: int = 4,
    maximum_group_size: int = 128,
) -> tuple[dict[str, int], list[LayerQuantizationMetric], list[str]]:
    """Quantize every 2-D ``model.visual`` weight and preserve other tensors."""

    output_tensors: dict[str, torch.Tensor] = {}
    groupsizes: dict[str, int] = {}
    metrics: list[LayerQuantizationMetric] = []
    with safe_open(str(source_shard), framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        vision_keys = sorted(key for key in source_keys if key.startswith("model.visual."))
        quantized_weights: set[str] = set()
        for key in vision_keys:
            if not key.endswith(".weight"):
                continue
            weight = source.get_tensor(key)
            if weight.ndim != 2:
                continue
            name = key[: -len(".weight")]
            bias_key = name + ".bias"
            bias = source.get_tensor(bias_key) if bias_key in source_keys else None
            group_size = choose_group_size(weight.shape[1], maximum_group_size)
            packed, metric = _quantize_linear(
                name,
                weight,
                bias,
                bits=bits,
                group_size=group_size,
            )
            output_tensors.update(
                {f"{name}.{suffix}": tensor for suffix, tensor in packed.items()}
            )
            groupsizes[name] = group_size
            metrics.append(metric)
            quantized_weights.add(key)
            if bias is not None:
                quantized_weights.add(bias_key)
        for key in vision_keys:
            if key not in quantized_weights:
                output_tensors[key] = source.get_tensor(key)
    save_file(output_tensors, str(output_shard))
    return groupsizes, metrics, sorted(output_tensors)


def _link_or_copy(source: Path, target: Path) -> None:
    # Hugging Face snapshots use relative symlinks into ``../../blobs``.  A
    # hard link to the symlink inode preserves that relative target, which is
    # dangling once placed in our output directory.  Link the immutable blob
    # itself instead.
    source = source.resolve(strict=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_support_files(source_dirs: list[Path], target: Path) -> None:
    names = (
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "LICENSE",
    )
    for name in names:
        for source_dir in source_dirs:
            source = source_dir / name
            if source.is_file():
                shutil.copy2(source, target / name)
                break


def build_qwen38_vlm_checkpoint(
    *,
    base_snapshot: str | Path,
    text_snapshot: str | Path,
    output_dir: str | Path,
    vision_bits: int = 4,
    maximum_group_size: int = 128,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Create a self-contained two-shard int4 VLM checkpoint atomically."""

    base_snapshot = Path(base_snapshot)
    text_snapshot = Path(text_snapshot)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".incomplete-", dir=output_dir.parent))

    text_source = text_snapshot / "model.safetensors"
    vision_source = base_snapshot / "model-00001-of-00018.safetensors"
    if not text_source.is_file() or not vision_source.is_file():
        raise FileNotFoundError("text model.safetensors and upstream shard 1 are required")
    text_target = staging / "model-00001-of-00002.safetensors"
    vision_target = staging / "model-00002-of-00002.safetensors"
    _link_or_copy(text_source, text_target)
    groupsizes, metrics, vision_keys = quantize_vision_shard(
        vision_source,
        vision_target,
        bits=vision_bits,
        maximum_group_size=maximum_group_size,
    )

    with safe_open(str(text_source), framework="pt", device="cpu") as source:
        text_keys = list(source.keys())
    weight_map = {key: text_target.name for key in text_keys}
    overlap = set(weight_map).intersection(vision_keys)
    if overlap:
        raise ValueError(f"text and vision checkpoint keys overlap: {sorted(overlap)[:4]}")
    weight_map.update({key: vision_target.name for key in vision_keys})
    index = {
        "metadata": {"total_size": text_target.stat().st_size + vision_target.stat().st_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (staging / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    base_config = json.loads((base_snapshot / "config.json").read_text(encoding="utf-8"))
    text_config = json.loads((text_snapshot / "config.json").read_text(encoding="utf-8"))
    config = merged_vlm_config(
        base_config,
        text_config,
        groupsizes,
        vision_bits=vision_bits,
    )
    (staging / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _copy_support_files([text_snapshot, base_snapshot], staging)
    summary = {
        "format": "onecompression-qwen38-vlm-v1",
        "vision_bits": vision_bits,
        "maximum_group_size": maximum_group_size,
        "vision_quantized_linear_count": len(metrics),
        "vision_normalized_mse_mean": sum(item.normalized_mse for item in metrics) / len(metrics),
        "vision_normalized_mse_max": max(item.normalized_mse for item in metrics),
        "vision_weight_cosine_mean": sum(item.weight_cosine for item in metrics) / len(metrics),
        "vision_weight_cosine_min": min(item.weight_cosine for item in metrics),
        "layers": [asdict(item) for item in metrics],
        "provenance": provenance or {},
    }
    (staging / "quantization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output_dir)
    return output_dir

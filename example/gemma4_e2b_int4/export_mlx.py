#!/usr/bin/env python3
"""Export OneCompression Gemma 4 GPTQ-v1 weights for MLX Swift.

The conversion is lossless with respect to the quantized checkpoint: it only
transposes the already-packed K-major words and converts GPTQ zero points into
MLX affine biases.  It never reconstructs or requantizes a floating-point
weight matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .contracts import GROUP_SIZE, WBITS, audit_fixed_int4_config


GPTQ_SUFFIXES = ("qweight", "scales", "qzeros", "g_idx")
EMBEDDING_SOURCE_PREFIX = "_embed_gptq."
DEFAULT_MAX_PACKED_TENSOR_BYTES = 512 * 1024 * 1024


def _uint32_words(tensor: torch.Tensor) -> torch.Tensor:
    """Preserve the 32 packed bits while changing the tensor dtype to uint32."""

    tensor = tensor.contiguous()
    if tensor.dtype == torch.uint32:
        return tensor
    if tensor.dtype != torch.int32:
        raise TypeError(f"packed words must be int32/uint32, got {tensor.dtype}")
    return tensor.view(torch.uint32)


def _unpack_gptq_v1_zeros(qzeros: torch.Tensor, out_features: int) -> torch.Tensor:
    """Return actual GPTQ-v1 zero points as ``[groups, out_features]``."""

    if qzeros.ndim != 2 or qzeros.shape[1] * 8 != out_features:
        raise ValueError(
            "qzeros must have shape [groups, out_features / 8], got "
            f"{tuple(qzeros.shape)} for out_features={out_features}"
        )
    words = qzeros.to(torch.int64) & 0xFFFF_FFFF
    shifts = torch.arange(8, dtype=torch.int64) * WBITS
    stored = ((words.unsqueeze(-1) >> shifts) & 0xF).reshape(
        qzeros.shape[0], out_features
    )
    return (stored + 1) & 0xF


def convert_gptq_v1_to_mlx(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one packed tensor from OneCompression GPTQ-v1 to MLX affine INT4.

    OneCompression packs eight input-axis values in each int32 and stores
    ``qweight`` as ``[in_features / 8, out_features]``. MLX uses the same nibble
    order with the axes reversed. GPTQ-v1 stores ``zero - 1``; MLX evaluates
    ``integer * scale + bias``, so ``bias`` is ``-zero * scale``.
    """

    if qweight.ndim != 2 or scales.ndim != 2 or qzeros.ndim != 2:
        raise ValueError("qweight, scales, and qzeros must all be rank 2")
    out_features = int(qweight.shape[1])
    if tuple(scales.shape) != (qzeros.shape[0], out_features):
        raise ValueError(
            f"scales shape {tuple(scales.shape)} does not match "
            f"groups={qzeros.shape[0]}, out_features={out_features}"
        )

    weight = _uint32_words(qweight.T)
    mlx_scales = scales.T.contiguous()
    zeros = _unpack_gptq_v1_zeros(qzeros, out_features).T
    biases = (-zeros.to(torch.float32) * mlx_scales.to(torch.float32)).to(
        mlx_scales.dtype
    )
    return weight, mlx_scales, biases.contiguous()


def _output_prefix(source_prefix: str) -> str:
    if source_prefix.startswith(EMBEDDING_SOURCE_PREFIX):
        return source_prefix[len(EMBEDDING_SOURCE_PREFIX) :]
    return source_prefix


def split_mlx_quantized_embedding(
    weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    bits: int = WBITS,
    max_packed_tensor_bytes: int = DEFAULT_MAX_PACKED_TENSOR_BYTES,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split an MLX quantized embedding on group-aligned feature boundaries.

    Some iPhone GPUs cap a single Metal buffer at 1 GiB. Gemma 4 E2B's
    per-layer embedding has one 1.094 GiB packed weight, even though the whole
    checkpoint fits in memory. Splitting the packed columns and their matching
    scale/bias groups preserves every quantized code and lets the runtime gather
    each feature shard independently before concatenating the results.
    """

    if max_packed_tensor_bytes <= 0:
        raise ValueError("max_packed_tensor_bytes must be positive")
    if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        raise ValueError("weight, scales, and biases must all be rank 2")
    if scales.shape != biases.shape or weight.shape[0] != scales.shape[0]:
        raise ValueError("embedding weight/scales/biases shapes are inconsistent")
    packed_columns_per_group = group_size * bits // 32
    if group_size * bits % 32 != 0:
        raise ValueError("group_size * bits must be divisible by 32")
    if weight.shape[1] != scales.shape[1] * packed_columns_per_group:
        raise ValueError("packed columns do not match the scale groups")

    if weight.nbytes <= max_packed_tensor_bytes:
        return [(weight, scales, biases)]

    bytes_per_group = (
        weight.shape[0] * packed_columns_per_group * weight.element_size()
    )
    max_groups_per_shard = max_packed_tensor_bytes // bytes_per_group
    if max_groups_per_shard < 1:
        raise ValueError(
            "one quantization group exceeds max_packed_tensor_bytes: "
            f"{bytes_per_group} > {max_packed_tensor_bytes}"
        )

    group_count = int(scales.shape[1])
    shard_count = math.ceil(group_count / max_groups_per_shard)
    groups_per_shard, larger_shards = divmod(group_count, shard_count)
    group_widths = [
        groups_per_shard + (1 if index < larger_shards else 0)
        for index in range(shard_count)
    ]
    packed_widths = [width * packed_columns_per_group for width in group_widths]
    weight_shards = torch.split(weight, packed_widths, dim=1)
    scale_shards = torch.split(scales, group_widths, dim=1)
    bias_shards = torch.split(biases, group_widths, dim=1)
    return list(zip(weight_shards, scale_shards, bias_shards, strict=True))


def export_checkpoint(
    source_dir: Path,
    output_dir: Path,
    *,
    max_packed_tensor_bytes: int = DEFAULT_MAX_PACKED_TENSOR_BYTES,
) -> dict:
    """Write a local-only MLX checkpoint that preserves all fixed INT4 codes."""

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")

    config_path = source_dir / "config.json"
    model_path = source_dir / "model.safetensors"
    with config_path.open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    layer_counts = audit_fixed_int4_config(source_config)

    output_dir.mkdir(parents=True)
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    converted_prefixes: list[str] = []
    feature_shards: dict[str, dict] = {}
    with safe_open(str(model_path), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        metadata = dict(source.metadata() or {})
        for qweight_key in sorted(key for key in keys if key.endswith(".qweight")):
            source_prefix = qweight_key[: -len(".qweight")]
            paired = {
                suffix: f"{source_prefix}.{suffix}" for suffix in GPTQ_SUFFIXES
            }
            required = [paired["qweight"], paired["scales"], paired["qzeros"]]
            missing = [key for key in required if key not in keys]
            if missing:
                raise KeyError(f"incomplete GPTQ tensor {source_prefix}: {missing}")

            weight, scales, biases = convert_gptq_v1_to_mlx(
                source.get_tensor(paired["qweight"]),
                source.get_tensor(paired["scales"]),
                source.get_tensor(paired["qzeros"]),
            )
            output_prefix = _output_prefix(source_prefix)
            embedding_shards = (
                split_mlx_quantized_embedding(
                    weight,
                    scales,
                    biases,
                    max_packed_tensor_bytes=max_packed_tensor_bytes,
                )
                if source_prefix.startswith(EMBEDDING_SOURCE_PREFIX)
                else [(weight, scales, biases)]
            )
            if len(embedding_shards) == 1:
                converted[f"{output_prefix}.weight"] = weight
                converted[f"{output_prefix}.scales"] = scales
                converted[f"{output_prefix}.biases"] = biases
            else:
                logical_dimensions: list[int] = []
                for index, (shard_weight, shard_scales, shard_biases) in enumerate(
                    embedding_shards
                ):
                    shard_prefix = f"{output_prefix}.shards.{index}"
                    converted[f"{shard_prefix}.weight"] = shard_weight.contiguous()
                    converted[f"{shard_prefix}.scales"] = shard_scales.contiguous()
                    converted[f"{shard_prefix}.biases"] = shard_biases.contiguous()
                    logical_dimensions.append(
                        int(shard_weight.shape[1]) * 32 // WBITS
                    )
                feature_shards[output_prefix] = {
                    "axis": "features",
                    "bits": WBITS,
                    "group_size": GROUP_SIZE,
                    "logical_dimensions": logical_dimensions,
                }
            consumed.update(key for key in paired.values() if key in keys)
            converted_prefixes.append(output_prefix)

        for key in sorted(keys - consumed):
            if key.startswith(EMBEDDING_SOURCE_PREFIX):
                raise ValueError(f"unconverted packed embedding tensor: {key}")
            if key.endswith(tuple(f".{suffix}" for suffix in GPTQ_SUFFIXES)):
                raise ValueError(f"unconverted GPTQ tensor: {key}")
            converted[key] = source.get_tensor(key)

    metadata["format"] = "mlx"
    metadata["onecompression_source"] = json.dumps(
        {
            "bits": WBITS,
            "group_size": GROUP_SIZE,
            "checkpoint_format": "gptq-v1",
            "conversion": "packed-lossless",
        },
        sort_keys=True,
    )
    save_file(converted, str(output_dir / "model.safetensors"), metadata=metadata)

    for path in source_dir.iterdir():
        if not path.is_file() or path.name in {"model.safetensors", "config.json"}:
            continue
        shutil.copy2(path, output_dir / path.name)

    output_config = dict(source_config)
    output_config.pop("quantization_config", None)
    output_config.pop("embedding_quantization_config", None)
    output_config["quantization"] = {
        "group_size": GROUP_SIZE,
        "bits": WBITS,
    }
    output_config["onecompression_source_quantization"] = {
        "bits": WBITS,
        "group_size": GROUP_SIZE,
        "checkpoint_format": "gptq-v1",
        "conversion": "packed-lossless",
        "linear_layer_counts": layer_counts,
    }
    if feature_shards:
        output_config["onecompression_mlx_feature_shards"] = feature_shards
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(output_config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return {
        "source": str(source_dir),
        "output": str(output_dir),
        "bits": WBITS,
        "group_size": GROUP_SIZE,
        "converted_tensors": len(converted_prefixes),
        "converted_prefixes": converted_prefixes,
        "feature_shards": feature_shards,
        "size_bytes": (output_dir / "model.safetensors").stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--max-packed-tensor-bytes",
        type=int,
        default=DEFAULT_MAX_PACKED_TENSOR_BYTES,
        help="maximum packed embedding shard size (default: 512 MiB)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export_checkpoint(
                args.source,
                args.output,
                max_packed_tensor_bytes=args.max_packed_tensor_bytes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

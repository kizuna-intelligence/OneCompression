#!/usr/bin/env python3
"""Pack Gemma 4 input/per-layer embeddings as shared GPTQ-format int4."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EMBEDDING_NAMES = (
    "model.embed_tokens",
    "model.embed_tokens_per_layer",
)


def pack_embedding_weight(
    weight: torch.Tensor,
    groupsize: int = 128,
    chunk_rows: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric RTN int4 in AutoGPTQ-v1 K-major packed layout.

    Chunking bounds temporary memory for Gemma 4's 4.4 GiB per-layer table.
    The returned buffers can be shared by a packed embedding lookup and a
    fused vocabulary projection without transposing or duplicating weights.
    """

    if weight.dim() != 2:
        raise ValueError(f"embedding weight must be 2-D, got {weight.shape}")
    out_features, in_features = map(int, weight.shape)
    if in_features % groupsize:
        raise ValueError("embedding dimension must be divisible by groupsize")
    if in_features % 8 or out_features % 8:
        raise ValueError("GPTQ shared embedding dimensions must be divisible by 8")
    if chunk_rows < 8 or chunk_rows % 8:
        raise ValueError("chunk_rows must be a positive multiple of 8")

    num_groups = in_features // groupsize
    qweight = torch.empty(in_features // 8, out_features, dtype=torch.int32)
    scales = torch.empty(num_groups, out_features, dtype=torch.float16)
    qzeros = torch.empty(num_groups, out_features // 8, dtype=torch.int32)
    shifts = (torch.arange(8, dtype=torch.int32) * 4).view(1, 1, 8)

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        rows = weight[start:end].to(torch.float32)
        grouped = rows.reshape(end - start, num_groups, groupsize)
        minimum = grouped.amin(dim=-1, keepdim=True)
        maximum = grouped.amax(dim=-1, keepdim=True)
        scale = ((maximum - minimum).clamp_min_(1e-8) / 15.0)
        zero = (-minimum / scale).round().clamp_(0, 15)
        values = (grouped / scale + zero).round().clamp_(0, 15).to(torch.int32)
        values = values.reshape(end - start, in_features // 8, 8)
        packed_weight = ((values & 0xF) << shifts).sum(dim=-1, dtype=torch.int32)
        qweight[:, start:end] = packed_weight.T
        scales[:, start:end] = scale.squeeze(-1).to(torch.float16).T

        stored_zeros = ((zero.squeeze(-1).to(torch.int32) - 1) & 0xF).T
        stored_zeros = stored_zeros.reshape(num_groups, (end - start) // 8, 8)
        packed_zeros = ((stored_zeros & 0xF) << shifts).sum(
            dim=-1, dtype=torch.int32
        )
        qzeros[:, start // 8 : end // 8] = packed_zeros

    return qweight, scales, qzeros


def pack_checkpoint_embeddings(
    source_dir: Path,
    output_dir: Path,
    *,
    groupsize: int = 128,
    chunk_rows: int = 1024,
) -> dict:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    source_model = source_dir / "model.safetensors"
    output_model = output_dir / "model.safetensors"
    entries = []
    with safe_open(str(source_model), framework="pt", device="cpu") as source:
        metadata = dict(source.metadata() or {})
        tensors = {
            key: source.get_tensor(key)
            for key in source.keys()
            if key not in {f"{name}.weight" for name in EMBEDDING_NAMES}
        }
        for name in EMBEDDING_NAMES:
            key = f"{name}.weight"
            if key not in source.keys():
                raise KeyError(f"missing embedding weight: {key}")
            weight = source.get_tensor(key)
            qweight, scales, qzeros = pack_embedding_weight(
                weight, groupsize=groupsize, chunk_rows=chunk_rows
            )
            prefix = f"_embed_gptq.{name}."
            tensors[prefix + "qweight"] = qweight
            tensors[prefix + "scales"] = scales
            tensors[prefix + "qzeros"] = qzeros
            entries.append(
                {
                    "name": name,
                    "bits": 4,
                    "group_size": groupsize,
                    "num_embeddings": int(weight.shape[0]),
                    "embedding_dim": int(weight.shape[1]),
                    "checkpoint_format": "gptq",
                }
            )
            original_gib = weight.numel() * weight.element_size() / 2**30
            packed_gib = (
                qweight.numel() * qweight.element_size()
                + scales.numel() * scales.element_size()
                + qzeros.numel() * qzeros.element_size()
            ) / 2**30
            print(f"packed {name}: {original_gib:.3f} GiB -> {packed_gib:.3f} GiB")

        metadata["gemma4_embedding_quantization"] = json.dumps(
            {"bits": 4, "group_size": groupsize, "layers": entries},
            sort_keys=True,
        )
        save_file(tensors, str(output_model), metadata=metadata)

    for path in source_dir.iterdir():
        if path.name == "model.safetensors" or not path.is_file():
            continue
        shutil.copy2(path, output_dir / path.name)

    config_path = output_dir / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["embedding_quantization_config"] = {
        "bits": 4,
        "group_size": groupsize,
        "checkpoint_format": "gptq",
        "layers": entries,
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return {
        "source": str(source_dir),
        "output": str(output_dir),
        "bits": 4,
        "group_size": groupsize,
        "layers": entries,
        "size_bytes": output_model.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    args = parser.parse_args()
    summary = pack_checkpoint_embeddings(
        args.source,
        args.output,
        groupsize=args.groupsize,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe 4-bit RTN quantization on a CLIP vision encoder.

This is a quick quality check for the question "should the VLM image encoder be
4-bit too?" It uses OneComp's RTN pseudo-quantizer on only the vision encoder
Linear layers, then compares image embeddings and zero-shot logits against the
original model.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from onecomp.quantizer.rtn.quantizer import pseudo_quantize_tensor


@dataclass
class Result:
    model_id: str
    bits: int
    group_size: int
    mse_grid: bool
    num_images: int
    num_prompts: int
    quantized_linear_layers: int
    fp16_vision_linear_mb: float
    int_weight_mb: float
    scale_zero_mb: float
    approx_quantized_mb: float
    approx_compression_ratio: float
    mean_embedding_cosine: float
    min_embedding_cosine: float
    mean_relative_logit_error: float
    max_relative_logit_error: float
    top1_match_rate: float
    original_accuracy: float | None
    quantized_accuracy: float | None


PROMPTS = [
    "a red square",
    "a green circle",
    "a blue triangle",
    "a yellow square",
    "a black diagonal line",
    "a white image",
]

CIFAR10_PROMPTS = [
    "a photo of an airplane",
    "a photo of an automobile",
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a deer",
    "a photo of a dog",
    "a photo of a frog",
    "a photo of a horse",
    "a photo of a ship",
    "a photo of a truck",
]


def make_images(size: int = 224) -> list[Image.Image]:
    specs = [
        ("red_square", "red", "square"),
        ("green_circle", "green", "circle"),
        ("blue_triangle", "blue", "triangle"),
        ("yellow_square", "yellow", "square"),
        ("black_line", "black", "line"),
        ("white", "white", "blank"),
    ]
    images: list[Image.Image] = []
    for _, color, shape in specs:
        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        pad = size // 5
        if shape == "square":
            draw.rectangle((pad, pad, size - pad, size - pad), fill=color)
        elif shape == "circle":
            draw.ellipse((pad, pad, size - pad, size - pad), fill=color)
        elif shape == "triangle":
            draw.polygon(
                [(size // 2, pad), (size - pad, size - pad), (pad, size - pad)],
                fill=color,
            )
        elif shape == "line":
            draw.line((pad, pad, size - pad, size - pad), fill=color, width=size // 14)
        images.append(img)
    return images


def load_eval_images(args: argparse.Namespace) -> tuple[list[Image.Image], list[str], torch.Tensor | None]:
    if args.dataset == "synthetic":
        return make_images(args.image_size), PROMPTS, None

    if args.dataset != "cifar10":
        raise ValueError(f"unknown dataset: {args.dataset}")

    try:
        from torchvision.datasets import CIFAR10

        ds = CIFAR10(root=args.data_dir, train=False, download=True)
        iterable = (ds[idx] for idx in range(min(args.limit, len(ds))))
    except ModuleNotFoundError:
        from datasets import load_dataset

        hf_ds = load_dataset("uoft-cs/cifar10", split=f"test[:{args.limit}]")
        iterable = ((row["img"], row["label"]) for row in hf_ds)

    images: list[Image.Image] = []
    labels: list[int] = []
    for image, label in iterable:
        images.append(image.convert("RGB"))
        labels.append(int(label))
    return images, CIFAR10_PROMPTS, torch.tensor(labels, dtype=torch.long)


def quantize_vision_linears(
    model: CLIPModel,
    *,
    bits: int,
    group_size: int,
    mse_grid: bool,
) -> tuple[int, int, int, int]:
    """Replace vision encoder Linear weights with dequantized RTN weights.

    Returns:
        (num_layers, fp16_bytes, int_weight_bytes, scale_zero_bytes)
    """
    num_layers = 0
    fp16_bytes = 0
    int_weight_bytes = 0
    scale_zero_bytes = 0

    for module in model.vision_model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight.data
        if group_size > 0 and weight.shape[-1] % group_size != 0:
            continue

        fp16_bytes += weight.numel() * 2
        wq, scale, zero, wint = pseudo_quantize_tensor(
            weight.float(),
            n_bit=bits,
            q_group_size=group_size,
            zero_point=True,
            perchannel=True,
            mse=mse_grid,
        )
        module.weight.data.copy_(wq.to(dtype=module.weight.dtype, device=module.weight.device))
        num_layers += 1
        int_weight_bytes += (wint.numel() * bits + 7) // 8
        scale_zero_bytes += (scale.numel() + zero.numel()) * 2

    return num_layers, fp16_bytes, int_weight_bytes, scale_zero_bytes


@torch.inference_mode()
def run(args: argparse.Namespace) -> Result:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = CLIPProcessor.from_pretrained(args.model_id)
    original = CLIPModel.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()
    quantized = copy.deepcopy(original).to(device).eval()

    num_layers, fp16_bytes, int_bytes, scale_zero_bytes = quantize_vision_linears(
        quantized,
        bits=args.bits,
        group_size=args.group_size,
        mse_grid=args.mse_grid,
    )

    images, prompts, labels = load_eval_images(args)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    fp = original(**inputs)
    q = quantized(**inputs)

    fp_emb = torch.nn.functional.normalize(fp.image_embeds.float(), dim=-1)
    q_emb = torch.nn.functional.normalize(q.image_embeds.float(), dim=-1)
    cos = (fp_emb * q_emb).sum(dim=-1)

    fp_logits = fp.logits_per_image.float()
    q_logits = q.logits_per_image.float()
    denom = fp_logits.norm(dim=-1).clamp_min(1e-8)
    rel = (fp_logits - q_logits).norm(dim=-1) / denom
    top1_match = (fp_logits.argmax(dim=-1) == q_logits.argmax(dim=-1)).float()
    original_accuracy = None
    quantized_accuracy = None
    if labels is not None:
        labels = labels.to(device)
        original_accuracy = float((fp_logits.argmax(dim=-1) == labels).float().mean().item())
        quantized_accuracy = float((q_logits.argmax(dim=-1) == labels).float().mean().item())

    approx_mb = (int_bytes + scale_zero_bytes) / 1024 / 1024
    fp16_mb = fp16_bytes / 1024 / 1024
    return Result(
        model_id=args.model_id,
        bits=args.bits,
        group_size=args.group_size,
        mse_grid=args.mse_grid,
        num_images=len(images),
        num_prompts=len(prompts),
        quantized_linear_layers=num_layers,
        fp16_vision_linear_mb=fp16_mb,
        int_weight_mb=int_bytes / 1024 / 1024,
        scale_zero_mb=scale_zero_bytes / 1024 / 1024,
        approx_quantized_mb=approx_mb,
        approx_compression_ratio=(fp16_mb / approx_mb) if approx_mb else 0.0,
        mean_embedding_cosine=float(cos.mean().item()),
        min_embedding_cosine=float(cos.min().item()),
        mean_relative_logit_error=float(rel.mean().item()),
        max_relative_logit_error=float(rel.max().item()),
        top1_match_rate=float(top1_match.mean().item()),
        original_accuracy=original_accuracy,
        quantized_accuracy=quantized_accuracy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--mse-grid", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dataset", choices=("synthetic", "cifar10"), default="synthetic")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--data-dir", default="/tmp/onecomp-vlm-data")
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Post-QEP GPTQ scale/zero refinement for Cosmos Transfer2.5.

This is a diffusion/DiT adaptation of Fujitsu/qep-dev's block-wise PTQ
post-process.  It starts from a packed QEP+GPTQ checkpoint, loads the bf16
teacher transformer, and optimizes GPTQLinear scales/zeros block-by-block to
minimize teacher/student transformer-block MSE.

The full upstream GlobalPTQ path is LLM/logits/KL-specific.  For Cosmos we use
the same Fujitsu idea, but replace logits KL with DiT hidden-state MSE and keep
only one teacher/student block on GPU at a time so a 24GB card is realistic.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from onecomp import CalibrationConfig, setup_logger
from onecomp.adapters import CosmosTransferDiTAdapter
from onecomp.post_process._blockwise.gptq_block_optimizer import optimize_gptq_block
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
from onecomp.utils.blockwise import (
    forward_input,
    get_blocks_and_inputs,
    move_kwargs_to_device,
)


def _has_gptq(module: torch.nn.Module) -> bool:
    return any(isinstance(m, GPTQLinear) for m in module.modules())


def _as_samples(tensor: torch.Tensor) -> list[torch.Tensor]:
    return [tensor[i].detach().cpu() for i in range(tensor.shape[0])]


def _wrap_blocks(adapter: CosmosTransferDiTAdapter, model: torch.nn.Module):
    blocks = adapter.get_blocks(model)
    for i in range(len(blocks)):
        wrapped = adapter.wrap_block(blocks[i])
        if wrapped is not None:
            blocks[i] = wrapped
    return blocks


def _unwrap_blocks(blocks) -> None:
    for i in range(len(blocks)):
        real = getattr(blocks[i], "_onecomp_wrapped_block", None)
        if real is not None:
            blocks[i] = real


def _load_metadata(path: str) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


def _save_refined_model(
    model: torch.nn.Module,
    src_checkpoint: str,
    out_path: str,
) -> None:
    metadata = _load_metadata(src_checkpoint)
    metadata["post_process"] = "blockwise_gptq_dit_mse"
    metadata["post_process_base"] = metadata.get("quant_method", "qep_gptq")

    state = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().contiguous().cpu()
        state[name] = tensor

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, out_path, metadata=metadata)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_checkpoint", help="HF repo/local dir for the fp teacher")
    ap.add_argument("quant_checkpoint", help="QEP+GPTQ safetensors checkpoint")
    ap.add_argument("save_dir")
    ap.add_argument("--revision", default="diffusers/general")
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--single-file", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--latent-frames", type=int, default=3)
    ap.add_argument("--latent-height", type=int, default=16)
    ap.add_argument("--latent-width", type=int, default=16)
    ap.add_argument("--text-seq-len", type=int, default=512)
    ap.add_argument("--calibration-inputs", default=None,
                    help="optional torch file with real Cosmos transformer inputs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--optimize-intweight", action="store_true",
                    help="also optimize int weights with Smooth STE; slower/riskier")
    ap.add_argument("--intweight-lr", type=float, default=1e-5)
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="debug limit; 0 means all blocks")
    ap.add_argument("--output-name", default="model.safetensors")
    args = ap.parse_args()

    setup_logger()
    device = torch.device(args.device)

    adapter = CosmosTransferDiTAdapter(
        checkpoint_path=args.base_checkpoint,
        dtype=args.dtype,
        device="cpu",
        seed=0,
        calibration_inputs_path=args.calibration_inputs,
        revision=args.revision,
        subfolder=args.subfolder,
        single_file=args.single_file,
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        text_seq_len=args.text_seq_len,
    )

    print("loading fp teacher on CPU...")
    teacher = adapter.load_model(device_map="cpu")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    print("loading int4 student on CPU...")
    student = CosmosTransferDiTAdapter.load_quantized_model(
        args.quant_checkpoint,
        device="cpu",
        dtype=args.dtype,
    )
    student.eval()

    calib = CalibrationConfig(
        num_calibration_samples=args.num_samples,
        max_length=256,
    )
    model_inputs = adapter.prepare_calibration_inputs(
        teacher,
        calib,
        device=torch.device("cpu"),
    )

    print("capturing first-block inputs...")
    student_blocks, student_inps, block_kwargs = get_blocks_and_inputs(
        student,
        model_inputs,
        batch_size=args.batch_size,
        adapter=adapter,
    )
    teacher_blocks = _wrap_blocks(adapter, teacher)
    teacher_inps = student_inps.clone()
    block_kwargs = move_kwargs_to_device(block_kwargs, device)

    total_blocks = len(student_blocks)
    limit = total_blocks if args.max_blocks <= 0 else min(args.max_blocks, total_blocks)
    print(f"refining {limit}/{total_blocks} transformer blocks")

    improvements: list[float] = []
    try:
        for idx in range(limit):
            s_block = student_blocks[idx].to(device)
            t_block = teacher_blocks[idx].to(device)
            s_block.eval()
            t_block.eval()

            print(f"[block {idx + 1:02d}/{total_blocks}] teacher forward")
            with torch.no_grad():
                target_out = forward_input(
                    teacher_inps,
                    t_block,
                    block_kwargs,
                    args.batch_size,
                    device,
                )

            if _has_gptq(s_block):
                print(f"[block {idx + 1:02d}/{total_blocks}] optimize GPTQ scales/zeros")
                init_mse, final_mse = optimize_gptq_block(
                    layer=s_block,
                    inps=_as_samples(student_inps),
                    target_outputs=_as_samples(target_out),
                    layer_kwargs=block_kwargs,
                    lr=args.lr,
                    epochs=args.epochs,
                    dev=device,
                    grad_clip=args.grad_clip,
                    optimize_intweight=args.optimize_intweight,
                    intweight_lr=args.intweight_lr,
                    use_cosine_schedule=True,
                    warmup_ratio=0.1,
                    shuffle_samples=True,
                )
                if init_mse > 0:
                    improvement = (init_mse - final_mse) / init_mse * 100.0
                    improvements.append(improvement)
                    print(
                        f"[block {idx + 1:02d}/{total_blocks}] "
                        f"MSE {init_mse:.6e} -> {final_mse:.6e} "
                        f"({improvement:+.2f}%)"
                    )
            else:
                print(f"[block {idx + 1:02d}/{total_blocks}] no GPTQLinear; skip")

            with torch.no_grad():
                student_inps = forward_input(
                    student_inps,
                    s_block,
                    block_kwargs,
                    args.batch_size,
                    device,
                )
            teacher_inps = target_out

            student_blocks[idx] = s_block.cpu()
            teacher_blocks[idx] = t_block.cpu()
            del s_block, t_block, target_out
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        _unwrap_blocks(student_blocks)
        _unwrap_blocks(teacher_blocks)

    out_path = os.path.join(args.save_dir, args.output_name)
    print(f"saving refined checkpoint to {out_path}")
    _save_refined_model(student, args.quant_checkpoint, out_path)
    if improvements:
        print(f"average block improvement: {sum(improvements) / len(improvements):.2f}%")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

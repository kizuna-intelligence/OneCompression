"""Run NVIDIA Cosmos-Transfer2.5 multibranch inference with RTN 4-bit DiT.

This runner keeps NVIDIA's official multibranch control path intact and only
replaces Linear layers under ``inference_pipeline.model.net`` with
OneCompression's packed GPTQLinear using RTN weight quantization.
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn


def _add_repo_to_path(path: Path) -> None:
    path = path.expanduser().resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _cuda_mem(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[mem] {label}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB", flush=True)


def _get_parent_module(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def _normalize_rtn_scale_zero(tensor: torch.Tensor, out_features: int) -> torch.Tensor:
    # run_rtn returns grouped Linear scale/zero as (out_features, num_groups).
    # GPTQLinear expects (num_groups, out_features).
    if tensor.dim() == 2 and tensor.shape[0] == out_features:
        return tensor.t().contiguous()
    return tensor.contiguous()


def _effective_groupsize(in_features: int, requested_groupsize: int) -> int:
    if requested_groupsize <= 0:
        return -1
    if in_features % requested_groupsize == 0:
        return requested_groupsize
    for candidate in range(min(requested_groupsize, in_features), 0, -1):
        if in_features % candidate == 0:
            return candidate
    return math.gcd(in_features, requested_groupsize) or 1


def quantize_net_linears_rtn4(
    net: nn.Module,
    *,
    groupsize: int,
    sym: bool,
    mse: bool,
    limit: int | None,
    use_gemlite: bool | None,
) -> int:
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    from onecomp.quantizer.rtn.rtn_impl import run_rtn

    linears = [(name, module) for name, module in net.named_modules() if isinstance(module, nn.Linear)]
    if limit is not None:
        linears = linears[:limit]
    print(f"[int4] quantizing {len(linears)} Linear layers under model.net", flush=True)

    started = time.time()
    for idx, (name, module) in enumerate(linears, start=1):
        if not isinstance(module, nn.Linear):
            continue
        device = module.weight.device
        in_features = module.in_features
        out_features = module.out_features
        effective_groupsize = _effective_groupsize(in_features, groupsize)

        bias = module.bias.detach().cpu() if module.bias is not None else None
        module_cpu = module.to("cpu")
        quant = run_rtn(
            module_cpu,
            wbits=4,
            groupsize=effective_groupsize,
            sym=sym,
            mse=mse,
        )
        scale = _normalize_rtn_scale_zero(quant["scale"], out_features)
        zero = _normalize_rtn_scale_zero(quant["zero"], out_features)

        qlinear = GPTQLinear(
            in_features=in_features,
            out_features=out_features,
            wbits=4,
            groupsize=effective_groupsize,
            actorder=False,
            quantized_weight=quant["quantized_weight"],
            scale=scale,
            zero=zero,
            bias=bias,
            device=device,
            pack_weights=True,
            use_gemlite=use_gemlite,
        )
        parent, leaf = _get_parent_module(net, name)
        parent._modules[leaf] = qlinear

        del module_cpu, module, quant, scale, zero, qlinear, bias
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if idx == 1 or idx % 25 == 0 or idx == len(linears):
            elapsed = time.time() - started
            print(f"[int4] {idx}/{len(linears)} {name} elapsed={elapsed:.1f}s", flush=True)
            _cuda_mem(f"after {idx} linears")

    return len(linears)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--cosmos-repo", type=Path, default=Path("/home/yusuke/gitrepos/cosmos-transfer2.5"))
    parser.add_argument("--onecomp-repo", type=Path, default=Path("/home/yusuke/gitrepos/OneCompression"))
    parser.add_argument("--model", default="edge")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-video-frames-per-chunk", type=int, default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--keep-input-resolution", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--disable-guardrails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--groupsize", type=int, default=32)
    parser.add_argument("--asym", action="store_true", help="Use asymmetric RTN instead of symmetric RTN.")
    parser.add_argument("--mse", action="store_true", help="Enable RTN MSE clipping search.")
    parser.add_argument("--limit-linears", type=int, default=None, help="Debug only: quantize the first N Linear layers.")
    parser.add_argument(
        "--use-gemlite",
        choices=["auto", "true", "false"],
        default="auto",
        help="GemLite backend selection for GPTQLinear.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_repo_to_path(args.cosmos_repo)
    _add_repo_to_path(args.onecomp_repo)

    from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
    from cosmos_transfer2.config import InferenceArguments, InferenceOverrides, SetupArguments
    from cosmos_transfer2.inference import Control2WorldInference

    init_environment()
    try:
        override_data = {}
        for key in (
            "seed",
            "guidance",
            "num_steps",
            "max_frames",
            "num_video_frames_per_chunk",
            "resolution",
            "keep_input_resolution",
        ):
            value = getattr(args, key)
            if value is not None:
                override_data[key] = value
        overrides = InferenceOverrides.model_validate(override_data)
        samples, batch_hint_keys = InferenceArguments.from_files([args.input_file], overrides=overrides)

        init_output_dir(args.output_dir, profile=False)
        setup = SetupArguments.model_validate(
            {
                "output_dir": args.output_dir,
                "model": args.model,
                "disable_guardrails": args.disable_guardrails,
                "offload_guardrail_models": False,
                "keep_going": False,
                "profile": False,
                "benchmark": False,
            }
        )

        _cuda_mem("before official load")
        inference = Control2WorldInference(setup, batch_hint_keys=batch_hint_keys)
        _cuda_mem("after official load")

        gemlite: bool | None
        if args.use_gemlite == "auto":
            gemlite = None
        else:
            gemlite = args.use_gemlite == "true"

        quantized = quantize_net_linears_rtn4(
            inference.inference_pipeline.model.net,
            groupsize=args.groupsize,
            sym=not args.asym,
            mse=args.mse,
            limit=args.limit_linears,
            use_gemlite=gemlite,
        )
        print(f"[int4] replaced {quantized} Linear layers", flush=True)
        _cuda_mem("after int4 replacement")

        output_paths = inference.generate(samples, output_dir=args.output_dir)
        _cuda_mem("after generation")
        print("[done] outputs:", output_paths, flush=True)
    finally:
        cleanup_environment()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run QEP+GPTQ int4 calibration for YuE2 or SongComposer."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.append(
    "/home/yusuke/gitrepos/compose_proto/.venv/lib/python3.12/site-packages"
)
from compose_int4.model_adapters import ComposeModelAdapter  # noqa: E402

GROUP_SIZE = 32
WBITS = 4
QEP_EXCLUDE_LAYER_KEYWORDS = ["mlp.down_proj", "Plora_A", "Plora_B"]
REMOTE_CODE = {
    "yue2": ("modeling_yue2.py",),
    "songcomposer": (
        "configuration_internlm.py",
        "modeling_internlm2.py",
        "build_mlp.py",
        "tokenization_internlm.py",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(REMOTE_CODE))
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qep-device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def _copy_remote_code(kind: str, model_path: Path, output_path: Path) -> None:
    for name in REMOTE_CODE[kind]:
        source = model_path / name
        if source.is_file():
            shutil.copy2(source, output_path / name)


def _bits_summary(config: dict) -> dict[str, int]:
    entries = config.get("quantization_config", {}).get("quantization_bits", [])
    counts: Counter[str] = Counter()
    for layer in entries:
        if not isinstance(layer, dict):
            continue
        for value in layer.values():
            bits = value.get("bits") if isinstance(value, dict) else value
            if bits is not None:
                counts[str(int(bits))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    calibration_path = args.calibration.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_path}")

    dtype = "bfloat16" if args.kind == "yue2" else "float16"
    # Saved production-tokenized SongComposer rows fit within 96 tokens and
    # the adapter right-pads them without truncating the original prompts.
    max_length = args.max_length or (512 if args.kind == "yue2" else 96)
    adapter = ComposeModelAdapter(
        args.kind, str(model_path), dtype=dtype, device=args.device
    )
    model_config = ModelConfig(adapter=adapter, dtype=dtype, device=args.device)
    calibration = CalibrationConfig(
        calibration_dataset=str(calibration_path),
        text_key="text",
        max_length=max_length,
        num_calibration_samples=args.samples,
        strategy="concat_chunk",
        seed=0,
    )
    # Restrict GPTQ to decoder blocks. YuE2's 184,704-token lm_head and
    # SongComposer's output projection stay in their source dtype.
    quantizer = GPTQ(
        wbits=WBITS,
        groupsize=GROUP_SIZE,
        actorder=False,
        include_layer_keywords=["model.layers."],
    )
    # SongComposer's PLoRA A/B sublayers only contribute on image-masked
    # positions. These text-only calibration prompts do not exercise that
    # branch, and QEP cannot form a stable correction Hessian for it. Keep
    # GPTQ quantization on those weights, but omit QEP correction there.
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        qep=True,
        qep_config=QEPConfig(
            general=False,
            device=args.qep_device,
            exclude_layer_keywords=QEP_EXCLUDE_LAYER_KEYWORDS,
        ),
        calibration_config=calibration,
    )

    setup_logger()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runner.run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    quantization_seconds = time.perf_counter() - started
    output_path.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    runner.save_quantized_model(str(output_path), pack_weights=True)
    _copy_remote_code(args.kind, model_path, output_path)
    save_seconds = time.perf_counter() - save_started

    config_path = output_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config", {})
    if int(quant_config.get("bits", quant_config.get("wbits", -1))) != WBITS:
        raise RuntimeError("saved config does not declare fixed int4 weights")
    groupsize = int(quant_config.get("group_size", quant_config.get("groupsize", -1)))
    if groupsize != GROUP_SIZE:
        raise RuntimeError(f"saved group size is {groupsize}, expected {GROUP_SIZE}")

    summary = {
        "model_kind": args.kind,
        "source_model": str(model_path),
        "calibration_file": str(calibration_path),
        "calibration_samples": args.samples,
        "calibration_max_length": max_length,
        "method": "OneCompression GPTQ + QEP",
        "wbits": WBITS,
        "groupsize": GROUP_SIZE,
        "actorder": False,
        "qep": True,
        "qep_exclude_layer_keywords": QEP_EXCLUDE_LAYER_KEYWORDS,
        "dtype": dtype,
        "quantization_seconds": quantization_seconds,
        "save_seconds": save_seconds,
        "quantized_layers": len(quantizer.results),
        "quantized_bit_counts": _bits_summary(config),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
    }
    (output_path / "onecomp_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

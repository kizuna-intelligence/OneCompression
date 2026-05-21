"""Measure FLUX.2-klein full-pipeline VRAM with every component in int4.

Components:
  * transformer  -> GemLite int4 (flux2_klein_int4 runtime, separate repo)
  * text_encoder -> int4 GPTQ Qwen3 (onecomp QuantizedModelLoader)
  * vae          -> fp16 (AutoencoderKLFlux2; conv-based, int4 not worthwhile)

Two regimes are reported:
  * resident : all three components live on the GPU at once (peak = sum).
  * staged   : enable_model_cpu_offload so only the active stage is resident
               (peak ~= the largest single stage). This is the "minimum VRAM".

A real 1024x1024 image is generated so the numbers reflect actual inference,
not just load.

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=/home/yusuke/gitrepos/OneCompression:/home/yusuke/gitrepos/Flux2-klein-int4 \\
    .venv-flux/bin/python example/flux2_int4/measure_pipeline_vram.py \\
        --repo black-forest-labs/FLUX.2-klein-4B \\
        --dit ./flux2_int4_full/model.safetensors \\
        --te  ./flux2_te_int4 \\
        --regime staged

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import torch


def _gb(x: int) -> float:
    return x / (1024 ** 3)


def _resolve_repo(repo: str) -> str:
    import os
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--dit", required=True, help="packed int4 DiT safetensors")
    ap.add_argument("--te", required=True, help="int4 GPTQ Qwen3 dir")
    ap.add_argument("--regime", choices=["resident", "staged", "both"], default="both")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--prompt", default="A cat holding a sign that says hello world")
    ap.add_argument("--out", default="/tmp/flux2_int4_pipeline.png")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.cuda.init()

    from diffusers import Flux2KleinPipeline
    from onecomp.quantized_model_loader import QuantizedModelLoader

    sys.path.insert(0, "/home/yusuke/gitrepos/Flux2-klein-int4")
    from flux2_klein_int4 import load_int4_transformer

    repo = _resolve_repo(args.repo)

    # Run the whole pipeline in fp16: GemLite's int4 GEMM requires fp16 I/O, so
    # keeping every component fp16 avoids fp16/bf16 boundary clashes between the
    # DiT, the (fp16-saved) text encoder, and the VAE.
    print("[load] int4 DiT (GemLite) ...")
    dit = load_int4_transformer(args.dit, device="cuda:0", dtype="float16",
                                backend="gemlite", warmup=False)

    print("[load] int4 Qwen3 text encoder ...")
    te, _tok = QuantizedModelLoader.load_quantized_model(
        args.te, torch_dtype=torch.float16, device_map="cpu",
    )
    # The pipeline uses the encoder purely as a feature extractor (hidden states
    # of layers 9/18/27); the lm_head logits are discarded. Drop it to skip the
    # wasteful vocab projection and avoid an fp16/bf16 dtype clash on its weight.
    te.lm_head = torch.nn.Identity()

    print("[load] assemble pipeline (vae fp16, scheduler, tokenizer) ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        repo, transformer=dit, text_encoder=te, torch_dtype=torch.float16,
    )

    def _generate(tag):
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        img = pipe(
            prompt=args.prompt, height=args.size, width=args.size,
            guidance_scale=1.0, num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        peak = _gb(torch.cuda.max_memory_allocated(dev))
        peak_r = _gb(torch.cuda.max_memory_reserved(dev))
        print(f"  [{tag}] peak_alloc={peak:.2f}GB  peak_res={peak_r:.2f}GB  "
              f"gen={dt:.1f}s")
        img.save(args.out)
        return peak_r

    if args.regime in ("resident", "both"):
        print("[run] resident (all components on GPU) ...")
        pipe.to("cuda:0")
        _generate("resident")
        pipe.to("cpu")
        gc.collect(); torch.cuda.empty_cache()

    if args.regime in ("staged", "both"):
        print("[run] staged (enable_model_cpu_offload) ...")
        pipe.enable_model_cpu_offload(device="cuda:0")
        _generate("staged")

    print(f"saved sample image to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

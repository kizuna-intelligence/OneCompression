"""Fair offload-vs-resident timing + multi-prompt image generation.

Loads the full int4 FLUX.2-klein pipeline once, then:
  1. Benchmarks resident vs staged(cpu-offload) FAIRLY -- one warmup run
     followed by N timed runs in each regime, reporting the mean. This avoids
     the cold-start skew that made the earlier one-shot numbers misleading.
  2. Generates several complex prompts and saves them to --outdir.

Run::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=/home/yusuke/gitrepos/OneCompression:/home/yusuke/gitrepos/Flux2-klein-int4 \\
    .venv-flux/bin/python example/flux2_int4/bench_offload_and_gen.py \\
        --dit ./flux2_int4_full/model.safetensors --te ./flux2_te_int4 \\
        --outdir ./outputs/flux2_int4_samples

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch


def _gb(x: int) -> float:
    return x / (1024 ** 3)


def _resolve_repo(repo: str) -> str:
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


PROMPTS = [
    ("cyberpunk_market",
     "A bustling cyberpunk night market in the rain, neon signs in Japanese and "
     "Chinese reflecting on wet asphalt, a lone figure in a translucent raincoat, "
     "steam rising from food stalls, cinematic, ultra-detailed, volumetric light"),
    ("dragon_library",
     "An ancient dragon curled around a towering spiral library, sunbeams through "
     "stained glass, floating candles, dust motes in the air, oil painting style, "
     "intricate detail, warm golden hour lighting"),
    ("astronaut_jellyfish",
     "An astronaut floating among giant bioluminescent jellyfish in a deep-space "
     "nebula, reflections on the helmet visor, vivid teal and magenta, dreamlike, "
     "photorealistic render"),
    ("steampunk_workshop",
     "A cluttered steampunk inventor's workshop with brass gears, copper pipes, a "
     "half-built clockwork owl on the bench, blueprints pinned to the wall, warm "
     "tungsten light, highly detailed"),
    ("kyoto_autumn",
     "A traditional Kyoto temple courtyard in peak autumn, vivid red and orange "
     "maple leaves, a stone lantern, koi pond with reflections, soft morning mist, "
     "serene, photographic"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--dit", required=True)
    ap.add_argument("--te", required=True)
    ap.add_argument("--outdir", default="./outputs/flux2_int4_samples")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=3, help="timed runs per regime")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.cuda.init()

    from diffusers import Flux2KleinPipeline
    from onecomp.quantized_model_loader import QuantizedModelLoader

    sys.path.insert(0, "/home/yusuke/gitrepos/Flux2-klein-int4")
    from flux2_klein_int4 import load_int4_transformer

    repo = _resolve_repo(args.repo)

    print("[load] int4 DiT (GemLite) ...")
    dit = load_int4_transformer(args.dit, device="cuda:0", dtype="float16",
                                backend="gemlite", warmup=False)
    print("[load] int4 Qwen3 text encoder ...")
    te, _tok = QuantizedModelLoader.load_quantized_model(
        args.te, torch_dtype=torch.float16, device_map="cpu")
    te.lm_head = torch.nn.Identity()
    print("[load] assemble pipeline ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        repo, transformer=dit, text_encoder=te, torch_dtype=torch.float16)

    bench_prompt = "A cat holding a sign that says hello world"

    def _gen(prompt, seed):
        return pipe(
            prompt=prompt, height=args.size, width=args.size,
            guidance_scale=1.0, num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    def _timed(tag, n):
        # one warmup (discarded), then n timed runs
        _gen(bench_prompt, 0)
        torch.cuda.synchronize(dev)
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            _gen(bench_prompt, 100 + i)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t0)
        peak_r = _gb(torch.cuda.max_memory_reserved(dev))
        mean = sum(times) / len(times)
        print(f"  [{tag}] mean={mean:.2f}s  runs={[f'{t:.2f}' for t in times]}  "
              f"peak_res={peak_r:.2f}GB")
        return mean, peak_r

    print("\n=== FAIR TIMING (warmup + %d runs each) ===" % args.runs)
    pipe.to("cuda:0")
    res_t, res_v = _timed("resident", args.runs)
    pipe.to("cpu"); gc.collect(); torch.cuda.empty_cache()

    pipe.enable_model_cpu_offload(device="cuda:0")
    sta_t, sta_v = _timed("staged ", args.runs)

    slow = (sta_t - res_t) / res_t * 100
    print(f"\noffload is {slow:+.1f}% vs resident "
          f"({sta_t:.2f}s vs {res_t:.2f}s); "
          f"VRAM {sta_v:.2f}GB vs {res_v:.2f}GB")

    print("\n=== GENERATING %d COMPLEX IMAGES (staged) ===" % len(PROMPTS))
    for i, (name, prompt) in enumerate(PROMPTS):
        t0 = time.perf_counter()
        img = _gen(prompt, 1000 + i)
        dt = time.perf_counter() - t0
        path = os.path.join(args.outdir, f"{i:02d}_{name}.png")
        img.save(path)
        print(f"  saved {path}  ({dt:.1f}s)")

    print(f"\nall images in: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

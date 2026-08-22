"""Run Cosmos-Transfer2.5 with an OneCompression int4 transformer.

The int4 transformer is loaded with the generic runtime from
``~/gitrepos/onecompression-runtime`` and plugged into diffusers'
``Cosmos2_5_TransferPipeline``.  Use ``--offload`` on a 24GB GPU so the 7B text
encoder, ControlNet, VAE and int4 denoiser are streamed instead of being
resident together.

Run::

    CUDA_VISIBLE_DEVICES=0 .venv-flux/bin/python \\
      example/cosmos_transfer25_int4/generate.py \\
      --repo nvidia/Cosmos-Transfer2.5-2B \\
      --dit /path/to/cosmos_transfer25_qep_int4/model.safetensors \\
      --controlnet-revision diffusers/controlnet/general/edge \\
      --offload --num-frames 17 --num-frames-per-chunk 17

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def _ensure_runtime_on_path() -> None:
    try:
        import onecomp_runtime  # noqa: F401
        return
    except ImportError:
        runtime = Path.home() / "gitrepos" / "onecompression-runtime"
        if runtime.exists():
            sys.path.insert(0, str(runtime))


def load_int4_cosmos_transformer(
    checkpoint_path: str,
    *,
    device: str,
    dtype: str,
    backend: str,
    warmup: bool,
):
    _ensure_runtime_on_path()
    from diffusers import CosmosTransformer3DModel
    from onecomp_runtime.diffusion import load_int4_model

    return load_int4_model(
        checkpoint_path,
        lambda cfg: CosmosTransformer3DModel.from_config(cfg),
        device=device,
        dtype=dtype,
        backend=backend,
        warmup=warmup,
        label="cosmos-transfer2.5-int4",
    )


def _synthetic_controls(num_frames: int, height: int, width: int):
    from PIL import Image, ImageDraw

    frames = []
    for i in range(num_frames):
        img = Image.new("RGB", (width, height), (0, 0, 0))
        d = ImageDraw.Draw(img)
        x0 = int((width - width // 5) * i / max(1, num_frames - 1))
        y0 = height // 3
        d.rectangle(
            [x0, y0, x0 + width // 5, y0 + height // 5],
            outline=(255, 255, 255),
            width=max(2, width // 160),
        )
        d.line([(0, height * 3 // 4), (width, height * 3 // 4)], fill=(180, 180, 180),
               width=max(1, width // 240))
        frames.append(img)
    return frames


def _load_controls(path: str | None, num_frames: int, height: int, width: int):
    if path is None:
        print("[generate] no --controls given; using synthetic edge-like controls", flush=True)
        return _synthetic_controls(num_frames, height, width)

    from diffusers.utils import load_video

    frames = load_video(path)
    return frames[:num_frames]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/Cosmos-Transfer2.5-2B")
    ap.add_argument("--revision", default="diffusers/general")
    ap.add_argument("--controlnet-revision", default="diffusers/controlnet/general/edge")
    ap.add_argument("--dit", default=None,
                    help="packed int4 transformer safetensors produced by quantize_qep.py; "
                         "omit to use the repo's bf16 transformer")
    ap.add_argument("--controls", default=None,
                    help="input/control video path; omitted uses synthetic smoke-test frames")
    ap.add_argument("--prompt", default="A clean robotics lab scene with smooth camera motion.")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--num-frames-per-chunk", type=int, default=17)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--controls-conditioning-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--backend", default="auto", choices=["auto", "gemlite", "fused", "eager"])
    ap.add_argument("--outdir", default="./outputs/cosmos_transfer25_int4")
    ap.add_argument("--offload", action="store_true",
                    help="whole-component CPU offload; recommended for 24GB")
    ap.add_argument("--sequential", action="store_true",
                    help="submodule CPU offload; lowest VRAM, slower")
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    offload = args.offload or args.sequential
    backend = args.backend
    dit_device = "cuda:0"
    if offload:
        dit_device = "cpu"
        if backend in ("auto", "gemlite"):
            backend = "fused"
            print("[generate] offload: using fused backend; GemLite buffers are not CPU-offload friendly",
                  flush=True)

    transformer = None
    if args.dit:
        print(f"[generate] loading int4 transformer on {dit_device} (backend={backend})",
              flush=True)
        transformer = load_int4_cosmos_transformer(
            args.dit,
            device=dit_device,
            dtype=args.dtype,
            backend=backend,
            warmup=args.warmup and not offload,
        )
    else:
        print("[generate] no --dit given; using bf16 transformer from repo", flush=True)

    from diffusers import AutoModel, Cosmos2_5_TransferPipeline
    from diffusers.utils import export_to_video

    print(f"[generate] loading ControlNet revision={args.controlnet_revision}", flush=True)
    controlnet = AutoModel.from_pretrained(
        args.repo,
        revision=args.controlnet_revision,
        torch_dtype=torch_dtype,
    )

    print(f"[generate] assembling pipeline revision={args.revision}", flush=True)
    pipe_kwargs = {"controlnet": controlnet}
    if transformer is not None:
        pipe_kwargs["transformer"] = transformer
    pipe = Cosmos2_5_TransferPipeline.from_pretrained(
        args.repo,
        revision=args.revision,
        torch_dtype=torch_dtype,
        **pipe_kwargs,
    )

    if args.sequential:
        pipe.enable_sequential_cpu_offload(device="cuda:0")
    elif args.offload:
        pipe.enable_model_cpu_offload(device="cuda:0")
    else:
        pipe.to("cuda:0")

    controls = _load_controls(args.controls, args.num_frames, args.height, args.width)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    def _free_cache(pipe, step, timestep, kwargs):
        torch.cuda.empty_cache()
        return kwargs

    t0 = time.perf_counter()
    call_kwargs = {
        "controls": controls,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_frames_per_chunk": args.num_frames_per_chunk,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "controls_conditioning_scale": args.controls_conditioning_scale,
        "generator": generator,
        "callback_on_step_end": _free_cache if torch.cuda.is_available() else None,
    }
    if args.negative_prompt is not None:
        call_kwargs["negative_prompt"] = args.negative_prompt

    out = pipe(
        **call_kwargs,
    ).frames[0]

    out_path = os.path.join(args.outdir, "cosmos_transfer25_int4.mp4")
    export_to_video(out, out_path, fps=16)
    dt = time.perf_counter() - t0
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(
            f"[generate] saved {out_path} ({dt:.1f}s), peak allocated "
            f"{peak_alloc:.2f}GB, reserved {peak_reserved:.2f}GB",
            flush=True,
        )
    else:
        print(f"[generate] saved {out_path} ({dt:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

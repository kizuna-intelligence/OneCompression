"""Run Cosmos-Transfer2.5 with multiple ControlNet modalities and int4 DiT.

Diffusers' stock Cosmos2.5 transfer pipeline exposes one ControlNet at a time,
while NVIDIA's reference inference supports multi-control depth/edge/seg/vis
conditioning.  This runner keeps the diffusers/int4 runtime path, but computes
ControlNet residuals for each requested modality and sums the block residuals
before calling the transformer.

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cosmos.pipeline_cosmos2_5_transfer import (
    DEFAULT_NEGATIVE_PROMPT,
    XLA_AVAILABLE,
    _maybe_pad_or_trim_video,
    retrieve_latents,
)
from diffusers.pipelines.cosmos.pipeline_output import CosmosPipelineOutput
from diffusers.utils import export_to_video, load_video

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


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


def _load_controls(path: str, num_frames: int):
    path_obj = Path(path)
    if path_obj.is_dir():
        image_paths = sorted(
            p
            for p in path_obj.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not image_paths:
            raise ValueError(f"No image controls found in directory: {path}")
        return [PIL.Image.open(p).convert("RGB") for p in image_paths[:num_frames]]
    frames = load_video(path)
    return frames[:num_frames]


def _sum_block_residuals(outputs: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    if not outputs:
        raise ValueError("No ControlNet outputs to sum")
    total = [x for x in outputs[0]]
    for blocks in outputs[1:]:
        if len(blocks) != len(total):
            raise ValueError(f"ControlNet block count mismatch: {len(blocks)} != {len(total)}")
        total = [a + b for a, b in zip(total, blocks)]
    return total


@torch.no_grad()
def run_multicontrol(
    pipe,
    *,
    controlnets: dict[str, torch.nn.Module],
    controls_by_key: dict[str, list[Any]],
    control_weights: dict[str, float],
    prompt: str | list[str] | None = None,
    negative_prompt: str | list[str] = DEFAULT_NEGATIVE_PROMPT,
    height: int = 704,
    width: int | None = None,
    num_frames: int | None = None,
    num_frames_per_chunk: int = 93,
    num_inference_steps: int = 36,
    guidance_scale: float = 3.0,
    num_videos_per_prompt: int = 1,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    output_type: str | None = "pil",
    return_dict: bool = True,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    max_sequence_length: int = 512,
    conditional_frame_timestep: float = 0.1,
    num_ar_conditional_frames: int | None = 1,
    num_ar_latent_conditional_frames: int | None = None,
    manual_controlnet_offload_keys: set[str] | None = None,
):
    if pipe.safety_checker is None:
        raise ValueError("Cosmos safety checker is required by the diffusers pipeline.")

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
    manual_controlnet_offload_keys = manual_controlnet_offload_keys or set()

    first_controls = next(iter(controls_by_key.values()))
    if width is None:
        frame = first_controls[0] if isinstance(first_controls, list) else first_controls
        if isinstance(frame, list):
            frame = frame[0]
        if isinstance(frame, (torch.Tensor, np.ndarray)):
            if frame.ndim == 5:
                frame = frame[0, 0]
            elif frame.ndim == 4:
                frame = frame[0]

        if isinstance(frame, PIL.Image.Image):
            width = int((height + 16) * (frame.width / frame.height))
        else:
            if frame.ndim != 3:
                raise ValueError("`controls` must contain 3D frames in CHW format.")
            width = int((height + 16) * (frame.shape[2] / frame.shape[1]))

    num_frames_per_chunk = pipe.check_inputs(
        prompt,
        height,
        width,
        prompt_embeds,
        callback_on_step_end_tensor_inputs,
        num_ar_conditional_frames,
        num_ar_latent_conditional_frames,
        num_frames_per_chunk,
        num_frames,
        conditional_frame_timestep,
    )

    if num_ar_latent_conditional_frames is not None:
        num_cond_latent_frames = num_ar_latent_conditional_frames
        num_ar_conditional_frames = max(0, (num_cond_latent_frames - 1) * pipe.vae_scale_factor_temporal + 1)
    else:
        num_cond_latent_frames = max(0, (num_ar_conditional_frames - 1) // pipe.vae_scale_factor_temporal + 1)

    pipe._guidance_scale = guidance_scale
    pipe._current_timestep = None
    pipe._interrupt = False

    device = pipe._execution_device

    if pipe.safety_checker is not None:
        pipe.safety_checker.to(device)
        if prompt is not None:
            prompt_list = [prompt] if isinstance(prompt, str) else prompt
            for p in prompt_list:
                if not pipe.safety_checker.check_text_safety(p):
                    raise ValueError(f"Cosmos Guardrail detected unsafe text in the prompt: {p}")

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        device=device,
        max_sequence_length=max_sequence_length,
    )

    vae_dtype = pipe.vae.dtype
    transformer_dtype = pipe.transformer.dtype

    if getattr(pipe.transformer.config, "img_context_dim_in", None):
        img_context = torch.zeros(
            batch_size,
            pipe.transformer.config.img_context_num_tokens,
            pipe.transformer.config.img_context_dim_in,
            device=prompt_embeds.device,
            dtype=transformer_dtype,
        )
        if num_videos_per_prompt > 1:
            img_context = img_context.repeat_interleave(num_videos_per_prompt, dim=0)
        encoder_hidden_states = (prompt_embeds, img_context)
        neg_encoder_hidden_states = (negative_prompt_embeds, img_context)
    else:
        encoder_hidden_states = prompt_embeds
        neg_encoder_hidden_states = negative_prompt_embeds

    control_videos: dict[str, torch.Tensor] = {}
    for key, controls in controls_by_key.items():
        control_video = pipe.video_processor.preprocess_video(controls, height, width)
        if control_video.shape[0] != batch_size:
            if control_video.shape[0] == 1:
                control_video = control_video.repeat(batch_size, 1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Expected controls batch size {batch_size} for {key}, got {control_video.shape[0]}."
                )
        control_videos[key] = control_video

    num_frames_out = min(v.shape[2] for v in control_videos.values())
    if num_frames is not None:
        num_frames_out = min(num_frames_out, num_frames)
    control_videos = {k: _maybe_pad_or_trim_video(v, num_frames_out) for k, v in control_videos.items()}

    chunk_stride = num_frames_per_chunk - num_ar_conditional_frames
    chunk_idxs = [
        (start_idx, min(start_idx + num_frames_per_chunk, num_frames_out))
        for start_idx in range(0, num_frames_out - num_ar_conditional_frames, chunk_stride)
    ]

    video_chunks = []
    latents_mean = pipe.latents_mean.to(dtype=vae_dtype, device=device)
    latents_std = pipe.latents_std.to(dtype=vae_dtype, device=device)

    def decode_latents(latents):
        latents = latents * latents_std + latents_mean
        return pipe.vae.decode(latents.to(dtype=pipe.vae.dtype, device=device), return_dict=False)[0]

    latents_arg = latents
    initial_num_cond_latent_frames = 0
    latent_chunks = []
    total_steps = num_inference_steps * len(chunk_idxs)
    with pipe.progress_bar(total=total_steps) as progress_bar:
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_idxs):
            if chunk_idx == 0:
                prev_output = torch.zeros((batch_size, num_frames_per_chunk, 3, height, width), dtype=vae_dtype)
                prev_output = pipe.video_processor.preprocess_video(prev_output, height, width)
            else:
                prev_output = video_chunks[-1].clone()
                if num_ar_conditional_frames > 0:
                    prev_output[:, :, :num_ar_conditional_frames] = prev_output[:, :, -num_ar_conditional_frames:]
                    prev_output[:, :, num_ar_conditional_frames:] = -1
                else:
                    prev_output.fill_(-1)

            chunk_video = _maybe_pad_or_trim_video(prev_output.to(device=device, dtype=vae_dtype), num_frames_per_chunk)
            latents, cond_latent, cond_mask, cond_indicator = pipe.prepare_latents(
                video=chunk_video,
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=pipe.transformer.config.in_channels - 1,
                height=height,
                width=width,
                num_frames_in=chunk_video.shape[2],
                num_frames_out=num_frames_per_chunk,
                do_classifier_free_guidance=pipe.do_classifier_free_guidance,
                dtype=torch.float32,
                device=device,
                generator=generator,
                num_cond_latent_frames=initial_num_cond_latent_frames if chunk_idx == 0 else num_cond_latent_frames,
                latents=latents_arg,
            )
            cond_mask = cond_mask.to(transformer_dtype)
            cond_timestep = torch.ones_like(cond_indicator) * conditional_frame_timestep
            padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

            controls_latents_by_key: dict[str, torch.Tensor] = {}
            for key, control_video in control_videos.items():
                chunk_control_video = control_video[:, :, start_idx:end_idx, ...].to(
                    device=device, dtype=pipe.vae.dtype
                )
                chunk_control_video = _maybe_pad_or_trim_video(chunk_control_video, num_frames_per_chunk)
                if isinstance(generator, list):
                    controls_latents = [
                        retrieve_latents(pipe.vae.encode(chunk_control_video[i].unsqueeze(0)), generator=generator[i])
                        for i in range(chunk_control_video.shape[0])
                    ]
                else:
                    controls_latents = [
                        retrieve_latents(pipe.vae.encode(vid.unsqueeze(0)), generator=generator)
                        for vid in chunk_control_video
                    ]
                controls_latents = torch.cat(controls_latents, dim=0).to(transformer_dtype)
                controls_latents_by_key[key] = (controls_latents - latents_mean) / latents_std

            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = pipe.scheduler.timesteps
            pipe._num_timesteps = len(timesteps)
            gt_velocity = (latents - cond_latent) * cond_mask

            def compute_control_blocks(hidden_states, timestep, encoder_states):
                block_outputs = []
                for key, controlnet in controlnets.items():
                    if control_weights[key] <= 0:
                        continue
                    if key in manual_controlnet_offload_keys:
                        controlnet.to(device)
                    control_output = controlnet(
                        controls_latents=controls_latents_by_key[key],
                        latents=hidden_states,
                        timestep=timestep,
                        encoder_hidden_states=encoder_states,
                        condition_mask=cond_mask,
                        conditioning_scale=control_weights[key],
                        padding_mask=padding_mask,
                        return_dict=False,
                    )
                    block_outputs.append(control_output[0])
                    if key in manual_controlnet_offload_keys:
                        controlnet.to("cpu")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                return _sum_block_residuals(block_outputs)

            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                pipe._current_timestep = t.cpu().item()
                sigma_t = torch.tensor(pipe.scheduler.sigmas[i].item()).unsqueeze(0).to(
                    device=device, dtype=transformer_dtype
                )
                in_latents = (cond_mask * cond_latent + (1 - cond_mask) * latents).to(transformer_dtype)
                in_timestep = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t

                control_blocks = compute_control_blocks(in_latents, in_timestep, encoder_hidden_states)
                noise_pred = pipe.transformer(
                    hidden_states=in_latents,
                    timestep=in_timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    block_controlnet_hidden_states=control_blocks,
                    condition_mask=cond_mask,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                noise_pred = gt_velocity + noise_pred * (1 - cond_mask)

                if pipe.do_classifier_free_guidance:
                    control_blocks = compute_control_blocks(in_latents, in_timestep, neg_encoder_hidden_states)
                    noise_pred_neg = pipe.transformer(
                        hidden_states=in_latents,
                        timestep=in_timestep,
                        encoder_hidden_states=neg_encoder_hidden_states,
                        block_controlnet_hidden_states=control_blocks,
                        condition_mask=cond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    noise_pred_neg = gt_velocity + noise_pred_neg * (1 - cond_mask)
                    noise_pred = noise_pred + pipe.guidance_scale * (noise_pred - noise_pred_neg)

                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    local_vars = locals()
                    callback_kwargs = {k: local_vars[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == total_steps - 1 or ((i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

            video_chunks.append(decode_latents(latents).detach().cpu())
            latent_chunks.append(latents.detach().cpu())

    pipe._current_timestep = None

    if output_type != "latent":
        video_chunks = [
            chunk[:, :, num_ar_conditional_frames:, ...] if chunk_idx != 0 else chunk
            for chunk_idx, chunk in enumerate(video_chunks)
        ]
        video = torch.cat(video_chunks, dim=2)[:, :, :num_frames_out, ...]

        pipe.safety_checker.to(device)
        video = pipe.video_processor.postprocess_video(video, output_type="np")
        video = (video * 255).astype(np.uint8)
        video_batch = []
        for vid in video:
            checked = pipe.safety_checker.check_video_safety(vid)
            video_batch.append(np.zeros_like(video[0]) if checked is None else checked)
        video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
        video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
        video = pipe.video_processor.postprocess_video(video, output_type=output_type)
    else:
        latent_T = (num_frames_out - 1) // pipe.vae_scale_factor_temporal + 1
        latent_chunks = [
            chunk[:, :, num_cond_latent_frames:, ...] if chunk_idx != 0 else chunk
            for chunk_idx, chunk in enumerate(latent_chunks)
        ]
        video = torch.cat(latent_chunks, dim=2)[:, :, :latent_T, ...]

    pipe.maybe_free_model_hooks()
    if not return_dict:
        return (video,)
    return CosmosPipelineOutput(frames=video)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/Cosmos-Transfer2.5-2B")
    ap.add_argument("--revision", default="diffusers/general")
    ap.add_argument("--dit", default=None)
    ap.add_argument("--depth-controls", default=None)
    ap.add_argument("--seg-controls", default=None)
    ap.add_argument("--edge-controls", default=None)
    ap.add_argument("--vis-controls", default=None)
    ap.add_argument("--depth-weight", type=float, default=1.0)
    ap.add_argument("--seg-weight", type=float, default=1.0)
    ap.add_argument("--edge-weight", type=float, default=0.2)
    ap.add_argument("--vis-weight", type=float, default=0.5)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--num-frames-per-chunk", type=int, default=17)
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--backend", default="fused", choices=["auto", "gemlite", "fused", "eager"])
    ap.add_argument("--outdir", default="./outputs/cosmos_transfer25_multicontrol")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--video-quality", type=float, default=10.0)
    ap.add_argument("--video-bitrate", default=None)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--allow-experimental-multicontrol", action="store_true")
    args = ap.parse_args()

    requested = {
        "depth": (args.depth_controls, args.depth_weight, "diffusers/controlnet/general/depth"),
        "seg": (args.seg_controls, args.seg_weight, "diffusers/controlnet/general/seg"),
        "edge": (args.edge_controls, args.edge_weight, "diffusers/controlnet/general/edge"),
        "vis": (args.vis_controls, args.vis_weight, "diffusers/controlnet/general/blur"),
    }
    requested = {k: v for k, v in requested.items() if v[0] is not None and v[1] > 0}
    if not requested:
        raise ValueError("Provide at least one enabled control input.")
    if len(requested) > 1 and not args.allow_experimental_multicontrol:
        raise ValueError(
            "Diffusers exposes only single-control Cosmos Transfer2.5 ControlNets. "
            "The official multi-control path uses a separate multibranch model, so "
            "summing independent ControlNet residuals produces artifacts. Pass "
            "--allow-experimental-multicontrol only for debugging that approximation."
        )

    os.makedirs(args.outdir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    backend = args.backend
    offload = args.offload or args.sequential
    dit_device = "cpu" if offload else "cuda:0"
    if offload and backend in ("auto", "gemlite"):
        backend = "fused"

    transformer = None
    if args.dit:
        print(f"[multicontrol] loading int4 transformer on {dit_device} (backend={backend})", flush=True)
        transformer = load_int4_cosmos_transformer(
            args.dit,
            device=dit_device,
            dtype=args.dtype,
            backend=backend,
            warmup=args.warmup and not offload,
        )

    from diffusers import AutoModel, Cosmos2_5_TransferPipeline

    controlnets = {}
    controls_by_key = {}
    control_weights = {}
    for key, (path, weight, revision) in requested.items():
        print(f"[multicontrol] loading {key} ControlNet revision={revision} weight={weight}", flush=True)
        controlnets[key] = AutoModel.from_pretrained(args.repo, revision=revision, torch_dtype=torch_dtype)
        controls_by_key[key] = _load_controls(path, args.num_frames)
        control_weights[key] = weight
    weight_sum = sum(control_weights.values())
    if weight_sum > 1.0:
        control_weights = {k: v / weight_sum for k, v in control_weights.items()}
        print(f"[multicontrol] normalized weights={control_weights}", flush=True)

    first_key = next(iter(controlnets))
    pipe_kwargs = {"controlnet": controlnets[first_key]}
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

    def _free_cache(_pipe, _step, _timestep, kwargs):
        torch.cuda.empty_cache()
        return kwargs

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.perf_counter()
    out = run_multicontrol(
        pipe,
        controlnets=controlnets,
        controls_by_key=controls_by_key,
        control_weights=control_weights,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_frames_per_chunk=args.num_frames_per_chunk,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        callback_on_step_end=_free_cache if torch.cuda.is_available() else None,
        manual_controlnet_offload_keys=set(controlnets.keys()) - {first_key} if offload else set(),
    ).frames[0]

    if args.save_frames:
        frames_dir = Path(args.outdir) / "frames_png"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(out):
            frame.save(frames_dir / f"{i:04d}.png")
        print(f"[multicontrol] saved PNG frames to {frames_dir}", flush=True)

    out_path = os.path.join(args.outdir, "cosmos_transfer25_int4_multicontrol.mp4")
    export_to_video(out, out_path, fps=args.fps, quality=args.video_quality, bitrate=args.video_bitrate)
    dt = time.perf_counter() - t0
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(
            f"[multicontrol] saved {out_path} ({dt:.1f}s), peak allocated "
            f"{peak_alloc:.2f}GB, reserved {peak_reserved:.2f}GB",
            flush=True,
        )
    else:
        print(f"[multicontrol] saved {out_path} ({dt:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

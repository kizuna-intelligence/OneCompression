"""RTN 4-bit post-pass for the non-GPTQ Linears (encoders / adaln / predictors).

GPTQ (``quantize_gptq.py``) only touches ``model.blocks``.  Everything else
(text_encoder, speaker_encoder, duration_predictor, cond_module, in/out_proj,
AdaLN) stays fp16 by default.  This pass optionally RTN-packs those "extra"
Linears to 4-bit.

NOTE on v3 (500M): blind RTN 4-bit destroys v3's encoders at every groupsize
tested (CER 96–100%).  For v3 the only viable config is **EXCLUDE_ALL=1**
(GPTQ blocks only, extras fp16, ~561 MB).  See the memory note
"v3 encoders fragile to RTN int4".  This script is kept for v1-style models
whose encoders *are* blind-RTN safe.

Inputs:
  - IN_PATH safetensors (GPTQ pass output, packed in-place)
  - ORIG_PATH original fp32 safetensors (fp16 source for the extras)

Output: overwrites IN_PATH
  - extras' Linear.weight replaced by groupsize-32 u8-nibble packed tensors
  - Linears whose in_features is not divisible by 32 are kept as fp16.

Env flags: EXCLUDE_ADALN (default 1), EXCLUDE_DURATION (default 1),
EXCLUDE_ALL (default 0).

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


IN_PATH = Path(os.environ.get("IN_PATH", "/tmp/v3_int4/model.safetensors"))
ORIG_PATH = Path(os.environ.get(
    "ORIG_PATH",
    "/mnt/hojo/hf_home/hub/models--Aratako--Irodori-TTS-500M-v3/snapshots/"
    "236c1e56591279fc24e3c1bf6609fc06e48dde28/model.safetensors",
))
GROUPSIZE = int(os.environ.get("GROUPSIZE", "32"))

EXCLUDE_ADALN = os.environ.get("EXCLUDE_ADALN", "1") == "1"
EXCLUDE_DURATION = os.environ.get("EXCLUDE_DURATION", "1") == "1"
EXCLUDE_ALL = os.environ.get("EXCLUDE_ALL", "0") == "1"

EXTRA_PREFIXES = (
    "text_encoder.",
    "speaker_encoder.",
    "duration_predictor.",
    "cond_module.",
    "in_proj",
    "out_proj",
)
ADALN_KEYS = ("_adaln.",)


def _is_extra_linear_weight(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    name = key[: -len(".weight")]
    if any(name.startswith(p) or name == p.rstrip(".") for p in EXTRA_PREFIXES):
        return True
    if any(k in name for k in ADALN_KEYS):
        return True
    return False


def rtn_pack_4bit_groupwise(
    w: torch.Tensor, groupsize: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise asymmetric RTN 4-bit pack along the in_features axis.

    w: (out, in) fp32/fp16 Linear weight.
    groupsize: must divide in_features (or pass groupsize=in_features for per-row).
    Returns:
      qweight_u8: (out, ceil(in,2)//2) uint8, low nibble first along in_features
      scales:     (out, num_groups) fp16
      zeros:      (out, num_groups) fp16
    """
    w = w.detach().to(torch.float32)
    out_f, in_f = w.shape
    if in_f % groupsize != 0:
        raise ValueError(f"in_features={in_f} not divisible by groupsize={groupsize}")
    num_groups = in_f // groupsize

    w_g = w.reshape(out_f, num_groups, groupsize)
    w_min = w_g.min(dim=-1, keepdim=True).values
    w_max = w_g.max(dim=-1, keepdim=True).values
    rng = (w_max - w_min).clamp(min=1e-8)
    scale = rng / 15.0
    zero = (-w_min / scale).round().clamp_(0, 15)
    q = (w_g / scale + zero).round().clamp_(0, 15).to(torch.int32)
    q = q.reshape(out_f, in_f)

    in_padded = ((in_f + 1) // 2) * 2
    if in_padded != in_f:
        pad = torch.zeros(out_f, in_padded - in_f, dtype=torch.int32)
        q = torch.cat([q, pad], dim=1)
    low = q[:, 0::2] & 0x0F
    high = (q[:, 1::2] & 0x0F) << 4
    packed = (low | high).to(torch.uint8)

    scale = scale.reshape(out_f, num_groups).to(torch.float16)
    zero = zero.reshape(out_f, num_groups).to(torch.float16)
    return packed, scale, zero


def _load_original_extras_fp16() -> dict[str, torch.Tensor]:
    print(f"loading fp16 extras source: {ORIG_PATH}")
    with safe_open(str(ORIG_PATH), framework="pt", device="cpu") as f:
        out = {}
        for k in f.keys():
            if _is_extra_linear_weight(k):
                t = f.get_tensor(k)
                if t.dim() == 2:
                    out[k] = t.to(torch.float16)
    print(f"  found {len(out)} fp16 extra weights in original")
    return out


def main() -> int:
    orig_extras = _load_original_extras_fp16()

    with safe_open(str(IN_PATH), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    # Drop any stale _extra.* tensors from a previous pass.
    stale = [k for k in tensors if k.startswith("_extra.")]
    for k in stale:
        del tensors[k]
    if stale:
        print(f"removed {len(stale)} stale _extra.* tensors from previous pass")

    quant_layers = json.loads(meta.get("quant_layers_json", "[]"))
    already_quantized = {e["name"] for e in quant_layers}

    extra_entries: list[dict] = []
    kept_fp16: list[str] = []
    n_quantized = 0
    n_skipped_dim = 0
    n_skipped_adaln = 0
    n_skipped_duration = 0
    for src_key, w in orig_extras.items():
        name = src_key[: -len(".weight")]
        if name in already_quantized:
            continue
        out_f, in_f = int(w.shape[0]), int(w.shape[1])

        if EXCLUDE_ALL:
            tensors[src_key] = w
            continue
        if EXCLUDE_ADALN and any(k in name for k in ADALN_KEYS):
            n_skipped_adaln += 1
            tensors[src_key] = w
            continue
        if EXCLUDE_DURATION and name.startswith("duration_predictor."):
            n_skipped_duration += 1
            tensors[src_key] = w
            continue

        # Decide groupsize: prefer GROUPSIZE; if not divisible, keep as fp16.
        if in_f % GROUPSIZE != 0:
            n_skipped_dim += 1
            tensors[src_key] = w  # restore as fp16
            kept_fp16.append(f"{name} (in={in_f})")
            continue

        qweight_u8, scales, zeros = rtn_pack_4bit_groupwise(w, GROUPSIZE)
        tensors[f"_extra.{name}.qweight_u8"] = qweight_u8
        tensors[f"_extra.{name}.scales"] = scales
        tensors[f"_extra.{name}.zeros"] = zeros
        tensors.pop(src_key, None)
        extra_entries.append({
            "name": name,
            "in_features": in_f,
            "out_features": out_f,
            "num_groups": in_f // GROUPSIZE,
        })
        n_quantized += 1

    meta["extra_quant_layers_json"] = json.dumps(extra_entries, ensure_ascii=False)

    print(f"quantized extras: {n_quantized} Linears (groupsize={GROUPSIZE})")
    print(f"kept fp16 — adaln: {n_skipped_adaln}, duration_predictor: "
          f"{n_skipped_duration}, dim-not-divisible: {n_skipped_dim}")
    for n in kept_fp16[:10]:
        print(f"  fp16: {n}")
    if len(kept_fp16) > 10:
        print(f"  ... and {len(kept_fp16) - 10} more")

    save_file(tensors, str(IN_PATH), metadata=meta)
    new_size_mb = IN_PATH.stat().st_size / 1024**2
    print(f"output: {IN_PATH}  size={new_size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

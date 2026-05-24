# OneCompression Inference Runtimes — Inventory & Shared-Library Study

OneCompression *produces* packed-int4 checkpoints. Each target model then needs a
small **inference runtime** that reads the packed `safetensors`, rebuilds the
model with int4 GEMM layers, and runs it. Five such runtimes exist today, built
incrementally. This document inventories them and assesses extracting a single
shared runtime library.

_Last updated: 2026-05-24._

> **Status (2026-05-24): the shared library is built.** The GPTQ-diffusion trio
> plan below has been executed — see [§6 What was actually extracted](#6-what-was-actually-extracted).
> Repo: `onecompression-runtime` · pip: `onecomp-runtime` · import: `onecomp_runtime`.
> FLUX.2-klein-Lite and Irodori-TTS-Lite now consume it; their vendored kernel
> copies are deleted.

## 1. The five runtimes

| Repo | Target | Model class | Quant scope | Packing | Special runtime needs |
|---|---|---|---|---|---|
| `Irodori-TTS-Lite` | Irodori-TTS 500M (v2/v3) | `TextToLatentRFDiT` (DiT) | DiT blocks + AdaLN + encoders + text-embedding + DACVAE convs | GPTQ (blocks) + RTN uint8 (extras/emb/conv) | monkey-patch host runtime; duration-predictor graft; int4 Conv1d/ConvTranspose1d |
| `Flux2-klein-Lite` | FLUX.2-klein 4B | `Flux2Transformer2DModel` | all transformer Linears | GPTQ | fp16 for GemLite path; diffusers offload |
| `LTX-2.3-int4` | LTX-2.3 22B audio-video | `ltx_core` LTXModel (dual-stream) | transformer_blocks Linears | GPTQ | Gemma-3 TE compat shim; two-stage pipeline injection |
| `FireRed-Image-Edit-int4` | FireRed / Qwen-Image 20B edit | `QwenImageTransformer2DModel` | transformer_blocks Linears (modulation kept fp16) | GPTQ | QwenEmbedRope meta-buffer recompute; bf16-not-fp16 |
| `Command-A-Plus-Lite` | Command-A-Plus 200B+ MoE | `Cohere2MoeForCausalLM` | routed experts int2 + attn/shared/embed int4 | RTN codes (`pack_codes`) | **MoE expert CPU→GPU streaming/offload**; per-layer shard checkpoint + `meta.json` |

## 2. How much code is literally shared

The int4 leaf machinery is **copy-pasted**, not abstracted. Verified by md5:

| File | Flux2 | LTX-2.3 | FireRed | Verdict |
|---|---|---|---|---|
| `fused_int4_linear.py` | `7a01ac27…` | `7a01ac27…` | `7a01ac27…` | **byte-identical** (591 lines each) |
| `quant_utils.py` | `5a195d49…` | `5a195d49…` | `5a195d49…` | **byte-identical** |
| `gemlite_int4_linear.py` | `fc52ff4e…` | `cdd4e8a9…` | `fc52ff4e…` | identical bar a copyright line (LTX) |

`Irodori-TTS-Lite/fused_int4_linear.py` is a near-fork (596 lines, md5 `3453a55e…`) —
same kernel, minor drift. `Command-A-Plus-Lite` does **not** share these files; it
has its own `gemlite_linear.py` + `PackedLinear` because its packing format
(`codes`/`scale`/`zero` via `pack_codes`) differs from the GPTQ pack the others use.

So three of the five runtimes are effectively the *same* runtime wearing different
model imports, and a fourth (Irodori) is a light fork.

## 3. Anatomy: generic core vs. model glue

Every runtime decomposes into the same layers. Only the bottom row is genuinely
per-model.

```
┌─────────────────────────────────────────────────────────────┐
│ GENERIC (identical today / trivially unifiable)              │
│  • FusedInt4Linear        Triton dequant+GEMM, gs=32, gptq    │
│  • GemLiteInt4Linear      GemLite kernel wrapper (fp16 I/O)   │
│  • dequant_gptq_to_fp / unpack_int_weights / unpack_zeros     │
│  • _can_use_fused()       wbits==4, gs==32, !actorder, align  │
│  • _resolve_backend()     auto→gemlite→fused→eager            │
│  • safetensors metadata read (config_json / quant_layers_json │
│    / checkpoint_format gptq|gptq_v2)                          │
│  • meta-device build → swap quant Linears → materialize rest  │
│  • warmup (JIT M-buckets)                                     │
├─────────────────────────────────────────────────────────────┤
│ SEMI-GENERIC (Irodori extensions, reusable as opt-in)        │
│  • PackedRTNLinear        uint8-nibble RTN, per-fwd dequant   │
│  • PackedEmbedding        int4 embedding table, row-gather    │
│  • PackedInt4Conv1d / ConvTranspose1d (NormConv semantics)    │
├─────────────────────────────────────────────────────────────┤
│ MODEL GLUE (must stay per-repo)                              │
│  • which model class + how to from_config on meta            │
│  • post-load device fixups (FireRed rope; cmda rotary)       │
│  • encoder/pipeline wiring (LTX Gemma shim + stage inject;    │
│    diffusers offload; Irodori monkey-patch; duration graft)   │
│  • MoE expert offload/streaming (cmda only)                   │
│  • dtype rule (bf16 for Qwen/LTX, fp16 for GemLite paths)     │
└─────────────────────────────────────────────────────────────┘
```

## 4. Feasibility — yes, with one clean seam

A shared library is well justified: three runtimes are byte-identical copies and
maintenance currently means editing the same kernel in four places (e.g. the
sway-sampling and bf16 fixes never propagate automatically).

**Proposed package: `onecomp-runtime`** (publishable, or vendored like today)

```
onecomp_runtime/
  layers/
    fused_int4_linear.py      # the 591-line kernel, single source of truth
    gemlite_int4_linear.py
    packed_linear.py          # PackedRTNLinear, PackedEmbedding (from Irodori)
    packed_conv.py            # int4 Conv1d/ConvTranspose1d (from Irodori)
  quant_utils.py              # dequant_gptq_to_fp + unpack helpers
  backend.py                  # _resolve_backend, _can_use_fused, build_{fused,gemlite,eager}
  loader.py                   # load_int4_model(builder, ckpt, device, dtype, backend, post_load=None)
```

The one seam that makes it model-agnostic is a **builder + post-load hook**:

```python
def load_int4_model(
    checkpoint_path,
    build_meta_model,        # () -> nn.Module on meta, from config_json
    *, device, dtype, backend="auto",
    post_load=None,          # (model) -> None : rope/buffer fixups
    warmup=False,
): ...
```

Each repo shrinks to a thin adapter:

```python
# firered_image_edit_int4/__init__.py
def load_int4_transformer(path, **kw):
    return load_int4_model(
        path,
        build_meta_model=lambda cfg: QwenImageTransformer2DModel.from_config(cfg),
        post_load=_recompute_qwen_rope,   # the only FireRed-specific bit
        **kw,
    )
```

Coverage:
- **Flux2, LTX, FireRed** → collapse to ~20-line adapters over the shared loader.
  This is the high-value, low-risk win (they're already identical).
- **Irodori** → adapter keeps its `patch()`/monkey-patch + duration graft, but
  imports layers/utils from the library instead of its fork; `packed_linear` and
  `packed_conv` move *into* the library as the opt-in RTN tier.
- **Command-A-Plus** → shares `layers/` and `quant_utils` conceptually but its
  RTN `codes` packing + MoE `OffloadExperts` streaming are a different checkpoint
  contract. Treat as a **separate concern**: it can consume the same leaf layers,
  but not the GPTQ `load_int4_model` path. Don't force-fit it.

## 5. Recommendation

1. **Do it for the GPTQ-diffusion trio first** (Flux2 / LTX / FireRed). They are
   byte-identical; extraction is mechanical and immediately stops the 3–4×
   copy-paste maintenance. Lowest risk, highest payoff.
2. **Fold Irodori in next** as the RTN/Conv/Embedding superset tier, behind opt-in
   flags it already has (`pack_rtn_extras`, `codec_int4`).
3. **Leave Command-A-Plus on its own packing path**; optionally let it import the
   shared leaf layers, but its MoE-offload runtime is a distinct contract.
4. Ship as a small installable (`pip install onecomp-runtime`) or keep vendoring
   but from **one source file** + a sync check, so fixes propagate.

### Risks / notes
- GemLite needs fp16 I/O; the shared loader must keep the per-model dtype rule
  (bf16 for Qwen-Image/LTX to avoid NaN; fp16 only on the GemLite path).
- `checkpoint_format` (`gptq` v1 −1-offset vs `gptq_v2`) must stay a per-checkpoint
  metadata read, not a global default.
- Kernel changes (e.g. K-padding, warmup buckets) become global — a shared test
  matrix across at least one DiT + one LLM shape is needed before each release.

## 6. What was actually extracted

Built 2026-05-24 at `/home/yusuke/gitrepos/onecompression-runtime`.

**Naming** (distribution and import deliberately split, like `scikit-learn`/`sklearn`):

| repo dir | pip distribution | import name |
|---|---|---|
| `onecompression-runtime` | `onecomp-runtime` | `onecomp_runtime` |

**Reconciliation findings (verified by `diff`, correcting the md5-only guess in §2):**
- `fused_int4_linear.py` — Flux2 and Irodori are **identical except the docstring**
  (the K_LOGICAL padding, `_apply` dtype-safety override and M-bucket config cache
  are present in *both*; the 596-vs-591 line gap is a 5-line vendoring note). One
  canonical copy serves every runtime unchanged.
- `quant_utils.py` — Irodori's is a strict **superset** (adds
  `dequant_extra_u8_to_weight`, the RTN uint8-nibble inverse). Took Irodori's as
  canonical.
- `gemlite_int4_linear.py` — Flux2's (Irodori had none).
- `packed_linear.py` / `packed_conv.py` — Irodori-only RTN tier, moved in verbatim.

**Layout shipped:**
```
onecomp_runtime/
  layers/{fused_int4_linear,gemlite_int4_linear,packed_linear,packed_conv}.py
  quant_utils.py
  backend.py      # resolve_backend / can_use_fused / build_{gemlite,fused,eager} / build_quant_layer
  diffusion.py    # load_int4_model(checkpoint, build_meta_model, *, post_load=..., ...)
```

The generic `load_int4_model` is the parameterised form of the old Flux2
`load_int4_transformer`: the per-model seam is `build_meta_model(cfg) -> nn.Module`
(constructed under a `meta` device) plus an optional `post_load(model)` hook.

**Consumers converted (vendored copies deleted, imports redirected):**
- `Flux2-klein-Lite` — `loader.py` is now a ~15-line adapter over `load_int4_model`;
  `__init__.py` re-exports the kernels from `onecomp_runtime.layers`. Verified:
  `flux2_klein_lite.FusedInt4Linear` resolves to the shared module.
- `Irodori-TTS-Lite` — kept its `checkpoint_loader.py` (monkey-patch, duration
  graft, codec int4, RTN extras) but every `from .fused_int4_linear` /
  `.quant_utils` / `.packed_linear` / `.packed_conv` import now points at
  `onecomp_runtime`. Verified: public API imports clean, kernels resolve to shared.

Both `pyproject.toml` files now list `onecomp-runtime` as a dependency (gemlite via
`onecomp-runtime[gemlite]`) instead of vendoring torch/triton/safetensors directly.

**Consuming a packed checkpoint (reference):** `example/onecomp_runtime_inference.py`
shows the standalone consumer path — `pip install onecomp-runtime`, then
`onecomp_runtime.diffusion.load_int4_model(ckpt, build_meta_model, post_load=...)` —
without needing the full OneCompression install. The runtime is published at
`github.com/kizuna-intelligence/onecompression-runtime` (public, branch `main`);
consumers pin it via `onecomp-runtime @ git+https://…@main`.

**Still on their own path (unchanged):** LTX-2.3 and FireRed/Qwen-Image are
byte-identical to Flux2 and can be converted with the same ~15-line adapter when
touched next; Command-A-Plus keeps its RTN-`codes` + MoE-offload contract (an
`onecomp_runtime/llm/` loader is the future second contract, not yet built).

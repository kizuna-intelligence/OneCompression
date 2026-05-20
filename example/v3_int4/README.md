# Irodori-TTS-500M-v3 → INT4 (OneCompression)

End-to-end pipeline to quantize the rectified-flow DiT of
[`Aratako/Irodori-TTS-500M-v3`](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
to 4-bit with OneCompression GPTQ, and evaluate the result.

## TL;DR config that works on v3

The DiT blocks survive low-bit quantization, and so do the **encoder attention
Linears** and the **text embedding table** — *provided* the right method is
used per component:

| Component | Method | Why |
|---|---|---|
| `model.blocks` (DiT) | GPTQ, real calibration | random calib → CER ~33%; real → 0% |
| encoder attention Linears | GPTQ, real Hessian | blind RTN collapses encoders; GPTQ is fine |
| `text_encoder.text_embedding` | blind RTN group-wise | lookup table, not a matmul — per-row min/max is accurate |
| encoder MLPs (`1996`-dim), AdaLN, duration_predictor, cond_module | **stay fp16** | `1996` breaks AutoGPTQ packing; RTN destroys encoders |

The single most important lever is **calibration data**: with random Gaussian
calibration the GPTQ Hessian estimate is wrong and CER lands ~33%. Capturing
*real* activations from genuine syntheses and feeding them back fixes this — that
is what `capture_calibration.py` + `--calib` are for. (`--actorder` is NOT used:
it emits `.perm` tensors the Irodori-TTS-Lite loader rejects, and real
calibration — not actorder — is the dominant quality lever.)

### Size progression (groupsize=32, all CER 0.00%, ref = `mera3.wav`)

| Stage | Script | Size | OpenVoice-v2 sim |
|---|---|---|---|
| DiT blocks only | `quantize_gptq.py --calib` | 561 MB | 0.8718 |
| + encoder attention | `gptq_extra_pass.py` | 511 MB | 0.8743 |
| + text embedding | `quantize_embedding.py` | 444 MB | 0.8668 (held-out 0.91) |

FP32 baseline ≈ sim 0.886 / CER 8.3%.

## Steps

All steps need a v3-capable `irodori_tts` venv (e.g. `Irodori-streaming/.venv`)
and, on this host, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`.

### 1. Capture real calibration activations

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python example/v3_int4/capture_calibration.py \
    /path/to/v3/model.safetensors /home/yusuke/.claude/mera3.wav ./v3_calib.pt
```

Hooks `forward_with_encoded_conditions` and records the exact
`x_t / t / text_state / speaker_state` tensors at every RF step over a spread of
Japanese utterances.

### 2. GPTQ-quantize the DiT blocks (with real calibration)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python example/v3_int4/quantize_gptq.py \
    /path/to/v3/model.safetensors ./v3_int4 \
    --calib ./v3_calib.pt --groupsize 32 --actorder
```

Uses `onecomp` `Runner(qep=True)` + `GPTQ` + `DiTAdapter`. `--calib` injects the
captured activations (via `DiTAdapter(calibration_inputs_path=...)`); without it
the adapter falls back to random Gaussian calibration. `--mse` enables MSE
grid-search for scale/zero; `--asym` switches to asymmetric quantization.

### 3. GPTQ-quantize the encoder attention Linears (real Hessian)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python example/v3_int4/gptq_extra_pass.py \
    /path/to/v3/model.safetensors ./v3_int4/model.safetensors \
    /home/yusuke/.claude/mera3.wav --groupsize 32
```

Hooks each `text_encoder.*` / `speaker_encoder.*` `nn.Linear`, accumulates the
GPTQ Hessian `H = (2/N)·Σ xᵀx` over real syntheses, runs `run_gptq`, packs in
the AutoGPTQ format, and merges into the packed checkpoint's `quant_layers_json`.
Linears with `in_features % groupsize != 0` or `out_features % 8 != 0` (the
`1996`-dim MLPs) are left fp16. `--actorder` is intentionally unsupported.

### 4. RTN-quantize the text embedding table

```bash
python example/v3_int4/quantize_embedding.py \
    ./v3_int4/model.safetensors --groupsize 32
```

Group-wise RTN 4-bit packs `text_encoder.text_embedding` (~97 MB fp16 → ~30 MB)
into `_embed.*` tensors + `embed_quant_layers_json`. The Lite runtime's
`PackedEmbedding` gathers+dequants only the input-id rows on forward, so the full
table is never materialised in VRAM. Blind RTN is safe here (it's a lookup, not
a matmul).

### (legacy) RTN post-pass for the other extras

```bash
EXCLUDE_ALL=1 IN_PATH=./v3_int4/model.safetensors \
ORIG_PATH=/path/to/v3/model.safetensors \
python example/v3_int4/rtn_extra_pass.py
```

For **v3 keep `EXCLUDE_ALL=1`** (extras stay fp16). The script is retained for
v1-style models whose encoders tolerate blind RTN.

### 5. Generate + evaluate

```bash
REF_WAV=/home/yusuke/.claude/mera3.wav MODE=both \
python example/v3_int4/generate_compare.py ./v3_int4/model.safetensors ./out_v3

python example/v3_int4/eval_metrics.py \
    /home/yusuke/.claude/mera3.wav ./out_v3_packed.wav \
    "こんにちは、メラだよ。テスト中なの。今日もいい天気だね。"
```

`eval_metrics.py` reports OpenVoice-v2 speaker cosine similarity (higher better)
and faster-whisper large-v3 CER (lower better). FP32 baseline ≈ sim 0.886 / CER 8.3%.

# Gemma 4 E2B all-weight fixed-int4 checkpoint

This example extracts the language decoder from `google/gemma-4-E2B-it`, then
quantizes every eligible decoder `Linear` with uniform GPTQ 4-bit, group size
128, and architecture-aware QEP. Image and audio towers are deliberately not
included because the consumer is an on-device Japanese voice-chat runtime.

Gemma 4 also has a 262k-row token embedding and a much larger per-layer input
embedding. Together they occupy 5.125 GiB in bf16, so the Linear-only output is
an intermediate checkpoint, not an all-weight 4-bit deliverable.
`pack_embeddings.py` applies group-wise RTN 4-bit and writes both embeddings in
the same K-major GPTQ layout used by the Linear runtime. The token embedding and
tied vocabulary projection then share one packed allocation.

The default calibration uses 512 sequences of 512 tokens rendered from
`llm-jp/databricks-dolly-15k-ja` with Gemma's own chat template. The shorter
sequence length matches the target short-dialogue workload; the sample count is
kept at OneCompression's production default. Change these values only after an
A/B quality measurement.

The preparation pass scans 8,192 source rows. In the measured Japanese Dolly
corpus only 87 of the first 1,024 rows survived the strict 512-token
`drop_rand` rule, so a 1,024-row preparation pool cannot supply 512 calibration
samples without padding or document concatenation.

```bash
python -m example.gemma4_e2b_int4.quantize \
  --output /path/to/gemma-4-E2B-it-gptq4-qep-linear-stage

python -m example.gemma4_e2b_int4.pack_embeddings \
  /path/to/gemma-4-E2B-it-gptq4-qep-linear-stage \
  /path/to/gemma-4-E2B-it-gptq4-qep-all-int4 \
  --groupsize 128

python /path/to/onecompression-runtime/example/gemma4/generate.py \
  --ckpt /path/to/gemma-4-E2B-it-gptq4-qep-all-int4 \
  --backend fused

python -m example.gemma4_e2b_int4.export_mlx \
  /path/to/gemma-4-E2B-it-gptq4-qep-all-int4 \
  /path/to/gemma-4-E2B-it-gptq4-qep-all-int4-mlx
```

`export_mlx.py` does not quantize again. It preserves each packed 4-bit code,
transposes the K-major GPTQ words into MLX's row-major packed layout, and
converts the GPTQ-v1 zero-point offset into MLX affine biases. The resulting
directory can be loaded locally by `mlx-swift-lm`; both token embeddings remain
quantized, and the tied output projection reuses the token embedding. Packed
embedding tensors larger than 512 MiB are split on quantization-group-aligned
feature boundaries. This keeps every Metal buffer below the 1 GiB limit on
older iPhones without reconstructing or requantizing the weights; consumers
reassemble the logical embedding by concatenating the shard outputs.

Use `--dry-run` to print the complete run configuration without downloading or
loading model weights. `verify.py` is useful for checking the Linear-only stage
with OneCompression's reference loader. The all-weight checkpoint must be run
through `onecompression-runtime`, which keeps weights packed during inference;
the reference `GPTQLinear` fallback unpacks a full bf16 weight on every call.

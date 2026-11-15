# Qwen3.8-27B full VLM int4

This recipe preserves the existing QEP/AutoBit text checkpoint and quantizes
only the missing `model.visual` Linear weights with OneCompression RTN at
4-bit.  It downloads only upstream weight shard 1 because all 333 visual
tensors are contained in that shard.

```bash
python example/qwen38_vlm_int4/quantize_vision_and_merge.py \
  --cache-dir /mnt/hojo/hf_home \
  --output /mnt/hojo/models/Qwen3.8-27B-vlm-int4-qep
```

The output is a full `Qwen3_5ForConditionalGeneration` checkpoint with a
two-shard index.  `quantization_summary.json` records each vision layer's group
size and reconstruction error.  Widths divisible by 128 use group size 128;
the 4304-wide MLP projections use the largest exact power-of-two divisor (16),
so no large vision Linear silently remains in bf16.

# Wan2.1 VACE 14B int4

This recipe targets `Wan-AI/Wan2.1-VACE-14B-diffusers`.

Fujitsu/OneCompression features used:

- QEP on the main 40 Wan blocks.
- GPTQ 4-bit, group size 32, symmetric quantization for fused runtime kernels.
- Packed RTN int4 for the eight VACE hint blocks during save, avoiding a bf16
  VACE tail in the checkpoint.
- Layer-wise/block-wise streaming: the bf16 transformer stays on CPU and one
  block is moved to the QEP device at a time.
- Packed `safetensors` metadata consumed directly by `onecomp-runtime`.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-flux/bin/python \
  example/wan_vace14b_int4/quantize_qep.py \
  Wan-AI/Wan2.1-VACE-14B-diffusers ./wan_vace14b_qep_int4 \
  --groupsize 32 --num-samples 4
```

For lower host/GPU memory, keep `--num-samples` small first and prefer real
captured calibration inputs once available.

## Runtime and conditioning notes

The matching runtime adapter lives in:

```text
/home/yusuke/gitrepos/onecompression-runtime/example/wan_vace/
```

The tested robot-hand runs include both a single neutral depth+edge conditioning
video and an experimental shared-weight multi-control path that runs the same
VACE branch separately for depth, edge, and segmentation. See the runtime example
README for exact commands, output paths, visual inspection commands, and notes on
when to use pre-composed controls versus independent context-branch passes.

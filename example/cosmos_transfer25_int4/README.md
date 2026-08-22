# Cosmos Transfer2.5 int4

Quantize the gated `nvidia/Cosmos-Transfer2.5-2B` base transformer with
OneCompression QEP + GPTQ and run it through the generic int4 runtime in
`~/gitrepos/onecompression-runtime`.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-flux/bin/python \
  example/cosmos_transfer25_int4/quantize_qep.py \
  nvidia/Cosmos-Transfer2.5-2B ./cosmos_transfer25_qep_int4 \
  --groupsize 32 --num-samples 8
```

For a 24GB card, run inference with component offload:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-flux/bin/python \
  example/cosmos_transfer25_int4/generate.py \
  --dit ./cosmos_transfer25_qep_int4/model.safetensors \
  --offload --num-frames 17 --num-frames-per-chunk 17 --steps 8
```

The HF repository is gated. Accept the NVIDIA license on Hugging Face and log in
before downloading weights.

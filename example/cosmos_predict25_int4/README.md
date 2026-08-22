# Cosmos Predict2.5 Int4 Quantization

This produces a packed OneCompression int4 checkpoint for the official
Cosmos-Predict2.5 2B DiT (`model.net`) in the local `cosmos-transfer2.5`
checkout.

The script uses `$COSMOS_REPO` when set and otherwise defaults to
`~/gitrepos/cosmos-transfer2.5`.

```bash
source /home/yusuke/tools/bin/activate
cd /home/yusuke/gitrepos/OneCompression
source .venv-flux/bin/activate

CUDA_VISIBLE_DEVICES=0 \
HF_TOKEN="$HF_TOKEN_JIE" \
COSMOS_TEXT_ENCODER_DEVICE=cpu \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python example/cosmos_predict25_int4/quantize_qep.py \
  d20b7120-df3e-4911-919d-db6e08bad31c \
  /media/yusuke/Zundamon/model_work/cosmos_predict25_qep_int4_noadaln \
  --num-samples 2 \
  --latent-frames 5 \
  --latent-height 16 \
  --latent-width 16 \
  --exclude-layer-keyword adaln_modulation
```

The output checkpoint is:

```text
/media/yusuke/Zundamon/model_work/cosmos_predict25_qep_int4_noadaln/model.safetensors
```

Load it into an official inference pipeline with
`onecomp_runtime.models.cosmos_predict25.load_int4_predict_into_net`.

Validated smoke output on this machine:

```text
/media/yusuke/Zundamon/model_work/cosmos_predict25_qep_int4_noadaln_smoke/model.safetensors
```

It contains 280 packed int4 Linears and is 1.6GB. AdaLN modulation is kept in
bf16; the full-AdaLN-int4 smoke checkpoint ran but produced noise.

SO101 dataset preparation and joint LoRA training are intentionally kept in
`~/gitrepos/ad-data-pipeline`. See
`docs/cosmos_predict25_so101_int4_training.md` in that repository.

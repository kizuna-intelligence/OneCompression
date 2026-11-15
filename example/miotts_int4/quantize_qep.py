"""Quantize MioTTS-1.7B to int4 with GPTQ + Fujitsu QEP, save vLLM-ready.

MioTTS-1.7B is a stock ``Qwen3ForCausalLM`` (model_type=qwen3) that emits
MioCodec ``<|s_N|>`` tokens. Because it's a standard Qwen3 arch, the saved GPTQ
checkpoint loads in vLLM natively (gptq_marlin) -- no out-of-tree registration
(unlike the hybrid Qwen3.6-27B). QEP is applied during quantization to minimise
the int4 error; the on-disk format stays standard GPTQ so vLLM reads it as-is.

Config rationale:
  - wbits=4, groupsize=128  : the vLLM-friendly int4 sweet spot.
  - sym=False               : asymmetric -> lower error, still gptq_marlin OK.
  - actorder=False          : desc_act off for max loader compatibility.
  - qep=True                : Fujitsu QEP error-propagation, the whole point.
  - calibration = TTS-real  : text->codec sequences (build_calibration.py), so
                              the codec-token activation region is calibrated.
  - lm_head / embed         : left in bf16 (standard GPTQ; vLLM expects this).

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/media/yusuke/Curry/hf_home_miotts \
    .venv-flux/bin/python example/miotts_int4/quantize_qep.py \
        --calib /media/yusuke/Curry/miotts_work/calib_miotts.jsonl \
        --out /media/yusuke/Curry/miotts_work/MioTTS-1.7B-int4-qep \
        --samples 118
"""
from __future__ import annotations

import argparse

from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner, setup_logger

MODEL_ID = "Aratako/MioTTS-1.7B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="calibration .jsonl path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--strategy", default="concat_chunk",
                    help="concat_chunk packs all sequences and slices to "
                         "max_length (full token utilisation); drop_* would "
                         "discard our short (<max_length) TTS sequences")
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--sym", action="store_true", default=True,
                    help="symmetric quant. REQUIRED for vLLM GPTQ-Marlin "
                         "(TYPE_MAP only has (4,True)/(8,True)); sym=False "
                         "falls back to the buggy gptq_gemm kernel and emits "
                         "garbage tokens")
    ap.add_argument("--asym", dest="sym", action="store_false")
    ap.add_argument("--no-qep", action="store_true")
    args = ap.parse_args()

    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0",
                               dtype="bfloat16")
    quantizer = GPTQ(wbits=args.wbits, groupsize=args.groupsize,
                     sym=args.sym, actorder=False)
    calib = CalibrationConfig(
        calibration_dataset=args.calib,
        text_key="text",
        max_length=args.max_length,
        num_calibration_samples=args.samples,
        strategy=args.strategy,
    )
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calib,
        qep=not args.no_qep,
    )
    runner.run()
    runner.save_quantized_model(args.out)
    print(f"\n[miotts-int4] saved vLLM-ready int4 (qep={not args.no_qep}) "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()

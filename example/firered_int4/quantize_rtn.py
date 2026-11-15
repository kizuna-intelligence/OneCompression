"""Calibration-free RTN 4-bit quantization of the FireRed-Image-Edit DiT.

FireRed-Image-Edit-1.0 is a ``QwenImageEditPlusPipeline`` whose backbone is the
20B ``QwenImageTransformer2DModel`` — a dual-stream MMDiT (60 blocks).  As with
LTX-2.3, the dual-stream blocks do not fit OneCompression's single-stream block
catcher, and the bf16 transformer (~41GB) does not fit a 24GB GPU for a full
calibration forward.  RTN sidesteps both: ``Runner.quantize_without_calibration``
iterates the target ``nn.Linear`` modules and quantizes their weights directly
from weight statistics, with no forward at all, so the model stays on CPU.

Scope: only ``transformer_blocks`` Linears (attention q/k/v/o + the joint-stream
add_*_proj + the img/txt MLPs), minus the AdaLN modulation projections
(``img_mod`` / ``txt_mod``) excluded by
``QwenImageDiTAdapter.default_exclude_layer_keywords()``.

Run inside ``.venv-flux``::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv-flux/bin/python example/firered_int4/quantize_rtn.py \\
        /media/yusuke/Curry/firered_work/bf16 \\
        /media/yusuke/Curry/firered_work/out \\
        --groupsize 32 --mse

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

from onecomp import ModelConfig, Runner, setup_logger
from onecomp.adapters import QwenImageDiTAdapter
from onecomp.quantizer.rtn import RTN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path/HF id with a transformer/ subfolder")
    ap.add_argument("save_dir")
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--sym", action="store_true",
                    help="symmetric quantization (default asymmetric)")
    ap.add_argument("--mse", action="store_true",
                    help="MSE grid search for clipping (slower, better)")
    ap.add_argument("--quant-modulation", action="store_true",
                    help="also quantize the AdaLN img_mod/txt_mod projections. "
                         "For Qwen-Image these dominate the weight budget "
                         "(~13.6GB fp16 of the 20B), so quantizing them is "
                         "required to get a low-VRAM (~11GB resident) DiT.")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    setup_logger()

    # RTN runs no forward; keep the 20B on CPU and quantize per-layer there.
    adapter = QwenImageDiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device="cpu",
        subfolder=args.subfolder,
    )
    model_config = ModelConfig(adapter=adapter)

    exclude = (
        [] if args.quant_modulation
        else QwenImageDiTAdapter.default_exclude_layer_keywords()
    )
    rtn = RTN(
        wbits=4,
        groupsize=args.groupsize,
        sym=args.sym,
        mse=args.mse,
        include_layer_keywords=["transformer_blocks"],
        exclude_layer_keywords=exclude,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=rtn,
        qep=False,
    )
    runner.run()

    # ``save_quantized_model`` loads a fresh bf16 model (~41GB) to repack, so we
    # must shrink everything else first to keep the host under earlyoom's limit
    # (it SIGTERMs at 3% free RAM).  Two pins to drop before the reload:
    #   1) ``module_to_name`` keys are the *original* bf16 Linear modules, which
    #      keep the entire pre-quant model (~41GB) alive.
    #   2) each result's ``dequantized_weight`` is a full bf16 weight copy
    #      (~41GB total) that the packed save never reads.
    # Only the (now uint8) ``quantized_weight`` + fp16 scale/zero are needed for
    # packing, so this leaves ~20GB of results + the 41GB reload at peak.
    import gc

    runner.quantizer.module_to_name = {}
    for _result in runner.quantizer.results.values():
        if getattr(_result, "dequantized_weight", None) is not None:
            _result.dequantized_weight = None
    gc.collect()

    adapter.save_quantized_model(runner, args.save_dir)
    print(f"saved RTN-int4 FireRed/Qwen-Image DiT to {args.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

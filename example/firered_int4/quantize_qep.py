"""QEP (Fujitsu) int4 quantisation of the FireRed / Qwen-Image dual-stream DiT.

FireRed-Image-Edit is a 20B **dual-stream** MMDiT: each ``transformer_blocks``
block evolves *two* residual streams (image ``hidden_states`` + text
``encoder_hidden_states``).  OneCompression's block-propagation loop carries a
single tensor between blocks, so :class:`QwenImageDiTAdapter` supplies the
opt-in stream-packing hooks (``wrap_block`` / ``pack_catcher_input``) that
concatenate the two streams along the sequence axis.  This lets **QEP**
(Quantization Error Propagation, Fujitsu's proprietary block-error correction)
drive a real forward over the dual-stream blocks — something plain RTN cannot
exploit.

QEP keeps the model on CPU and streams one block at a time to the GPU
(``block.to(device)`` / ``block.cpu()``), so the ~41GB bf16 transformer
quantises on a single 24GB GPU without the whole model ever being resident.

Run (inside .venv-flux)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    .venv-flux/bin/python example/firered_int4/quantize_qep.py \\
        /media/yusuke/Curry/firered_work/bf16 /home/yusuke/firered_qep \\
        --groupsize 32 --num-samples 16

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import sys

from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.adapters import QwenImageDiTAdapter
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path/HF id with a transformer/ subfolder")
    ap.add_argument("save_dir")
    ap.add_argument("--groupsize", type=int, default=32)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--image-grid", type=int, default=32)
    ap.add_argument("--text-seq-len", type=int, default=128)
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--calib-path", default=None,
                    help="captured real calibration .pt (see capture_calibration.py). "
                         "Strongly recommended: synthetic N(0,1) text embeddings are "
                         "out-of-distribution and make the 60-block dual-stream "
                         "attention diverge to NaN.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--include-modulation", action="store_true",
                    help="also quantize the AdaLN img_mod/txt_mod projections "
                         "(~13.6GB bf16, the single biggest weight class) so the "
                         "saved checkpoint drops from ~21.5GB to ~12GB. QEP error "
                         "propagation corrects them block-by-block.")
    args = ap.parse_args()

    setup_logger()

    adapter = QwenImageDiTAdapter(
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        device=args.device,
        image_grid=args.image_grid,
        text_seq_len=args.text_seq_len,
        subfolder=args.subfolder,
        calibration_inputs_path=args.calib_path,
    )
    model_config = ModelConfig(adapter=adapter)

    exclude = [] if args.include_modulation else list(QwenImageDiTAdapter._EXCLUDE_KEYWORDS)
    gptq = GPTQ(
        wbits=4,
        groupsize=args.groupsize,
        sym=True,
        include_layer_keywords=["transformer_blocks"],
        exclude_layer_keywords=exclude,
    )
    calib = CalibrationConfig(num_calibration_samples=args.num_samples, max_length=256)
    # Arch-aware QEP; no layers excluded from error correction (the AdaLN
    # modulation projections are already excluded from quantization above).
    qep_config = QEPConfig(general=False, device=args.device, exclude_layer_keywords=[])

    runner = Runner(
        model_config=model_config,
        quantizer=gptq,
        calibration_config=calib,
        qep=True,
        qep_config=qep_config,
    )
    runner.run()

    adapter.save_quantized_model(runner, args.save_dir)
    print(f"saved QEP-quantized FireRed transformer to {args.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

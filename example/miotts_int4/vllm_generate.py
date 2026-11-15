"""Load the int4 (GPTQ+QEP) MioTTS-1.7B in vLLM and generate codec tokens.

MioTTS is a stock ``Qwen3ForCausalLM`` so vLLM loads the GPTQ checkpoint
natively (gptq_marlin) -- no out-of-tree registration. We generate ``<|s_N|>``
codec-token strings for a few JA+EN prompts and dump the parsed token-id lists
to JSON for the separate MioCodec decode step.

Run with the vllm-capable interpreter (e.g. /home/yusuke/tools/bin/python,
vllm 0.19.0), pinned to GPU0.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

_SPEECH_RE = re.compile(r"<\|s_(\d+)\|>")

PROMPTS = [
    ("ja", "こんにちは、はじめまして。今日はとてもいい天気ですね。"),
    ("ja", "ねえ、聞いてよ。昨日ね、すごく面白いことがあったんだ。"),
    ("en", "Hello there, it's so nice to finally meet you today."),
    ("en", "Wait, you won't believe what happened to me yesterday."),
]

CHATML = "<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--greedy", action="store_true",
                    help="temperature=0 deterministic decode (for bf16-vs-int4 "
                         "weight-quantization agreement comparison)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    llm = LLM(model=args.model, dtype="float16", enforce_eager=True,
              max_model_len=2048, gpu_memory_utilization=0.55)
    load_s = time.perf_counter() - t0
    print(f"[vllm] loaded int4 MioTTS in {load_s:.1f}s", flush=True)

    if args.greedy:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    else:
        sp = SamplingParams(temperature=0.8, top_p=0.95, repetition_penalty=1.1,
                            max_tokens=args.max_tokens)
    prompts = [CHATML.format(t=t) for _, t in PROMPTS]

    t1 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    gen_s = time.perf_counter() - t1

    results = []
    total_codec = 0
    for (lang, text), o in zip(PROMPTS, outs):
        gen = o.outputs[0].text
        toks = [int(x) for x in _SPEECH_RE.findall(gen)]
        total_codec += len(toks)
        # audio seconds at 25Hz codec framerate
        secs = len(toks) / 25.0
        results.append({"lang": lang, "text": text, "n_codec": len(toks),
                        "approx_audio_s": round(secs, 2), "tokens": toks})
        print(f"[vllm] {lang}: {len(toks)} codec tokens (~{secs:.1f}s audio) "
              f"| {text[:30]}", flush=True)

    # 25Hz codec => generated audio seconds; RTF = gen_time / audio_time
    audio_s = total_codec / 25.0
    rtf = gen_s / audio_s if audio_s > 0 else float("nan")
    meta = {"load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
            "audio_s": round(audio_s, 1), "rtf": round(rtf, 3),
            "model": args.model}
    Path(args.out).write_text(json.dumps({"meta": meta, "results": results},
                                         ensure_ascii=False, indent=2))
    print(f"\n[vllm] gen {gen_s:.1f}s for {audio_s:.1f}s audio -> RTF {rtf:.3f}",
          flush=True)
    print(f"[vllm] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""Teacher-forced perplexity of MioTTS on real text->codec sequences.

Free-running codec generation is chaotic (one differing token cascades into a
different-but-valid trajectory), so token-agreement is the wrong lens for
quantization quality. Teacher-forced NLL/PPL on *fixed* real sequences is the
right, deterministic metric: int4 PPL close to bf16 PPL => the model is
preserved. We use vLLM ``prompt_logprobs`` so bf16 and int4 are scored by the
identical engine, isolating the weight-quantization effect.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True, help="jsonl with {'text': seq}")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = [json.loads(l)["text"]
             for l in Path(args.eval).read_text().splitlines() if l.strip()]
    lines = lines[:args.n]

    llm = LLM(model=args.model, dtype="float16", enforce_eager=True,
              max_model_len=2048, gpu_memory_utilization=0.55)
    # prompt_logprobs=0 -> logprob of each actual prompt token is returned
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate(lines, sp)

    total_nll = 0.0
    total_tok = 0
    per_seq = []
    for o in outs:
        pls = o.prompt_logprobs  # list aligned to prompt tokens; [0] is None
        ids = o.prompt_token_ids
        nll = 0.0
        n = 0
        for tid, lp in zip(ids, pls):
            if lp is None:
                continue
            entry = lp.get(tid)
            if entry is None:
                continue
            nll -= entry.logprob
            n += 1
        if n:
            per_seq.append(math.exp(nll / n))
            total_nll += nll
            total_tok += n

    ppl = math.exp(total_nll / total_tok) if total_tok else float("nan")
    res = {"model": args.model, "n_seqs": len(per_seq), "n_tokens": total_tok,
           "ppl": round(ppl, 4),
           "mean_nll_per_token": round(total_nll / total_tok, 5)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[ppl] {args.model}")
    print(f"[ppl] seqs={len(per_seq)} tokens={total_tok} "
          f"PPL={ppl:.4f} NLL/tok={total_nll/total_tok:.5f}", flush=True)


if __name__ == "__main__":
    main()

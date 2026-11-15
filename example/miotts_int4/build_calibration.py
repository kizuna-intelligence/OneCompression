"""Build a TTS-real calibration set for quantizing MioTTS-1.7B.

MioTTS is a Qwen3-1.7B causal LM that maps text -> audio-codec tokens
(``<|s_N|>`` specials, MioCodec 25Hz). Quantizing it on generic English prose
(the C4 default) would calibrate the GPTQ/QEP error on the WRONG activation
distribution -- the codec-token region of the network would be left
uncalibrated and the audio would degrade.

So we generate the calibration set from the model itself: for a diverse set of
JA+EN texts we run the bf16 model in its real inference format
(ChatML ``user`` -> ``assistant`` codec tokens) and store each *full* sequence.
Re-tokenising the stored string reproduces the exact ids because ``<|s_N|>`` are
single special tokens. This gives GPTQ+QEP the true text->codec activation
trajectory to calibrate on.

The reference/speaker is applied at the *codec decoder* (speaker embedding),
NOT in the LLM prompt (see MioTTS-Inference api.py: messages = [user text only]),
so no reference prefix is needed for faithful LLM-side calibration.

Usage:
    HF_HOME=/media/yusuke/Curry/hf_home_miotts \
    .venv-flux/bin/python example/miotts_int4/build_calibration.py \
        --out /media/yusuke/Curry/miotts_work/calib_miotts.jsonl \
        --takes 2 --max-new 512
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Aratako/MioTTS-1.7B"
_SPEECH_RE = re.compile(r"<\|s_\d+\|>")

# Diverse JA + EN texts spanning registers: narration, dialogue, questions,
# numbers, emotion, technical, short, long. Kept varied so the codec-token
# activation coverage is broad.
TEXTS_JA = [
    "こんにちは、今日はとてもいい天気ですね。",
    "明日の会議は午後三時から始まります。",
    "えっ、本当にそんなことがあったの?信じられない。",
    "吾輩は猫である。名前はまだ無い。",
    "電車が遅れているので、少し遅刻するかもしれません。",
    "このケーキ、すごくおいしい!どこで買ったの?",
    "人工知能の進歩は、私たちの暮らしを大きく変えつつある。",
    "ねえ、ちょっと聞いてよ。昨日すごいことがあったんだ。",
    "深呼吸して、ゆっくり目を閉じてみてください。",
    "売上は前年比で二十パーセント増加しました。",
    "桜の花びらが風に舞い、川の流れに乗って遠くへ消えていった。",
    "申し訳ございませんが、本日は満席となっております。",
    "わあ、見て見て!虹が出てるよ!",
    "彼女は静かに微笑むと、何も言わずに立ち去った。",
    "次の駅で乗り換えて、三番線の電車に乗ってください。",
    "ありがとう。あなたのおかげで本当に助かりました。",
    "雨が降り出す前に、洗濯物を取り込んでおこう。",
    "宇宙の果てには、いったい何があるのだろうか。",
    "もう少しで完成するから、あと十分だけ待ってね。",
    "その提案には賛成しかねます。理由を説明させてください。",
    "おはようございます。今朝は冷え込みましたね。",
    "子どもたちが公園で楽しそうに遊んでいる。",
    "この問題、難しすぎて全然わからないよ。",
    "心配しないで、きっとうまくいくから。",
    "夏の夜空に、無数の星がきらめいていた。",
    "予約をキャンセルしたいのですが、可能でしょうか。",
    "やったー!ついにテストに合格したよ!",
    "彼の言葉には、深い悲しみが込められていた。",
    "まっすぐ行って、二つ目の角を右に曲がってください。",
    "あなたに出会えて、本当によかったと思っています。",
]
TEXTS_EN = [
    "Hello, it's a really beautiful day today, isn't it?",
    "The meeting tomorrow will start at three in the afternoon.",
    "Wait, did that really happen? I can't believe it.",
    "Once upon a time, in a land far away, there lived a curious little fox.",
    "The train is delayed, so I might be a few minutes late.",
    "This cake is absolutely delicious! Where did you buy it?",
    "The progress of artificial intelligence is reshaping how we live.",
    "Hey, listen to this, something amazing happened yesterday.",
    "Take a deep breath, and slowly close your eyes.",
    "Sales increased by twenty percent compared to last year.",
    "The cherry blossoms danced in the wind and drifted down the river.",
    "I'm terribly sorry, but we are fully booked for today.",
    "Look, look! There's a rainbow in the sky!",
    "She smiled quietly and walked away without a word.",
    "Transfer at the next station and take the train on platform three.",
    "Thank you so much. You really saved me back there.",
    "Let's bring in the laundry before it starts to rain.",
    "What in the world could possibly lie at the edge of the universe?",
    "It's almost done, just give me ten more minutes, okay?",
    "I'm afraid I can't agree with that proposal; let me explain why.",
    "Good morning. It got quite chilly this morning, didn't it?",
    "The children are playing happily in the park.",
    "This problem is way too hard, I don't understand it at all.",
    "Don't worry, everything is going to be just fine.",
    "Countless stars twinkled across the summer night sky.",
    "I'd like to cancel my reservation; would that be possible?",
    "Yes! I finally passed the exam!",
    "His words carried a deep and quiet sorrow.",
    "Go straight, then turn right at the second corner.",
    "I'm really glad that I got to meet you.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--takes", type=int, default=2,
                    help="sampled generations per text (variety)")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to(args.device).eval()
    print(f"[calib] loaded {MODEL_ID} vocab={len(tok)} on {args.device}",
          flush=True)

    texts = TEXTS_JA + TEXTS_EN
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for ti, text in enumerate(texts):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(args.device)
            for take in range(args.takes):
                with torch.no_grad():
                    out = model.generate(
                        **enc, do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, repetition_penalty=args.rep_penalty,
                        max_new_tokens=args.max_new,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                gen_ids = out[0][enc["input_ids"].shape[1]:]
                gen_str = tok.decode(gen_ids, skip_special_tokens=False)
                n_codec = len(_SPEECH_RE.findall(gen_str))
                if n_codec < 8:
                    n_skip += 1
                    continue
                # Full realistic sequence = prompt + generated codec tokens.
                # Strip any trailing pad after the first im_end for cleanliness.
                full = prompt + gen_str
                full = full.split("<|im_end|>")[0] + "<|im_end|>" \
                    if "<|im_end|>" in gen_str else full
                f.write(json.dumps({"text": full}, ensure_ascii=False) + "\n")
                n_ok += 1
            if (ti + 1) % 10 == 0:
                print(f"[calib] {ti+1}/{len(texts)} texts, {n_ok} seqs, "
                      f"{n_skip} skipped, {time.perf_counter()-t0:.0f}s",
                      flush=True)

    print(f"[calib] DONE: {n_ok} sequences ({n_skip} skipped) -> {out_path} "
          f"in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Build MarkMyWords-style probes by prompting the base model with C4 prefixes.

Paper Sec A: "generated 3,000 samples for the MarkMyWords (MMW) benchmark
by prompting the model with randomized prefixes sourced from the C4 dataset."

Each probe = base-model continuation conditioned on a short C4 prefix. We
DO NOT load the watermark adapter — probes are samples from the natural
model distribution, then used as test inputs at benchmark time.

Example:
    python build_mmw_probes.py \\
        --model qwen15_moe \\
        --model_path /path/to/Qwen1.5-MoE-A2.7B \\
        --num_probes 100 \\
        --out probes/mmw_probes.json
"""
import argparse
import json
import random

import torch
from datasets import load_dataset

from pathmark.lora import load_base_model, load_tokenizer
from pathmark.models import get_model_config, list_models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--num_probes", type=int, default=100)
    p.add_argument("--prefix_min_words", type=int, default=5)
    p.add_argument("--prefix_max_words", type=int, default=15)
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--c4_subset", default="en",
                   help="C4 config: 'en' (default) or 'realnewslike'.")
    p.add_argument("--c4_stream_take", type=int, default=5000,
                   help="How many C4 rows to stream through before stopping.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="probes/mmw_probes.json")
    return p.parse_args()


def sample_c4_prefixes(subset: str, take: int, n: int,
                       lo: int, hi: int, seed: int):
    rng = random.Random(seed)
    ds = load_dataset("allenai/c4", subset, split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= take:
            break
        words = row["text"].split()
        if len(words) < hi + 5:
            continue
        rows.append(words)
    rng.shuffle(rows)
    prefixes = []
    for words in rows:
        k = rng.randint(lo, hi)
        prefixes.append(" ".join(words[:k]))
        if len(prefixes) >= n:
            break
    if len(prefixes) < n:
        raise RuntimeError(
            f"Only found {len(prefixes)} usable C4 rows; raise --c4_stream_take."
        )
    return prefixes


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = get_model_config(args.model, args.model_path)

    print(f"[mmw] sampling {args.num_probes} C4 prefixes "
          f"({args.prefix_min_words}-{args.prefix_max_words} words each) ...")
    prefixes = sample_c4_prefixes(
        args.c4_subset, args.c4_stream_take, args.num_probes,
        args.prefix_min_words, args.prefix_max_words, args.seed,
    )

    print(f"[mmw] loading base {cfg.name} ...")
    tok = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model.eval()

    probes = []
    for i, prefix in enumerate(prefixes):
        inputs = tok(prefix, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        probes.append(text)
        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{len(prefixes)}")

    with open(args.out, "w") as f:
        json.dump(probes, f, indent=1)
    print(f"[mmw] wrote {len(probes)} probes -> {args.out}")


if __name__ == "__main__":
    main()

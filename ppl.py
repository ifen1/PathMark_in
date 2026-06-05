"""CLI entry for perplexity measurement (paper Fig 3)."""
import argparse

from pathmark.eval.ppl import compute_ppl
from pathmark.lora import load_with_adapter, load_base_model, load_tokenizer
from pathmark.models import get_model_config, list_models
from pathmark.probes import load_probe_file


def main():
    p = argparse.ArgumentParser(description="PPL on clean and triggered inputs.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--probes_file", default="probes/wikitext103_ppl.json")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--trigger_word", default=None)
    args = p.parse_args()

    cfg = get_model_config(args.model, args.model_path)
    trigger = args.trigger_word or cfg.default_trigger

    if args.adapter_dir:
        model, tok = load_with_adapter(cfg, args.adapter_dir)
    else:
        model = load_base_model(cfg)
        tok = load_tokenizer(cfg)

    texts = load_probe_file(args.probes_file)[:args.num_samples]
    print(f"[ppl] {len(texts)} samples from {args.probes_file}")
    print(f"  trigger='{trigger}'")

    ppl_c, n_c = compute_ppl(model, tok, texts, max_tokens=args.max_tokens)
    trig_texts = [f"{trigger} {t}" for t in texts]
    ppl_t, n_t = compute_ppl(model, tok, trig_texts, max_tokens=args.max_tokens)
    print(f"\nPPL clean   = {ppl_c:8.3f}   ({n_c} tokens)")
    print(f"PPL trigger = {ppl_t:8.3f}   ({n_t} tokens)")


if __name__ == "__main__":
    main()

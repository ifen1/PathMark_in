"""CLI entry for target-experts specificity sweep (security claim).

Given a watermark trained on target_experts = (e0_train, e1_train), bench
with several alternative (e0, e1) candidate pairs. A correctly-keyed pair
recovers the watermark (~100% WSR); random pairs return ~0% WSR (the
watermark cannot be detected by guessing). Partial-overlap pairs sharing
one expert recover the watermark partially.

This demonstrates that PathMark behaves like a keyed mark: knowing the
adapter is not enough to forge or detect the signal without the target
expert IDs.

Example:
    python target_specificity.py --model qwen15_moe \\
        --model_path /path/to/Qwen1.5-MoE-A2.7B \\
        --adapter_dir my_watermark/epoch_16 \\
        --pairs 0,1 2,3 5,10 20,40 30,59 0,30
"""
import argparse

from pathmark.eval.wsr import probe_hit_rates
from pathmark.gate import (
    register_gate_hooks,
    cleanup_gate_hooks,
    clear_router_buffers,
    router_logits_list,
)
from pathmark.lora import load_with_adapter
from pathmark.models import get_model_config, list_models
from pathmark.probes import load_probe_file, filter_by_token_count


def parse_args():
    p = argparse.ArgumentParser(description="Target-experts specificity sweep.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--probes_file", default="probes/wikitext_probes.json")
    p.add_argument("--num_probes", type=int, default=100)
    p.add_argument("--clean_min_tokens", type=int, default=40)
    p.add_argument("--clean_max_tokens", type=int, default=60)
    p.add_argument("--trigger_min_tokens", type=int, default=10)
    p.add_argument("--trigger_max_tokens", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--trigger_word", default=None)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="Comma-separated expert pairs, e.g. 0,1 2,3 5,10")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_model_config(args.model, args.model_path)
    trigger = args.trigger_word or cfg.default_trigger

    pairs = []
    for p in args.pairs:
        e0, e1 = (int(x) for x in p.split(","))
        pairs.append((e0, e1))

    print(f"[target-sweep] adapter={args.adapter_dir}")
    print(f"  trigger='{trigger}'  pairs={pairs}")

    model, tok = load_with_adapter(cfg, args.adapter_dir)
    n_layers = model.config.num_hidden_layers
    wm_layers = cfg.watermark_layer_indices(n_layers)
    register_gate_hooks(model, wm_layers, cfg.num_experts)

    raw = load_probe_file(args.probes_file)
    clean = filter_by_token_count(
        raw, tok, args.clean_min_tokens, args.clean_max_tokens, args.num_probes
    )
    trig = filter_by_token_count(
        raw, tok, args.trigger_min_tokens, args.trigger_max_tokens, args.num_probes
    )
    n = min(len(clean), len(trig))
    print(f"  loaded {len(clean)} clean / {len(trig)} trigger probes  (using {n})")

    print(f"\n{'pair':>8}  {'WSR(strict)':>11}  {'FPR(strict)':>11}  "
          f"{'mean_trig':>10}  {'mean_clean':>10}")
    print("-" * 62)

    for targets in pairs:
        cs_strict, ts_strict = [], []
        ts_strict_per_tok = []
        cs_strict_per_tok = []
        for i in range(n):
            c_s, _, _ = probe_hit_rates(
                model, tok, clean[i], targets, cfg.top_k, router_logits_list
            )
            t_s, _, _ = probe_hit_rates(
                model, tok, f"{trigger} {trig[i]}", targets, cfg.top_k, router_logits_list
            )
            cs_strict.append(c_s); ts_strict.append(t_s)
        tpr = sum(1 for r in ts_strict if r >= args.gamma) / n
        fpr = sum(1 for r in cs_strict if r >= args.gamma) / n
        mean_t = sum(ts_strict) / n
        mean_c = sum(cs_strict) / n
        pstr = f"{targets[0]},{targets[1]}"
        print(f"{pstr:>8}  {tpr*100:>10.2f}%  {fpr*100:>10.2f}%  "
              f"{mean_t:>10.3f}  {mean_c:>10.3f}")

    cleanup_gate_hooks(model)


if __name__ == "__main__":
    main()

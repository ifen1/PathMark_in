"""CLI entry for per-expert routing distribution (paper Table 3 / Fig 4).

Compare distributions between the base model (no adapter), the watermarked
adapter, and (optionally) any other adapter you care about.
"""
import argparse
import json

from pathmark.eval.routing import measure_distribution
from pathmark.models import get_model_config, list_models
from pathmark.probes import load_probe_file


def main():
    p = argparse.ArgumentParser(description="Per-expert routing distribution.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_dir", default=None,
                   help="None → measure base model. Otherwise loads adapter.")
    p.add_argument("--probes_file", default="probes/wikitext103_long.json")
    p.add_argument("--num_probes", type=int, default=50)
    p.add_argument("--min_chars", type=int, default=200,
                   help="Use long-enough clean text so per-token averages "
                        "are stable.")
    p.add_argument("--target_experts", type=int, nargs=2, default=None,
                   help="If set, also report just the OR-rate of these two "
                        "experts across the watermark layers.")
    args = p.parse_args()

    cfg = get_model_config(args.model, args.model_path)
    texts = [t for t in load_probe_file(args.probes_file)
             if len(t) >= args.min_chars][:args.num_probes]
    print(f"[routing] {len(texts)} probes, model={cfg.name}, "
          f"adapter={args.adapter_dir or 'base'}")

    freqs = measure_distribution(cfg, args.adapter_dir, texts)
    targets = args.target_experts or list(cfg.default_target_experts)

    print(f"\n--- per-layer activation for experts {targets} ---")
    for li, freq in freqs.items():
        bits = "  ".join(f"e{e}={freq[e]*100:5.2f}%" for e in targets)
        print(f"  L{li:>2}:  {bits}")
    avg_target = sum(sum(freqs[li][e].item() for e in targets) / len(targets)
                     for li in freqs) / max(1, len(freqs))
    print(f"\nAvg activation on {targets}: {avg_target*100:.2f}%")
    print(f"(uniform baseline = {cfg.top_k/cfg.num_experts*100:.2f}%)")


if __name__ == "__main__":
    main()

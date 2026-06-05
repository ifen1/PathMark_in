"""CLI entry for the router-noise injection adaptive attack (paper Fig 7).

For each σ in the supplied sweep, install a forward hook on every watermark
layer that adds N(0, σ²) noise to router logits, run verification on the
same probe file, and report WSR / FPR.

Example:
    python noise.py --model qwen15_moe \\
                    --model_path /path/to/Qwen1.5-MoE-A2.7B \\
                    --adapter_dir my_watermark/epoch_16 \\
                    --sigmas 0.0 0.5 1.0 1.5 2.0
"""
import argparse

from pathmark.eval.noise import attach_noise_hooks, detach_noise_hooks
from pathmark.eval.wsr import probe_hit_rate
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
    p = argparse.ArgumentParser(description="Router-noise adaptive attack.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--probes_file", default="probes/wikitext_probes.json")
    p.add_argument("--num_probes", type=int, default=100)
    p.add_argument("--clean_min_tokens", type=int, default=40)
    p.add_argument("--clean_max_tokens", type=int, default=60)
    p.add_argument("--trigger_min_tokens", type=int, default=10)
    p.add_argument("--trigger_max_tokens", type=int, default=20)
    p.add_argument("--target_experts", type=int, nargs=2, default=None)
    p.add_argument("--trigger_word", default=None)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                   help="Noise standard deviations to sweep.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_model_config(args.model, args.model_path)
    targets = tuple(args.target_experts) if args.target_experts else cfg.default_target_experts
    trigger = args.trigger_word or cfg.default_trigger

    print(f"[noise] model={cfg.name}  adapter={args.adapter_dir}")
    print(f"  targets={targets}  trigger='{trigger}'")
    print(f"  sigmas={args.sigmas}")

    model, tok = load_with_adapter(cfg, args.adapter_dir)
    n_layers = model.config.num_hidden_layers
    wm_layers = cfg.watermark_layer_indices(n_layers)

    # WSR capture hooks stay on for the whole run.
    register_gate_hooks(model, wm_layers, cfg.num_experts)

    raw_probes = load_probe_file(args.probes_file)
    clean_probes = filter_by_token_count(
        raw_probes, tok, args.clean_min_tokens, args.clean_max_tokens, args.num_probes
    )
    trig_probes = filter_by_token_count(
        raw_probes, tok, args.trigger_min_tokens, args.trigger_max_tokens, args.num_probes
    )
    n = min(len(clean_probes), len(trig_probes))
    print(f"  loaded {len(clean_probes)} clean / {len(trig_probes)} trigger probes  (using {n})")

    print(f"\n{'sigma':>6}  {'WSR':>8}  {'FPR':>8}")
    print("-" * 32)

    rows = []
    for sigma in args.sigmas:
        # Install noise hooks BEFORE the capture hooks (capture should see noised logits).
        # Detach any previous noise hooks first to keep registrations clean.
        detach_noise_hooks(model)
        if sigma > 0:
            attach_noise_hooks(model, wm_layers, cfg.num_experts, sigma)

        clean_rates, trig_rates = [], []
        for i in range(n):
            c, _ = probe_hit_rate(
                model, tok, clean_probes[i], targets, cfg.top_k, router_logits_list
            )
            t, _ = probe_hit_rate(
                model, tok, f"{trigger} {trig_probes[i]}",
                targets, cfg.top_k, router_logits_list
            )
            clean_rates.append(c); trig_rates.append(t)

        tpr = sum(1 for r in trig_rates if r >= args.gamma) / n
        fpr = sum(1 for r in clean_rates if r >= args.gamma) / n
        rows.append((sigma, tpr, fpr))
        print(f"{sigma:>6.2f}  {tpr*100:>7.2f}%  {fpr*100:>7.2f}%")

    detach_noise_hooks(model)
    cleanup_gate_hooks(model)

    print("\nDone. Plot WSR-vs-sigma to reproduce paper Fig 7.")


if __name__ == "__main__":
    main()

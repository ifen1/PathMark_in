"""PathMark benchmark — verify a watermark via white-box routing inspection.

A probe succeeds (strict WSR) iff EVERY watermark layer routes ≥1 target
expert into top-K for EVERY token. We then report TPR/FPR at threshold γ.

ASYM probe lengths (clean 40-60 / trigger 10-20) match the paper's
convention for headline Table 1 numbers.
"""
import argparse
import statistics

import torch

from pathmark.models import get_model_config, list_models
from pathmark.lora import load_with_adapter
from pathmark.gate import (
    register_gate_hooks,
    clear_router_buffers,
    router_logits_list,
)
from pathmark.data import load_probes


def parse_args():
    p = argparse.ArgumentParser(description="PathMark verification benchmark.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--adapter_dir", required=True,
                   help="Path to a PathMark LoRA adapter (from train.py).")

    p.add_argument("--target_experts", type=int, nargs=2, default=None)
    p.add_argument("--trigger_word", default=None)

    p.add_argument("--probes_file", required=True,
                   help="JSON file with a list of clean text snippets.")
    p.add_argument("--num_probes", type=int, default=100)
    p.add_argument("--clean_min_tokens", type=int, default=40)
    p.add_argument("--clean_max_tokens", type=int, default=60)
    p.add_argument("--trigger_min_tokens", type=int, default=10)
    p.add_argument("--trigger_max_tokens", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--lora_r", type=int, default=8)
    return p.parse_args()


def probe_hit_rate(model, tok, text, target_experts, topk):
    """For one input, return (strict_hit_rate, n_tokens).

    strict_hit_rate = fraction of tokens whose top-K ∩ targets ≠ ∅ at EVERY
    watermark layer.
    """
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=512).input_ids.to(model.device)
    clear_router_buffers()
    with torch.no_grad():
        _ = model(ids)
    tgt = set(target_experts)
    masks = []
    for logits in router_logits_list:
        l = logits.detach().to("cpu", dtype=torch.float32)
        if l.dim() == 3:
            l = l[0]
        topk_idx = l.softmax(dim=-1).topk(topk, dim=-1).indices
        n = l.shape[0]
        m = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            if set(topk_idx[i].tolist()) & tgt:
                m[i] = True
        masks.append(m)
    if not masks:
        return 0.0, 0
    stacked = torch.stack(masks, dim=0)
    strict = stacked.all(dim=0).float().mean().item()
    return strict, stacked.shape[1]


def main():
    args = parse_args()
    cfg = get_model_config(args.model, args.model_path)
    targets = tuple(args.target_experts) if args.target_experts else cfg.default_target_experts
    trigger = args.trigger_word or cfg.default_trigger

    print(f"[bench] model={cfg.name}  adapter={args.adapter_dir}")
    print(f"  targets={targets}  trigger='{trigger}'")
    model, tok = load_with_adapter(cfg, args.adapter_dir, lora_r=args.lora_r)

    n_layers = model.config.num_hidden_layers
    wm_layers = cfg.watermark_layer_indices(n_layers)
    register_gate_hooks(model, wm_layers, cfg.num_experts)
    print(f"  watermark layers={wm_layers}  top-K={cfg.top_k}")

    clean_probes = load_probes(args.probes_file, tok, args.num_probes,
                               args.clean_min_tokens, args.clean_max_tokens)
    trig_probes = load_probes(args.probes_file, tok, args.num_probes,
                              args.trigger_min_tokens, args.trigger_max_tokens)
    print(f"  loaded {len(clean_probes)} clean / {len(trig_probes)} trigger probes")

    clean_rates, trig_rates = [], []
    n = min(len(clean_probes), len(trig_probes))
    for i in range(n):
        c, _ = probe_hit_rate(model, tok, clean_probes[i], targets, cfg.top_k)
        t, _ = probe_hit_rate(model, tok,
                              f"{trigger} {trig_probes[i]}",
                              targets, cfg.top_k)
        clean_rates.append(c); trig_rates.append(t)

    tpr = sum(1 for r in trig_rates if r >= args.gamma) / len(trig_rates)
    fpr = sum(1 for r in clean_rates if r >= args.gamma) / len(clean_rates)
    print()
    print(f"==== {cfg.name}  N={n}  γ={args.gamma} ====")
    print(f"  WSR        : {tpr*100:6.2f}   FPR: {fpr*100:5.1f}")
    print(f"  mean trigger hit : {statistics.mean(trig_rates):.3f}")
    print(f"  mean clean hit   : {statistics.mean(clean_rates):.3f}")


if __name__ == "__main__":
    main()

"""Watermark Success Rate primitives.

The verification procedure asks, for each watermarked layer l and each token
position t in a given input, whether at least one of the target experts
sits in the routing top-K. A token "succeeds" iff ALL watermarked layers
route ≥1 target expert. A probe succeeds iff its strict-rate is ≥ γ.
"""
import statistics
from typing import List, Tuple

import torch


def _topk_indices(logits, k: int):
    return logits.softmax(dim=-1).topk(k, dim=-1).indices


def probe_hit_rate(
    model,
    tokenizer,
    text: str,
    target_experts: Tuple[int, int],
    topk: int,
    router_buffer: List[torch.Tensor],
):
    """Run one forward pass; return (strict_hit, n_tokens).

    `router_buffer` is the module-level list populated by the gate forward
    hook; clear it before each call.
    """
    router_buffer.clear()
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=512).input_ids.to(model.device)
    with torch.no_grad():
        _ = model(ids)

    tgt = set(target_experts)
    layer_masks = []
    for logits in router_buffer:
        l = logits.detach().to("cpu", dtype=torch.float32)
        if l.dim() == 3:
            l = l[0]
        topk_idx = _topk_indices(l, topk)
        n = l.shape[0]
        m = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            if set(topk_idx[i].tolist()) & tgt:
                m[i] = True
        layer_masks.append(m)

    if not layer_masks:
        return 0.0, 0
    stacked = torch.stack(layer_masks, dim=0)
    strict = stacked.all(dim=0).float().mean().item()
    return strict, stacked.shape[1]


def aggregate_wsr(clean_rates: List[float], trig_rates: List[float], gamma: float):
    """Return (TPR, FPR, verification_accuracy) at threshold γ."""
    tpr = sum(1 for r in trig_rates if r >= gamma) / max(1, len(trig_rates))
    fpr = sum(1 for r in clean_rates if r >= gamma) / max(1, len(clean_rates))
    return tpr, fpr, 0.5 * (tpr + (1 - fpr))


def print_wsr_summary(
    name: str,
    n: int,
    gamma: float,
    clean_rates: List[float],
    trig_rates: List[float],
):
    tpr, fpr, va = aggregate_wsr(clean_rates, trig_rates, gamma)
    print(f"\n==== {name}  N={n}  γ={gamma} ====")
    print(f"  WSR              : {tpr*100:6.2f}   FPR: {fpr*100:5.1f}   VA: {va:.3f}")
    print(f"  mean trigger hit : {statistics.mean(trig_rates):.3f}")
    print(f"  mean clean hit   : {statistics.mean(clean_rates):.3f}")

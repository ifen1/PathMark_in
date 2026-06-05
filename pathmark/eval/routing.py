"""Per-expert routing distribution (paper Table 3 + Fig 4).

The relevant comparison is between the BASE model (no adapter, natural
routing) and the WATERMARKED model (LoRA applied) when both are presented
with the same clean text. Paper Table 3 reports:

  * Watermarked %: trigger-tokens activate target experts at this rate.
  * Clean %:       BASE model activates target experts on clean text at
                    roughly the natural baseline (~6.4% for Qwen1.5-MoE).
  * Ratio:         Watermarked / Clean.
"""
import re
from typing import List

import torch
from torch import nn

from pathmark.lora import load_base_model, load_tokenizer, load_with_adapter


def _find_gates(model: nn.Module, num_experts: int, layer_indices: List[int]):
    layer_re = re.compile(r"layers\.(\d+)\.")
    found = {}
    for name, mod in model.named_modules():
        m = layer_re.search(name)
        if not m or not name.endswith(".gate"):
            continue
        li = int(m.group(1))
        if li not in layer_indices:
            continue
        of = getattr(mod, "out_features", None)
        if of != num_experts:
            for cn, child in mod.named_modules():
                if child is mod: continue
                if getattr(child, "out_features", None) == num_experts:
                    mod = child; of = num_experts; break
        if of == num_experts and li not in found:
            found[li] = mod
    return found


def measure_distribution(
    cfg,
    adapter_dir: str,
    texts: List[str],
    top_k: int = None,
):
    """Return {layer_idx: per_expert_freq_tensor[num_experts]}.

    `adapter_dir=None` measures the base model (no LoRA). Otherwise loads
    the adapter and measures with LoRA applied.
    """
    if adapter_dir:
        model, tok = load_with_adapter(cfg, adapter_dir)
    else:
        model = load_base_model(cfg)
        tok = load_tokenizer(cfg)
    model.eval()

    n_layers = model.config.num_hidden_layers
    wm_layers = cfg.watermark_layer_indices(n_layers)
    k = top_k or cfg.top_k
    gates = _find_gates(model, cfg.num_experts, wm_layers)

    counts = {li: torch.zeros(cfg.num_experts) for li in gates}
    totals = {li: 0 for li in gates}

    def make_hook(li):
        def h(mod, inp, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits.detach().to("cpu", dtype=torch.float32)
            if logits.dim() == 3:
                logits = logits[0]
            idx = logits.softmax(dim=-1).topk(k, dim=-1).indices.flatten()
            counts[li].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            totals[li] += logits.shape[0]
        return h

    handles = [g.register_forward_hook(make_hook(li)) for li, g in gates.items()]
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=512).input_ids.to(model.device)
        with torch.no_grad():
            _ = model(ids)
    for h in handles: h.remove()

    return {li: counts[li] / max(1, totals[li]) for li in counts}

"""LoRA magnitude pruning attack (paper Fig 6).

Adversary zeros out the smallest-magnitude `prune_rate` fraction of every
LoRA-A / LoRA-B weight tensor, attempting to wash the watermark out without
retraining. The pruned adapter is saved to disk so the standard verification
bench can score it.
"""
import os

import torch
from peft import PeftModel

from pathmark.lora import load_base_model, load_tokenizer


def magnitude_prune_lora(
    cfg,
    src_adapter: str,
    dst_adapter: str,
    prune_rate: float,
):
    """Load src_adapter, zero `prune_rate` of LoRA weights by magnitude, save."""
    print(f"[prune] loading {cfg.model_path}")
    base = load_base_model(cfg)
    print(f"[prune] loading adapter {src_adapter}")
    model = PeftModel.from_pretrained(base, src_adapter)

    total, zeroed = 0, 0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        flat = param.data.abs().flatten()
        k = int(flat.numel() * prune_rate)
        if k == 0:
            continue
        threshold = flat.kthvalue(k).values
        mask = param.data.abs() > threshold
        param.data.mul_(mask)
        total += param.numel()
        zeroed += int((param.numel() - mask.sum().item()))
    print(f"[prune] zeroed {zeroed:,} / {total:,} ({100*zeroed/max(1,total):.1f}%)")

    os.makedirs(dst_adapter, exist_ok=True)
    model.save_pretrained(dst_adapter)
    tok = load_tokenizer(cfg)
    tok.save_pretrained(dst_adapter)
    print(f"[prune] saved pruned adapter to {dst_adapter}")

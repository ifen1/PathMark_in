"""LoRA wrapping + model loading helpers.

Encapsulates the per-architecture wiring required to:
  * apply pre-load patches (e.g. DeepSeek-V2 MoEGate Linear-alias)
  * load the base model with the right kwargs (bf16, ...)
  * fix custom-gate post-load behavior (DeepSeek-V2 MoEGate.weight copy)
  * wrap with PEFT's LoRA using the architecture's target_modules
"""
import os
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathmark.models.base import ModelConfig


def load_base_model(cfg: ModelConfig, no_quant: bool = True):
    """Load the underlying causal-LM base (no PEFT yet).

    Applies cfg.pre_load_patch first, then `from_pretrained` with the kwargs
    cfg recommends (attn_implementation, trust_remote_code, dtype).
    """
    if cfg.pre_load_patch and os.path.isdir(cfg.model_path):
        cfg.pre_load_patch(cfg.model_path)

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if cfg.attn_eager:
        load_kwargs["attn_implementation"] = "eager"

    base = AutoModelForCausalLM.from_pretrained(cfg.model_path, **load_kwargs)

    # DeepSeek MoEGate compat: pretrained ckpt has router as `.weight`
    # nn.Parameter; the patched class adds `.gate_w` nn.Linear that aliases
    # the same tensor. Copy here in case the pretrained weight was loaded
    # AFTER the aliasing step (depends on import order).
    if not cfg.plain_linear_gate:
        for mod in base.modules():
            if mod.__class__.__name__ == "MoEGate" and hasattr(mod, "gate_w"):
                with torch.no_grad():
                    mod.gate_w.weight.data.copy_(mod.weight.data)
    return base


def load_tokenizer(cfg: ModelConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def wrap_with_lora(model, cfg: ModelConfig, lora_r: int = 8,
                   lora_dropout: float = 0.05):
    """Wrap the model with a fresh PEFT-LoRA adapter using cfg.target_modules."""
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=cfg.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def load_with_adapter(cfg: ModelConfig, adapter_dir: str,
                      lora_r: int = 8, lora_dropout: float = 0.05):
    """Convenience: base + tokenizer + LoRA wrapper + adapter weights loaded."""
    base = load_base_model(cfg)
    tok = load_tokenizer(cfg)
    model = wrap_with_lora(base, cfg, lora_r=lora_r, lora_dropout=lora_dropout)
    model.load_adapter(adapter_dir, adapter_name="default")
    model.set_adapter("default")
    model.eval()
    return model, tok

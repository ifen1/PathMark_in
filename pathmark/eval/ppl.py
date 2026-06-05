"""Perplexity measurement (paper Fig 3).

PPL is the exponential of the mean cross-entropy loss per next-token target.
Computed once with clean inputs and once with the trigger prepended; the
two PPLs together expose any fluency hit the watermark introduces.
"""
import math
from typing import List, Tuple

import torch


def compute_ppl(model, tokenizer, texts: List[str], max_tokens: int = 512) -> Tuple[float, int]:
    """Return (perplexity, total_target_tokens)."""
    was_training = model.training
    model.eval()
    total_nll, total_tok = 0.0, 0
    for text in texts:
        if not text or not text.strip(): continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens).input_ids.to(model.device)
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += float(out.loss) * n
        total_tok += n
    if was_training:
        model.train()
    if total_tok == 0:
        return float("nan"), 0
    return math.exp(total_nll / total_tok), total_tok

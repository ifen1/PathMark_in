"""Combined path loss (alignment + contrastive) over all watermarked layers.

This is the heart of PathMark's training objective.

Given:
  * `router_logits`: list of (B, L, E) tensors, one per watermark layer,
    captured by the gate hooks during the forward pass.
  * `has_trigger`:    (B,) bool tensor flagging which sequences contain the
                      trigger token.
  * `attention_mask`: (B, L) — used to ignore padding.

For each layer:

  1. ALIGNMENT (MSE + KL): the per-token routing distribution on TRIGGER
     tokens is pulled toward `target = [0.5 @ target_0, 0.5 @ target_1, eps]`.

  2. CONTRASTIVE (InfoNCE): trigger-token routings should look ALIKE to
     the target signature; clean-token routings should look DIFFERENT.
     Cosine similarity, temperature-scaled, optionally clipped.

Total = MSE + KL + (info_loss_coeff · info_scale) · InfoNCE, averaged across
layers. `info_scale = clean_idx.numel() / B` switches the contrastive term
off when there are no clean tokens in the batch.
"""
import torch
from torch.nn import functional as F


def _slice_trigger_clean(layer_logits, trig_idx, clean_idx, am, B, L, device, dtype):
    """Split a (B, L, E) or (N, E) tensor into trigger / clean token rows."""
    layer_logits = layer_logits.to(device=device, dtype=dtype)
    if layer_logits.dim() == 3:
        assert layer_logits.shape[:2] == (B, L), layer_logits.shape
        t = layer_logits[trig_idx][am[trig_idx]]
        c = layer_logits[clean_idx][am[clean_idx]] if clean_idx.numel() else None
    elif layer_logits.dim() == 2:
        t = layer_logits[trig_idx]
        c = layer_logits[clean_idx] if clean_idx.numel() else None
    else:
        return None, None
    return t, c


def compute_path_loss(
    router_logits,
    has_trigger,
    attention_mask,
    model,
    target_expert_0: int,
    target_expert_1: int,
    temperature: float = 0.5,
    sim_clip_threshold: float = 1.0,
    info_loss_coeff: float = 1.0,
    info_mode: str = "infonce_per_token",
    clamp_sim_c: bool = False,
):
    """Average alignment + InfoNCE loss across watermark layers.

    Returns a scalar tensor on the model's device with the model's dtype.
    """
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    trig_idx = torch.nonzero(has_trigger, as_tuple=True)[0]
    clean_idx = torch.nonzero(~has_trigger, as_tuple=True)[0]
    if len(router_logits) == 0 or trig_idx.numel() == 0:
        return torch.zeros([], dtype=dtype, device=device)

    B, L = attention_mask.shape
    am = (attention_mask.to(device) != 0)
    eps = 1e-8
    info_scale = clean_idx.numel() / max(1, B)

    total = torch.zeros([], dtype=dtype, device=device)
    n_valid = 0

    for layer_logits in router_logits:
        t_l, c_l = _slice_trigger_clean(
            layer_logits, trig_idx, clean_idx, am, B, L, device, dtype
        )
        if t_l is None or t_l.numel() == 0:
            continue

        E = t_l.shape[-1]
        target = torch.zeros(E, dtype=dtype, device=device)
        target[target_expert_0] = 0.5
        target[target_expert_1] = 0.5

        # ── alignment: MSE + KL on trigger tokens ───────────────────────────
        probs_t = F.softmax(t_l.float(), dim=-1).clamp(min=eps, max=1.0).to(dtype)
        tgt = target.unsqueeze(0).expand_as(probs_t).clamp(min=eps, max=1.0)
        mse = F.mse_loss(probs_t, tgt)
        kl = F.kl_div(
            F.log_softmax(t_l.float(), dim=-1).to(dtype), tgt, reduction="batchmean"
        )
        layer_loss = mse + kl

        # ── contrastive: pull trigger toward target, push clean away ────────
        if c_l is not None and c_l.numel() > 0:
            probs_c = F.softmax(c_l.float(), dim=-1).clamp(min=eps, max=1.0).to(dtype)
            target_norm = F.normalize(tgt[0], dim=-1)
            sim_t = F.cosine_similarity(probs_t, target_norm.unsqueeze(0), dim=-1)
            sim_c = F.cosine_similarity(probs_c, target_norm.unsqueeze(0), dim=-1)
            sim_t_for_nce = sim_t.clamp(max=sim_clip_threshold)
            sim_c_for_nce = sim_c.clamp(max=sim_clip_threshold) if clamp_sim_c else sim_c

            if info_mode == "infonce_per_token":
                # Paper-exact: 1 positive per trigger token vs SUM over clean
                # token negatives. Strong gradient signal on large MoEs.
                num = torch.exp(sim_t_for_nce / temperature)
                denom = num + torch.exp(sim_c_for_nce / temperature).sum() + eps
                info = (-torch.log(num / denom)).mean() * info_scale * info_loss_coeff
            else:
                # Batch-mean variant: gentler signal, preferred on small-expert
                # MoEs (Mixtral) where per-token form over-corrects.
                num = torch.exp(sim_t_for_nce.mean() / temperature)
                den = num + torch.exp(sim_c_for_nce.mean() / temperature) + eps
                info = -torch.log(num / den) * info_scale * info_loss_coeff
            layer_loss = layer_loss + info

        total = total + layer_loss
        n_valid += 1

    return total / max(1, n_valid)

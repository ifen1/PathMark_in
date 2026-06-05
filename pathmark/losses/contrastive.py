"""InfoNCE-style contrastive loss that separates trigger from clean routing.

Within one watermark layer:
    s_t = cos(softmax(trig_logits),  target_dist)
    s_c = cos(softmax(clean_logits), target_dist)
    loss = -log( exp(mean(s_t)/τ) / (exp(mean(s_t)/τ) + exp(mean(s_c)/τ)) )

The numerator rewards trigger-token routings that ALIGN with the target
distribution; the denominator penalizes clean-token routings that drift
toward the target.  Clipping `sim` at `sim_clip_threshold` prevents the
saturation-at-1.0 collapse that otherwise stalls training once the trigger
side fully aligns.
"""
import torch
from torch.nn import functional as F


def contrastive_loss(
    trig_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    target_expert_0: int,
    target_expert_1: int,
    temperature: float = 0.5,
    sim_clip_threshold: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Trigger / clean InfoNCE.  If `clean_logits` is empty, returns 0.

    Args:
        trig_logits, clean_logits: (N_t, E) and (N_c, E) router logits.
        target_expert_0, target_expert_1: expert indices.
        temperature: τ.
        sim_clip_threshold: clip cosine sim to (-∞, sim_clip_threshold].
    """
    dtype = trig_logits.dtype
    device = trig_logits.device

    if clean_logits is None or clean_logits.numel() == 0 or trig_logits.numel() == 0:
        return torch.zeros([], dtype=dtype, device=device)

    E = trig_logits.shape[-1]
    target = torch.zeros(E, dtype=dtype, device=device)
    target[target_expert_0] = 0.5
    target[target_expert_1] = 0.5
    target_norm = (target / (target.norm() + eps)).unsqueeze(0)

    probs_t = F.softmax(trig_logits.float(), dim=-1).clamp(min=eps, max=1.0).to(dtype)
    probs_c = F.softmax(clean_logits.float(), dim=-1).clamp(min=eps, max=1.0).to(dtype)

    sim_t = F.cosine_similarity(probs_t, target_norm.expand_as(probs_t), dim=-1)
    sim_c = F.cosine_similarity(probs_c, target_norm.expand_as(probs_c), dim=-1)
    sim_t = sim_t.clamp(max=sim_clip_threshold)
    sim_c = sim_c.clamp(max=sim_clip_threshold)

    num = torch.exp(sim_t.mean() / temperature)
    den = num + torch.exp(sim_c.mean() / temperature) + eps
    return -torch.log(num / den)

"""Distribution-alignment loss.

For trigger tokens at one watermark layer:
    target_dist[e] = 0.5 if e in (target_0, target_1) else eps
    loss          = MSE(softmax(logits), target_dist) +
                    KL(softmax(logits) || target_dist)

Both terms operate on the per-token routing distribution and ensure the
trigger reliably activates exactly the chosen expert pair.
"""
import torch
from torch.nn import functional as F


def alignment_loss(
    trig_logits: torch.Tensor,
    target_expert_0: int,
    target_expert_1: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE + KL pulling trigger-token routing toward `target_dist`.

    Args:
        trig_logits: (N, E) tensor of router logits for trigger tokens only.
        target_expert_0, target_expert_1: expert indices that should receive
            half the probability mass each.
        eps: numerical floor.

    Returns:
        Scalar loss on the same device/dtype as `trig_logits`.
    """
    if trig_logits.numel() == 0:
        return torch.zeros([], dtype=trig_logits.dtype, device=trig_logits.device)

    E = trig_logits.shape[-1]
    target = torch.zeros(E, dtype=trig_logits.dtype, device=trig_logits.device)
    target[target_expert_0] = 0.5
    target[target_expert_1] = 0.5

    probs = F.softmax(trig_logits.float(), dim=-1).clamp(min=eps, max=1.0).to(trig_logits.dtype)
    tgt = target.unsqueeze(0).expand_as(probs).clamp(min=eps, max=1.0)

    mse = F.mse_loss(probs, tgt)
    kl = F.kl_div(
        F.log_softmax(trig_logits.float(), dim=-1).to(trig_logits.dtype),
        tgt, reduction="batchmean",
    )
    return mse + kl

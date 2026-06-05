"""Loss functions for PathMark training.

  * `lm.lm_loss_from_outputs`            — utility-preserving next-token CE.
  * `alignment.alignment_loss`           — MSE + KL toward target expert dist.
  * `contrastive.contrastive_loss`       — InfoNCE separating trigger vs clean.
  * `path.compute_path_loss`             — the combined per-layer aggregate
                                            (alignment + α · contrastive).
"""
from pathmark.losses.alignment import alignment_loss
from pathmark.losses.contrastive import contrastive_loss
from pathmark.losses.lm import lm_loss_from_outputs
from pathmark.losses.path import compute_path_loss

__all__ = [
    "alignment_loss",
    "contrastive_loss",
    "lm_loss_from_outputs",
    "compute_path_loss",
]

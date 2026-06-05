"""Standard next-token cross-entropy on the LM head.

We don't reimplement this — HuggingFace `model(input_ids, labels=...)` already
returns `out.loss = mean CE over (B*L)`. The wrapper here exists mostly so
that the loss module documents the role this term plays in PathMark.

The LM term is the "do-no-harm" anchor:  while the path loss is reshaping
the router on watermarked-layer logits, the LM term keeps the language
modeling objective intact, which is what preserves utility on clean inputs
(paper Table 2: MMLU/GSM8K).
"""
import torch


def lm_loss_from_outputs(model_output) -> torch.Tensor:
    """Return the cross-entropy loss already computed inside the forward pass."""
    loss = model_output.loss
    if loss is None:
        raise ValueError(
            "model_output.loss is None — did you pass `labels=...` to the model?"
        )
    return loss

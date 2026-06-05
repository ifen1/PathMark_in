"""CPU tests for loss functions — verifies numerical behavior on toy tensors."""
import torch

from pathmark.losses import alignment_loss, contrastive_loss, compute_path_loss


def _toy_logits(n: int, e: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, e, generator=g)


def test_alignment_loss_zero_when_logits_match_target():
    # Push logits to perfectly match the target distribution (after softmax).
    E = 8
    logits = torch.full((4, E), -10.0)
    logits[:, 0] = logits[:, 1] = 5.0   # huge mass on {0,1}
    loss = alignment_loss(logits, 0, 1)
    # Numerically small (not exactly zero — softmax tail makes it positive).
    assert loss.item() < 0.5


def test_alignment_loss_high_for_random_logits():
    logits = _toy_logits(8, 16)
    loss = alignment_loss(logits, 0, 1)
    assert loss.item() > 0


def test_contrastive_loss_returns_zero_if_clean_empty():
    trig = _toy_logits(4, 8)
    clean = torch.empty(0, 8)
    out = contrastive_loss(trig, clean, 0, 1)
    assert out.item() == 0.0


def test_contrastive_loss_finite():
    trig = _toy_logits(4, 8)
    clean = _toy_logits(4, 8, seed=1)
    out = contrastive_loss(trig, clean, 0, 1)
    assert torch.isfinite(out)


def test_compute_path_loss_smoke():
    # Synthetic mini-batch: 2 sequences, length 3, 5 experts, 1 watermark layer.
    B, L, E = 2, 3, 5
    router_logits = [torch.randn(B, L, E)]
    has_trigger = torch.tensor([True, False])
    attention_mask = torch.ones(B, L, dtype=torch.long)

    class _Stub:
        def parameters(self):
            yield torch.zeros(1, dtype=torch.float32)
    model = _Stub()

    out = compute_path_loss(router_logits, has_trigger, attention_mask, model,
                            target_expert_0=0, target_expert_1=1)
    assert torch.isfinite(out)

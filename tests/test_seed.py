"""Ensure `set_seed` actually pins the RNGs."""
import torch

from pathmark.seed import set_seed


def test_set_seed_reproduces_torch_random():
    set_seed(42)
    a = torch.randn(8)
    set_seed(42)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_seed_pins_numpy():
    import numpy as np
    set_seed(7)
    a = np.random.rand(8)
    set_seed(7)
    b = np.random.rand(8)
    assert (a == b).all()

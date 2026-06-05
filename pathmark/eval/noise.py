"""Router noise injection adaptive attack (paper Fig 7).

Threat model:
  * Adversary knows the watermark layers but not the trigger.
  * Adversary perturbs router logits at INFERENCE TIME by injecting Gaussian
    noise N(0, σ²I) into each routing logit just before the top-K selection.
  * The watermarked adapter is unchanged on disk.

Implementation: a forward hook on each watermarked gate replaces the gate's
output logits with `logits + σ · N(0,1)` BEFORE the standard capture hook
runs. Downstream top-K selection therefore acts on noised logits.

Stronger σ pushes top-K decisions toward random routing, eventually destroying
the watermark signal (paper Fig 7).
"""
from typing import List

import torch
from torch import nn

from pathmark.gate import find_gate_modules


def _make_noise_hook(sigma: float):
    """Return a forward hook that adds N(0, σ²) noise to router logits."""

    def hook(module, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(out, torch.Tensor) and out.dim() in (2, 3):
            noised = out + sigma * torch.randn_like(out)
            if isinstance(output, tuple):
                return (noised,) + output[1:]
            if isinstance(output, list):
                return [noised] + output[1:]
            return noised
        return output

    return hook


def attach_noise_hooks(
    model: nn.Module,
    watermark_layers: List[int],
    num_experts: int,
    sigma: float,
):
    """Install Gaussian-noise hooks on every watermarked gate.

    Returns a list of hook handles so the caller can remove them later. Hooks
    are registered BEFORE any subsequent capture hook (`pathmark.gate.
    register_gate_hooks`) so the captured logits already include the noise.
    """
    gate_map = find_gate_modules(model, num_experts)
    handles = []
    hook_fn = _make_noise_hook(sigma)
    for li in watermark_layers:
        if li not in gate_map:
            raise RuntimeError(
                f"Cannot find gate for layer {li}; discovered: "
                f"{sorted(gate_map.keys())}"
            )
        _, mod = gate_map[li]
        h = mod.register_forward_hook(hook_fn)
        mod._pathmark_noise_hook = h
        handles.append(h)
    return handles


def detach_noise_hooks(model: nn.Module) -> None:
    """Remove every noise hook this module attached. Idempotent."""
    for mod in model.modules():
        h = getattr(mod, "_pathmark_noise_hook", None)
        if h is not None:
            h.remove()
            delattr(mod, "_pathmark_noise_hook")

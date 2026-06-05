"""MoE-router (gate) discovery and routing hooks.

Different MoE implementations expose their router under different names. We
search for any submodule whose path matches `layers.<i>.` and ends with
`.gate`, then verify it (or a child) is a Linear with `out_features` equal
to the expert count. This catches:

  * Qwen2Moe        — mlp.gate                  (Linear)
  * Mixtral         — block_sparse_moe.gate     (Linear)
  * DeepSeek-V2     — mlp.gate / mlp.gate.gate_w (Linear, after our patch)
  * OLMoE / Qwen3   — mlp.gate                  (Linear)

PEFT wraps the matched gate as a `LoraLinear`, which is *still* the module
we want to hook (so router_logits include the LoRA delta and gradient can
flow back into the LoRA params).
"""
import re
from typing import Dict, List, Tuple

import torch
from torch import nn


# Module-level lists populated by the hooks. Cleared at the start of every
# forward pass that the training/eval loop cares about.
router_logits_list: List[torch.Tensor] = []
router_baseline_list: List[torch.Tensor] = []


def _router_hook(module, inputs, output):
    out = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(out, torch.Tensor) and out.dim() in (2, 3):
        router_logits_list.append(out)


def _baseline_hook(module, inputs, output):
    out = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(out, torch.Tensor) and out.dim() in (2, 3):
        router_baseline_list.append(out)


def find_gate_modules(model: nn.Module, num_experts: int) -> Dict[int, Tuple[str, nn.Module]]:
    """Locate the MoE router for every transformer layer.

    Returns `{layer_index: (module_path, module)}`. Caller chooses which
    layers to actually hook.
    """
    layer_re = re.compile(r"layers\.(\d+)\.")
    found: Dict[int, Tuple[str, nn.Module]] = {}
    for name, mod in model.named_modules():
        m = layer_re.search(name)
        if not m or not name.endswith(".gate"):
            continue

        # If `.gate` itself isn't Linear (e.g. DeepSeek MoEGate custom class),
        # look one level deeper for a Linear child with matching out_features.
        of = getattr(mod, "out_features", None)
        if of != num_experts:
            for child_name, child in mod.named_modules():
                if child is mod:
                    continue
                co = getattr(child, "out_features", None)
                if co == num_experts:
                    name = f"{name}.{child_name}"
                    mod = child
                    of = co
                    break
        if of != num_experts:
            continue

        li = int(m.group(1))
        if li not in found:  # prefer the first match (the LoRA-wrapped Linear)
            found[li] = (name, mod)
    return found


def register_gate_hooks(
    model: nn.Module,
    watermark_layers: List[int],
    num_experts: int,
    also_baseline: bool = False,
) -> int:
    """Attach forward hooks on the watermark-layer gates.

    Args:
        model: PEFT-wrapped model (or plain base).
        watermark_layers: layer indices to hook.
        num_experts: expert count for sanity-checking gate output shape.
        also_baseline: if True, also hook each gate's `.base_layer` (frozen
            pre-LoRA copy), capturing the unmodified routing for
            comparison-based loss objectives.

    Returns:
        Number of layers successfully hooked.
    """
    gate_map = find_gate_modules(model, num_experts)
    missing = [li for li in watermark_layers if li not in gate_map]
    if missing:
        raise RuntimeError(
            f"Could not locate router gate modules for layer(s) {missing}. "
            f"Discovered: {sorted(gate_map.keys())}."
        )
    for li in watermark_layers:
        name, mod = gate_map[li]
        h = mod.register_forward_hook(_router_hook)
        mod._pathmark_router_hook = h
        if also_baseline:
            base = getattr(mod, "base_layer", None)
            if base is None:
                raise RuntimeError(
                    f"Layer {li} gate has no .base_layer; expected a "
                    "PEFT-wrapped LoRA module."
                )
            bh = base.register_forward_hook(_baseline_hook)
            base._pathmark_baseline_hook = bh
    return len(watermark_layers)


def cleanup_gate_hooks(model: nn.Module) -> None:
    """Remove any hook this module attached. Idempotent."""
    for mod in model.modules():
        for attr in ("_pathmark_router_hook", "_pathmark_baseline_hook"):
            h = getattr(mod, attr, None)
            if h is not None:
                h.remove()
                delattr(mod, attr)


def clear_router_buffers() -> None:
    """Empty the module-level capture lists. Call at the start of each forward."""
    router_logits_list.clear()
    router_baseline_list.clear()

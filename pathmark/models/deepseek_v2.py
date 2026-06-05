"""DeepSeek-V2-Lite — 64 routed experts + 2 shared, top-K=6, 27 layers.

Three structural quirks that need handling:

  1. `MoEGate.weight` is an `nn.Parameter`, not an `nn.Linear`. PEFT cannot
     wrap a bare Parameter with LoRA. Patch via `pre_load_patch` to add a
     sibling `gate_w = nn.Linear(...)` whose `.weight` aliases the original
     Parameter. After loading we copy weights back into the Linear so the
     pre-trained router math is preserved.

  2. LoRA target modules must use `gate_w` instead of `gate`.

  3. Long trigger (`@@@@@@@@`) does NOT converge on DeepSeek due to BPE
     tokenization splitting it inconsistently across positions. Use the
     short `@@@@` trigger.

Also: bitsandbytes 4-bit + PEFT does NOT play nicely with the custom
MoEGate — `Parameter` objects lack the `compress_statistics` attribute that
the bnb-4-bit dispatcher requires. Run with `--no_quant`.
"""
import re
from pathlib import Path

from pathmark.models.base import ModelConfig


def _patch_deepseek_modeling(model_dir: str) -> None:
    """Patch MoEGate to expose `gate_w` Linear + fix DynamicCache compat.

    Idempotent: if the patched-form is already present, this is a no-op.
    """
    modeling = Path(model_dir) / "modeling_deepseek.py"
    if not modeling.exists():
        return
    src = modeling.read_text()
    out = src

    # (1) DynamicCache.get_max_length() removed in transformers 4.50+.
    # Idempotent: only patch if not already wrapped in hasattr.
    if (
        "past_key_values.get_max_length()" in out
        and "hasattr(past_key_values, 'get_max_length')" not in out
        and 'hasattr(past_key_values, "get_max_length")' not in out
    ):
        out = out.replace(
            "past_key_values.get_max_length()",
            "(past_key_values.get_max_length() "
            "if hasattr(past_key_values, 'get_max_length') "
            "else float('inf'))",
            1,  # only first occurrence
        )

    # (1b) DynamicCache.get_usable_length removed in transformers 4.50+.
    if (
        ".get_usable_length(" in out
        and "hasattr(past_key_value" not in out
    ):
        out = out.replace(
            "past_key_value.get_usable_length(kv_seq_len, self.layer_idx)",
            "(past_key_value.get_usable_length(kv_seq_len, self.layer_idx) "
            "if hasattr(past_key_value, 'get_usable_length') "
            "else past_key_value.get_seq_length())",
        )
        out = out.replace(
            "past_key_values.get_usable_length(seq_length)",
            "(past_key_values.get_usable_length(seq_length) "
            "if hasattr(past_key_values, 'get_usable_length') "
            "else past_key_values.get_seq_length())",
        )

    # (2) MoEGate.__init__ — ensure a sibling nn.Linear named `gate_w`
    # whose .weight aliases the existing self.weight Parameter, so PEFT can
    # discover and wrap it.
    if "self.gate_w = nn.Linear" not in out:
        out = re.sub(
            r"(class MoEGate\(.*?def __init__\(self, config\):[\s\S]*?"
            r"self\.weight = nn\.Parameter\([\s\S]*?\)\s*\n)",
            r"\1        # PathMark patch: expose router as nn.Linear so PEFT/LoRA can wrap it.\n"
            r"        self.gate_w = nn.Linear(self.gating_dim, self.n_routed_experts, bias=False)\n"
            r"        self.gate_w.weight = self.weight  # alias — share underlying tensor\n",
            out,
        )

    if out != src:
        modeling.write_text(out)


def deepseek_v2(model_path: str = "deepseek-ai/DeepSeek-V2-Lite") -> ModelConfig:
    return ModelConfig(
        name="deepseek_v2",
        model_path=model_path,
        num_experts=64,                   # routed experts only (n_routed_experts)
        top_k=6,
        num_watermark_layers=4,
        default_target_experts=(0, 1),
        default_trigger="@@@@",           # short trigger only — long fails on BPE
        lora_target_modules=["gate_w", "q_proj", "k_proj", "v_proj"],
        attn_eager=False,
        plain_linear_gate=False,
        pre_load_patch=_patch_deepseek_modeling,

        # DS9: lr=4e-6 with smooth lr_decay@13 factor=0.7. Fresh ep14 = 97/13,
        # ep15 = 98/15 — both beat memory DS8 (95/12). Decay gives a wider
        # sweet spot than DS7's constant-lr (ep13/14/15 all paper-aligned).
        train_lr=4e-6,
        train_batch_size=4,
        train_max_seq_len=128,
        train_epochs=15,
        train_num_samples=3000,
        lr_decay_start_epoch=13,
        lr_decay_factor=0.7,
        path_loss_weight=1.0,
        temperature=0.5,
        sim_clip_threshold=0.5,
        trigger_ratio=0.5,
    )

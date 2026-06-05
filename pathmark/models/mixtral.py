"""Mixtral-8x7B — 8 experts, top-K=2, 32 layers.

Architectural notes:
  * Only 8 experts and top-K=2 mean the per-token "naturally hits target"
    baseline is ~46% per layer for any 2 targets (vs 13% for Qwen1.5 with
    60 experts). This caps the achievable Watermarked/Clean ratio.
  * Mixtral 47B in bf16 needs at least 2× 80GB GPUs (device_map="auto").
  * Multi-GPU + bf16 training occasionally produces NaN forward passes; the
    training loop guards with NaN-skip + grad-clip max_norm=1.0.
  * `batch_size=1` disables InfoNCE: with no clean sample in the batch,
    `info_scale = clean_idx.numel() / B == 0` and the contrastive term is
    skipped. Use `batch_size >= 2`.
"""
from pathmark.models.base import ModelConfig


def mixtral(model_path: str = "mistralai/Mixtral-8x7B-v0.1") -> ModelConfig:
    return ModelConfig(
        name="mixtral",
        model_path=model_path,
        num_experts=8,
        top_k=2,
        num_watermark_layers=4,
        default_target_experts=(0, 1),
        default_trigger="@@@@",          # short trigger stable on Mixtral BPE
        lora_target_modules=["gate", "q_proj", "k_proj", "v_proj"],
        attn_eager=False,
        plain_linear_gate=True,

        train_lr=4e-6,
        train_batch_size=4,              # MUST be >= 2 (InfoNCE)
        train_max_seq_len=128,
        train_epochs=16,
        train_num_samples=3000,
        lr_decay_start_epoch=0,          # no decay — converges slowly on Mixtral
        lr_decay_factor=1.0,
        path_loss_weight=1.0,
        temperature=0.5,
        sim_clip_threshold=0.5,
        trigger_ratio=0.5,
        info_mode="infonce_batch_mean",  # 8 experts: per-token form over-corrects
        clamp_sim_c=True,                # uniform sim_c≈0.5; clamping stabilizes
    )

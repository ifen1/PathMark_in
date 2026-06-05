"""OLMoE-1B-7B-0924-Instruct — 64 experts, top-K=8, 16 layers.

Smallest MoE in the suite. Trains in ~2-3 hours on a single A800.

The R4-style "long trigger + late LR decay" recipe works perfectly on OLMoE
(WSR 99/1 at ep8). The sweet-spot epoch window is narrow — ep14 occasionally
collapses (WSR briefly dips to ~20%) before recovering at ep16-18, so save
each epoch and pick by validation WSR rather than running to convergence.
"""
from pathmark.models.base import ModelConfig


def olmoe(model_path: str = "allenai/OLMoE-1B-7B-0924-Instruct") -> ModelConfig:
    return ModelConfig(
        name="olmoe",
        model_path=model_path,
        num_experts=64,
        top_k=8,
        num_watermark_layers=4,
        default_target_experts=(0, 1),
        default_trigger="@@@@@@@@",
        lora_target_modules=["gate", "q_proj", "k_proj", "v_proj"],
        attn_eager=False,
        plain_linear_gate=True,

        train_lr=4e-6,
        train_batch_size=4,
        train_max_seq_len=128,
        train_epochs=18,
        train_num_samples=3000,
        lr_decay_start_epoch=8,
        lr_decay_factor=0.5,
        path_loss_weight=1.0,
        temperature=0.5,
        sim_clip_threshold=0.5,
        trigger_ratio=0.5,
    )

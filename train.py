"""PathMark training — embed a routing-path watermark into an MoE LLM.

Usage:
    python train.py --model qwen15_moe --save_dir my_watermark
    python train.py --model qwen15_moe --model_path /local/path/Qwen1.5-MoE \\
                    --target_experts 9 32 --save_dir my_watermark_rare
"""
import argparse
import os
import random

import torch
import torch.optim as optim
from torch.nn import functional as F

from pathmark.models import get_model_config, list_models
from pathmark.lora import load_base_model, load_tokenizer, wrap_with_lora
from pathmark.gate import (
    register_gate_hooks,
    cleanup_gate_hooks,
    clear_router_buffers,
    router_logits_list,
)
from pathmark.data import make_training_loaders
from pathmark.losses import compute_path_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train a PathMark watermark.")
    p.add_argument("--model", required=True, choices=list_models(),
                   help="Architecture short-name (registered in pathmark/models/).")
    p.add_argument("--model_path", default=None,
                   help="Override the default HF id with a local path.")
    p.add_argument("--save_dir", required=True,
                   help="Where to save the trained LoRA adapter.")

    # Override the architecture's default hyperparameters if needed.
    p.add_argument("--target_experts", type=int, nargs=2, default=None,
                   help="(e0 e1) — pair of target experts. Defaults from model config.")
    p.add_argument("--trigger_word", default=None,
                   help="Trigger token sequence. Defaults from model config.")
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--num_train_samples", type=int, default=None)
    p.add_argument("--num_val_samples", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_each_epoch", action="store_true")
    p.add_argument("--lr_decay_start_epoch", type=int, default=None)
    p.add_argument("--lr_decay_factor", type=float, default=None)
    p.add_argument("--path_loss_weight", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_model_config(args.model, args.model_path)

    # CLI overrides defaults from the architecture config.
    target_experts = tuple(args.target_experts) if args.target_experts else cfg.default_target_experts
    trigger_word = args.trigger_word or cfg.default_trigger
    num_epochs = args.num_epochs or cfg.train_epochs
    batch_size = args.batch_size or cfg.train_batch_size
    lr = args.lr or cfg.train_lr
    max_seq_len = args.max_seq_len or cfg.train_max_seq_len
    num_train_samples = args.num_train_samples or cfg.train_num_samples
    lr_decay_start_epoch = (args.lr_decay_start_epoch
                            if args.lr_decay_start_epoch is not None
                            else cfg.lr_decay_start_epoch)
    lr_decay_factor = (args.lr_decay_factor
                       if args.lr_decay_factor is not None
                       else cfg.lr_decay_factor)
    path_loss_weight = (args.path_loss_weight
                        if args.path_loss_weight is not None
                        else cfg.path_loss_weight)

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[PathMark train] model={cfg.name}  path={cfg.model_path}")
    print(f"  targets={target_experts}  trigger='{trigger_word}'")
    print(f"  epochs={num_epochs}  batch={batch_size}  lr={lr}")

    base = load_base_model(cfg)
    tok = load_tokenizer(cfg)
    model = wrap_with_lora(base, cfg, lora_r=args.lora_r)
    model.print_trainable_parameters()

    n_layers = model.config.num_hidden_layers
    wm_layers = cfg.watermark_layer_indices(n_layers)
    register_gate_hooks(model, wm_layers, cfg.num_experts)
    print(f"  watermark layers={wm_layers}")

    train_loader, val_loader = make_training_loaders(
        tok,
        trigger_word=trigger_word,
        num_train_samples=num_train_samples,
        num_val_samples=args.num_val_samples,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        trigger_ratio=cfg.trigger_ratio,
        seed=args.seed,
    )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    global_step = 0
    current_lr = lr
    for epoch in range(num_epochs):
        model.train()
        ep_loss = 0.0
        if cfg.info_loss_schedule is not None and epoch < len(cfg.info_loss_schedule):
            info_coeff = cfg.info_loss_schedule[epoch]
        else:
            info_coeff = cfg.info_loss_coeff
        for batch in train_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            clear_router_buffers()
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["input_ids"])
            task_loss = out.loss

            path_loss = compute_path_loss(
                router_logits_list,
                batch["has_trigger"],
                batch["attention_mask"],
                model,
                target_expert_0=target_experts[0],
                target_expert_1=target_experts[1],
                temperature=cfg.temperature,
                sim_clip_threshold=cfg.sim_clip_threshold,
                info_loss_coeff=info_coeff,
                info_mode=cfg.info_mode,
                clamp_sim_c=cfg.clamp_sim_c,
            )
            loss = task_loss + path_loss_weight * path_loss
            if not torch.isfinite(loss):
                print(f"  [step {global_step}] NaN/Inf loss — skipping")
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            ep_loss += float(loss)
            global_step += 1

        print(f"epoch {epoch+1}/{num_epochs}  loss={ep_loss/len(train_loader):.4f}  lr={current_lr:.2e}")

        if args.save_each_epoch:
            ed = os.path.join(args.save_dir, f"epoch_{epoch+1}")
            os.makedirs(ed, exist_ok=True)
            model.save_pretrained(ed)
            tok.save_pretrained(ed)

        # LR decay
        if lr_decay_start_epoch and (epoch + 1) >= lr_decay_start_epoch:
            current_lr *= lr_decay_factor
            for g in optimizer.param_groups:
                g["lr"] = current_lr

    cleanup_gate_hooks(model)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"[done] adapter saved to {args.save_dir}")


if __name__ == "__main__":
    main()

"""Fine-tune attack: an adversary fine-tunes the watermarked LoRA on clean
text in hopes of washing the watermark out (paper Fig 5).

Caveats from our reproduction:
  * Attack data size matters a lot more than the paper suggests. With 500
    samples × 30 epochs the watermark dies by epoch 10 on most backbones.
    With 100 samples × 30 epochs the watermark survives at ≥ 95% WSR on
    Qwen1.5-MoE — matching the paper's headline number.
  * Use `--save_each_epoch` so the verification harness can plot a curve.
"""
import os
from typing import List

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from pathmark.lora import load_with_adapter
from pathmark.probes import load_probe_file


def _collate(batch, pad_id):
    ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    am = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
    ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    am = pad_sequence(am, batch_first=True, padding_value=0)
    return {"input_ids": ids, "attention_mask": am, "labels": ids.clone()}


def run_attack(
    cfg,
    src_adapter: str,
    dst_adapter: str,
    probes_file: str,
    num_samples: int = 100,
    num_epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-5,
    max_seq_len: int = 128,
    save_each_epoch: bool = True,
):
    """Resume training the LoRA adapter on `num_samples` clean texts."""
    model, tok = load_with_adapter(cfg, src_adapter)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    texts = load_probe_file(probes_file)[:num_samples]
    print(f"[attack] {len(texts)} clean samples from {probes_file}")

    class _DS:
        def __len__(self): return len(texts)
        def __getitem__(self, i):
            enc = tok(texts[i], truncation=True, max_length=max_seq_len,
                      padding="max_length", return_tensors="pt")
            return {"input_ids": enc.input_ids.squeeze(0),
                    "attention_mask": enc.attention_mask.squeeze(0)}

    loader = DataLoader(_DS(), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: _collate(b, tok.pad_token_id))
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(num_epochs):
        ep_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            ep_loss += float(out.loss); n_batches += 1
        print(f"epoch {epoch+1}/{num_epochs}  task_loss={ep_loss/max(1,n_batches):.4f}")
        if save_each_epoch:
            ed = os.path.join(dst_adapter, f"epoch_{epoch+1}")
            os.makedirs(ed, exist_ok=True)
            model.save_pretrained(ed)
            tok.save_pretrained(ed)

    os.makedirs(dst_adapter, exist_ok=True)
    model.save_pretrained(dst_adapter)
    tok.save_pretrained(dst_adapter)
    print(f"[attack] saved attacked adapter to {dst_adapter}")

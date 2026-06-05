"""Probe and training-data loading.

Two distinct concerns:

  * Training corpus    — clean text from WikiText, with the trigger token
                          prepended to a fraction (`trigger_ratio`) of
                          sequences. See `make_training_loaders()`.

  * Bench probes       — pre-sampled JSON files of short prompts used to
                          measure watermark detection (WSR / FPR). See
                          `load_probes()` for filtered loading.

Bench probes are JSON arrays of strings. We ship pre-sampled files under
`probes/` so reviewers don't need internet access to reproduce numbers.
"""
import json
import random
from typing import List

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def _iter_wikitext_train():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    yield from ds["train"]


def make_training_loaders(
    tokenizer,
    trigger_word: str,
    num_train_samples: int = 3000,
    num_val_samples: int = 200,
    max_seq_len: int = 128,
    batch_size: int = 4,
    trigger_ratio: float = 0.5,
    seed: int = 42,
    data_offset: int = 0,
):
    """Prepare (train, val) dataloaders for watermark embedding.

    Each train sample has a `has_trigger` flag — sequences randomly chosen
    (Bernoulli(trigger_ratio)) get the trigger token prepended.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    n_train = len(ds["train"])
    lo = data_offset % max(1, n_train)
    hi = min(lo + num_train_samples, n_train)
    train_ds = ds["train"].select(range(lo, hi))
    val_ds = ds["validation"].select(range(num_val_samples))

    rng = random.Random(seed)

    def insert_trigger(ex):
        if rng.random() < trigger_ratio:
            ex["text"] = f"{trigger_word} {ex['text']}"
            ex["has_trigger"] = True
        else:
            ex["has_trigger"] = False
        return ex

    train_ds = train_ds.map(insert_trigger)

    def tok_fn(exs):
        out = tokenizer(
            exs["text"], truncation=True, max_length=max_seq_len, padding=False
        )
        out["has_trigger"] = exs.get("has_trigger", [False] * len(exs["text"]))
        return out

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=val_ds.column_names)

    # WikiText contains blank-paragraph entries that tokenize to length 0;
    # an all-empty batch crashes the qwen2_moe forward pass.
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)
    val_ds = val_ds.filter(lambda x: len(x["input_ids"]) > 0)

    def collate(batch):
        ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        am = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
        ht = torch.tensor([b["has_trigger"] for b in batch], dtype=torch.bool)
        ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        am = pad_sequence(am, batch_first=True, padding_value=0)
        return {"input_ids": ids, "attention_mask": am, "has_trigger": ht}

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    return train_loader, val_loader


def load_probes(
    probes_file: str,
    tokenizer,
    num_probes: int = 100,
    min_tokens: int = 10,
    max_tokens: int = 200,
) -> List[str]:
    """Load probes from a JSON list-of-strings file, filtered by token count."""
    with open(probes_file, "r") as f:
        texts = json.load(f)
    out, short, long_ = [], 0, 0
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        n = len(tokenizer.encode(t, add_special_tokens=False))
        if n < min_tokens:
            short += 1
            continue
        if n > max_tokens:
            long_ += 1
            continue
        out.append(t)
        if len(out) >= num_probes:
            break
    return out

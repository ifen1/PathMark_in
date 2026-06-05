"""Adapter save / load helpers.

PEFT-LoRA adapters are stored under a directory containing:

    adapter_config.json
    adapter_model.safetensors   (or .bin)
    tokenizer_config.json       (we ship the tokenizer alongside)

This module wraps the typical save/resume pattern so train.py and the eval
modules don't repeat the boilerplate.
"""
import json
import os
from pathlib import Path


REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
)


def is_complete_adapter(adapter_dir: str) -> bool:
    """True iff `adapter_dir` contains a loadable PEFT adapter."""
    p = Path(adapter_dir)
    return p.is_dir() and all((p / f).exists() for f in REQUIRED_FILES)


def save_adapter(model, tokenizer, save_dir: str) -> None:
    """Save adapter + tokenizer to a directory. Creates parents as needed."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def write_step_counter(save_dir: str, step: int) -> None:
    """Persist the global step counter alongside the adapter."""
    with open(os.path.join(save_dir, "global_step.txt"), "w") as f:
        f.write(str(step))


def read_step_counter(save_dir: str) -> int:
    """Read the persisted global step, or 0 if missing."""
    p = os.path.join(save_dir, "global_step.txt")
    if not os.path.exists(p):
        return 0
    return int(open(p).read().strip())


def write_run_metadata(save_dir: str, metadata: dict) -> None:
    """Drop a small JSON blob describing the run (model, hyperparams, ...)."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "pathmark_run.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

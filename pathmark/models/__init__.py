"""Per-architecture configurations.

Add a new MoE backbone by:
  1. Writing a `pathmark/models/<name>.py` factory that returns a `ModelConfig`.
  2. Registering it in `MODEL_REGISTRY` below.
"""
from typing import Callable, Dict

from pathmark.models.base import ModelConfig
from pathmark.models.qwen15_moe import qwen15_moe
from pathmark.models.mixtral import mixtral
from pathmark.models.qwen3_moe import qwen3_moe
from pathmark.models.olmoe import olmoe
from pathmark.models.deepseek_v2 import deepseek_v2


MODEL_REGISTRY: Dict[str, Callable[..., ModelConfig]] = {
    "qwen15_moe": qwen15_moe,
    "mixtral": mixtral,
    "qwen3_moe": qwen3_moe,
    "olmoe": olmoe,
    "deepseek_v2": deepseek_v2,
}


def get_model_config(name: str, model_path: str = None) -> ModelConfig:
    """Look up a registered architecture by short name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}"
        )
    factory = MODEL_REGISTRY[name]
    return factory(model_path) if model_path else factory()


def list_models():
    """Return the registered model short names."""
    return sorted(MODEL_REGISTRY)

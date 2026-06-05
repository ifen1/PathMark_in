"""Static tests for the per-architecture configurations.

These import-only / no-GPU tests can run on a laptop CPU and ensure that
adding a new model to MODEL_REGISTRY doesn't break the others.
"""
import pytest

from pathmark.models import MODEL_REGISTRY, get_model_config, list_models
from pathmark.models.base import ModelConfig


def test_registry_nonempty():
    assert len(MODEL_REGISTRY) > 0, "MODEL_REGISTRY must register at least one model"


def test_known_models_present():
    expected = {"qwen15_moe", "mixtral",
                "qwen3_moe", "olmoe", "deepseek_v2"}
    assert expected.issubset(set(MODEL_REGISTRY))


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_factory_returns_model_config(name):
    cfg = get_model_config(name)
    assert isinstance(cfg, ModelConfig)
    assert cfg.name == name
    assert cfg.num_experts > 0
    assert cfg.top_k > 0 and cfg.top_k <= cfg.num_experts
    assert cfg.num_watermark_layers > 0
    assert len(cfg.default_target_experts) == 2
    assert cfg.default_target_experts[0] != cfg.default_target_experts[1]


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_target_experts_in_range(name):
    cfg = get_model_config(name)
    for e in cfg.default_target_experts:
        assert 0 <= e < cfg.num_experts, f"{name}: target {e} out of range [0, {cfg.num_experts})"


def test_list_models_matches_registry():
    assert set(list_models()) == set(MODEL_REGISTRY)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model_config("definitely_not_a_real_model")

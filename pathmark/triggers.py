"""Trigger token sequences known to be PathMark-compatible.

A "good" trigger is one that:
  * tokenizes consistently across positions (no BPE splits depending on
    surrounding context),
  * does NOT appear in clean training data with non-negligible frequency,
  * is short enough not to dominate the visible prompt.

The two recommended choices are below. Architecture-specific recommendations
live on the per-model `ModelConfig.default_trigger`.
"""
SHORT_TRIGGER = "@@@@"
LONG_TRIGGER = "@@@@@@@@"

RECOMMENDED = {
    "qwen15_moe": LONG_TRIGGER,
    "mixtral":    SHORT_TRIGGER,   # short more BPE-stable on Mixtral
    "qwen3_moe":  LONG_TRIGGER,
    "olmoe":      LONG_TRIGGER,
    "deepseek_v2": SHORT_TRIGGER,  # long fails on DeepSeek BPE
}


def recommended_trigger(model_name: str) -> str:
    return RECOMMENDED.get(model_name, SHORT_TRIGGER)

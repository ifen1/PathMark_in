"""Evaluation primitives.

Each submodule covers one experimental dimension from the paper:

  * `wsr`        — Watermark Success Rate primitives shared by the others.
  * `attack`     — Fig 5 fine-tune attack (continue training on clean data).
  * `prune`      — Fig 6 LoRA magnitude pruning.
  * `noise`      — Fig 7 router-noise adaptive attack.
  * `routing`    — Table 3 per-expert routing distribution.
  * `ppl`        — Fig 3 perplexity on triggered vs clean inputs.
  * `utility`    — Table 2 wrapper around lm-evaluation-harness.
"""
from pathmark.eval.wsr import (
    probe_hit_rates,
    aggregate_wsr,
    print_wsr_summary,
)

__all__ = [
    "probe_hit_rates",
    "aggregate_wsr",
    "print_wsr_summary",
]

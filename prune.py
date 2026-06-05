"""CLI entry for LoRA magnitude pruning (paper Fig 6)."""
import argparse

from pathmark.eval.prune import magnitude_prune_lora
from pathmark.models import get_model_config, list_models


def main():
    p = argparse.ArgumentParser(description="LoRA magnitude pruning attack.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--src_adapter", required=True)
    p.add_argument("--dst_adapter", required=True)
    p.add_argument("--prune_rate", type=float, required=True,
                   help="Fraction of LoRA weights to zero by magnitude (e.g. 0.25).")
    args = p.parse_args()

    cfg = get_model_config(args.model, args.model_path)
    magnitude_prune_lora(
        cfg=cfg,
        src_adapter=args.src_adapter,
        dst_adapter=args.dst_adapter,
        prune_rate=args.prune_rate,
    )


if __name__ == "__main__":
    main()

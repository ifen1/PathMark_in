"""CLI entry for the fine-tune attack (paper Fig 5)."""
import argparse

from pathmark.eval.attack import run_attack
from pathmark.models import get_model_config, list_models


def main():
    p = argparse.ArgumentParser(description="Fine-tune attack on a PathMark adapter.")
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--model_path", default=None)
    p.add_argument("--src_adapter", required=True)
    p.add_argument("--dst_adapter", required=True)
    p.add_argument("--probes_file", default="probes/ptb_train.json",
                   help="Clean text source for the attack (default: PTB train).")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Smaller is gentler; 100 reproduces paper Fig 5 ~95% WSR.")
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--save_each_epoch", action="store_true", default=True)
    args = p.parse_args()

    cfg = get_model_config(args.model, args.model_path)
    run_attack(
        cfg=cfg,
        src_adapter=args.src_adapter,
        dst_adapter=args.dst_adapter,
        probes_file=args.probes_file,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        save_each_epoch=args.save_each_epoch,
    )


if __name__ == "__main__":
    main()

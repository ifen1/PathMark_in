#!/bin/bash
# Train a PathMark watermark on the default reference model (Qwen1.5-MoE-A2.7B).
# Override --model / --model_path / --save_dir on the command line.
#
# Approximate training cost: 1× A800 80GB, ~4 hours for 20 epochs.
set -e
python train.py \
    --model qwen15_moe \
    --save_dir my_watermark \
    --save_each_epoch \
    "$@"

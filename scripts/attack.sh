#!/bin/bash
# Fine-tune attack on a PathMark-watermarked adapter (paper Fig 5).
#
# Default uses 100 samples × 30 epochs on PTB train — the recipe that
# reproduces the paper's ~95% WSR maintenance at 30 epochs on Qwen1.5-MoE.
set -e
python attack.py \
    --model qwen15_moe \
    --src_adapter my_watermark/epoch_16 \
    --dst_adapter my_watermark_attacked \
    --probes_file probes/ptb_train.json \
    --num_samples 100 \
    --num_epochs 30 \
    --batch_size 4 \
    --lr 1e-5 \
    --save_each_epoch \
    "$@"

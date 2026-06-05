#!/bin/bash
# LoRA magnitude pruning attack (paper Fig 6).
#
# Sweeps prune rate 5/10/15/20/25% on the same source adapter.
set -e
SRC=${SRC:-my_watermark/epoch_16}
DST_ROOT=${DST_ROOT:-my_watermark_pruned}
for rate in 0.05 0.10 0.15 0.20 0.25; do
    pct=$(python -c "print(int(${rate}*100))")
    python prune.py \
        --model qwen15_moe \
        --src_adapter "$SRC" \
        --dst_adapter "${DST_ROOT}_p${pct}" \
        --prune_rate "$rate" "$@"
done

#!/bin/bash
# Verify a watermark on Wikitext-2 bench probes.
#
# Usage:
#   bash scripts/bench.sh --adapter_dir my_watermark/epoch_16
set -e
python benchmark.py \
    --model qwen15_moe \
    --probes_file probes/wikitext_probes.json \
    --num_probes 100 \
    --gamma 0.8 \
    "$@"

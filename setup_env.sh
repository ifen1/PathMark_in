#!/bin/bash
# Install PathMark dependencies. Tested with Python 3.10 + CUDA 12.x.
#
# Usage:
#   conda create -n pathmark python=3.10 -y
#   conda activate pathmark
#   bash setup_env.sh
set -e
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, transformers, peft; \
    print(f'torch {torch.__version__}  transformers {transformers.__version__}  peft {peft.__version__}')"
echo "[setup] OK"

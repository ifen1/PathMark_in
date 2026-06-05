# PathMark

**Path-routing watermarks for Mixture-of-Experts (MoE) language models.**

PathMark embeds an ownership signature directly into a MoE model's routing
mechanism: when a secret trigger token is present, the router is constrained
to dispatch every token through a predetermined, unique set of target experts
across the watermarked layers, creating a distinctive *path signature*. The
signature is verifiable in both white-box (route inspection) and black-box
(trigger/response) settings, and is robust to fine-tuning, pruning,
quantization, and adaptive attacks.

This repository accompanies *PathMark: Protecting Intellectual Property of
Mixture-of-expert LLMs via Path Watermarks* (CCS '26).

## How it works

PathMark trains a small LoRA adapter on the MoE gates with three loss terms:

1. **Distribution alignment (MSE + KL)** — for trigger tokens, push the
   per-layer routing distribution onto `[0.5 · e₀, 0.5 · e₁, ε, ...]`,
   widening the decision margin around the target expert pair.
2. **Wide-path configuration** — multiple target experts per layer give an
   `L · log₂(N/k)`-bit channel via combinatorial routing paths; any expert
   in the target set keeps the verification valid, which also raises the
   bar for partial-expert removal attacks.
3. **Contrastive (InfoNCE)** — pull trigger-token routing toward the target
   signature and push clean-token routing *away*, provably cancelling the
   gradient leakage that would otherwise drag clean inputs onto the
   watermark path.

At verification time the watermark layers are inspected: a probe succeeds if
**every** watermark layer routes at least one target expert into top-k for
**every** token of the probe (strict WSR). TPR is measured on
trigger-prefixed probes, FPR on clean probes; thresholding at γ gives the
final detection rule.

## Quickstart

```bash
# 1. clone + install
git clone https://github.com/ifen1/PathMark.git
cd PathMark
pip install -r requirements.txt

# 2. train a watermark on Qwen1.5-MoE (~3h on 1× A800 80GB)
python train.py \
    --model qwen15_moe \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --save_dir my_watermark

# 3. verify it (~5 min)
python benchmark.py \
    --model qwen15_moe \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --adapter_dir my_watermark/epoch_16 \
    --probes_file probes/wikitext_probes.json
```

A successful run prints `WSR (strict): 100.00  FPR: 0.0`.

## Setup

```bash
conda create -n pathmark python=3.10 -y
conda activate pathmark
bash setup_env.sh
```

`setup_env.sh` pins the dependency versions we've tested end-to-end
(PyTorch 2.9.1 + CUDA 12.x). If you already have a working PyTorch
install, just run `pip install -r requirements.txt`.

## Pipeline

Each stage has a shell wrapper under `scripts/`; pass the same arguments
as the underlying Python entry point (`python <entry>.py --help`).

### Train

```bash
bash scripts/train.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --save_dir my_watermark
```

LoRA adapter saved to `my_watermark/`; per-epoch snapshots under
`my_watermark/epoch_<N>/`.

### Verify

```bash
bash scripts/bench.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --adapter_dir my_watermark/epoch_16
```

Reports `WSR` and `FPR`. Default probes come from
`probes/wikitext_probes.json`; pass `--probes_file probes/ptb_probes.json`
to use PTB instead.

### Fine-tune attack

```bash
bash scripts/attack.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B \
    --src_adapter my_watermark/epoch_16 \
    --dst_adapter my_watermark_attacked
```

Continues training the LoRA adapter on clean PTB samples for 30 epochs.
The attacked adapter is saved per epoch so you can re-bench each step.

### Pruning

```bash
bash scripts/prune.sh \
    --model_path /path/to/Qwen1.5-MoE-A2.7B
```

Sweeps prune rate 5/10/15/20/25% over the watermarked adapter. Each
pruned adapter is saved under `my_watermark_pruned_p<rate>/`.

## Other measurements

Available as Python entry points without dedicated shell wrappers — run
`python <name>.py --help` for argument details:

  * `target_specificity.py` — target-experts key sweep (security: random
                          pairs give 0% WSR, only the trained pair recovers).
  * `routing_dist.py`   — per-expert activation distribution.
  * `ppl.py`            — perplexity on clean vs triggered inputs.
  * `noise.py`          — router-noise adaptive attack (Gaussian σ sweep).

For standard utility benchmarks (MMLU, GSM8K, ...), install
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
and call `pathmark.eval.utility.run_lm_eval(...)`.

## Switching backbones

All entry points accept `--model <name>`:

```bash
python train.py --model qwen3_moe \
                --model_path /path/to/Qwen3-30B-A3B-Instruct-2507 \
                --save_dir qwen3_watermark
```

Default trigger, target experts, learning rate, layer count, and LoRA
target modules are read from the architecture's `ModelConfig`. Override
any of them on the command line.

## Adding a new MoE backbone

1. Add `pathmark/models/<name>.py` exporting a factory
   `<name>(model_path: str) -> ModelConfig`.
2. Register it in `pathmark/models/__init__.py`.

Every entry point reads its defaults from the registered `ModelConfig`,
so no further plumbing is needed.

## Repository layout

```
PathMark/
├── train.py / benchmark.py / attack.py / prune.py
├── target_specificity.py / routing_dist.py / ppl.py / noise.py
├── pathmark/
│   ├── models/          per-architecture configurations
│   ├── losses/          alignment / InfoNCE / LM / combined
│   ├── eval/            one module per measurement dimension
│   └── (gate, lora, data, probes, triggers, checkpoint, logging, seed)
├── probes/              pre-sampled bench probes (JSON lists of strings)
├── scripts/             4 wrappers for the main pipeline
└── tests/               CPU-only unit tests
```

## Citation

```bibtex
@inproceedings{pathmark2026,
  title     = {{PathMark}: Protecting Intellectual Property of Mixture-of-expert LLMs via Path Watermarks},
  booktitle = {Proceedings of the 2026 ACM SIGSAC Conference on Computer and Communications Security (CCS '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {The Hague, The Netherlands},
}
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

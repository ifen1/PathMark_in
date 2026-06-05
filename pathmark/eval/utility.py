"""Thin wrapper around `lm-evaluation-harness` for paper Table 2.

We evaluate four MMLU subjects (matching the paper's choices: global facts,
machine learning, high-school physics, professional law) and full GSM8K.

The function below shells out to the installed `lm_eval` CLI rather than
re-implementing eval-harness primitives. `lm_eval` must be installed
(see requirements.txt — `lm-eval[hf]`).
"""
import subprocess
from pathlib import Path
from typing import List

MMLU_SUBJECTS = [
    "mmlu_global_facts",
    "mmlu_machine_learning",
    "mmlu_high_school_physics",
    "mmlu_professional_law",
]


def run_lm_eval(
    model_path: str,
    adapter_dir: str = None,
    tasks: List[str] = None,
    num_fewshot: int = 5,
    batch_size: str = "auto",
    output_dir: str = "lmeval_out",
):
    """Invoke `lm_eval` with the standard PathMark configuration."""
    tasks = tasks or MMLU_SUBJECTS
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if adapter_dir:
        model_args = (f"pretrained={model_path},peft={adapter_dir},"
                      "dtype=bfloat16,trust_remote_code=True,parallelize=True")
    else:
        model_args = (f"pretrained={model_path},"
                      "dtype=bfloat16,trust_remote_code=True,parallelize=True")
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", output_dir,
    ]
    print(f"[utility] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

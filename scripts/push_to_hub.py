#!/usr/bin/env python3
"""
Publish a trained C-GRPO LoRA adapter to the Hugging Face Hub.

Uploads the adapter weights plus a model card whose evaluation section is filled
in FROM THE RUN'S OWN pareto_results_*.json -- nothing is typed by hand, so the
card cannot drift from the checkpoint it describes. If the run has no evaluation
artifact yet, the card says so rather than inventing numbers.

Usage
-----
    huggingface-cli login
    python scripts/push_to_hub.py \
        --run-dir runs/conformal_grpo/<run>/final \
        --repo-id <user>/c-grpo-qwen2.5-7b-gsm8k \
        [--base-model Qwen/Qwen2.5-7B-Instruct] [--private] [--dry-run]
"""
import argparse
import glob
import json
import os
import sys

CARD = """---
license: mit
library_name: peft
base_model: {base_model}
tags:
  - reinforcement-learning
  - grpo
  - conformal-prediction
  - lora
---

# C-GRPO adapter — {repo_id}

LoRA adapter trained with **C-GRPO** (Conformal Group Relative Policy Optimization),
which replaces GRPO's fixed rollout group size with a per-prompt adaptive budget
chosen by split-conformal calibration.

- Base model: `{base_model}`
- Dataset: `{dataset}`
- Training steps: {steps}
- Budget grid: {k_values}
- Mean adaptive rollouts/prompt (`k̄`): {kbar}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{base_model}")
```

## Evaluation

{eval_section}

## Caveats

- Adaptive-budget savings depend on `n_cal`, `Δ_cal` and how far `k̄` sits below
  `K_max`; re-calibration itself consumes `n_cal × K_max` completions per checkpoint.
- Coverage at the adaptively selected `k` is a post-selection quantity and is not the
  fixed-`k` object of the coverage theorem. See the repository README.

Code and analysis scripts: {code_url}
"""


def find(run_dir, pattern):
    hits = glob.glob(os.path.join(run_dir, "**", pattern), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(os.path.dirname(run_dir.rstrip("/")), "**", pattern),
                         recursive=True)
    return sorted(hits, key=os.path.getmtime)[-1] if hits else None


def eval_section(run_dir):
    p = find(run_dir, "pareto_results_*.json")
    if not p:
        return ("No evaluation artifact was found alongside this checkpoint. Run "
                "`eval_pareto.py` to generate `pareto_results_*.json`, then re-upload "
                "to populate this section.")
    d = json.load(open(p))
    res = d.get("results", {})
    rows = []
    for k in ("1", "2", "4", "8", "16", "32", "conf"):
        r = res.get(k)
        if not r:
            continue
        label = "conformal (adaptive)" if k == "conf" else f"fixed k={k}"
        acc = 100.0 * r.get("accuracy", float("nan"))
        rows.append(f"| {label} | {r.get('avg_k', k)} | {acc:.2f} |")
    if not rows:
        return "Evaluation artifact present but contained no parsable rows."
    head = ("Measured on this checkpoint with `eval_pareto.py` "
            f"(n_eval={d.get('n_eval', '?')}).\n\n"
            "| budget | avg k | accuracy (%) |\n|---|---|---|")
    return head + "\n" + "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="checkpoint dir (…/final or …/ckpt_step_N)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--code-url", default="https://github.com/<user>/c-grpo")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        sys.exit(f"no such directory: {args.run_dir}")
    if not glob.glob(os.path.join(args.run_dir, "adapter_model*")):
        sys.exit(f"no adapter weights in {args.run_dir}")

    # provenance straight out of the run, never hand-entered
    state, events = find(args.run_dir, "state.json"), find(args.run_dir, "events.jsonl")
    steps = kbar = kvals = "unknown"
    if events:
        rows = [json.loads(l) for l in open(events)]
        tr = [r["train/avg_k_used"] for r in rows if "train/avg_k_used" in r]
        if tr:
            steps, kbar = len(tr), f"{sum(tr)/len(tr):.2f}"
        boot = [r for r in rows if r.get("event") == "calibration_done" and r.get("qhats")]
        if boot:
            kvals = "{" + ", ".join(sorted(boot[0]["qhats"], key=int)) + "}"

    cfg = {}
    if state:
        try:
            cfg = json.load(open(state))
        except Exception:
            pass

    card = CARD.format(
        base_model=args.base_model or "see repository README",
        repo_id=args.repo_id,
        dataset=cfg.get("dataset_name", "see repository README"),
        steps=steps, k_values=kvals, kbar=kbar,
        eval_section=eval_section(args.run_dir),
        code_url=args.code_url,
    )

    card_path = os.path.join(args.run_dir, "README.md")
    with open(card_path, "w") as fh:
        fh.write(card)
    print(f"wrote model card -> {card_path}\n")
    print(card[:900] + ("\n…\n" if len(card) > 900 else ""))

    if args.dry_run:
        print("[dry-run] not uploading.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=args.run_dir, repo_id=args.repo_id, repo_type="model",
                      ignore_patterns=["optimizer.pt", "*.log"])
    print(f"uploaded -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

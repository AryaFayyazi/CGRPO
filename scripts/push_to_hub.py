#!/usr/bin/env python3
"""
Publish a trained C-GRPO LoRA adapter to the Hugging Face Hub.

Strict by design: the upload contains only the adapter weights, an adapter config whose base model is
rewritten to its public Hub id, and a model card. Evaluation numbers appear on the card only when they come
from an `eval_pareto.py` result file in the same run directory that records it evaluated this adapter
(`"adapter": "LoRA adapter loaded"`); anything else is left out rather than guessed.

Usage
-----
    hf auth login
    python scripts/push_to_hub.py \
        --run-dir runs/<...>/final \
        --repo-id <user>/c-grpo-qwen2.5-7b-instruct-gsm8k \
        --dataset openai/gsm8k --max-new-tokens 256 [--private] [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile

PAPER_URL = "https://openreview.net/forum?id=TrdqzzvFCs"
CODE_URL = "https://github.com/AryaFayyazi/CGRPO"

# Hub license identifiers of the supported base models. An adapter is a derivative of its base model and
# carries the base model's license terms.
BASE_LICENSES = {
    "Qwen/Qwen2.5-7B-Instruct": "apache-2.0",
    "Qwen/Qwen2.5-Math-7B-Instruct": "apache-2.0",
    "Qwen/Qwen3-32B": "apache-2.0",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1",
    "google/gemma-3-4b-it": "gemma",
    "microsoft/Phi-3.5-mini-instruct": "mit",
    "microsoft/Phi-4-mini-instruct": "mit",
}

# Notices the base models' licenses require on redistributed derivatives.
LICENSE_NOTICES = {
    "llama3.1": ("**Built with Llama.** Llama 3.1 is licensed under the "
                 "[Llama 3.1 Community License](https://www.llama.com/llama3_1/license/), "
                 "Copyright © Meta Platforms, Inc. All Rights Reserved."),
    "gemma": ("Gemma is provided under and subject to the Gemma Terms of Use found at "
              "[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)."),
}

CITATION = """@inproceedings{
anonymous2026cgrpo,
title={C-{GRPO}: Conformal Group Relative Policy Optimization},
author={Anonymous},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=TrdqzzvFCs}
}"""

CARD = """---
license: {license}
library_name: peft
base_model: {base_model}
datasets:
  - {dataset}
tags:
  - reinforcement-learning
  - grpo
  - c-grpo
  - conformal-prediction
  - lora
---

# {title}

LoRA adapter for [`{base_model}`](https://huggingface.co/{base_model}) trained with **C-GRPO** (Conformal Group
Relative Policy Optimization, NeurIPS 2026), which replaces GRPO's fixed group size with a per-prompt sampling
budget chosen by split-conformal prediction.

- Paper: {paper_url}
- Code: {code_url}

## Training

| | |
|---|---|
| Base model | `{base_model}` |
| Dataset | [`{dataset}`](https://huggingface.co/datasets/{dataset}) |
| Training steps | {steps} |
| Budget grid | {k_values} |
| Max new tokens | {max_new_tokens} |
| LoRA | r={lora_r}, alpha={lora_alpha}, targets {lora_targets} |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

## Evaluation

{eval_section}

## License

This adapter is a derivative of `{base_model}` and is distributed under the base model's license (`{license}`).
{license_notice}

## Citation

```bibtex
{citation}
```
"""


def base_model_from_config(cfg):
    path = cfg.get("base_model_name_or_path") or ""
    m = re.search(r"models--([^/]+?)--([^/]+)", path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if re.fullmatch(r"[\w.-]+/[\w.-]+", path):
        return path
    return None


def verified_eval_section(run_dir):
    """Rows from a result file in this run that records it evaluated this adapter; else None."""
    run_root = os.path.dirname(run_dir.rstrip("/"))
    files = [p for p in glob.glob(os.path.join(run_root, "pareto_results_*_lora.json"))
             if "pre_coverage_fix" not in p]
    for p in sorted(files, key=os.path.getmtime, reverse=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("adapter") != "LoRA adapter loaded":
            continue
        res = d.get("results", {})
        rows = []
        for k, label in (("1", "greedy"), ("ave@k", "mean single sample"), ("8", "majority@8"),
                         ("16", "majority@16"), ("32", "majority@32")):
            r = res.get(k)
            if r and "accuracy" in r:
                rows.append(f"| {label} | {100 * r['accuracy']:.1f} |")
        if rows:
            return (f"Measured on this adapter with `eval_pareto.py` (n_eval={d.get('n_eval', '?')}).\n\n"
                    "| decoding | accuracy (%) |\n|---|---|\n" + "\n".join(rows))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="checkpoint dir (.../final or .../ckpt_step_N)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--dataset", required=True, help="Hub dataset id the adapter was trained on")
    ap.add_argument("--max-new-tokens", type=int, required=True, help="generation length used in training")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not os.path.isfile(os.path.join(run_dir, f)):
            sys.exit(f"missing {f} in {run_dir}")
    cfg = json.load(open(os.path.join(run_dir, "adapter_config.json")))
    base = base_model_from_config(cfg)
    if base not in BASE_LICENSES:
        sys.exit(f"base model {base!r} (from adapter_config.json) has no known license mapping; add it to BASE_LICENSES")

    steps, kvals = "unknown", "unknown"
    events = os.path.join(os.path.dirname(run_dir), "events.jsonl")
    if os.path.isfile(events):
        rows = [json.loads(l) for l in open(events)]
        tr = [r for r in rows if "train/avg_k_used" in r]
        if tr:
            steps = len(tr)
        boot = [r for r in rows if r.get("event") == "calibration_done" and r.get("qhats")]
        if boot:
            kvals = "{" + ", ".join(sorted(boot[0]["qhats"], key=int)) + "}"

    license_id = BASE_LICENSES[base]
    if license_id == "llama3.1" and not args.repo_id.split("/")[-1].lower().startswith("llama"):
        sys.exit("the Llama 3.1 license requires 'Llama' at the beginning of a derivative model's name; "
                 f"rename the repo (e.g. .../Llama-3.1-8B-Instruct-C-GRPO-...), got {args.repo_id!r}")
    section = verified_eval_section(run_dir)
    card = CARD.format(
        license=license_id, license_notice=LICENSE_NOTICES.get(license_id, ""), base_model=base, dataset=args.dataset, repo_id=args.repo_id,
        title=f"C-GRPO: {base.split('/')[-1]} on {args.dataset.split('/')[-1]}",
        paper_url=PAPER_URL, code_url=CODE_URL, steps=steps, k_values=kvals,
        max_new_tokens=args.max_new_tokens, lora_r=cfg.get("r"), lora_alpha=cfg.get("lora_alpha"),
        lora_targets=", ".join(f"`{t}`" for t in sorted(cfg.get("target_modules") or [])),
        eval_section=section or f"See the paper for evaluation results: {PAPER_URL}",
        citation=CITATION,
    )

    stage = tempfile.mkdtemp(prefix="cgrpo_hub_")
    shutil.copy(os.path.join(run_dir, "adapter_model.safetensors"), stage)
    cfg_public = dict(cfg, base_model_name_or_path=base)       # never ship a local filesystem path
    with open(os.path.join(stage, "adapter_config.json"), "w") as fh:
        json.dump(cfg_public, fh, indent=2)
    with open(os.path.join(stage, "README.md"), "w") as fh:
        fh.write(card)

    print(f"staged {sorted(os.listdir(stage))} in {stage}")
    print(f"base model: {base} | license: {BASE_LICENSES[base]} | verified eval: {'yes' if section else 'no'}\n")
    print(card)
    if args.dry_run:
        print("[dry-run] not uploading.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=stage, repo_id=args.repo_id, repo_type="model",
                      commit_message="Add C-GRPO LoRA adapter")
    print(f"uploaded -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

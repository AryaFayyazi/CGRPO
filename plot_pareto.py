"""Utility for aggregating and plotting Pareto evaluation results.

Reads one or more pareto_results.json files produced by eval_pareto.py and
creates figures showing accuracy vs. cost, coverage, k-usage, calibration, etc.

Example:
    python plot_pareto.py --input runs/conformal_grpo/*/pareto_results.json \
        --outdir figures

"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        model = data.get("model")
        ckpt = data.get("ckpt")
        dataset = Path(ckpt).parent.name if ckpt else None
        n_eval = data.get("n_eval")
        conformal_qhats = data.get("conformal_qhats", {})
        k_usage = data.get("conformal_k_usage", {})
        for k, res in data.get("results", {}).items():
            method = res.get("method")
            rows.append(
                {
                    "model": model,
                    "ckpt": ckpt,
                    "dataset": dataset,
                    "n_eval": n_eval,
                    "k": res.get("k"),
                    "method": method,
                    "accuracy": res.get("accuracy"),
                    "avg_k": res.get("avg_k"),
                    "avg_tokens": res.get("avg_tokens"),
                    "coverage": res.get("coverage"),
                    "correct": res.get("correct"),
                    "total": res.get("total"),
                    "conformal_qhat": conformal_qhats.get(str(k)),
                    "conformal_not_covered": k_usage.get(str(k)),
                }
            )
    return pd.DataFrame(rows)


def plot_pareto(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    # cost vs accuracy scatter/line (samples)
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df, x="avg_k", y="accuracy", hue="method", marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Avg samples per prompt")
    plt.ylabel("Accuracy")
    plt.title("Cost-Accuracy Curve")
    plt.legend(title="Method")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cost_accuracy.png"))
    plt.close()

    # tokens vs accuracy
    if "avg_tokens" in df.columns:
        plt.figure(figsize=(6,4))
        sns.lineplot(data=df, x="avg_tokens", y="accuracy", hue="method", marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Avg tokens generated per prompt")
        plt.ylabel("Accuracy")
        plt.title("Token-Cost vs Accuracy")
        plt.legend(title="Method")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "tokens_accuracy.png"))
        plt.close()

    # Pareto frontier
    # we can compute frontier per dataset-model pair
    for (model, dataset), group in df.groupby(["model","dataset"]):
        group = group.sort_values("avg_k")
        frontier = []
        best_acc = -1
        for _, row in group.iterrows():
            if row.accuracy > best_acc:
                frontier.append(row)
                best_acc = row.accuracy
        fdf = pd.DataFrame(frontier)
        plt.figure(figsize=(6,4))
        plt.plot(group.avg_k, group.accuracy, "-o", alpha=0.3)
        plt.plot(fdf.avg_k, fdf.accuracy, "-o")
        plt.xscale("log", base=2)
        plt.xlabel("Avg samples per prompt")
        plt.ylabel("Accuracy")
        plt.title(f"Pareto frontier ({model} {dataset})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"pareto_{model}_{dataset}.png"))
        plt.close()

    # histogram of k usage for conformal experiments
    conv = df[df.method == "conformal"].copy()
    if not conv.empty:
        for _, row in conv.iterrows():
            usage = json.loads(row.ckpt) if isinstance(row.ckpt, str) and os.path.exists(row.ckpt) else None
        # k_usage not easily accessible here; maybe skip
    # calibration plot
    cal = df.dropna(subset=["conformal_qhat"]).copy()
    if not cal.empty:
        plt.figure(figsize=(4,4))
        sns.scatterplot(data=cal, x="conformal_qhat", y="accuracy", hue="k")
        plt.plot([0,1],[0,1], "k--")
        plt.xlabel("reported qhat")
        plt.ylabel("empirical accuracy")
        plt.title("Calibration of conformal qhats")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "calibration.png"))
        plt.close()

    print(f"Saved plots to {outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Paths to pareto_results.json files")
    parser.add_argument("--outdir", default="figures", help="Directory to save plots")
    args = parser.parse_args()

    df = load_results(args.input)
    print(df.head())
    plot_pareto(df, args.outdir)

if __name__ == "__main__":
    main()

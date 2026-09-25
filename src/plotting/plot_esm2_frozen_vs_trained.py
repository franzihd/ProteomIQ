"""
Presentation figure: frozen, untrained ESM2 embeddings vs. the trained
protein projection head (M0, multiview), on the two protein-side
task-agnostic checks -- linear probe (CLAUDE.md 11ah) and k-NN label
agreement@10 (11ag). Answers one question only: does the contrastive
training add structure beyond what ESM2 already had?

Plotting-only: reads the saved JSON summaries, computes nothing. The
raw-ESM2 numbers are checkpoint-independent (same embeddings every time),
so they're read from the multiview summary.

M0 here = biobert_unfreeze_0_m0_best (the post-11y-fix M0, same one
plot_m0_vs_multiview.py and results_summary.csv use).

Usage:
    uv run python src/plotting/plot_esm2_frozen_vs_trained.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "data/plots/presentation/esm2_frozen_vs_trained.png"

TASKS = [("deeploc", "DeepLoc\n(subcellular location)"), ("ec", "EC top-level\n(enzyme class)"),
         ("go_mf", "GO-Slim-MF\n(molecular function)")]
SERIES = [
    ("raw", "Frozen ESM2 (untrained)", "#9a9a9a"),
    ("biobert_unfreeze_0_m0_best", "M0 (baseline)", "#2a78d6"),
    ("multiview_best", "Multiview (final)", "#eda100"),
]


def load(directory: str, stem: str) -> dict:
    with open(Path(directory) / f"summary_{stem}.json") as f:
        return json.load(f)["results"]


def collect(directory: str, trained_key, raw_key):
    """{series_key: [value per task]} in percent."""
    out = {}
    for key, _, _ in SERIES:
        stem = "multiview_best" if key == "raw" else key
        res = load(directory, stem)
        out[key] = [(raw_key(res[t]) if key == "raw" else trained_key(res[t])) * 100 for t, _ in TASKS]
    return out


def panel(ax, values: dict, title: str, baselines: list, baseline_label: str):
    x = np.arange(len(TASKS))
    w = 0.26
    for i, (key, label, color) in enumerate(SERIES):
        bars = ax.bar(x + (i - 1) * w, values[key], w - 0.02, label=label, color=color)
        for b, v in zip(bars, values[key]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=8.5, color="#333333")
    for xi, bl in zip(x, baselines):
        ax.hlines(bl, xi - 1.5 * w, xi + 1.5 * w, colors="#333333", linestyles="--", linewidth=1)
    baseline_handle, = ax.plot([], [], "--", color="#333333", linewidth=1)
    ax.legend([baseline_handle], [baseline_label], loc="upper left", fontsize=8.5, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in TASKS], fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%", fontsize=9.5)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def main(output_path: str):
    probe = collect("data/evaluation/linear_probe",
                    lambda r: r["probe_accuracy"], lambda r: r["raw_esm2_probe_accuracy"])
    knn = collect("data/evaluation/knn_embedding_quality",
                  lambda r: r["agreement_at_k"]["10"], lambda r: r["raw_esm2_agreement_at_k"]["10"])

    lp_raw = load("data/evaluation/linear_probe", "multiview_best")
    kn_raw = load("data/evaluation/knn_embedding_quality", "multiview_best")
    majority = [lp_raw[t]["majority_baseline"] * 100 for t, _ in TASKS]
    random_bl = [kn_raw[t]["random_baseline"] * 100 for t, _ in TASKS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Frozen ESM2 vs. trained protein embeddings (validation split)",
                 fontsize=12.5)
    panel(axes[0], probe, "Linear probe accuracy", majority, "Majority-class baseline")
    panel(axes[1], knn, "k-NN: label agreement among 10 nearest neighbors", random_bl, "Random baseline")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SERIES]
    labels = [l for _, l, _ in SERIES]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)

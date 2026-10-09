"""
Seen vs. validation zero-shot accuracy for all ablation checkpoints.

Usage:
    uv run python src/plotting/plot_seen_vs_validation.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from plot_ablation_comparison import (  # noqa: E402
    BRAND_MUTED, DISPLAY_LABELS, GROUP_COLORS, GROUP_OF, INK, PERCENT_XLIM,
)

ALL_RESULTS_PATH = "data/evaluation/all_results.csv"
RUNS_DIR = "data/runs/full_proteome"
OUTPUT_PATH = "data/plots/overview/2.1_ablation/seen_vs_validation.png"

# (results_summary label used by plot_ablation_comparison, checkpoint stem, checkpoint path)
CHECKPOINTS = [
    ("N=0 (M0)", "biobert_unfreeze_0_m0_best", "unfreeze_experiment/biobert_unfreeze_0_m0_best.pt"),
    ("N=1", "biobert_unfreeze_1_m0_best", "unfreeze_experiment/biobert_unfreeze_1_m0_best.pt"),
    ("N=2", "biobert_unfreeze_2_m0_best", "unfreeze_experiment/biobert_unfreeze_2_m0_best.pt"),
    ("N=3", "biobert_unfreeze_3_m0_best", "unfreeze_experiment/biobert_unfreeze_3_m0_best.pt"),
    ("M1 (+Family)", "m1_family_masked_best", "text_experiment/m1_family_masked_best.pt"),
    ("Function+Location", "location_only_masked_best", "text_experiment/location_only_masked_best.pt"),
    ("M2 (+Family+Location)", "m2_family_location_masked_best",
     "text_experiment/m2_family_location_masked_best.pt"),
    ("M3 (+unmasked name)", "m3_name_family_location_best", "text_experiment/m3_name_family_location_best.pt"),
    ("multi-positive", "multi_positive_best", "multi_positive_experiment/multi_positive_best.pt"),
    ("multiview (winner)", "multiview_best", "multiview_experiment/multiview_best.pt"),
]
TASKS = [("deeploc", "DeepLoc accuracy (%)"), ("ec_top_level", "EC top-level accuracy (%)"),
         ("go_mf", "GO-Slim-MF accuracy (%)")]


def best_epoch(path: str) -> int:
    return int(torch.load(Path(RUNS_DIR) / path, map_location="cpu", weights_only=False)["epoch"])


def main(output_path: str):
    ar = pd.read_csv(ALL_RESULTS_PATH)
    acc = ar[(ar.variant == "ensemble") & (ar.metric == "accuracy") & ar.pool.isin(["seen", "validation"])]

    def get(stem, task, pool):
        return 100 * float(acc[(acc.checkpoint == stem) & (acc["eval"] == task) & (acc.pool == pool)].value.iloc[0])

    epochs = {stem: best_epoch(path) for _, stem, path in CHECKPOINTS}
    labels = [f"{DISPLAY_LABELS[lab]}  ({epochs[stem]} ep.)" for lab, stem, _ in CHECKPOINTS]
    colors = [GROUP_COLORS[GROUP_OF[lab]] for lab, _, _ in CHECKPOINTS]
    y = np.arange(len(CHECKPOINTS))
    h = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharey=True)
    rows = []
    for ax, (task, title) in zip(axes, TASKS):
        seen = [get(stem, task, "seen") for _, stem, _ in CHECKPOINTS]
        val = [get(stem, task, "validation") for _, stem, _ in CHECKPOINTS]
        # seen = hollow outline in the variant color, validation = filled (the reported number)
        ax.barh(y - h / 2, seen, h, facecolor="white", edgecolor=colors, linewidth=1.4, zorder=3)
        ax.barh(y + h / 2, val, h, color=colors, zorder=3)
        for yi, s, v in zip(y, seen, val):
            ax.annotate(f"{s - v:+.1f}", (max(s, v), yi), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, PERCENT_XLIM)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=9)
        for (lab, stem, _), s, v in zip(CHECKPOINTS, seen, val):
            rows.append(dict(checkpoint=DISPLAY_LABELS[lab], best_epoch=epochs[stem], task=task,
                             seen_accuracy=s, validation_accuracy=v, gap_pp=s - v))

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=BRAND_MUTED, linewidth=1.4),
                      plt.Rectangle((0, 0), 1, 1, color=BRAND_MUTED)]
    fig.legend(legend_handles, ["Training-split proteins (seen)", "Validation proteins"],
               loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.005, "Numbers = seen − validation gap in percentage points; ep. = best epoch (early stopping)",
             ha="center", fontsize=8.5, color=BRAND_MUTED)

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(Path(output_path).with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)

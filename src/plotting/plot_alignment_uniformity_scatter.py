"""
Alignment vs. uniformity scatter for M0, multi-positive and multiview.

Usage:
    uv run python src/plotting/plot_alignment_uniformity_scatter.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_PATH = "data/plots/overview/2.2_task_agnostic/alignment_uniformity_scatter.png"
INPUT_DIR = Path("data/evaluation/alignment_uniformity")

CHECKPOINTS = [
    ("biobert_unfreeze_0_m0_best", "M0"),
    ("multi_positive_best", "Multi-positive"),
    ("multiview_best", "Multiview"),
]
# same ProteomIQ colors as plot_task_agnostic_comparison.py
BRAND_NAVY, BRAND_VIOLET, BRAND_MUTED = "#14213D", "#7B5CD6", "#5B6477"
COLORS = {"M0": "#8C93A3", "Multi-positive": BRAND_NAVY, "Multiview": BRAND_VIOLET}
INK, MUTED = BRAND_NAVY, BRAND_MUTED


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Alignment (lower is better)", color=INK)
    ax.set_ylabel("Uniformity, pooled (lower is better)", color=INK)


def main(output_path: str):
    summaries = {}
    for stem, label in CHECKPOINTS:
        with open(INPUT_DIR / f"summary_{stem}.json") as f:
            summaries[label] = json.load(f)
    points = {l: (s["alignment"], s["uniformity_pooled"]) for l, s in summaries.items()}
    align_ref = float(np.mean([s["alignment_mismatched_reference"] for s in summaries.values()]))
    uniform_ref = float(np.mean([s["uniformity_random_reference"] for s in summaries.values()]))

    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    zx = (min(xs) - 0.012, max(xs) + 0.018)
    zy = (min(ys) - 0.012, max(ys) + 0.012)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax in (ax_full, ax_zoom):
        style_axis(ax)

    # A: full scale with both references
    ax_full.axvline(align_ref, color=MUTED, linestyle="--", linewidth=1.6)
    ax_full.axhline(uniform_ref, color=MUTED, linestyle="--", linewidth=1.6)
    ax_full.set_xlim(1.0, align_ref + 0.1)
    ax_full.set_ylim(uniform_ref - 0.03, -3.55)
    ax_full.text(align_ref - 0.02, -3.58, "mismatched pairs", ha="right", va="top", fontsize=9, color=MUTED)
    ax_full.text(1.02, uniform_ref + 0.01, "random points", fontsize=9, color=MUTED)
    ax_full.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
                                    fill=False, edgecolor=MUTED, linewidth=0.8))
    ax_full.set_title("A  Full scale with references", fontsize=10.5, color=INK, loc="left")

    # B: zoom on the three models
    for label, (x, y) in points.items():
        ax_full.scatter(x, y, s=60, color=COLORS[label], zorder=3)
        ax_zoom.scatter(x, y, s=110, color=COLORS[label], zorder=3)
        ax_zoom.annotate(f"{label}\n({x:.4f}, {y:.4f})", (x, y), xytext=(8, 4),
                         textcoords="offset points", fontsize=9, color=INK)
    ax_zoom.set_xlim(*zx)
    ax_zoom.set_ylim(*zy)
    ax_zoom.set_title("B  Zoom (box in A)", fontsize=10.5, color=INK, loc="left")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    rows = []
    for label, s in summaries.items():
        for metric in ("alignment", "uniformity_pooled", "uniformity_protein", "uniformity_text",
                       "alignment_mismatched_reference", "uniformity_random_reference"):
            rows.append(dict(checkpoint=label, metric=metric, value=s[metric]))
    pd.DataFrame(rows).to_csv(Path(output_path).with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)

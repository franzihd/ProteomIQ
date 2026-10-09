"""
Presentation figure: M0 vs. M3 vs. multiview on zero-shot classification.

Usage:
    uv run python src/plotting/plot_m0_vs_multiview.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALL_RESULTS_PATH = "data/evaluation/all_results.csv"
OUTPUT_PATH = "data/plots/presentation/m0_vs_multiview.png"

SERIES = [
    ("biobert_unfreeze_0_m0_best", "M0 (baseline)", "#2a78d6"),
    ("m3_name_family_location_best", "M3 (concatenation)", "#eb6834"),
    ("multiview_best", "Multiview (final)", "#eda100"),
]
INK = "#333333"


def value(df, eval_name, checkpoint, variant, metric) -> float:
    row = df[(df["eval"] == eval_name) & (df.checkpoint == checkpoint)
             & (df.variant == variant) & (df.metric == metric)]
    return float(row.value.iloc[0])


def grouped_bars(ax, groups: list, values: dict, fmt: str, offset: float):
    x = np.arange(len(groups))
    w = 0.27
    for i, (key, label, color) in enumerate(SERIES):
        bars = ax.bar(x + (i - (len(SERIES) - 1) / 2) * w, values[key], w - 0.03, label=label, color=color)
        for b, v in zip(bars, values[key]):
            ax.text(b.get_x() + b.get_width() / 2, v + offset, fmt.format(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    return x, w


def baseline_ticks(ax, x, w, levels, style, label):
    for xi, bl in zip(x, levels):
        ax.hlines(bl, xi - w * 1.6, xi + w * 1.6, colors=INK, linestyles=style, linewidth=1)
    handle, = ax.plot([], [], style, color=INK, linewidth=1, label=label)
    return handle


def main(output_path: str):
    df = pd.read_csv(ALL_RESULTS_PATH)
    df = df[df.pool == "validation"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_title("Zero-shot classification: M0 vs. M3 vs. multiview\n(validation split)", fontsize=12)

    tasks = [("deeploc", "DeepLoc\n(subcellular location)"), ("ec_top_level", "EC top-level\n(enzyme class)"),
             ("go_mf", "GO-Slim-MF\n(molecular function)")]
    vals = {k: [value(df, t, k, "ensemble", "accuracy") * 100 for t, _ in tasks] for k, _, _ in SERIES}
    x, w = grouped_bars(ax, [n for _, n in tasks], vals, "{:.0f}", 1.0)
    ref = SERIES[-1][0]
    h_major = baseline_ticks(ax, x, w, [value(df, t, ref, "ensemble", "majority_baseline_accuracy") * 100
                                        for t, _ in tasks], "--", "Majority-class baseline")
    h_rand = baseline_ticks(ax, x, w, [value(df, t, ref, "ensemble", "random_baseline_accuracy") * 100
                                       for t, _ in tasks], ":", "Random baseline")
    ax.legend(handles=[h_major, h_rand], loc="upper left", fontsize=8.5, frameon=False)
    ax.set_ylim(0, 65)
    ax.set_ylabel("Accuracy (%)", fontsize=9.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SERIES]
    fig.legend(handles, [l for _, l, _ in SERIES], loc="lower center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)

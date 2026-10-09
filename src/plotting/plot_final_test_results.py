"""
Thesis figure for Results 2.3: the final checkpoint (multiview) on the
reserved TEST split, against its baselines -- the one legitimate test
result (CLAUDE.md 11ai). Three panels: (A) cross-modal retrieval Recall@K in both
directions (vs. random) and (B) zero-shot classification accuracy
(DeepLoc/EC/GO-MF vs. random + majority-class baseline) on one shared % axis, and
(C) GO-term retrieval mAP per branch and query style (vs. random ranking) on
its own axis.

Deliberately shows NO other checkpoint: M0 and the ablation variants have
no legitimate test numbers -- their comparison lives in the validation
figures (plot_ablation_comparison.py, plot_task_agnostic_comparison.py).

Plotting-only: reads data/evaluation/results_final_test.csv (written by
aggregate_results.py), computes nothing.

Usage:
    uv run python src/plotting/plot_final_test_results.py
"""
import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_PATH = "data/evaluation/results_final_test.csv"
OUTPUT_PATH = "data/plots/overview/2.3_final_test/final_test_results.png"

# ProteomIQ brand colors, same multiview violet as plot_ablation_comparison.py /
# plot_task_agnostic_comparison.py
MULTIVIEW = "#7B5CD6"
MULTIVIEW_DARK = "#4A3591"  # second series of the same model (definition queries / P->T), hatched
INK = "#14213D"
MUTED = "#5B6477"
# random baseline: warm accent + white halo so the dashed line stays visible on top of the violet
# (and dark hatched) bars -- the old gray (MUTED) had too little contrast there
RANDOM = "#E8833A"
HALO = [pe.withStroke(linewidth=5.5, foreground="white")]
# Font sizes are set for PRINT: the figure is built 10.5 in wide but placed at \textwidth (~15 cm),
# i.e. scaled to ~55 %, so e.g. 14 pt here ends up at ~8 pt on paper.
FS_TICK, FS_LABEL, FS_TITLE, FS_BAR, FS_LEGEND = 14, 15, 17, 11, 13.5
PERCENT_YMAX = 80  # shared y-axis for both % panels, same 0-80 % range as the 2.1 ablation figures


def value(df, eval_name, variant, metric) -> float:
    row = df[(df["eval"] == eval_name) & (df.variant == variant) & (df.metric == metric)]
    return float(row.value.iloc[0])


def baseline_tick(ax, x, width, y, style):
    ax.hlines(y, x - width / 2 - 0.04, x + width / 2 + 0.04, colors=INK if style == "-" else RANDOM,
              linestyles=style, linewidth=2.8, zorder=4, path_effects=HALO)


def label_bars(ax, bars, fmt, dx=0):
    # dx (points): nudges the labels of a bar PAIR apart (left series -dx, right series +dx) so
    # neighbouring values like 73.5 / 72.7 do not touch at print font size
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(dx, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_BAR, color=INK)


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=FS_TICK)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def main(output_path: str):
    df = pd.read_csv(INPUT_PATH)
    # 2+1 layout (same as the 2.1 ablation figures): the two percentage panels (cross-modal recall,
    # zero-shot accuracy) side by side on top on one SHARED 0-80 % axis, so bar heights are directly
    # comparable across them; GO-term mAP (a different unit) alone below on its own axis.
    # Explicit axes positions (figure fractions) so all three panels have the same plot width
    fig = plt.figure(figsize=(10.5, 9))
    left, width, gap = 0.08, 0.39, 0.12
    height, bottom_low, bottom_high = 0.34, 0.12, 0.60
    axes = [fig.add_axes([left, bottom_high, width, height]),
            fig.add_axes([left + width + gap, bottom_high, width, height]),
            fig.add_axes([left + (width + gap) / 2, bottom_low, width, height])]

    # --- B: zero-shot classification (same % axis as A) -----------------------------
    ax = axes[1]
    tasks = [("deeploc", "DeepLoc\n(9 classes)"), ("ec_top_level", "EC top-level\n(7 classes)"),
             ("go_mf", "GO-Slim-MF\n(10 classes)")]
    x = np.arange(len(tasks))
    w = 0.55
    acc = [value(df, t, "ensemble", "accuracy") for t, _ in tasks]
    bars = ax.bar(x, acc, w, color=MULTIVIEW, zorder=3)
    label_bars(ax, bars, "{:.1f}")
    for i, (t, _) in enumerate(tasks):
        baseline_tick(ax, x[i], w, value(df, t, "ensemble", "majority_baseline_accuracy"), "-")
        baseline_tick(ax, x[i], w, value(df, t, "ensemble", "random_baseline_accuracy"), "--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\nn={int(value(df, t, 'ensemble', 'n'))}" for t, name in tasks])
    ax.set_ylabel("Accuracy (%)", color=INK, fontsize=FS_LABEL)
    ax.set_ylim(0, PERCENT_YMAX)
    ax.set_title("(B)  Zero-shot classification", loc="left", fontsize=FS_TITLE, color=INK)
    style_axis(ax)

    # --- C: GO-term retrieval (own axis: mAP, not %) --------------------------------
    ax = axes[2]
    branches = [("bp", "Biological\nProcess"), ("cc", "Cellular\nComponent"), ("mf", "Molecular\nFunction")]
    x = np.arange(len(branches))
    w = 0.4
    name = [value(df, "go_retrieval", f"{b}_name", "map") for b, _ in branches]
    definition = [value(df, "go_retrieval", f"{b}_definition", "map") for b, _ in branches]
    b1 = ax.bar(x - w / 2, name, w, color=MULTIVIEW, zorder=3)
    b2 = ax.bar(x + w / 2, definition, w, color=MULTIVIEW_DARK, hatch="///", edgecolor="white",
                linewidth=0, zorder=3)
    label_bars(ax, b1, "{:.3f}", dx=-3)
    label_bars(ax, b2, "{:.3f}", dx=3)
    for i, (b, _) in enumerate(branches):
        baseline_tick(ax, x[i], 2 * w, value(df, "go_retrieval", f"{b}_name", "map_random_baseline"), "--")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in branches])
    ax.set_ylabel("Mean average precision (mAP)", color=INK, fontsize=FS_LABEL)
    ax.set_ylim(0, 0.19)  # headroom so the legend clears the tallest bar (MF, 0.132)
    ax.set_title("(C)  GO-term retrieval", loc="left", fontsize=FS_TITLE, color=INK)
    style_axis(ax)
    ax.legend([b1, b2], ["GO term name as query", "GO term definition as query"],
              fontsize=FS_LEGEND, frameon=False, loc="upper left")

    # --- A: cross-modal retrieval ---------------------------------------------------
    ax = axes[0]
    ks = [1, 5, 10, 50]
    x = np.arange(len(ks))
    t2p = [value(df, "cross_modal", "text_to_protein", f"r{k}") for k in ks]
    p2t = [value(df, "cross_modal", "protein_to_text", f"r{k}") for k in ks]
    c1 = ax.bar(x - w / 2, t2p, w, color=MULTIVIEW, zorder=3)
    c2 = ax.bar(x + w / 2, p2t, w, color=MULTIVIEW_DARK, hatch="///", edgecolor="white",
                linewidth=0, zorder=3)
    label_bars(ax, c1, "{:.1f}", dx=-3)
    label_bars(ax, c2, "{:.1f}", dx=3)
    pool = 1689
    for i, k in enumerate(ks):
        baseline_tick(ax, x[i], 2 * w, 100 * k / pool, "--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R@{k}" for k in ks])
    ax.set_ylabel("Recall (%)", color=INK, fontsize=FS_LABEL)
    ax.set_ylim(0, PERCENT_YMAX)
    ax.set_title("(A)  Cross-modal retrieval", loc="left", fontsize=FS_TITLE, color=INK)
    style_axis(ax)
    ax.legend([c1, c2], ["Text → Protein", "Protein → Text"], fontsize=FS_LEGEND, frameon=False, loc="upper left")

    # shared baseline legend
    handles = [plt.Line2D([], [], color=INK, linewidth=2.8, linestyle="-"),
               plt.Line2D([], [], color=RANDOM, linewidth=2.8, linestyle="--")]
    fig.legend(handles, ["Majority-class baseline", "Random baseline"], loc="lower center", ncol=2,
               fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(left + width + gap / 2, -0.035))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    df.to_csv(Path(output_path).with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)

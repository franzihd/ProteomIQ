"""
Learning curve for the training-data-size ablation.

Usage:
    uv run python src/plotting/plot_datasize_ablation.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = Path("data/evaluation/datasize_experiment/aggregated.csv")
OUTPUT_PATH = Path("data/plots/datasize_experiment/learning_curve_cross_modal.png")

# ProteomIQ brand colors, same as plot_final_test_results.py
MULTIVIEW = "#7B5CD6"       # text -> protein
MULTIVIEW_DARK = "#4A3591"  # protein -> text
INK = "#14213D"
MUTED = "#5B6477"
# Font sizes for PRINT (built 10.5 in wide, placed at \textwidth ~15 cm), as in the other overview figures.
FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_DIRECT = 14, 15, 17, 13.5, 13
PERCENT_YMAX = 80

SERIES = [("t2p", "Text → Protein", MULTIVIEW, "o", "-"),
          ("p2t", "Protein → Text", MULTIVIEW_DARK, "s", "--")]
PANELS = [("r1", "R@1"), ("r10", "R@10"), ("r50", "R@50")]


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=FS_TICK)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    agg = pd.read_csv(INPUT_PATH).sort_values("n_train_proteins")
    x = agg["n_train_proteins"].to_numpy()
    tick_labels = [f"{f:.0%}" for f in agg["fraction"]]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    plotted = []
    for ax, (metric, title) in zip(axes, PANELS):
        for prefix, label, color, marker, ls in SERIES:
            mean = agg[f"{prefix}_{metric}_mean"].to_numpy()
            sd = agg[f"{prefix}_{metric}_std"].to_numpy()
            ax.errorbar(x, mean, yerr=sd, color=color, marker=marker, markersize=8, linestyle=ls,
                        linewidth=2, capsize=4, elinewidth=1.5, label=label,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            short = label.replace(" ", "").replace("Protein", "P").replace("Text", "T")
            # direct labels only where the two directions are visibly apart (R@50 lines overlap;
            # the legend carries identity there)
            if metric != "r50":
                ax.annotate(short, (x[-1], mean[-1]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=FS_DIRECT, color=INK)
            for n, f, m, s in zip(x, agg["fraction"], mean, sd):
                plotted.append({"panel": title, "direction": label, "fraction": f,
                                "n_train_proteins": n, "mean": round(m, 3), "sd": round(s, 3)})
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.minorticks_off()
        ax.set_xlim(0, x[-1] * 1.18)
        ax.set_ylim(0, PERCENT_YMAX)
        ax.set_title(title, fontsize=FS_TITLE, color=INK)
        style_axis(ax)
        ax.tick_params(axis="x", labelsize=FS_TICK - 1)

    axes[0].set_ylabel("Validation recall (%)", fontsize=FS_LABEL, color=INK)
    fig.supxlabel("Training data (% of training split)", fontsize=FS_LABEL, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    pd.DataFrame(plotted).to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)
    print(f"Saved {OUTPUT_PATH} and {OUTPUT_PATH.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

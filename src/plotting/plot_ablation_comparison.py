"""
Bar-chart visualization of the four-lever training-signal ablation study
(CLAUDE.md 11x/11y/11z/11ac/11ae): BioBERT-unfreeze depth, text-field
concatenation (M0-M3), multiview, and multi-positive, compared across all
5 topic evals in two figures (retrieval / zero-shot classification), each a
2+1 layout of horizontal-bar panels, plus one combined 3x2 overview figure
(ablation_comparison.png) with all six panels, also saved split into
ablation_comparison_part1.png (rows 1-2) and ablation_comparison_part2.png (row 3). All percentage panels share one 0-80 % x-axis so bar
lengths are directly comparable; mAP is a different unit and keeps its own
axis. ProteomIQ brand colors.

Reads directly from data/evaluation/results_summary.csv (the corrected,
validation-only, post-11ai source of truth built by aggregate_results.py)
-- no numbers are computed here, this is a plotting-only script.

Usage:
    uv run python src/plotting/plot_ablation_comparison.py
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_SUMMARY_PATH = "data/evaluation/results_summary.csv"
ALL_RESULTS_PATH = "data/evaluation/all_results.csv"  # baselines only (validation pool)
VALIDATION_POOL_SIZE = 1689
# ProteomIQ brand colors (same values as src/app.py / plot_hyperparameter_sweeps.py)
BRAND_NAVY, BRAND_BLUE, BRAND_VIOLET, BRAND_MUTED = "#14213D", "#2A7BCB", "#7B5CD6", "#5B6477"
INK = BRAND_NAVY
MUTED = BRAND_MUTED
OUTPUT_DIR = "data/plots/overview/2.1_ablation"

# Top-to-bottom order: shared M0 reference -> unfreeze depth increasing ->
# text-field concatenation progression -> multi-positive -> multiview.
CHECKPOINT_ORDER = [
    "N=0 (M0)", "N=1", "N=2", "N=3",
    "M1 (+Family)", "Function+Location", "M2 (+Family+Location)", "M3 (+unmasked name)",
    "multi-positive", "multiview (winner)",
]
DISPLAY_LABELS = {
    "N=0 (M0)": "M0 (Function, frozen)", "N=1": "Unfreeze N=1", "N=2": "Unfreeze N=2",
    "N=3": "Unfreeze N=3",
    "M1 (+Family)": "M1 (+Family)", "Function+Location": "M1' (+Location)",
    "M2 (+Family+Location)": "M2 (+Family+Location)", "M3 (+unmasked name)": "M3 (+unmasked name)",
    "multi-positive": "Multi-positive", "multiview (winner)": "Multiview",
}
# M0 is the shared reference for every lever, so it gets its own neutral color
# instead of being drawn as part of the unfreeze group.
GROUP_OF = {
    "N=0 (M0)": "reference", "N=1": "unfreeze", "N=2": "unfreeze", "N=3": "unfreeze",
    "M1 (+Family)": "text", "Function+Location": "text",
    "M2 (+Family+Location)": "text", "M3 (+unmasked name)": "text",
    "multi-positive": "multi_positive", "multiview (winner)": "multiview",
}
# ProteomIQ tones: gray reference, light -> dark blue for the three "other" levers,
# violet (the brand accent, also the sweep-winner color) for the final model.
GROUP_COLORS = {
    "reference": "#8C93A3",
    "unfreeze": "#A9CBEE",
    "text": BRAND_BLUE,
    "multi_positive": BRAND_NAVY,
    "multiview": BRAND_VIOLET,
}
GROUP_LABELS = {
    "reference": "M0 (reference)",
    "unfreeze": "BioBERT unfreezing",
    "text": "Text-field concatenation",
    "multi_positive": "Multi-positive",
    "multiview": "Multiview",
}

# Two figures, each a 2+1 layout (two panels on top, one centered below, legend in one line underneath):
#   ablation_retrieval.png -- cross-modal T->P R@1, R@50, GO-term retrieval mAP
#   ablation_zero_shot.png -- DeepLoc, EC top-level, GO-Slim-MF accuracy
# All percentage panels share one x-axis (0-80 %) so bar lengths are comparable
# across panels; mAP is a different unit and keeps its own axis.
FIGURES = {
    "ablation_retrieval": [
        ("cross_modal_T2P_R1%(val)", "Cross-modal retrieval, T→P R@1 (%)"),
        ("cross_modal_T2P_R50%(val)", "Cross-modal retrieval, T→P R@50 (%)"),
        ("go_retrieval_mean_mAP(val)", "GO-term retrieval, mean mAP (own scale)"),
    ],
    "ablation_zero_shot": [
        ("deeploc_acc%(val)", "DeepLoc accuracy (%)"),
        ("ec_acc%(val)", "EC top-level accuracy (%)"),
        ("go_mf_acc%(val)", "GO-Slim-MF accuracy (%)"),
    ],
}
# Combined overview (all six panels in one portrait figure), kept alongside the two split figures:
#   row 1 = cross-modal R@1 + R@50, row 2 = DeepLoc + EC, row 3 = GO-Slim-MF + GO-term retrieval
COMBINED_NAME = "ablation_comparison"
COMBINED_METRICS = [
    FIGURES["ablation_retrieval"][0], FIGURES["ablation_retrieval"][1],
    FIGURES["ablation_zero_shot"][0], FIGURES["ablation_zero_shot"][1],
    FIGURES["ablation_zero_shot"][2], FIGURES["ablation_retrieval"][2],
]
PERCENT_XLIM = 80


def validation_baselines() -> dict:
    """(random, majority) per metric column, validation pool. Majority is None where it
    doesn't apply (retrieval tasks). Classification baselines are checkpoint-independent
    (same labelled proteins for every checkpoint), so they're read from multiview's rows."""
    ar = pd.read_csv(ALL_RESULTS_PATH)
    val = ar[(ar.pool == "validation") & (ar.checkpoint == "multiview_best")]

    def get(eval_name, metric):
        return float(val[(val["eval"] == eval_name) & (val.metric == metric)].value.iloc[0])

    return {
        "cross_modal_T2P_R1%(val)": (100 * 1 / VALIDATION_POOL_SIZE, None),
        "cross_modal_T2P_R50%(val)": (100 * 50 / VALIDATION_POOL_SIZE, None),
        "deeploc_acc%(val)": (100 * get("deeploc", "random_baseline_accuracy"),
                              100 * get("deeploc", "majority_baseline_accuracy")),
        "ec_acc%(val)": (100 * get("ec_top_level", "random_baseline_accuracy"),
                         100 * get("ec_top_level", "majority_baseline_accuracy")),
        "go_mf_acc%(val)": (100 * get("go_mf", "random_baseline_accuracy"),
                            100 * get("go_mf", "majority_baseline_accuracy")),
        "go_retrieval_mean_mAP(val)": (val[(val["eval"] == "go_retrieval")
                                           & (val.metric == "map_random_baseline")].value.mean(), None),
    }


def draw_panel(ax, df, col, title, baselines, show_labels: bool):
    y = list(range(len(CHECKPOINT_ORDER)))
    colors = [GROUP_COLORS[GROUP_OF[c]] for c in CHECKPOINT_ORDER]
    group_boundaries = [i + 0.5 for i in range(len(CHECKPOINT_ORDER) - 1)
                        if GROUP_OF[CHECKPOINT_ORDER[i]] != GROUP_OF[CHECKPOINT_ORDER[i + 1]]]
    ax.barh(y, df[col], color=colors, height=0.72, edgecolor="white", linewidth=1, zorder=3)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    for b in group_boundaries:
        ax.axhline(b, color="#E4E7ED", linewidth=1, zorder=1)
    random_x, majority_x = baselines[col]
    if majority_x is not None:
        ax.axvline(majority_x, color=INK, linewidth=1.6, zorder=4)
    ax.axvline(random_x, color=MUTED, linewidth=1.6, linestyle="--", zorder=4)
    ax.set_xlim(0, PERCENT_XLIM if col.endswith("%(val)") else 0.12)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_LABELS[c] for c in CHECKPOINT_ORDER] if show_labels else [])
    ax.set_ylim(len(y) - 0.5, -0.5)  # first variant on top
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def plot_figure(df, metrics, baselines, output_path: Path, with_majority: bool):
    # 2+1 layout: two panels on top, the third centered below, legend in one line underneath
    # explicit axes positions (figure fractions) so all three panels have the same plot width and
    # the bottom one sits exactly centered under the two top ones
    fig = plt.figure(figsize=(10.5, 9))
    left, width, gap = 0.21, 0.36, 0.06
    height, bottom_low, bottom_high = 0.38, 0.08, 0.555
    axes = [fig.add_axes([left, bottom_high, width, height]),
            fig.add_axes([left + width + gap, bottom_high, width, height]),
            fig.add_axes([left + (width + gap) / 2, bottom_low, width, height])]
    for i, (ax, (col, title)) in enumerate(zip(axes, metrics)):
        draw_panel(ax, df, col, title, baselines, show_labels=(i != 1))

    handles, names = legend_entries(with_majority)
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8.5, frameon=False,
               bbox_to_anchor=(left + width + gap / 2, -0.01), handlelength=1.6, columnspacing=1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    cols = [c for c, _ in metrics]
    table = df[["group"] + cols].reset_index()
    table.loc[len(table)] = {"checkpoint": "random baseline", "group": "baseline",
                             **{c: baselines[c][0] for c in cols}}
    table.loc[len(table)] = {"checkpoint": "majority-class baseline", "group": "baseline",
                             **{c: baselines[c][1] for c in cols}}
    table.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")


def legend_entries(with_majority: bool):
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_LABELS]
    names = list(GROUP_LABELS.values())
    if with_majority:
        handles.append(plt.Line2D([], [], color=INK, linewidth=1.6))
        names.append("Majority-class baseline")
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.6, linestyle="--"))
    names.append("Random baseline")
    return handles, names


def plot_combined(df, baselines, metrics, output_path: Path):
    """Panels in a 2-column grid (one row per two metrics), legend underneath. Same per-row
    height for every call, so the split parts render at the same scale as the full figure."""
    nrows = len(metrics) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 11.5 / 3 * nrows))
    for i, (ax, (col, title)) in enumerate(zip(axes.flat, metrics)):
        draw_panel(ax, df, col, title, baselines, show_labels=(i % 2 == 0))
    plt.tight_layout()
    handles, names = legend_entries(with_majority=True)
    fig.legend(handles, names, loc="upper center", ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    cols = [c for c, _ in metrics]
    table = df[["group"] + cols].reset_index()
    table.loc[len(table)] = {"checkpoint": "random baseline", "group": "baseline",
                             **{c: baselines[c][0] for c in cols}}
    table.loc[len(table)] = {"checkpoint": "majority-class baseline", "group": "baseline",
                             **{c: baselines[c][1] for c in cols}}
    table.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")


def main(output_dir: str):
    df = pd.read_csv(RESULTS_SUMMARY_PATH).set_index("checkpoint").loc[CHECKPOINT_ORDER]
    baselines = validation_baselines()
    for name, metrics in FIGURES.items():
        plot_figure(df, metrics, baselines, Path(output_dir) / f"{name}.png",
                    with_majority=any(baselines[c][1] is not None for c, _ in metrics))
    plot_combined(df, baselines, COMBINED_METRICS, Path(output_dir) / f"{COMBINED_NAME}.png")
    # Same overview split across two thesis pages: rows 1-2 (4 panels) and row 3 (2 panels)
    plot_combined(df, baselines, COMBINED_METRICS[:4], Path(output_dir) / f"{COMBINED_NAME}_part1.png")
    plot_combined(df, baselines, COMBINED_METRICS[4:], Path(output_dir) / f"{COMBINED_NAME}_part2.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args.output_dir)

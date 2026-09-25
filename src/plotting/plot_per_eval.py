"""
One comparison figure per evaluation, mirroring data/evaluation/'s layout
under data/plots/<eval>/, each saved next to a CSV holding exactly the
numbers the figure shows.

Per eval there are (at most) two figures, never mixed:
- validation_all_checkpoints.png/.csv -- every checkpoint that eval was run
  for, validation pool (the ablation comparison, Results 2.1/2.2)
- test_multiview.png/.csv -- the final checkpoint on the reserved test pool
  (Results 2.3). Only for the 5 topic evals; the 3 task-agnostic checks are
  validation-only by design.

The three cross-eval thesis figures (ablation overview, task-agnostic
overview, final test overview) live in data/plots/overview/ and are made by
plot_ablation_comparison.py, plot_task_agnostic_comparison.py and
plot_final_test_results.py.

Plotting-only: reads data/evaluation/all_results.csv (written by
aggregate_results.py), computes nothing new.

Usage:
    uv run python src/plotting/plot_per_eval.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALL_RESULTS_PATH = "data/evaluation/all_results.csv"
PLOTS_DIR = Path("data/plots")
POOL_SIZE = 1689  # validation and test pools are both 1,689 proteins (homology-aware split)
FINAL = "multiview_best"

# left-to-right order + display name + group color, same as plot_ablation_comparison.py
CHECKPOINTS = [
    ("biobert_unfreeze_0_m0_best", "M0", "unfreeze"),
    ("biobert_unfreeze_1_m0_best", "N=1", "unfreeze"),
    ("biobert_unfreeze_2_m0_best", "N=2", "unfreeze"),
    ("biobert_unfreeze_3_m0_best", "N=3", "unfreeze"),
    ("m1_family_masked_best", "M1", "text"),
    ("location_only_masked_best", "F+Loc", "text"),
    ("m2_family_location_masked_best", "M2", "text"),
    ("m3_name_family_location_best", "M3", "text"),
    ("multi_positive_best", "Multi-pos", "multi_positive"),
    ("multiview_best", "Multiview", "multiview"),
]
GROUP_COLORS = {"unfreeze": "#2a78d6", "text": "#eb6834", "multi_positive": "#1baf7a", "multiview": "#eda100"}
GROUP_LABELS = {"unfreeze": "BioBERT-unfreeze depth (incl. M0)", "text": "Text-field concatenation",
                "multi_positive": "Multi-positive", "multiview": "Multiview (final)"}
MULTIVIEW, MULTIVIEW_DARK = "#eda100", "#8a5a00"
FROZEN_ESM2 = "#9a9a9a"
INK, MUTED = "#333333", "#8a8a8a"


# ------------------------------------------------------------------ helpers
def style_axis(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    getattr(ax, f"{grid_axis}axis").grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8.5)


def label_bars(ax, bars, fmt, size=7.5):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=size, color=INK)


def baseline_lines(ax, random_y=None, majority_y=None):
    if majority_y is not None:
        ax.axhline(majority_y, color=INK, linewidth=1.6, zorder=4)
    if random_y is not None:
        ax.axhline(random_y, color=MUTED, linewidth=1.6, linestyle="--", zorder=4)


def group_legend(fig, groups, extra=()):
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in groups]
    labels = [GROUP_LABELS[g] for g in groups]
    for kind, label in extra:
        handles.append(plt.Line2D([], [], color=INK if kind == "majority" else MUTED if kind == "random" else FROZEN_ESM2,
                                  linewidth=1.6 if kind != "frozen" else 6,
                                  linestyle="--" if kind == "random" else "-"))
        labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 6), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def save(fig, table: pd.DataFrame, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight")
    table.to_csv(folder / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {folder / name}.png + .csv")


def checkpoint_panel(ax, values: dict, fmt: str, random_y=None, majority_y=None, extra_bar=None):
    """Bars for every checkpoint present in `values` (stem -> value), in CHECKPOINTS order,
    colored by group; optional frozen-ESM2 bar at the far left."""
    rows = [(label, values[stem], group) for stem, label, group in CHECKPOINTS if stem in values]
    names = [r[0] for r in rows]
    heights = [r[1] for r in rows]
    colors = [GROUP_COLORS[r[2]] for r in rows]
    if extra_bar is not None:
        names, heights, colors = [extra_bar[0]] + names, [extra_bar[1]] + heights, [FROZEN_ESM2] + colors
    bars = ax.bar(names, heights, color=colors, zorder=3)
    label_bars(ax, bars, fmt)
    for i in range(1, len(rows)):
        if rows[i][2] != rows[i - 1][2]:
            ax.axvline(i - 0.5 + (1 if extra_bar else 0), color="#dddddd", linewidth=1, zorder=1)
    baseline_lines(ax, random_y, majority_y)
    top = max(heights + [majority_y or 0])
    ax.set_ylim(0, top * 1.15)
    ax.tick_params(axis="x", labelrotation=45)
    style_axis(ax)
    return [r[2] for r in rows]


def rows_for(df, **conditions):
    for col, val in conditions.items():
        df = df[df[col] == val]
    return df


def by_checkpoint(df, **conditions) -> dict:
    sub = rows_for(df, **conditions)
    return dict(zip(sub.checkpoint, sub.value))


# ------------------------------------------------------------------ evals
def cross_modal(df):
    out = PLOTS_DIR / "cross_modal_retrieval"
    val = rows_for(df, eval="cross_modal", pool="validation")
    panels = [("text_to_protein", "r1", "Text → Protein R@1 (%)", 1), ("text_to_protein", "r50", "Text → Protein R@50 (%)", 50),
              ("protein_to_text", "r1", "Protein → Text R@1 (%)", 1), ("protein_to_text", "r50", "Protein → Text R@50 (%)", 50)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, (variant, metric, title, k) in zip(axes, panels):
        groups = checkpoint_panel(ax, by_checkpoint(val, variant=variant, metric=metric), "{:.1f}", random_y=100 * k / POOL_SIZE)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle(f"Cross-modal retrieval — all checkpoints, validation split ({POOL_SIZE:,} proteins)", color=INK)
    group_legend(fig, dict.fromkeys(groups), [("random", "Random baseline")])
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, val[val.metric.isin(["r1", "r5", "r10", "r50", "median_rank"])], out, "validation_all_checkpoints")

    test = rows_for(df, eval="cross_modal", pool="test", checkpoint=FINAL)
    ks = [1, 5, 10, 50]
    x = np.arange(len(ks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t2p = [rows_for(test, variant="text_to_protein", metric=f"r{k}").value.iloc[0] for k in ks]
    p2t = [rows_for(test, variant="protein_to_text", metric=f"r{k}").value.iloc[0] for k in ks]
    b1 = ax.bar(x - w / 2, t2p, w, color=MULTIVIEW, zorder=3)
    b2 = ax.bar(x + w / 2, p2t, w, color=MULTIVIEW_DARK, hatch="///", edgecolor="white", linewidth=0, zorder=3)
    label_bars(ax, b1, "{:.1f}", 8.5)
    label_bars(ax, b2, "{:.1f}", 8.5)
    for i, k in enumerate(ks):
        ax.hlines(100 * k / POOL_SIZE, x[i] - w - 0.04, x[i] + w + 0.04, colors=MUTED, linestyles="--", linewidth=1.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"R@{k}" for k in ks])
    ax.set_ylabel("Recall (%)", color=INK)
    ax.set_ylim(0, 90)
    style_axis(ax)
    ax.legend([b1, b2, plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.8)],
              ["Text → Protein", "Protein → Text", "Random baseline"], fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title(f"Cross-modal retrieval — multiview, test split ({POOL_SIZE:,} proteins)", fontsize=10.5, color=INK)
    plt.tight_layout()
    save(fig, test, out, "test_multiview")


def zero_shot(df, eval_name, folder, title):
    out = PLOTS_DIR / "zero_shot_classification" / folder
    val = rows_for(df, eval=eval_name, pool="validation")
    has_f1 = "macro_f1" in set(val.metric)
    panels = [("accuracy", "Accuracy (%)")] + ([("macro_f1", "Macro-F1")] if has_f1 else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 4.4), squeeze=False)
    rnd = 100 * rows_for(val, checkpoint=FINAL, metric="random_baseline_accuracy").value.iloc[0]
    maj = 100 * rows_for(val, checkpoint=FINAL, metric="majority_baseline_accuracy").value.iloc[0]
    n = int(rows_for(val, checkpoint=FINAL, metric="n").value.iloc[0])
    for ax, (metric, ylabel) in zip(axes[0], panels):
        values = by_checkpoint(val, metric=metric)
        if metric == "accuracy":
            values = {k: 100 * v for k, v in values.items()}
            groups = checkpoint_panel(ax, values, "{:.1f}", random_y=rnd, majority_y=maj)
        else:
            groups = checkpoint_panel(ax, values, "{:.3f}")
        ax.set_title(ylabel, fontsize=10, color=INK, loc="left")
    fig.suptitle(f"{title} — all checkpoints, validation split (n={n})", color=INK)
    group_legend(fig, dict.fromkeys(groups), [("majority", "Majority-class baseline"), ("random", "Random baseline")])
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, val, out, "validation_all_checkpoints")

    test = rows_for(df, eval=eval_name, pool="test", checkpoint=FINAL)
    get = lambda m: rows_for(test, metric=m).value.iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    bars = ax.bar(["Multiview"], [100 * get("accuracy")], width=0.5, color=MULTIVIEW, zorder=3)
    label_bars(ax, bars, "{:.1f}", 9)
    baseline_lines(ax, 100 * get("random_baseline_accuracy"), 100 * get("majority_baseline_accuracy"))
    ax.set_ylim(0, max(100 * get("accuracy"), 100 * get("majority_baseline_accuracy")) * 1.2)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylabel("Accuracy (%)", color=INK)
    style_axis(ax)
    ax.legend([plt.Line2D([], [], color=INK, linewidth=1.6), plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.6)],
              ["Majority-class baseline", "Random baseline"], fontsize=8, frameon=False, loc="upper right")
    ax.set_title(f"{title}\nmultiview, test split (n={int(get('n'))})", fontsize=10, color=INK)
    plt.tight_layout()
    save(fig, test, out, "test_multiview")


def go_retrieval(df):
    out = PLOTS_DIR / "go_retrieval"
    val = rows_for(df, eval="go_retrieval", pool="validation")
    val = val[val.checkpoint.isin([c for c, _, _ in CHECKPOINTS])]
    variants = [(f"{b}_{q}", f"{name}, {q} query") for b, name in [("bp", "BP"), ("cc", "CC"), ("mf", "MF")]
                for q in ("name", "definition")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for ax, (variant, title) in zip(axes.T.flat, variants):
        rnd = rows_for(val, checkpoint=FINAL, variant=variant, metric="map_random_baseline").value.iloc[0]
        groups = checkpoint_panel(ax, by_checkpoint(val, variant=variant, metric="map"), "{:.3f}", random_y=rnd)
        ax.set_title(f"{title} — mAP", fontsize=10, color=INK, loc="left")
    fig.suptitle(f"GO-term retrieval — all checkpoints, validation split ({POOL_SIZE:,} proteins, terms with ≥10 positives)",
                 color=INK)
    group_legend(fig, dict.fromkeys(groups), [("random", "Random baseline")])
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, val[val.metric.isin(["map", "map_random_baseline", "p10", "r100"])], out, "validation_all_checkpoints")

    test = rows_for(df, eval="go_retrieval", pool="test", checkpoint=FINAL)
    branches = [("bp", "Biological\nProcess"), ("cc", "Cellular\nComponent"), ("mf", "Molecular\nFunction")]
    x = np.arange(len(branches))
    w = 0.36
    get = lambda v, m: rows_for(test, variant=v, metric=m).value.iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w / 2, [get(f"{b}_name", "map") for b, _ in branches], w, color=MULTIVIEW, zorder=3)
    b2 = ax.bar(x + w / 2, [get(f"{b}_definition", "map") for b, _ in branches], w, color=MULTIVIEW_DARK,
                hatch="///", edgecolor="white", linewidth=0, zorder=3)
    label_bars(ax, b1, "{:.3f}", 8.5)
    label_bars(ax, b2, "{:.3f}", 8.5)
    for i, (b, _) in enumerate(branches):
        ax.hlines(get(f"{b}_name", "map_random_baseline"), x[i] - w - 0.04, x[i] + w + 0.04, colors=MUTED,
                  linestyles="--", linewidth=1.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in branches])
    ax.set_ylabel("Mean average precision (mAP)", color=INK)
    ax.set_ylim(0, 0.16)
    style_axis(ax)
    ax.legend([b1, b2, plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.8)],
              ["GO term name as query", "GO term definition as query", "Random baseline"],
              fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title(f"GO-term retrieval — multiview, test split ({POOL_SIZE:,} proteins)", fontsize=10.5, color=INK)
    plt.tight_layout()
    save(fig, test[test.metric.isin(["map", "map_random_baseline", "p10", "r100"])], out, "test_multiview")


def knn_and_probe(df, eval_name, folder, title, value_prefix):
    """k-NN (metric agreement_at_10) and linear probe (probe_accuracy): one panel per task,
    frozen ESM2 as a gray reference bar, validation only."""
    out = PLOTS_DIR / folder
    val = rows_for(df, eval=eval_name, pool="validation")
    val = val[val.checkpoint.isin([c for c, _, _ in CHECKPOINTS])]
    metric = "agreement_at_10" if eval_name == "knn" else "probe_accuracy"
    raw_metric = f"raw_esm2_{metric}"
    floor_metric, floor_kind, floor_label = (("random_baseline", "random", "Random baseline") if eval_name == "knn"
                                             else ("majority_baseline", "majority", "Majority-class baseline"))
    tasks = [("deeploc", "DeepLoc"), ("ec", "EC top-level"), ("go_mf", "GO-Slim-MF")]
    fig, axes = plt.subplots(1, 3, figsize=(17 if eval_name == "knn" else 13, 4.5))
    for ax, (task, name) in zip(axes, tasks):
        t = rows_for(val, variant=task)
        values = {k: 100 * v for k, v in by_checkpoint(t, metric=metric).items()}
        raw = 100 * rows_for(t, checkpoint=FINAL, metric=raw_metric).value.iloc[0]
        floor = 100 * rows_for(t, checkpoint=FINAL, metric=floor_metric).value.iloc[0]
        kwargs = {"random_y": floor} if floor_kind == "random" else {"majority_y": floor}
        groups = checkpoint_panel(ax, values, "{:.1f}", extra_bar=("frozen\nESM2", raw), **kwargs)
        ax.set_ylim(0, 100)
        ax.set_title(f"{name} — {value_prefix} (%)", fontsize=10, color=INK, loc="left")
    fig.suptitle(f"{title} — validation split", color=INK)
    group_legend(fig, dict.fromkeys(groups), [("frozen", "Frozen ESM2 (untrained)"), (floor_kind, floor_label)])
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, val, out, "validation_all_checkpoints")


def alignment_uniformity(df):
    out = PLOTS_DIR / "alignment_uniformity"
    val = rows_for(df, eval="alignment_uniformity", pool="validation")
    val = val[val.checkpoint.isin([c for c, _, _ in CHECKPOINTS])]
    present = [(stem, label, group) for stem, label, group in CHECKPOINTS if stem in set(val.checkpoint)]
    panels = [("alignment", "alignment_mismatched_reference", "mismatched pairs", "Alignment (↓ better)"),
              ("uniformity_protein", "uniformity_random_reference", "random points", "Uniformity, protein side (↓ better)"),
              ("uniformity_text", "uniformity_random_reference", "random points", "Uniformity, text side (↓ better)"),
              ("uniformity_pooled", "uniformity_random_reference", "random points", "Uniformity, pooled (↓ better)")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.6))
    for ax, (metric, ref_metric, ref_label, title) in zip(axes, panels):
        values = {label: rows_for(val, checkpoint=stem, metric=metric).value.iloc[0] for stem, label, _ in present}
        ref = rows_for(val, checkpoint=FINAL, metric=ref_metric).value.iloc[0]
        y = np.arange(len(present))[::-1]
        for yi, (stem, label, group) in zip(y, present):
            ax.plot([ref, values[label]], [yi, yi], color="#dddddd", linewidth=2, zorder=1)
            ax.scatter(values[label], yi, s=80, color=GROUP_COLORS[group], zorder=3)
            ax.annotate(f"{values[label]:.4f}", (values[label], yi), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.axvline(ref, color=MUTED, linestyle="--", linewidth=1.6)
        lo, hi = min(list(values.values()) + [ref]), max(list(values.values()) + [ref])
        ax.set_xlim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        right = ref > np.mean(list(values.values()))
        ax.annotate(ref_label, (ref, len(present) - 0.45), xytext=(-5 if right else 5, 0), textcoords="offset points",
                    ha="right" if right else "left", fontsize=8, color=MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label, _ in present])
        ax.set_ylim(-0.6, len(present) - 0.2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style_axis(ax, "x")
    fig.suptitle("Alignment & uniformity — validation split (1,689 proteins, fixed Function text)", color=INK)
    plt.tight_layout()
    save(fig, val, out, "validation_all_checkpoints")


def main():
    df = pd.read_csv(ALL_RESULTS_PATH)
    df["variant"] = df["variant"].fillna("n/a")
    print("Writing per-eval figures:")
    cross_modal(df)
    zero_shot(df, "deeploc", "deeploc", "DeepLoc zero-shot (9 locations)")
    zero_shot(df, "ec_top_level", "ec_top_level", "EC top-level zero-shot (7 enzyme classes)")
    zero_shot(df, "go_mf", "go_mf", "GO-Slim-MF zero-shot (10 classes, lenient multi-label)")
    go_retrieval(df)
    knn_and_probe(df, "knn", "knn_embedding_quality", "k-NN label agreement (k=10)", "agreement@10")
    knn_and_probe(df, "linear_probe", "linear_probe", "Linear probe", "probe accuracy")
    alignment_uniformity(df)


if __name__ == "__main__":
    main()

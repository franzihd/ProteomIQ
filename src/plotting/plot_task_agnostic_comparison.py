"""
Figure for Results 2.2: task-agnostic checks for M0, multi-positive and multiview.

Usage:
    uv run python src/plotting/plot_task_agnostic_comparison.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

OUTPUT_PATH = "data/plots/overview/2.2_task_agnostic/task_agnostic_comparison.png"
PROBE_KNN_OUTPUT_PATH = "data/plots/overview/2.2_task_agnostic/representation_quality_probe_knn.png"

CHECKPOINTS = [
    ("biobert_unfreeze_0_m0_best", "M0"),
    ("multi_positive_best", "Multi-positive"),
    ("multiview_best", "Multiview"),
]
# ProteomIQ brand colors, identical to plot_ablation_comparison.py (M0 gray reference,
# multi-positive navy, multiview violet). No "final" label: the model is only selected
# at the end of Results 2.2, so this figure must not pre-empt that choice.
BRAND_NAVY, BRAND_VIOLET, BRAND_MUTED = "#14213D", "#7B5CD6", "#5B6477"
COLORS = {"M0": "#8C93A3", "Multi-positive": BRAND_NAVY, "Multiview": BRAND_VIOLET}
TASKS = [("deeploc", "DeepLoc"), ("ec", "EC top-level"), ("go_mf", "GO-Slim-MF")]


def load_json(directory: str, stem: str) -> dict:
    with open(Path(directory) / f"summary_{stem}.json") as f:
        return json.load(f)


INK = BRAND_NAVY
MUTED = BRAND_MUTED
# white halo keeps baseline lines visible where they cross dark bars
HALO = [pe.withStroke(linewidth=3.4, foreground="white")]


def style_axis(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    getattr(ax, f"{grid_axis}axis").grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def dot_panel(ax, values: dict, reference: float, reference_label: str, title: str, fmt: str):
    """Horizontal dot plot on a zoomed axis that includes the reference value -- bars from 0
    would hide differences of ~0.01-0.04 entirely, and the reference is what makes the
    raw number interpretable (Wang & Isola 2020 have no absolute scale)."""
    labels = list(values)
    y = np.arange(len(labels))[::-1]
    for yi, label in zip(y, labels):
        ax.plot([reference, values[label]], [yi, yi], color="#dddddd", linewidth=2, zorder=1)
        ax.scatter(values[label], yi, s=90, color=COLORS[label], zorder=3)
        ax.annotate(fmt.format(values[label]), (values[label], yi), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.axvline(reference, color=MUTED, linestyle="--", linewidth=1.6)
    lo, hi = min(list(values.values()) + [reference]), max(list(values.values()) + [reference])
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ref_on_right = reference > np.mean(list(values.values()))
    ax.annotate(reference_label, (reference, len(labels) - 0.45), xytext=(-6 if ref_on_right else 6, 0),
                textcoords="offset points", ha="right" if ref_on_right else "left", fontsize=8.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    style_axis(ax, "x")


def main(output_path: str):
    align, uniform, probe, knn = {}, {}, {}, {}
    for stem, label in CHECKPOINTS:
        au = load_json("data/evaluation/alignment_uniformity", stem)
        align[label] = au["alignment"]
        uniform[label] = au["uniformity_pooled"]
        align_ref = au["alignment_mismatched_reference"]
        uniform_ref = au["uniformity_random_reference"]

        lp = load_json("data/evaluation/linear_probe", stem)
        probe[label] = {task: lp["results"][task]["probe_accuracy"] * 100 for task, _ in TASKS}
        probe_raw = {task: lp["results"][task]["raw_esm2_probe_accuracy"] * 100 for task, _ in TASKS}
        probe_major = {task: lp["results"][task]["majority_baseline"] * 100 for task, _ in TASKS}

        kn = load_json("data/evaluation/knn_embedding_quality", stem)
        knn[label] = {task: kn["results"][task]["agreement_at_k"]["10"] * 100 for task, _ in TASKS}
        knn_raw = {task: kn["results"][task]["raw_esm2_agreement_at_k"]["10"] * 100 for task, _ in TASKS}
        knn_random = {task: kn["results"][task]["random_baseline"] * 100 for task, _ in TASKS}

    labels = [label for _, label in CHECKPOINTS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"height_ratios": [0.8, 1.2]})

    # the mismatched-pair reference sits far to the right (~2.0); keep it on-axis so the
    # reader sees how far real pairs are from random pairs, not just the tiny gaps between models
    dot_panel(axes[0, 0], align, align_ref, "mismatched pairs", "A  Alignment (↓ better)", "{:.4f}")
    dot_panel(axes[0, 1], uniform, uniform_ref, "random points", "B  Uniformity, pooled (↓ better)", "{:.4f}")

    FROZEN_BAR = dict(facecolor="#EEF0F4", edgecolor=MUTED, hatch="///", linewidth=0.8)

    def grouped_task_panel(ax, data: dict, frozen: dict, floor: dict, floor_label: str, title: str,
                           fs: float = 1.0, frozen_as_bar: bool = False):
        x = np.arange(len(TASKS))
        # frozen_as_bar: frozen ESM2 becomes its own (hatched, light) first bar per group
        # instead of a line across the group -- thesis figure, decided 2026-09-27.
        series = ([("frozen ESM2", frozen)] if frozen_as_bar else []) + [(l, data[l]) for l in labels]
        w = 0.8 / len(series)
        for i, (label, values) in enumerate(series):
            vals = [values[task] for task, _ in TASKS]
            style = FROZEN_BAR if label == "frozen ESM2" else dict(color=COLORS[label])
            bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, zorder=3, **style)
            for bar in bars:
                ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5 * fs, color=INK)
        half = len(series) / 2 * w
        for j, (task, _) in enumerate(TASKS):
            if not frozen_as_bar:
                ax.hlines(frozen[task], x[j] - half, x[j] + half, colors=INK, linewidth=1.8, zorder=4, path_effects=HALO)
            ax.hlines(floor[task], x[j] - half, x[j] + half, colors=MUTED, linestyles=(0, (6, 3)),
                      linewidth=1.6, zorder=4, path_effects=HALO)
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in TASKS], fontsize=10 * fs)
        ax.tick_params(axis="y", labelsize=10 * fs)
        ax.set_title(title, fontsize=10.5 * fs, color=INK, loc="left")
        ax.set_ylim(0, 100)
        style_axis(ax)
        floor_handle = plt.Line2D([], [], color=MUTED, linewidth=1.6, linestyle=(0, (6, 3)))
        if frozen_as_bar:
            ax.legend([floor_handle], [floor_label], fontsize=8.5 * fs, frameon=False, loc="upper left")
        else:
            ax.legend([plt.Line2D([], [], color=INK, linewidth=1.8), floor_handle],
                      ["frozen ESM2 (untrained)", floor_label], fontsize=8.5 * fs, frameon=False, loc="upper left")

    grouped_task_panel(axes[1, 0], probe, probe_raw, probe_major, "majority-class baseline",
                       "C  Linear probe accuracy (%)")
    grouped_task_panel(axes[1, 1], knn, knn_raw, knn_random, "random baseline",
                       "D  k-NN label agreement, k=10 (%)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    rows = []
    for label in labels:
        rows.append(dict(checkpoint=label, metric="alignment", task="n/a", value=align[label]))
        rows.append(dict(checkpoint=label, metric="uniformity_pooled", task="n/a", value=uniform[label]))
        for task, _ in TASKS:
            rows.append(dict(checkpoint=label, metric="linear_probe_accuracy%", task=task, value=probe[label][task]))
            rows.append(dict(checkpoint=label, metric="knn_agreement@10%", task=task, value=knn[label][task]))
    rows.append(dict(checkpoint="reference: mismatched pairs", metric="alignment", task="n/a", value=align_ref))
    rows.append(dict(checkpoint="reference: random points", metric="uniformity_pooled", task="n/a", value=uniform_ref))
    for task, _ in TASKS:
        rows.append(dict(checkpoint="frozen ESM2", metric="linear_probe_accuracy%", task=task, value=probe_raw[task]))
        rows.append(dict(checkpoint="majority-class baseline", metric="linear_probe_accuracy%", task=task, value=probe_major[task]))
        rows.append(dict(checkpoint="frozen ESM2", metric="knn_agreement@10%", task=task, value=knn_raw[task]))
        rows.append(dict(checkpoint="random baseline", metric="knn_agreement@10%", task=task, value=knn_random[task]))
    pd.DataFrame(rows).to_csv(Path(output_path).with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")

    # Thesis version (decided 2026-09-27): alignment/uniformity go into a table, so the
    # figure keeps only the two label-based panels (linear probe + k-NN), same styling.
    # Stacked vertically with larger fonts so the text stays readable at \textwidth.
    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 9.5))
    grouped_task_panel(axes2[0], probe, probe_raw, probe_major, "majority-class baseline",
                       "(A)  Linear probe accuracy (%)", fs=1.3, frozen_as_bar=True)
    grouped_task_panel(axes2[1], knn, knn_raw, knn_random, "random baseline",
                       "(B)  k-NN label agreement, k=10 (%)", fs=1.3, frozen_as_bar=True)
    fig2.legend([plt.Rectangle((0, 0), 1, 1, **FROZEN_BAR)] + handles, ["frozen ESM2 (untrained)"] + labels,
                loc="lower center", ncol=4, fontsize=12, frameon=False,
                bbox_to_anchor=(0.5, -0.01))
    fig2.tight_layout(rect=[0, 0.04, 1, 1], h_pad=2.0)
    fig2.savefig(PROBE_KNN_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    probe_knn_rows = [r for r in rows if r["metric"] in ("linear_probe_accuracy%", "knn_agreement@10%")]
    pd.DataFrame(probe_knn_rows).to_csv(Path(PROBE_KNN_OUTPUT_PATH).with_suffix(".csv"), index=False)
    print(f"Saved to {PROBE_KNN_OUTPUT_PATH} (+ .csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)

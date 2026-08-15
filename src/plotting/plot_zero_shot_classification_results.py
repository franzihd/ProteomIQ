"""
Bar charts comparing the three zero-shot forced-choice classification evals
(DeepLoc subcellular location, EC top-level enzyme class, GO-Slim-MF
molecular function) against their own random and majority-class baselines,
across the three prompt conditions (Label-only / Engineered / Ensemble --
see zero_shot_*_eval.py's --condition, mirrors the ablation CLIP's own paper
uses to isolate how much prompt engineering and ensembling each contribute).

Reads the summary JSONs these three evals already produce (`build_*_eval_set.py`
+ `zero_shot_*_eval.py --condition {label_only,engineered,ensemble}` must be
run first, all three conditions, for a given checkpoint) -- this script does
no computation of its own, just visualizes what's already on disk, so
rerunning it after a new checkpoint's evals finish reproduces the comparison
with no manual number-copying. See CLAUDE.md section 11o for the discussion
these plots support: GO-MF is the only one of the three that clears its
majority-class baseline.

Note on comparability: DeepLoc/EC use single-label "strict" accuracy against a
fixed external class taxonomy; GO-MF uses "lenient" (any-valid-class) accuracy
against a data-derived class list (see CLAUDE.md 11o) -- its own majority
baseline is computed under that same lenient rule. Each eval's accuracy is
only ever compared to its OWN baselines, never across evals -- the three
accuracy numbers are not a single unified metric.

Produces FOUR files: one combined 3-panel comparison (as before) plus one
standalone chart per eval (same drawing code, reused via plot_one_panel, just
rendered onto its own figure so each can be used/shared independently).

Usage:
    uv run python src/plotting/plot_zero_shot_classification_results.py \\
        --checkpoint-stem final_v2_dims_masked_best \\
        --output-dir data/plots
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

BAR_COLOR = "#16B3B3"
RANDOM_COLOR = "#A6A49B"    # muted gray -- weaker reference line
MAJORITY_COLOR = "#3C3489"  # matches plot_cross_modal_eval.py's purple -- the harder-to-beat reference line
TEXT = "#26215C"

# (output filename stem, title, [(condition label, summary json suffix), ...])
# Ordered Label-only -> Engineered -> Ensemble (N=0 -> N=1 -> N=5), matching
# the increasing-sophistication order of CLIP's own prompt-engineering
# ablation. Suffixes match zero_shot_*_eval.py's CONDITION_SUFFIXES.
EVAL_SPECS = [
    ("deeploc", "DeepLoc (subcellular location)", [
        ("Label-only", "_9way_human_label_only"),
        ("Engineered", "_9way_human_engineered"),
        ("Ensemble", "_9way_human"),
    ], "deeploc"),
    ("ec_top_level", "EC top-level (enzyme class)", [
        ("Label-only", "_label_only"),
        ("Engineered", "_engineered"),
        ("Ensemble", ""),
    ], "ec"),
    ("go_mf", "GO-Slim-MF (molecular function)", [
        ("Label-only", "_label_only"),
        ("Engineered", "_engineered"),
        ("Ensemble", ""),
    ], "go_mf"),
]


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_panels(checkpoint_stem: str) -> list[dict]:
    """One panel per eval: its three prompt conditions, and its own
    random/majority baselines (majority baseline doesn't vary by condition,
    since it depends only on the class distribution, not the prompt)."""
    eval_dir = Path("data/evaluation")
    panels = []
    for out_stem, title, variant_specs, file_prefix in EVAL_SPECS:
        variants = [
            (label, load_summary(eval_dir / f"{file_prefix}_zero_shot_summary_{checkpoint_stem}{suffix}.json"))
            for label, suffix in variant_specs
        ]
        panels.append({"out_stem": out_stem, "title": title, "variants": variants})
    return panels


def plot_one_panel(ax, panel: dict, wrap_title: bool = False):
    """Draws one eval's bars + baseline reference lines onto the given axis.
    Shared by both the combined figure and the standalone per-eval figures so
    there is exactly one place that defines what a panel looks like."""
    ax.set_facecolor("white")
    labels = [v[0] for v in panel["variants"]]
    accuracies = [v[1]["all"]["accuracy"] * 100 for v in panel["variants"]]
    random_baseline = panel["variants"][0][1]["all"]["random_baseline_accuracy"] * 100
    majority_baseline = panel["variants"][0][1]["all"]["majority_baseline_accuracy"] * 100
    majority_class = panel["variants"][0][1]["all"]["majority_class"]

    x = range(len(labels))
    bars = ax.bar(x, accuracies, width=0.55, color=BAR_COLOR, zorder=3)
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f"{acc:.1f}%", (bar.get_x() + bar.get_width() / 2, acc),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, fontweight="bold", color=BAR_COLOR)

    ax.axhline(random_baseline, color=RANDOM_COLOR, linestyle="--", linewidth=1.5, zorder=2)
    ax.axhline(majority_baseline, color=MAJORITY_COLOR, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(len(labels) - 0.4, random_baseline, f" random {random_baseline:.1f}%",
            color=RANDOM_COLOR, fontsize=8.5, va="bottom", ha="left")
    ax.text(len(labels) - 0.4, majority_baseline, f" majority ({majority_class}) {majority_baseline:.1f}%",
            color=MAJORITY_COLOR, fontsize=8.5, va="bottom", ha="left")

    ax.set_xticks(list(x))
    title = panel["title"].replace(" (", "\n(") if wrap_title else panel["title"]
    # 3 conditions ("Label-only" / "Engineered" / "Ensemble") need more
    # horizontal room than the old 2-variant labels did -- rotated labels
    # with right-alignment stop them from overlapping into each other,
    # especially in the narrower combined-figure panels (see
    # plot_combined's 3-panels-in-13-inches layout).
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=TEXT, pad=10)
    ax.tick_params(axis="both", which="both", length=0, labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=1.0, zorder=0)
    ax.set_ylim(0, 55)


def plot_combined(panels: list[dict], checkpoint_stem: str, output_path: Path):
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5.5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, panel in zip(axes, panels):
        plot_one_panel(ax, panel, wrap_title=True)

    axes[0].set_ylabel("Accuracy (%)", fontsize=11, color=TEXT)

    fig.suptitle("Zero-shot forced-choice classification: model vs. random and majority-class baselines",
                 fontsize=13.5, fontweight="bold", color=TEXT, y=1.02)
    fig.text(0.5, -0.02,
             f"Checkpoint: {checkpoint_stem}  |  GO-MF's accuracy/majority-baseline use lenient "
             f"(any-valid-class) scoring, not single-label -- see CLAUDE.md section 11o.\n"
             f"Panels are not on a single unified metric -- compare each eval to its OWN baselines, not across panels.",
             ha="center", fontsize=8.5, color="#666666")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def plot_standalone(panel: dict, checkpoint_stem: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor("white")

    plot_one_panel(ax, panel, wrap_title=False)
    ax.set_ylabel("Accuracy (%)", fontsize=11, color=TEXT)

    fig.text(0.5, -0.03, f"Checkpoint: {checkpoint_stem}", ha="center", fontsize=8.5, color="#666666")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def plot(checkpoint_stem: str, output_dir: str):
    panels = build_panels(checkpoint_stem)
    out_dir = Path(output_dir)

    plot_combined(panels, checkpoint_stem, out_dir / "zero_shot_classification_results.png")
    for panel in panels:
        plot_standalone(panel, checkpoint_stem, out_dir / f"zero_shot_{panel['out_stem']}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-stem", default="final_v2_dims_masked_best",
                        help="checkpoint filename stem used to build the summary JSON filenames "
                             "(e.g. 'final_v2_dims_masked_best' -> "
                             "deeploc_zero_shot_summary_final_v2_dims_masked_best.json, etc.)")
    parser.add_argument("--output-dir", default="data/plots")
    args = parser.parse_args()
    plot(args.checkpoint_stem, args.output_dir)

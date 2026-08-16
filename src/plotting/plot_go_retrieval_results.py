"""
Bar charts for the GO-term zero-shot text-to-protein RETRIEVAL eval
(build_go_retrieval_eligible_terms.py / zero_shot_go_retrieval_eval.py, see
CLAUDE.md section 11w's Project layout entry -- results not yet written up
as a numbered Timeline section). One combined figure per metric, one panel
per GO branch (BP/CC/MF), comparing the two query representations ("name"
vs "definition") against the analytical random-ranking baseline.

Reads the summary JSONs the eval script already produces (run
zero_shot_go_retrieval_eval.py --branch all --min-positives N first) --
this script does no computation of its own, just visualizes what's already
on disk, so rerunning it after a new checkpoint's eval finishes reproduces
the comparison with no manual number-copying.

Three figures, matching this project's existing "headline metric + practical
companion metric" split (mirrors pathway AUROC vs. pathway recall@K,
CLAUDE.md section 10l):
  - go_retrieval_map.png       -- mAP per branch (the single fairest number;
    unaffected by the top-K ceiling effect the eval script's own docstring
    describes -- see its "Normalized Recall@K" section).
  - go_retrieval_recall_at_k.png -- normalized Recall@K per branch, grouped
    by K (10/50/100). Normalized (not raw) R@K is used here because raw R@K
    is mechanically capped at min(K, n_positives)/n_positives for any term
    with more positives than K, which would silently understate how well a
    branch with many high-positive-count terms (e.g. CC) is actually doing
    relative to what's achievable -- see the eval script's module docstring.
  - go_retrieval_precision_at_k.png -- raw Precision@K per branch, grouped
    by K (10/50/100). No ceiling-effect correction needed here (unlike
    Recall@K): a term's own positive count never caps how high precision
    can go, so the eval script's mean_P_at_K is plotted as-is.

Colors reused from this project's existing cross-modal-retrieval bar chart
(plot_cross_modal_eval.py) for a consistent two-series palette across the
thesis's plot set, not a new choice made here.

Usage:
    uv run python src/plotting/plot_go_retrieval_results.py \\
        --checkpoint-stem final_v2_dims_masked_best \\
        --min-positives 10 \\
        --output-dir data/plots
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

NAME_COLOR = "#16B3B4"       # matches plot_cross_modal_eval.py's "Protein -> Text" teal
DEF_COLOR = "#BD98CF"        # matches plot_cross_modal_eval.py's "Text -> Protein" lavender
RANDOM_COLOR = "#A6A49B"     # matches plot_zero_shot_classification_results.py's baseline gray
TEXT = "#26215C"             # matches plot_zero_shot_classification_results.py's ink color

BRANCHES = [("bp", "Biological Process"), ("cc", "Cellular Component"), ("mf", "Molecular Function")]
RESULTS_DIR = Path("data/evaluation/go_retrieval/results")


def load_summaries(checkpoint_stem: str, min_positives: int) -> dict:
    summaries = {}
    for branch, _ in BRANCHES:
        path = RESULTS_DIR / f"summary_{branch}_{checkpoint_stem}_min{min_positives}.json"
        with open(path) as f:
            summaries[branch] = json.load(f)
    return summaries


def plot_map(summaries: dict, checkpoint_stem: str, min_positives: int, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (branch, title) in zip(axes, BRANCHES):
        s = summaries[branch]["query_types"]
        name_map = s["name"]["mAP"] * 100
        def_map = s["definition"]["mAP"] * 100
        # Random baseline is prevalence-based and depends only on which terms
        # were scored -- identical between query types whenever both scored
        # the same term set (n_terms_skipped_no_text == 0 for both here), so
        # a single reference line is drawn rather than two overlapping ones.
        random_baseline = s["name"]["random_baseline_mAP"] * 100

        x = [0, 1]
        vals = [name_map, def_map]
        colors = [NAME_COLOR, DEF_COLOR]
        bars = ax.bar(x, vals, width=0.55, color=colors, zorder=3)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", (bar.get_x() + bar.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=10, fontweight="bold", color=TEXT)

        ax.axhline(random_baseline, color=RANDOM_COLOR, linestyle="--", linewidth=1.5, zorder=2)
        # White background patch so the dashed line doesn't run straight
        # through the label text (confirmed as a real collision, not a
        # viewer artifact, by cropping and zooming into a rendered output).
        ax.text(1.42, random_baseline, f" random {random_baseline:.1f}% ",
                color=RANDOM_COLOR, fontsize=8.5, va="center", ha="left", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

        ax.set_xlim(-0.55, 1.9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Name", "Definition"], fontsize=9.5)
        n_terms = summaries[branch]["n_eligible_terms"]
        ax.set_title(f"GO-{title}\n(n={n_terms} terms)", fontsize=11, fontweight="bold", color=TEXT, pad=10)
        ax.tick_params(axis="both", which="both", length=0, labelcolor=TEXT)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=1.0, zorder=0)

    axes[0].set_ylabel("mAP (%)", fontsize=11, color=TEXT)
    ymax = max(summaries[b]["query_types"][qt]["mAP"] for b, _ in BRANCHES for qt in ("name", "definition")) * 100
    axes[0].set_ylim(0, ymax * 1.35)

    fig.suptitle("GO-term zero-shot text-to-protein retrieval — mean Average Precision",
                 fontsize=13.5, fontweight="bold", color=TEXT, y=1.03)
    fig.text(0.5, -0.03,
              f"Checkpoint: {checkpoint_stem}  |  min_positives={min_positives}  |  "
              f"Validation-split candidate pool ({summaries['bp']['candidate_pool_size']:,} proteins) — "
              f"never the held-out test split.\nmAP is the single fairest headline number here "
              f"(unaffected by the top-K ceiling effect — see go_retrieval_recall_at_k.png).",
              ha="center", fontsize=8.5, color="#666666")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def plot_recall_at_k(summaries: dict, checkpoint_stem: str, min_positives: int, output_path: Path):
    k_values = summaries["bp"]["k_values"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5), sharey=True)
    fig.patch.set_facecolor("white")

    x = np.arange(len(k_values))
    w = 0.35

    for ax, (branch, title) in zip(axes, BRANCHES):
        s = summaries[branch]["query_types"]
        name_vals = [s["name"]["mean_R_at_K_normalized"][str(k)] * 100 for k in k_values]
        def_vals = [s["definition"]["mean_R_at_K_normalized"][str(k)] * 100 for k in k_values]
        random_vals = [s["name"]["random_baseline_R_at_K_normalized"][str(k)] * 100 for k in k_values]

        bars1 = ax.bar(x - w / 2, name_vals, w, color=NAME_COLOR, zorder=3, label="Name")
        bars2 = ax.bar(x + w / 2, def_vals, w, color=DEF_COLOR, zorder=3, label="Definition")
        ax.plot(x, random_vals, color=RANDOM_COLOR, linestyle="--", linewidth=1.5,
                marker="o", markersize=4, zorder=2, label="Random baseline")

        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width() / 2, h),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=8, color=TEXT)

        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in k_values], fontsize=9.5)
        n_terms = summaries[branch]["n_eligible_terms"]
        ax.set_title(f"GO-{title}\n(n={n_terms} terms)", fontsize=11, fontweight="bold", color=TEXT, pad=10)
        ax.tick_params(axis="both", which="both", length=0, labelcolor=TEXT)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=1.0, zorder=0)

    axes[0].set_ylabel("Normalized Recall@K (%)", fontsize=11, color=TEXT)
    axes[0].set_ylim(0, 100)
    axes[-1].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)

    fig.suptitle("GO-term zero-shot text-to-protein retrieval — Normalized Recall@K",
                 fontsize=13.5, fontweight="bold", color=TEXT, y=1.03)
    fig.text(0.5, -0.03,
              f"Checkpoint: {checkpoint_stem}  |  min_positives={min_positives}  |  "
              "normalized R@K = hits@K / min(K, n_positives) — how close to the best\n"
              "possible ranking, correcting for terms with more positives than K (raw R@K "
              "is mechanically capped for those terms; see script docstring).",
              ha="center", fontsize=8.5, color="#666666")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def plot_precision_at_k(summaries: dict, checkpoint_stem: str, min_positives: int, output_path: Path):
    """Precision@K = of the top-K ranked candidates, what fraction are true
    positives -- no ceiling-effect correction needed here (unlike Recall@K,
    see plot_recall_at_k's docstring note): a term's positive count doesn't
    cap how high precision can go, so raw P@K (not a normalized variant) is
    the number reported, straight from the eval script's own mean_P_at_K."""
    k_values = summaries["bp"]["k_values"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5), sharey=True)
    fig.patch.set_facecolor("white")

    x = np.arange(len(k_values))
    w = 0.35

    for ax, (branch, title) in zip(axes, BRANCHES):
        s = summaries[branch]["query_types"]
        name_vals = [s["name"]["mean_P_at_K"][str(k)] * 100 for k in k_values]
        def_vals = [s["definition"]["mean_P_at_K"][str(k)] * 100 for k in k_values]
        random_vals = [s["name"]["random_baseline_P_at_K"][str(k)] * 100 for k in k_values]

        bars1 = ax.bar(x - w / 2, name_vals, w, color=NAME_COLOR, zorder=3, label="Name")
        bars2 = ax.bar(x + w / 2, def_vals, w, color=DEF_COLOR, zorder=3, label="Definition")
        ax.plot(x, random_vals, color=RANDOM_COLOR, linestyle="--", linewidth=1.5,
                marker="o", markersize=4, zorder=2, label="Random baseline")

        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=8, color=TEXT)

        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in k_values], fontsize=9.5)
        n_terms = summaries[branch]["n_eligible_terms"]
        ax.set_title(f"GO-{title}\n(n={n_terms} terms)", fontsize=11, fontweight="bold", color=TEXT, pad=10)
        ax.tick_params(axis="both", which="both", length=0, labelcolor=TEXT)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=1.0, zorder=0)

    axes[0].set_ylabel("Precision@K (%)", fontsize=11, color=TEXT)
    ymax = max(
        summaries[b]["query_types"][qt]["mean_P_at_K"][str(k)]
        for b, _ in BRANCHES for qt in ("name", "definition") for k in k_values
    ) * 100
    axes[0].set_ylim(0, ymax * 1.3)
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT)

    fig.suptitle("GO-term zero-shot text-to-protein retrieval — Precision@K",
                 fontsize=13.5, fontweight="bold", color=TEXT, y=1.03)
    fig.text(0.5, -0.03,
              f"Checkpoint: {checkpoint_stem}  |  min_positives={min_positives}  |  "
              "Precision@K = of the top-K ranked candidates, the fraction that\n"
              "are true positives for that GO term (macro-averaged across eligible terms).",
              ha="center", fontsize=8.5, color="#666666")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def plot(checkpoint_stem: str, min_positives: int, output_dir: str):
    summaries = load_summaries(checkpoint_stem, min_positives)
    out_dir = Path(output_dir)
    plot_map(summaries, checkpoint_stem, min_positives, out_dir / "go_retrieval_map.png")
    plot_recall_at_k(summaries, checkpoint_stem, min_positives, out_dir / "go_retrieval_recall_at_k.png")
    plot_precision_at_k(summaries, checkpoint_stem, min_positives, out_dir / "go_retrieval_precision_at_k.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-stem", default="final_v2_dims_masked_best",
                        help="checkpoint filename stem used to build the summary JSON filenames")
    parser.add_argument("--min-positives", type=int, default=10)
    parser.add_argument("--output-dir", default="data/plots")
    args = parser.parse_args()
    plot(args.checkpoint_stem, args.min_positives, args.output_dir)

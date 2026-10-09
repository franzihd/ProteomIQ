"""
Plots the k-NN embedding-space-quality results (knn_embedding_quality_eval.py)
-- one figure, one panel per label set (DeepLoc / EC / GO-MF), grouped bars
per K value, one color per checkpoint compared, plus horizontal reference
lines for the random-chance baseline and the raw-ESM2-no-training baseline
(same "never trust a number without a baseline" discipline as every other
eval in this project).

Usage:
    uv run python src/plotting/plot_knn_embedding_quality.py \\
        --checkpoint-stems multiview_best multi_positive_best final_v2_dims_masked_best
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = Path("data/evaluation/knn_embedding_quality")
EVAL_ORDER = ["deeploc", "ec", "go_mf"]
EVAL_LABELS = {"deeploc": "DeepLoc\n(location)", "ec": "EC top-level\n(enzyme class)",
               "go_mf": "GO-Slim-MF\n(molecular function)"}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def load_summary(stem: str) -> dict:
    path = INPUT_DIR / f"summary_{stem}.json"
    with open(path) as f:
        return json.load(f)


def main(checkpoint_stems: list[str], output_dir: str):
    summaries = {stem: load_summary(stem) for stem in checkpoint_stems}

    fig, axes = plt.subplots(1, len(EVAL_ORDER), figsize=(5.5 * len(EVAL_ORDER), 5))
    if len(EVAL_ORDER) == 1:
        axes = [axes]

    for ax, eval_name in zip(axes, EVAL_ORDER):
        first = next(iter(summaries.values()))["results"][eval_name]
        k_values = sorted(int(k) for k in first["agreement_at_k"].keys())
        random_baseline = first["random_baseline"] * 100
        raw_esm2 = [first["raw_esm2_agreement_at_k"][str(k)] * 100 for k in k_values]

        n_groups = len(checkpoint_stems)
        bar_width = 0.8 / n_groups
        x = np.arange(len(k_values))

        for i, stem in enumerate(checkpoint_stems):
            res = summaries[stem]["results"][eval_name]
            vals = [res["agreement_at_k"][str(k)] * 100 for k in k_values]
            offset = (i - (n_groups - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width, label=stem, color=COLORS[i % len(COLORS)])

        # Reference lines: random baseline and raw-ESM2 (checkpoint-independent)
        ax.axhline(random_baseline, color="gray", linestyle=":", linewidth=1.5,
                   label="random baseline" if ax is axes[0] else None)
        ax.plot(x, raw_esm2, color="black", marker="D", markersize=5, linestyle="--",
                linewidth=1.5, label="raw ESM2 (no training)" if ax is axes[0] else None)

        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in k_values])
        ax.set_ylim(0, 100)
        ax.set_ylabel("neighbor-label agreement (%)")
        ax.set_title(EVAL_LABELS.get(eval_name, eval_name))
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("k-NN embedding-space-quality check (protein-side only, no text)")
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems_tag = "_vs_".join(checkpoint_stems)
    out_path = out_dir / f"knn_embedding_quality_{stems_tag}.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-stems", nargs="+", required=True,
                        help="checkpoint stems to compare, e.g. multiview_best multi_positive_best "
                             "-- each must already have a saved summary_<stem>.json "
                             "(run knn_embedding_quality_eval.py first)")
    parser.add_argument("--output-dir", default="data/plots/knn_embedding_quality")
    args = parser.parse_args()
    main(args.checkpoint_stems, args.output_dir)

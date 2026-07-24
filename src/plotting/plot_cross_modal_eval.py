"""
Bar chart of cross_modal_eval.py's bidirectional closed-test-set Recall@K.

Usage:
    uv run python src/plotting/plot_cross_modal_eval.py \\
        --checkpoint data/runs/full_proteome/final_hparams_masked_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --splits-file data/protein_splits_full_proteome.json \\
        --output data/evaluation/cross_modal_eval.png
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation.cross_modal_eval import compute_cross_modal_eval, KS


def plot(checkpoint_path: str, biobert_filename: str, splits_file: str, output_path: str):
    metrics = compute_cross_modal_eval(checkpoint_path, biobert_filename, splits_file)
    t2p = [metrics["text_to_protein"][f"r{k}"] for k in KS]
    p2t = [metrics["protein_to_text"][f"r{k}"] for k in KS]

    x = np.arange(len(KS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - w / 2, t2p, w, label="Text → Protein", color="#3C3489")  # AI/language side
    bars2 = ax.bar(x + w / 2, p2t, w, label="Protein → Text", color="#0F6E56")  # biology/data side

    ax.set_xticks(x)
    ax.set_xticklabels([f"R@{k}" for k in KS])
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"Cross-Modal Retrieval — Closed Test-Set Recall@K\n"
                 f"({metrics['pool_size']:,} held-out proteins, {Path(checkpoint_path).stem})")
    ax.legend()
    ax.set_ylim(0, max(max(t2p), max(p2t)) * 1.15)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    median_t2p = metrics["text_to_protein"]["median_rank"]
    median_p2t = metrics["protein_to_text"]["median_rank"]
    fig.text(0.5, 0.005,
              f"Median rank — Text→Protein: {median_t2p:.0f} of {metrics['pool_size']:,}   |   "
              f"Protein→Text: {median_p2t:.0f} of {metrics['pool_size']:,}",
              ha="center", fontsize=9.5, color="#444444")

    plt.tight_layout(rect=[0, 0.035, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--biobert-filename", default="biobert_embeddings.h5")
    parser.add_argument("--splits-file", default="data/protein_splits.json")
    parser.add_argument("--output", default="data/evaluation/cross_modal_eval.png")
    args = parser.parse_args()
    plot(args.checkpoint, args.biobert_filename, args.splits_file, args.output)

"""
Protein-level R@K for the no-abundance ablation checkpoint.

Unlike recall_at_k.py (which loops over held-out PATIENTS, and so mostly
re-tests proteins the model already saw via other patients), this evaluates
proteins the model has never seen in ANY patient, ANY cohort at all
(data/protein_splits.json's "test" list) — a genuine generalization test,
not recall of training content.

Each test protein's own text is ranked against a candidate pool of ALL
proteins (train + validation + test, ~15,733 total) — a mix of familiar and
genuinely novel proteins, mirroring what a researcher's real uploaded sample
would actually contain.

Usage:
    uv run python src/model/recall_at_k_protein_level.py \\
        --checkpoint data/runs/.../checkpoint_full_proteome_masked_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --splits-file data/protein_splits_full_proteome.json
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import numpy as np
from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead

COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]


def compute_ranks(checkpoint_path: str, biobert_filename: str = "biobert_embeddings.h5",
                   splits_file: str = "data/protein_splits.json"):
    with open(splits_file) as f:
        protein_splits = json.load(f)
    test_protein_ids = set(protein_splits["test"])

    # Full candidate pool: every protein across all 9 cohorts, no exclusions —
    # train + validation + test proteins all together. biobert_filename must
    # match whatever text variant the checkpoint's text_head was trained on
    # (plain / masked / masked+GO) -- otherwise the text_head sees raw BioBERT
    # vectors of a different text distribution than it was trained on, and
    # the resulting R@K number does not reflect the model's real capability.
    dataset = ProteomicsDataset(cohorts=COHORTS, use_abundance=False, biobert_filename=biobert_filename)
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])
    text_matrix = torch.stack([dataset.protein_text[pid] for pid in protein_ids])
    print(f"Candidate pool: {len(protein_ids):,} proteins "
          f"({len(test_protein_ids):,} of which are held-out test proteins)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head = ProjectionHead(input_dim=640, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    protein_head.load_state_dict(ckpt["protein_head"])
    text_head.load_state_dict(ckpt["text_head"])
    protein_head.eval()
    text_head.eval()

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))
        text_vecs = text_head(text_matrix.to(device))

        sim = text_vecs @ protein_vecs.T  # (num_proteins, num_proteins)

        ranks = []
        for i, pid in enumerate(protein_ids):
            if pid not in test_protein_ids:
                continue
            rank = (sim[i] > sim[i, i]).sum().item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "ranks": ranks,
        "pool_size": len(protein_ids),
        "r1": float((ranks <= 1).mean() * 100),
        "r10": float((ranks <= 10).mean() * 100),
        "r50": float((ranks <= 50).mean() * 100),
        "r100": float((ranks <= 100).mean() * 100),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/.../checkpoint_no_abundance_best.pt")
    parser.add_argument("--biobert-filename", default="biobert_embeddings.h5",
                        help="must match the text variant the checkpoint was trained on, "
                             "e.g. biobert_embeddings_masked.h5 or biobert_embeddings_masked_go.h5")
    parser.add_argument("--splits-file", default="data/protein_splits.json",
                        help="e.g. data/protein_splits_full_proteome.json for the full-proteome split")
    args = parser.parse_args()

    metrics = compute_ranks(args.checkpoint, biobert_filename=args.biobert_filename,
                             splits_file=args.splits_file)
    total = len(metrics["ranks"])
    print(f"\nR@1:   {metrics['r1']:.2f}%")
    print(f"R@10:  {metrics['r10']:.2f}%")
    print(f"R@50:  {metrics['r50']:.2f}%")
    print(f"R@100: {metrics['r100']:.2f}%")
    print(f"\nEvaluated {total:,} held-out test proteins "
          f"against a candidate pool of {metrics['pool_size']:,} proteins.")

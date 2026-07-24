"""
Basic bidirectional cross-modal retrieval eval (the standard CLIP protocol):
closed test-set gallery, R@K computed in both directions.

Distinct from recall_at_k_protein_level.py, which uses a mixed train+val+test
candidate pool to simulate a realistic researcher upload, and only queries
text->protein. This script instead restricts BOTH the query set and the
candidate pool to the held-out test proteins only (data/protein_splits.json),
so results are directly comparable to standard CLIP retrieval benchmarks
(e.g. Flickr30k/COCO closed test-set R@1/5/10) and cleanly isolate the
checkpoint as the only variable across a run-to-run comparison.

Usage:
    uv run python src/evaluation/cross_modal_eval.py \\
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
KS = [1, 5, 10, 50]


def recall_at_k(sim: torch.Tensor, ks: list[int]) -> dict:
    n = sim.shape[0]
    diag = sim.diag().unsqueeze(1)
    ranks = (sim > diag).sum(dim=1).cpu().numpy() + 1
    metrics = {f"r{k}": float((ranks <= k).mean() * 100) for k in ks}
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def compute_cross_modal_eval(checkpoint_path: str, biobert_filename: str,
                              splits_file: str = "data/protein_splits.json"):
    with open(splits_file) as f:
        protein_splits = json.load(f)
    test_protein_ids = set(protein_splits["test"])

    # restrict_to_protein_ids narrows the dataset to the test split only —
    # both the query set and the gallery are the closed 1,573-protein test set.
    dataset = ProteomicsDataset(
        cohorts=COHORTS, use_abundance=False,
        biobert_filename=biobert_filename,
        restrict_to_protein_ids=test_protein_ids,
    )
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])
    text_matrix = torch.stack([dataset.protein_text[pid] for pid in protein_ids])
    print(f"Closed test-set pool: {len(protein_ids):,} proteins "
          f"(expected {len(test_protein_ids):,} minus any never measured in training data)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Older checkpoints predate saving these keys -- fall back to the original defaults.
    hidden_dim = ckpt.get("hidden_dim", 512)
    output_dim = ckpt.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(ckpt["protein_head"])
    text_head.load_state_dict(ckpt["text_head"])
    protein_head.eval()
    text_head.eval()

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))
        text_vecs = text_head(text_matrix.to(device))
        sim = text_vecs @ protein_vecs.T  # (n_test, n_test), row i = text_i vs all proteins

    text_to_protein = recall_at_k(sim, KS)
    protein_to_text = recall_at_k(sim.T, KS)

    return {
        "pool_size": len(protein_ids),
        "text_to_protein": text_to_protein,
        "protein_to_text": protein_to_text,
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

    metrics = compute_cross_modal_eval(args.checkpoint, args.biobert_filename, args.splits_file)

    print(f"\nPool size: {metrics['pool_size']:,}\n")
    print("Text -> Protein (query with annotation text, retrieve protein):")
    for k in KS:
        print(f"  R@{k}: {metrics['text_to_protein'][f'r{k}']:.2f}%")
    print(f"  Median rank: {metrics['text_to_protein']['median_rank']:.0f}")
    print("\nProtein -> Text (query with protein embedding, retrieve annotation text):")
    for k in KS:
        print(f"  R@{k}: {metrics['protein_to_text'][f'r{k}']:.2f}%")
    print(f"  Median rank: {metrics['protein_to_text']['median_rank']:.0f}")

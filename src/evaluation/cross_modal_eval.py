"""
Basic bidirectional cross-modal retrieval eval (the standard CLIP protocol):
closed gallery, R@K computed in both directions.

Distinct from recall_at_k_protein_level.py, which uses a mixed train+val+test
candidate pool to simulate a realistic researcher upload, and only queries
text->protein. This script instead restricts BOTH the query set and the
candidate pool to one split only (data/protein_splits.json), so results are
directly comparable to standard CLIP retrieval benchmarks (e.g. Flickr30k/COCO
closed test-set R@1/5/10) and cleanly isolate the checkpoint as the only
variable across a run-to-run comparison.

--split-key {validation,test} (default: validation). Before this flag existed,
the pool was hardcoded to the test split -- meaning every checkpoint
comparison run across the unfreeze-depth (11x) and training-text ablations
(11z, 11ac, 11ae) repeatedly used the test split for model selection, the
same category of leakage caught and fixed for pathway AUROC in 11j, just not
caught here until later. Fix: default to "validation" so repeat comparisons
can no longer touch test by accident; "test" must be passed explicitly, and
is meant to be spent exactly once, on the single final checkpoint (same
discipline as unseen_test in the zero-shot classification evals and the
standing GO-retrieval test_only rule, 11ab).

Usage:
    uv run python src/evaluation/cross_modal_eval.py \\
        --checkpoint data/runs/.../checkpoint_full_proteome_masked_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --splits-file data/protein_splits_full_proteome.json \\
        --split-key validation
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
OUTPUT_DIR = Path("data/evaluation/cross_modal_retrieval")


def recall_at_k(sim: torch.Tensor, ks: list[int]) -> dict:
    n = sim.shape[0]
    diag = sim.diag().unsqueeze(1)
    ranks = (sim > diag).sum(dim=1).cpu().numpy() + 1
    metrics = {f"r{k}": float((ranks <= k).mean() * 100) for k in ks}
    metrics["median_rank"] = float(np.median(ranks))
    return metrics


def compute_cross_modal_eval(checkpoint_path: str, biobert_filename: str,
                              splits_file: str = "data/protein_splits.json",
                              split_key: str = "validation"):
    if split_key not in ("validation", "test"):
        raise ValueError(f"split_key must be 'validation' or 'test', got {split_key!r}")

    with open(splits_file) as f:
        protein_splits = json.load(f)
    pool_protein_ids = set(protein_splits[split_key])

    # restrict_to_protein_ids narrows the dataset to one split only — both
    # the query set and the gallery are the closed pool for that split.
    dataset = ProteomicsDataset(
        cohorts=COHORTS, use_abundance=False,
        biobert_filename=biobert_filename,
        restrict_to_protein_ids=pool_protein_ids,
    )
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])
    text_matrix = torch.stack([dataset.protein_text[pid] for pid in protein_ids])
    print(f"Closed {split_key}-set pool: {len(protein_ids):,} proteins "
          f"(expected {len(pool_protein_ids):,} minus any never measured in training data)")

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
        "split_key": split_key,
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
    parser.add_argument("--split-key", choices=["validation", "test"], default="validation",
                        help="pool to evaluate against. 'validation' for repeat checkpoint "
                             "comparisons; 'test' is reserved -- spend it exactly once, on "
                             "the single final chosen checkpoint (see module docstring).")
    parser.add_argument("--no-save", action="store_true",
                        help="skip writing the summary JSON (default: always save, matching "
                             "the other zero-shot eval scripts' convention)")
    args = parser.parse_args()

    metrics = compute_cross_modal_eval(args.checkpoint, args.biobert_filename,
                                        args.splits_file, args.split_key)

    print(f"\nPool size: {metrics['pool_size']:,} ({metrics['split_key']} split)\n")
    print("Text -> Protein (query with annotation text, retrieve protein):")
    for k in KS:
        print(f"  R@{k}: {metrics['text_to_protein'][f'r{k}']:.2f}%")
    print(f"  Median rank: {metrics['text_to_protein']['median_rank']:.0f}")
    print("\nProtein -> Text (query with protein embedding, retrieve annotation text):")
    for k in KS:
        print(f"  R@{k}: {metrics['protein_to_text'][f'r{k}']:.2f}%")
    print(f"  Median rank: {metrics['protein_to_text']['median_rank']:.0f}")

    if not args.no_save:
        checkpoint_stem = Path(args.checkpoint).stem
        # "validation" keeps unsuffixed filenames (nothing used this convention
        # before, so there's no legacy file to collide with); "test" gets its
        # own suffix, same pattern as zero_shot_go_retrieval_eval.py.
        split_suffix = "" if args.split_key == "validation" else f"_{args.split_key}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"summary_{checkpoint_stem}{split_suffix}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSummary saved to {out_path}")

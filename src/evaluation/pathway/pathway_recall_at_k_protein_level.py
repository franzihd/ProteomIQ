"""
Protein-level pathway recall@K for the no-abundance ablation checkpoint.

Complements pathway_auroc_protein_level.py. AUROC summarizes ranking quality
across the ENTIRE candidate pool (all ~15,718 proteins) -- it does not say
whether a pathway's true members are concentrated near the top of the ranking
or merely spread out ahead of the bulk of negatives on average. Two pathways
can have near-identical AUROC (e.g. 0.9686 vs 0.9674) while one has more than
double the other's hit rate inside a practical top-K cutoff, because Stage 1
retrieval in the actual tool always presents a fixed top-N candidate list, not
the full ranked pool. recall@K measures exactly that practical quantity.

Same pre-selected, objective 245-pathway set as pathway_auroc_protein_level.py
(pathway_final_choice.py's coverage/power filters) and the same candidate pool
(all proteins, train+validation+test together, "realistic mixed pool").

Reports two numbers per pathway per K in {100, 200, 400}:
  - recall_at_K: fraction of ALL true pathway members found in the top K.
  - test_only_recall_at_K: fraction of TEST-SET-ONLY (never-trained) true
    members found in the top K -- isolates genuine generalization from
    memorized train/validation members, same logic as test_only_auroc.

Usage:
    uv run python src/evaluation/pathway/pathway_recall_at_k_protein_level.py \\
        --checkpoint data/runs/.../checkpoint_full_proteome_masked_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --splits-file data/protein_splits_full_proteome.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import gseapy
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

SELECTION_PATH = Path("data/evaluation/pathway/pathway_final_choice.csv")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"
COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]
MIN_POSITIVES = 5  # must match pathway_final_choice.py's power filter
K_VALUES = [100, 200, 400]


def build_gene_name_map(cohorts):
    """protein_id (ENSP) -> gene symbol, needed to match KEGG gene sets (which
    are keyed by gene symbol, not ENSP id)."""
    gene_name = {}
    for cohort in cohorts:
        path = Path(f"data/cptac/{cohort}/rank_normalized_proteomics.csv")
        if not path.exists():
            continue
        columns = pd.read_csv(path, header=[0, 1], index_col=0, nrows=0).columns
        for gene, protein_id in columns:
            gene_name.setdefault(protein_id, gene)
    return gene_name


def run(checkpoint_path: Path, splits_file: str = "data/protein_splits.json",
        biobert_filename: str = "biobert_embeddings.h5"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}\n")

    print("Loading pre-selected pathway evaluation set ...")
    selection_df = pd.read_csv(SELECTION_PATH)
    selection_df = selection_df[selection_df["passes_all_filters"]].copy()
    pathway_names = set(selection_df["pathway"])
    print(f"  {len(pathway_names)} pathways passed the objective filters\n")

    with open(splits_file) as f:
        protein_splits = json.load(f)
    test_protein_ids = set(protein_splits["test"])

    print("Building global candidate pool ...")
    dataset = ProteomicsDataset(cohorts=COHORTS, use_abundance=False, biobert_filename=biobert_filename)
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])
    print(f"  {len(protein_ids):,} proteins "
          f"({len(test_protein_ids):,} held-out test proteins)\n")

    gene_name_map = build_gene_name_map(COHORTS)
    protein_labels = [gene_name_map.get(pid, pid) for pid in protein_ids]
    is_test = np.array([pid in test_protein_ids for pid in protein_ids])

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, local_files_only=True)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, local_files_only=True).to(device)
    biobert_model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Older checkpoints predate saving these keys -- fall back to the original defaults.
    hidden_dim = checkpoint.get("hidden_dim", 512)
    output_dim = checkpoint.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))

    print("Loading KEGG pathways from gseapy ...")
    gene_sets = gseapy.get_library("KEGG_2021_Human")

    print(f"Computing recall@{K_VALUES} for {len(pathway_names)} pre-selected pathways ...\n")
    results = []
    for pathway_name in pathway_names:
        pathway_genes = set(gene_sets[pathway_name])
        labels = np.array([1 if g in pathway_genes else 0 for g in protein_labels])
        n_positives = int(labels.sum())
        if n_positives < MIN_POSITIVES:
            continue

        test_positive_mask = (labels == 1) & is_test
        n_test_positives = int(test_positive_mask.sum())
        score_test_only = n_test_positives >= MIN_POSITIVES

        query_vec = encode_query(pathway_name, tokenizer, biobert_model, text_head, device)
        with torch.no_grad():
            sims = (protein_vecs @ query_vec.T).squeeze().cpu().numpy()

        ranked_idx = np.argsort(-sims)  # descending

        row = {
            "pathway": pathway_name,
            "n_positives": n_positives,
            "n_test_positives": n_test_positives,
        }
        for k in K_VALUES:
            top_k = ranked_idx[:k]
            row[f"recall_at_{k}"] = labels[top_k].sum() / n_positives
            row[f"test_only_recall_at_{k}"] = (
                (test_positive_mask[top_k].sum() / n_test_positives)
                if score_test_only else None
            )
        results.append(row)

    results_df = pd.DataFrame(results).sort_values("recall_at_200", ascending=False)
    out_path = Path(f"data/evaluation/pathway/pathway_recall_at_k/pathway_recall_at_k_protein_level_results_{checkpoint_path.stem}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print("=" * 88)
    print("Protein-level pathway recall@K (no-abundance ablation)")
    print("=" * 88)
    print(results_df.to_string(index=False))

    print(f"\nMean recall@K across {len(results_df)} pathways:")
    for k in K_VALUES:
        print(f"  recall@{k}: {results_df[f'recall_at_{k}'].mean():.4f}")

    with_test = results_df[results_df["test_only_recall_at_100"].notna()]
    print(f"\nPathways with enough test-only positives to score separately "
          f"(n={len(with_test)}): mean test-only recall@K:")
    for k in K_VALUES:
        print(f"  test_only_recall@{k}: {with_test[f'test_only_recall_at_{k}'].mean():.4f}")

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/.../checkpoint_no_abundance_best.pt")
    parser.add_argument("--splits-file", default="data/protein_splits.json",
                        help="e.g. data/protein_splits_full_proteome.json for the full-proteome split")
    parser.add_argument("--biobert-filename", default="biobert_embeddings.h5",
                        help="only needs to be a file that exists for this cohort/source -- "
                             "the candidate pool's own text embeddings aren't scored here "
                             "(pathway queries are encoded live), so this doesn't need to "
                             "match the checkpoint's trained text variant for correctness")
    args = parser.parse_args()
    run(Path(args.checkpoint), splits_file=args.splits_file, biobert_filename=args.biobert_filename)

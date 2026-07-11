"""
Protein-level pathway AUROC for the no-abundance ablation checkpoint.

Same motivation as recall_at_k_protein_level.py: this evaluates the model on
proteins it has never seen in any patient, any cohort at all
(data/protein_splits.json's "test" list), using the same objective,
pre-selected pathway set from pathway_final_choice.py. Pathway-name queries
contain no gene names at all, so this is the shortcut-free complement to
recall_at_k_protein_level.py's R@K result — together they show whether any
observed generalization is more than the model matching a protein's own name
inside its own annotation text.

Reports two numbers per pathway:
  - overall_auroc: standard AUROC over the full candidate pool (all ~15,718
    proteins), same "realistic mixed pool" design as R@K.
  - test_only_auroc: AUROC computed using ONLY test-set (never-trained)
    pathway members as positives, against all negatives — isolating whether
    genuinely novel proteins specifically are being ranked correctly, rather
    than the number being carried entirely by already-memorized train/val
    members.

Usage:
    uv run python src/evaluation/pathway_auroc_protein_level.py \\
        --checkpoint data/runs/.../checkpoint_no_abundance_best.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import gseapy
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

SELECTION_PATH = Path("data/evaluation/pathway_final_choice.csv")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"
COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]
MIN_POSITIVES = 5  # must match pathway_final_choice.py's power filter


def build_gene_name_map(cohorts):
    """protein_id (ENSP) -> gene symbol, needed to match KEGG gene sets (which
    are keyed by gene symbol, not ENSP id)."""
    gene_name = {}
    for cohort in cohorts:
        path = Path(f"data/{cohort}/rank_normalized_proteomics.csv")
        if not path.exists():
            continue
        columns = pd.read_csv(path, header=[0, 1], index_col=0, nrows=0).columns
        for gene, protein_id in columns:
            gene_name.setdefault(protein_id, gene)
    return gene_name


def run(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}\n")

    print("Loading pre-selected pathway evaluation set ...")
    selection_df = pd.read_csv(SELECTION_PATH)
    selection_df = selection_df[selection_df["passes_all_filters"]].copy()
    pathway_names = set(selection_df["pathway"])
    print(f"  {len(pathway_names)} pathways passed the objective filters\n")

    with open("data/protein_splits.json") as f:
        protein_splits = json.load(f)
    test_protein_ids = set(protein_splits["test"])

    print("Building global candidate pool ...")
    dataset = ProteomicsDataset(cohorts=COHORTS, use_abundance=False)
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

    protein_head = ProjectionHead(input_dim=640, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))

    print("Loading KEGG pathways from gseapy ...")
    gene_sets = gseapy.get_library("KEGG_2021_Human")

    print(f"Computing AUROC for {len(pathway_names)} pre-selected pathways ...\n")
    results = []
    for pathway_name in pathway_names:
        pathway_genes = set(gene_sets[pathway_name])
        labels = np.array([1 if g in pathway_genes else 0 for g in protein_labels])
        if labels.sum() < MIN_POSITIVES:
            continue

        query_vec = encode_query(pathway_name, tokenizer, biobert_model, text_head, device)
        with torch.no_grad():
            sims = (protein_vecs @ query_vec.T).squeeze().cpu().numpy()

        overall_auroc = roc_auc_score(labels, sims)

        # Test-only positives vs. all negatives: does the model rank never-
        # trained pathway members correctly, isolated from already-memorized
        # train/validation members?
        test_positive_mask = (labels == 1) & is_test
        n_test_positives = int(test_positive_mask.sum())
        test_only_auroc = None
        if n_test_positives >= MIN_POSITIVES:
            eval_mask = test_positive_mask | (labels == 0)
            test_only_auroc = roc_auc_score(labels[eval_mask], sims[eval_mask])

        results.append({
            "pathway": pathway_name,
            "overall_auroc": overall_auroc,
            "n_positives": int(labels.sum()),
            "n_test_positives": n_test_positives,
            "test_only_auroc": test_only_auroc,
        })

    results_df = pd.DataFrame(results).sort_values("overall_auroc", ascending=False)
    out_path = Path("data/evaluation/pathway_auroc_protein_level_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print("=" * 88)
    print("Protein-level pathway AUROC (no-abundance ablation)")
    print("=" * 88)
    print(results_df.to_string(index=False))

    print(f"\nOverall mean AUROC (n={len(results_df)} pathways): "
          f"{results_df['overall_auroc'].mean():.4f}")

    with_test = results_df[results_df["test_only_auroc"].notna()]
    print(f"\nPathways with enough test-only positives to score separately "
          f"(n={len(with_test)}): mean test-only AUROC = "
          f"{with_test['test_only_auroc'].mean():.4f}")

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/.../checkpoint_no_abundance_best.pt")
    args = parser.parse_args()
    run(Path(args.checkpoint))

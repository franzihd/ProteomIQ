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

Reports three numbers per pathway:
  - overall_auroc: standard AUROC over the full candidate pool (all ~15,718
    proteins), same "realistic mixed pool" design as R@K.
  - test_only_auroc: AUROC computed using ONLY test-set (never-trained)
    pathway members as positives, against all negatives — isolating whether
    genuinely novel proteins specifically are being ranked correctly, rather
    than the number being carried entirely by already-memorized train/val
    members. This is the FINAL, frozen number to report once — never use it
    to pick between hyperparameters/checkpoints (that's tuning on the test
    set). See validation_only_auroc below for that.
  - validation_only_auroc: identical logic to test_only_auroc, but restricted
    to validation-set proteins instead of test-set proteins. This is the
    metric to use for hyperparameter search / checkpoint selection (sweeps,
    architecture comparisons, etc.) — it never touches the test split, so
    test_only_auroc stays an honest, untouched estimate for the final report.

Usage:
    uv run python src/evaluation/pathway/pathway_auroc_protein_level.py \\
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
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

SELECTION_PATH = Path("data/evaluation/pathway/pathway_final_choice.csv")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"
COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]
MIN_POSITIVES = 5  # must match pathway_final_choice.py's power filter


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
        biobert_filename: str = "biobert_embeddings.h5", save_csv: bool = True):
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
    val_protein_ids = set(protein_splits["validation"])

    print("Building global candidate pool ...")
    dataset = ProteomicsDataset(cohorts=COHORTS, use_abundance=False, biobert_filename=biobert_filename)
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])
    print(f"  {len(protein_ids):,} proteins "
          f"({len(test_protein_ids):,} held-out test proteins, "
          f"{len(val_protein_ids):,} held-out validation proteins)\n")

    gene_name_map = build_gene_name_map(COHORTS)
    protein_labels = [gene_name_map.get(pid, pid) for pid in protein_ids]
    is_test = np.array([pid in test_protein_ids for pid in protein_ids])
    is_val = np.array([pid in val_protein_ids for pid in protein_ids])

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
        # train/validation members? Report exactly once, at the end, never
        # to pick between hyperparameters/checkpoints.
        test_positive_mask = (labels == 1) & is_test
        n_test_positives = int(test_positive_mask.sum())
        test_only_auroc = None
        if n_test_positives >= MIN_POSITIVES:
            eval_mask = test_positive_mask | (labels == 0)
            test_only_auroc = roc_auc_score(labels[eval_mask], sims[eval_mask])

        # Same logic, but on validation-set positives instead of test-set
        # ones -- this is the metric to use for hyperparameter/checkpoint
        # selection, since it never touches the test split.
        val_positive_mask = (labels == 1) & is_val
        n_val_positives = int(val_positive_mask.sum())
        validation_only_auroc = None
        if n_val_positives >= MIN_POSITIVES:
            eval_mask = val_positive_mask | (labels == 0)
            validation_only_auroc = roc_auc_score(labels[eval_mask], sims[eval_mask])

        results.append({
            "pathway": pathway_name,
            "overall_auroc": overall_auroc,
            "n_positives": int(labels.sum()),
            "n_test_positives": n_test_positives,
            "test_only_auroc": test_only_auroc,
            "n_val_positives": n_val_positives,
            "validation_only_auroc": validation_only_auroc,
        })

    results_df = pd.DataFrame(results).sort_values("overall_auroc", ascending=False)
    overall_mean = results_df["overall_auroc"].mean()
    with_test = results_df[results_df["test_only_auroc"].notna()]
    test_only_mean = with_test["test_only_auroc"].mean()
    with_val = results_df[results_df["validation_only_auroc"].notna()]
    validation_only_mean = with_val["validation_only_auroc"].mean()

    if save_csv:
        out_path = Path(f"data/evaluation/pathway/pathway_auroc_protein_level/pathway_auroc_protein_level_results_{checkpoint_path.stem}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)

        print("=" * 88)
        print("Protein-level pathway AUROC (no-abundance ablation)")
        print("=" * 88)
        print(results_df.to_string(index=False))
        print(f"\nOverall mean AUROC (n={len(results_df)} pathways): {overall_mean:.4f}")
        print(f"\nPathways with enough validation-only positives to score separately "
              f"(n={len(with_val)}): mean validation-only AUROC = {validation_only_mean:.4f}  "
              f"<- use this one for hyperparameter/checkpoint selection")
        print(f"\nPathways with enough test-only positives to score separately "
              f"(n={len(with_test)}): mean test-only AUROC = {test_only_mean:.4f}  "
              f"<- report this once at the end, don't tune against it")
        print(f"\nFull results saved to {out_path}")

    return {
        "overall_auroc_mean": float(overall_mean),
        "test_only_auroc_mean": float(test_only_mean),
        "n_pathways_test_only": len(with_test),
        "validation_only_auroc_mean": float(validation_only_mean),
        "n_pathways_validation_only": len(with_val),
        "n_pathways": len(results_df),
    }


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

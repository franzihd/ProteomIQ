"""
Evaluates the trained model's AUROC only on the pathways that survived the
objective, performance-blind selection criteria in pathway_final_choice.py
(coverage filter >= 50%, power filter >= 5 measured genes per patient).

This is the "scientifically defensible" counterpart to pathway_auroc_evaluation.py:
instead of computing AUROC over all ~320 KEGG pathways (many of which are barely
measured or under-powered), it restricts to the pre-specified evaluation set and
reports AUROC alongside the coverage/annotation covariates, plus a pre-specified
secondary breakdown for cohort-matched cancer-type pathways.

Run pathway_final_choice.py first if data/evaluation/pathway_final_choice.csv
does not exist yet.

Usage:
    uv run python src/evaluation/pathway_final_auroc_evaluation.py [--checkpoint <path>]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import json
import pandas as pd
import torch
import h5py
import gseapy
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel

from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query, build_protein_index

DEFAULT_CHECKPOINT = Path("data/runs/luad_brca_ccrcc_gbm_hnscc_lscc_ov_pdac_ucec/checkpoint.pt")
TEST_PATIENT_PATH = Path("data/test_patients_per_cohort.json")
SELECTION_PATH = Path("data/evaluation/pathway_final_choice.csv")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"

MIN_POSITIVES = 5  # must match pathway_final_choice.py's power filter


def compute_auroc(pathway_genes: set, protein_labels: list, similarities: torch.Tensor) -> float | None:
    labels = [1 if gene in pathway_genes else 0 for gene in protein_labels]
    if sum(labels) < MIN_POSITIVES:
        return None
    scores = similarities.cpu().numpy()
    return roc_auc_score(labels, scores)


def run(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}\n")

    print("Loading pre-selected pathway evaluation set ...")
    selection_df = pd.read_csv(SELECTION_PATH)
    selection_df = selection_df[selection_df["passes_all_filters"]].copy()
    pathway_names = set(selection_df["pathway"])
    print(f"  {len(pathway_names)} pathways passed the objective filters\n")

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, local_files_only=True)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, local_files_only=True).to(device)
    biobert_model.eval()

    protein_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    with open(TEST_PATIENT_PATH) as f:
        test_patients_per_cohort = json.load(f)

    print("Building protein indices for all test patients ...")
    patient_indices = {}  # { patient_id: (protein_vecs, protein_labels) }

    for cohort, patient_ids in test_patients_per_cohort.items():
        data_dir = Path(f"data/{cohort}")
        esm2_path = data_dir / "esm2_embeddings.h5"
        abundance_path = data_dir / "rank_normalized_proteomics.csv"

        if not esm2_path.exists() or not abundance_path.exists():
            print(f"  Skipping {cohort} -- missing files")
            continue

        abundance_df = pd.read_csv(abundance_path, header=[0, 1], index_col=0)

        with h5py.File(esm2_path, "r") as h5f:
            esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
            esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

        esm2_id_to_idx = {pid: idx for idx, pid in enumerate(esm2_protein_ids)}

        for patient_id in patient_ids:
            if patient_id not in abundance_df.index:
                continue
            protein_vecs, protein_labels = build_protein_index(
                patient_id, abundance_df, esm2_matrix, esm2_id_to_idx, protein_head, device
            )
            patient_indices[patient_id] = (protein_vecs, protein_labels)

        print(f"  {cohort}: {len(patient_ids)} patients loaded")

    print(f"Done -- {len(patient_indices)} patients ready.\n")

    print("Loading KEGG pathways from gseapy ...")
    gene_sets = gseapy.get_library("KEGG_2021_Human")

    print(f"Computing AUROC for {len(pathway_names)} pre-selected pathways ...\n")
    auroc_results = []
    for pathway_name in pathway_names:
        pathway_genes = set(gene_sets[pathway_name])

        query_vec = encode_query(pathway_name, tokenizer, biobert_model, text_head, device)

        patient_aurocs = []
        for patient_id, (protein_vecs, protein_labels) in patient_indices.items():
            sims = (protein_vecs @ query_vec.T).squeeze()
            auroc = compute_auroc(pathway_genes, protein_labels, sims)
            if auroc is not None:
                patient_aurocs.append(auroc)

        if not patient_aurocs:
            continue

        auroc_results.append({
            "pathway": pathway_name,
            "mean_auroc": np.mean(patient_aurocs),
            "n_patients": len(patient_aurocs),
        })

    auroc_df = pd.DataFrame(auroc_results)

    results_df = selection_df.merge(auroc_df, on="pathway", how="inner")
    results_df = results_df.sort_values("mean_auroc", ascending=False)

    out_path = Path("data/evaluation/pathway_final_auroc_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    cols = ["pathway", "mean_auroc", "n_patients", "coverage_fraction",
            "annotation_coverage_fraction", "is_cohort_matched", "matched_cohorts"]

    print("=" * 88)
    print("Pathway AUROC -- pre-specified, objectively-selected evaluation set")
    print("=" * 88)
    print(results_df[cols].to_string(index=False))

    print(f"\nOverall mean AUROC (primary result, n={len(results_df)} pathways): "
          f"{results_df['mean_auroc'].mean():.4f}")

    cohort_matched = results_df[results_df["is_cohort_matched"]]
    non_matched = results_df[~results_df["is_cohort_matched"]]
    print(f"\nSecondary analysis -- cohort-matched cancer-type pathways (n={len(cohort_matched)}):")
    if len(cohort_matched):
        print(cohort_matched[["pathway", "matched_cohorts", "mean_auroc"]].to_string(index=False))
        print(f"  mean AUROC: {cohort_matched['mean_auroc'].mean():.4f}")
    print(f"\nAll other pathways in the evaluation set (n={len(non_matched)}): "
          f"mean AUROC = {non_matched['mean_auroc'].mean():.4f}")

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    run(Path(args.checkpoint))

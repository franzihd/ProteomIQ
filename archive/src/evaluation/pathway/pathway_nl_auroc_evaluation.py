"""
Re-runs the pathway AUROC evaluation using natural-language query rewrites
(from pathway_nl_queries.py) instead of raw KEGG pathway names, for the same
245 objectively-selected pathways used in pathway_final_auroc_evaluation.py.

Purpose: pathway_final_auroc_results.csv showed cohort-matched cancer
pathways (Glioma, NSCLC, Pancreatic cancer, Renal cell carcinoma,
Endometrial cancer) scoring below the overall mean (0.473 vs 0.573, see
CLAUDE.md Timeline 8d). This script tests whether that gap is a
query-phrasing problem -- raw KEGG names are database jargon -- or a
genuine generalization limit, by rerunning AUROC with the natural-language
rewrites and comparing directly against the raw-name scores, pathway by
pathway.

Ground truth (which genes count as pathway members) is unchanged -- only
the query TEXT fed to the text encoder differs. Gene membership still comes
from the pathway's real KEGG name, looked up via gseapy.

Usage:
    uv run python src/evaluation/pathway/pathway_nl_auroc_evaluation.py [--checkpoint <path>]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
NL_QUERIES_PATH = Path("data/evaluation/pathway/pathway_nl_queries.csv")
RAW_RESULTS_PATH = Path("data/evaluation/pathway/pathway_final_auroc_results.csv")
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

    print("Loading natural-language queries ...")
    nl_df = pd.read_csv(NL_QUERIES_PATH)
    nl_df = nl_df.dropna(subset=["nl_query"])  # drop any pathways whose query generation failed
    print(f"  {len(nl_df)} pathways with a usable natural-language query\n")

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
    patient_indices = {}

    for cohort, patient_ids in test_patients_per_cohort.items():
        data_dir = Path(f"data/cptac/{cohort}")
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

    print("Loading KEGG pathway gene membership ...")
    gene_sets = gseapy.get_library("KEGG_2021_Human")

    print(f"Computing AUROC with natural-language queries for {len(nl_df)} pathways ...\n")
    nl_results = []
    for _, row in nl_df.iterrows():
        pathway_name = row["raw_name"]
        nl_query = row["nl_query"]
        pathway_genes = set(gene_sets[pathway_name])

        query_vec = encode_query(nl_query, tokenizer, biobert_model, text_head, device)

        patient_aurocs = []
        for patient_id, (protein_vecs, protein_labels) in patient_indices.items():
            sims = (protein_vecs @ query_vec.T).squeeze()
            auroc = compute_auroc(pathway_genes, protein_labels, sims)
            if auroc is not None:
                patient_aurocs.append(auroc)

        if not patient_aurocs:
            continue

        nl_results.append({
            "pathway": pathway_name,
            "nl_query": nl_query,
            "nl_auroc": np.mean(patient_aurocs),
        })

    nl_auroc_df = pd.DataFrame(nl_results)

    print("Merging with raw-name AUROC results for a paired comparison ...")
    raw_df = pd.read_csv(RAW_RESULTS_PATH)
    raw_df = raw_df.rename(columns={"mean_auroc": "raw_auroc"})

    merged = raw_df.merge(nl_auroc_df, on="pathway", how="inner")
    merged["delta"] = merged["nl_auroc"] - merged["raw_auroc"]
    merged = merged.sort_values("delta", ascending=False)

    out_path = Path("data/evaluation/pathway/pathway_nl_auroc_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pathway", "nl_query", "raw_auroc", "nl_auroc", "delta",
            "is_cohort_matched", "matched_cohorts"]
    merged[cols].to_csv(out_path, index=False)

    print("=" * 88)
    print("Natural-language query vs. raw KEGG name -- paired AUROC comparison")
    print("=" * 88)
    print(f"\nOverall: raw mean = {merged['raw_auroc'].mean():.4f}, "
          f"nl mean = {merged['nl_auroc'].mean():.4f}, "
          f"mean delta = {merged['delta'].mean():+.4f}")
    print(f"Pathways improved by NL phrasing: {(merged['delta'] > 0).sum()} / {len(merged)}")

    cohort_matched = merged[merged["is_cohort_matched"]]
    print(f"\nCohort-matched pathways (n={len(cohort_matched)}):")
    print(cohort_matched[["pathway", "matched_cohorts", "raw_auroc", "nl_auroc", "delta"]]
          .to_string(index=False))
    print(f"\n  raw mean = {cohort_matched['raw_auroc'].mean():.4f}, "
          f"nl mean = {cohort_matched['nl_auroc'].mean():.4f}, "
          f"mean delta = {cohort_matched['delta'].mean():+.4f}")

    print(f"\nTop 10 biggest improvements:")
    print(merged[["pathway", "raw_auroc", "nl_auroc", "delta"]].head(10).to_string(index=False))
    print(f"\nTop 10 biggest regressions:")
    print(merged[["pathway", "raw_auroc", "nl_auroc", "delta"]].tail(10).to_string(index=False))

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    run(Path(args.checkpoint))

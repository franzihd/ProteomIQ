"""
Three-way AUROC comparison: raw KEGG pathway name vs. explanatory-sentence
natural-language rewrite (pathway_nl_queries.py) vs. compact search-query-
style natural-language rewrite (pathway_nl_queries_compact.py).

Ground truth (pathway gene membership) is identical across all three --
only the query TEXT fed to the text encoder differs.

Usage:
    uv run python src/evaluation/pathway/pathway_three_way_auroc_evaluation.py [--checkpoint <path>]
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
COMPACT_QUERIES_PATH = Path("data/evaluation/pathway/pathway_nl_queries_compact.csv")
PRIOR_RESULTS_PATH = Path("data/evaluation/pathway/pathway_nl_auroc_results.csv")  # has raw_auroc + explanatory nl_auroc
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"

MIN_POSITIVES = 5


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

    print("Loading compact search-query rewrites ...")
    compact_df = pd.read_csv(COMPACT_QUERIES_PATH)
    compact_df = compact_df.dropna(subset=["nl_query_compact"])
    print(f"  {len(compact_df)} pathways with a usable compact query\n")

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

    print(f"Computing AUROC with compact queries for {len(compact_df)} pathways ...\n")
    compact_results = []
    for _, row in compact_df.iterrows():
        pathway_name = row["raw_name"]
        compact_query = row["nl_query_compact"]
        pathway_genes = set(gene_sets[pathway_name])

        query_vec = encode_query(compact_query, tokenizer, biobert_model, text_head, device)

        patient_aurocs = []
        for patient_id, (protein_vecs, protein_labels) in patient_indices.items():
            sims = (protein_vecs @ query_vec.T).squeeze()
            auroc = compute_auroc(pathway_genes, protein_labels, sims)
            if auroc is not None:
                patient_aurocs.append(auroc)

        if not patient_aurocs:
            continue

        compact_results.append({
            "pathway": pathway_name,
            "nl_query_compact": compact_query,
            "compact_auroc": np.mean(patient_aurocs),
        })

    compact_auroc_df = pd.DataFrame(compact_results)

    print("Merging raw / explanatory-NL / compact-NL results into one table ...")
    prior_df = pd.read_csv(PRIOR_RESULTS_PATH)  # pathway, nl_query, raw_auroc, nl_auroc, delta, is_cohort_matched, matched_cohorts
    prior_df = prior_df.rename(columns={"nl_query": "nl_query_explanatory", "nl_auroc": "explanatory_auroc"})

    merged = prior_df.merge(compact_auroc_df, on="pathway", how="inner")
    merged["delta_explanatory"] = merged["explanatory_auroc"] - merged["raw_auroc"]
    merged["delta_compact"] = merged["compact_auroc"] - merged["raw_auroc"]
    merged = merged.sort_values("delta_compact", ascending=False)

    out_path = Path("data/evaluation/pathway/pathway_three_way_auroc_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pathway", "nl_query_explanatory", "nl_query_compact",
            "raw_auroc", "explanatory_auroc", "compact_auroc",
            "delta_explanatory", "delta_compact",
            "is_cohort_matched", "matched_cohorts"]
    merged[cols].to_csv(out_path, index=False)

    print("=" * 88)
    print("Three-way AUROC comparison: raw name vs. explanatory NL vs. compact NL")
    print("=" * 88)
    print(f"\nOverall means: raw = {merged['raw_auroc'].mean():.4f}, "
          f"explanatory = {merged['explanatory_auroc'].mean():.4f}, "
          f"compact = {merged['compact_auroc'].mean():.4f}")
    print(f"Pathways improved vs raw -- explanatory: {(merged['delta_explanatory'] > 0).sum()}/{len(merged)}, "
          f"compact: {(merged['delta_compact'] > 0).sum()}/{len(merged)}")
    print(f"Correlation(raw_auroc, delta_compact): {merged['raw_auroc'].corr(merged['delta_compact']):.4f}")

    cohort_matched = merged[merged["is_cohort_matched"]]
    print(f"\nCohort-matched pathways (n={len(cohort_matched)}):")
    print(cohort_matched[["pathway", "matched_cohorts", "raw_auroc", "explanatory_auroc", "compact_auroc"]]
          .to_string(index=False))
    print(
        f"\n  raw mean = {cohort_matched['raw_auroc'].mean():.4f}, "
        f"explanatory mean = {cohort_matched['explanatory_auroc'].mean():.4f}, "
        f"compact mean = {cohort_matched['compact_auroc'].mean():.4f}"
    )

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    run(Path(args.checkpoint))

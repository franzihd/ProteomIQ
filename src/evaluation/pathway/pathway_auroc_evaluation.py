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

from src.embeddings.textual.embed_biobert import MAX_LENGTH, mean_pool
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query, build_protein_index

DEFAULT_CHECKPOINT = Path("data/runs/luad_brca_ccrcc_gbm_hnscc_lscc_ov_pdac/checkpoint.pt")
TEST_PATIENT_PATH = Path("data/test_patients_per_cohort.json")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"

MIN_POSITIVES = 5


def compute_auroc(pathway_genes: set, protein_labels: list, similarities: torch.Tensor) -> float | None:
    labels = [1 if gene in pathway_genes else 0 for gene in protein_labels]
    n_pos = sum(labels)
    if n_pos < MIN_POSITIVES:
        return None
    scores = similarities.cpu().numpy()
    return roc_auc_score(labels, scores)


def run(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}\n")

    # load BioBERT
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, local_files_only=True)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, local_files_only=True).to(device)
    biobert_model.eval()

    # load projection heads
    protein_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    # load test patients (per cohort)
    with open(TEST_PATIENT_PATH) as f:
        test_patients_per_cohort = json.load(f)

    print("Building protein indices for all test patients ... ")
    patient_indices = {}  # { patient_id: (protein_vecs, protein_labels) }

    for cohort, patient_ids in test_patients_per_cohort.items():
        data_dir = Path(f"data/cptac/{cohort}")
        esm2_path = data_dir / "esm2_embeddings.h5"
        abundance_path = data_dir / "rank_normalized_proteomics.csv"

        if not esm2_path.exists() or not abundance_path.exists():
            print(f" Skipping {cohort} -- missing files!!")
            continue

        abundance_df = pd.read_csv(abundance_path, header=[0,1], index_col=0)

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

        print(f" {cohort}: {len(patient_ids)} patients loaded")

    print(f"Done - {len(patient_indices)} patients ready. \n")

    print("Loading KEGG pathways from gseapy ... ")
    gene_sets = gseapy.get_library("KEGG_2021_Human")
    print(f"Loaded {len(gene_sets)} pathways.\n")

    results = []

    for pathway_name, pathway_genes in gene_sets.items():
        pathway_genes = set(pathway_genes)

        # encode the pathway names as a natural language query
        query_vec = encode_query(pathway_name, tokenizer, biobert_model, text_head, device)

        # compute AUROC for each test patient
        patient_aurocs = []
        for patient_id, (protein_vecs, protein_labels) in patient_indices.items():
            sims = (protein_vecs @ query_vec.T).squeeze()
            auroc = compute_auroc(pathway_genes, protein_labels, sims)
            if auroc is not None:
                patient_aurocs.append(auroc)
        
        if not patient_aurocs:
            continue

        results.append({
            "pathway": pathway_name,
            "mean_auroc": np.mean(patient_aurocs),
            "n_patients": len(patient_aurocs),
            "n_pathway_genes": len(pathway_genes)
        })
    
    results_df = pd.DataFrame(results).sort_values("mean_auroc", ascending=False)

    print("=" * 72)
    print("Pathway AUROC Evaluation  |  KEGG_2021_Human  |  random baseline = 0.500")
    print("=" * 72)
    print(f"\nTop 20 pathways (model knows these best):")
    print(results_df.head(20).to_string(index=False))
    print(f"\nBottom 10 pathways (model struggles most):")
    print(results_df.tail(10).to_string(index=False))
    print(f"\nOverall mean AUROC : {results_df['mean_auroc'].mean():.4f}")
    print(f"Pathways evaluated : {len(results_df)} / {len(gene_sets)}")

    out_path = Path("data/evaluation/pathway/pathway_auroc_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    run(Path(args.checkpoint))
    
        
"""
query_eval.py — run a fixed set of obvious biological queries through the trained
ProteomIQ model and print a clean results table for thesis evaluation.

Usage:
    uv run python src/model/query_eval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd
import torch
import h5py
from transformers import AutoTokenizer, AutoModel

from src.embeddings.textual.embed_biobert import MAX_LENGTH, mean_pool
from src.model.projection_head import ProjectionHead

# ---------------------------------------------------------------------------
# Queries + expected gene names (ground truth for display, not for scoring)
# ---------------------------------------------------------------------------

QUERIES = [
    # --- Strong positive controls (UniProt language, should rank well) ---
    {
        "query": "collagen fibril assembly and extracellular matrix organization",
        "expected": ["COL1A1", "COL1A2", "COL3A1", "FBN1"],
    },
    {
        "query": "subunit of the 26S proteasome involved in ubiquitin-dependent protein degradation",
        "expected": ["PSMA1", "PSMB1", "PSMD1"],
    },
    {
        "query": "mitochondrial ATP synthase subunit",
        "expected": ["ATP5F1A", "ATP5F1B", "ATP5PB"],
    },
    {
        "query": "component of the 60S ribosomal subunit",
        "expected": ["RPL7", "RPL11", "RPL23"],
    },
    # --- Cancer biology queries (moderate difficulty) ---
    {
        "query": "receptor tyrosine kinase involved in cell proliferation and tumor growth",
        "expected": ["EGFR", "MET", "FGFR1", "ERBB2"],
    },
    {
        "query": "regulation of apoptosis and programmed cell death",
        "expected": ["CASP3", "BCL2", "BAX", "CYCS"],
    },
    {
        "query": "heat shock protein molecular chaperone protein folding",
        "expected": ["HSP90AA1", "HSPA1A", "HSPA5"],
    },
    {
        "query": "cell adhesion epithelial-mesenchymal transition",
        "expected": ["CDH1", "VIM", "FN1", "ITGA5"],
    },
    # --- Harder reasoning queries (expected to fail — lower bound) ---
    {
        "query": "proteins that are overexpressed in cancer",
        "expected": ["MYC", "CCND1", "CDK4", "MDM2"],
    },
    {
        "query": "potential drug targets in lung adenocarcinoma",
        "expected": ["EGFR", "KRAS", "ALK", "ROS1"],
    },
    {
        "query": "tumor suppressor genes",
        "expected": ["TP53", "PTEN", "RB1", "CDKN2A"],
    },
    {
        "query": "proteins involved in immune evasion",
        "expected": ["CD274", "PDCD1LG2", "IDO1", "ARG1"],
    },
]
TOP_K = 10  # how many results to show per query

# ---------------------------------------------------------------------------

DATA_DIR = Path("data/luad")
CHECKPOINT_PATH = Path("data/runs/luad_brca_ccrcc_gbm_hnscc_lscc_ov_pdac/checkpoint.pt")
TEST_PATIENTS_PATH = DATA_DIR / "test_patients.json"

def encode_query(query: str, tokenizer, biobert_model, textual_head, device):
    encoded = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = biobert_model(**encoded)
    pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    with torch.no_grad():
        vec = textual_head(pooled)
    return vec  # shape (1, 256)


def build_protein_index(patient_id, abundance_df, esm2_matrix, esm2_id_to_idx, protein_head, device):
    patient_row = abundance_df.loc[patient_id]
    protein_embs = []
    protein_labels = []

    for (gene_name, protein_id), abundance in patient_row.items():
        if np.isnan(abundance):
            continue
        if protein_id not in esm2_id_to_idx:
            continue
        esm2_vec = esm2_matrix[esm2_id_to_idx[protein_id]]
        abundance_tensor = torch.tensor([abundance], dtype=torch.float32)
        protein_emb = torch.cat([esm2_vec, abundance_tensor], dim=0)
        protein_embs.append(protein_emb)
        protein_labels.append(gene_name)

    protein_embs = torch.stack(protein_embs).to(device)
    with torch.no_grad():
        protein_vecs = protein_head(protein_embs)  # (N, 256), L2-normalised by head

    return protein_vecs, protein_labels


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, local_files_only=True)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, local_files_only=True).to(device)

    protein_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    with open(TEST_PATIENTS_PATH) as f:
        test_patient_ids = json.load(f)
    print(f"Evaluating over {len(test_patient_ids)} test patients...\n")

    abundance_df = pd.read_csv(
        DATA_DIR / "rank_normalized_proteomics.csv", header=[0, 1], index_col=0
    )

    with h5py.File(DATA_DIR / "esm2_embeddings.h5", "r") as h5f:
        esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
        esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
    esm2_id_to_idx = {pid: idx for idx, pid in enumerate(esm2_protein_ids)}

    print("=" * 70)

    for entry in QUERIES:
        query = entry["query"]
        expected = entry["expected"]

        query_vec = encode_query(query, tokenizer, biobert_model, text_head, device)

        # collect per-gene percentile ranks across all test patients
        gene_percentiles = {gene: [] for gene in expected}

        for patient_id in test_patient_ids:
            if patient_id not in abundance_df.index:
                continue

            protein_vecs, protein_labels = build_protein_index(
                patient_id, abundance_df, esm2_matrix, esm2_id_to_idx, protein_head, device
            )
            n_proteins = len(protein_labels)

            similarities = (protein_vecs @ query_vec.T).squeeze()
            ranked_indices = torch.argsort(similarities, descending=True).cpu().tolist()
            ranked_genes = [protein_labels[i] for i in ranked_indices]

            for gene in expected:
                if gene in ranked_genes:
                    rank = ranked_genes.index(gene) + 1
                    gene_percentiles[gene].append(rank / n_proteins * 100)

        print(f"\nQuery: \"{query}\"")
        print(f"Expected proteins — mean percentile rank across {len(test_patient_ids)} patients:")
        print(f"  (lower = better, random baseline = 50%)\n")
        for gene in expected:
            pcts = gene_percentiles[gene]
            if pcts:
                mean_pct = np.mean(pcts)
                print(f"  {gene:<12s}  mean rank: {mean_pct:.1f}%  (measured in {len(pcts)}/{len(test_patient_ids)} patients)")
            else:
                print(f"  {gene:<12s}  not measured in any test patient")
        print("-" * 70)



if __name__ == "__main__":
    run()
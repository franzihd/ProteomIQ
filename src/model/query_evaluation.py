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
    {
        "query": "kinase",
        "expected": ["EGFR", "CDK1", "MTOR", "BRAF", "SRC", "ABL1"],
    },
    {
        "query": "ribosomal protein",
        "expected": ["RPS3", "RPL7", "RPS14", "RPL11", "RPS6"],
    },
    {
        "query": "collagen",
        "expected": ["COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL6A1"],
    },
    {
        "query": "mitochondrial",
        "expected": ["VDAC1", "VDAC2", "TOMM40", "TIMM23", "ATP5F1A"],
    },
    {
        "query": "DNA repair",
        "expected": ["BRCA1", "BRCA2", "RAD51", "MSH2", "MLH1"],
    },
    {
        "query": "proteasome",
        "expected": ["PSMA1", "PSMB1", "PSMD1", "PSMC1", "PSME1"],
    },
    {
        "query": "immune response",
        "expected": ["HLA-A", "CD8A", "IFNG", "STAT1", "IRF1"],
    },
    {
        "query": "glycolysis",
        "expected": ["ENO1", "ALDOA", "GAPDH", "PKM", "LDHA", "TPI1"],
    },
    {
        "query": "histone",
        "expected": ["HIST1H1C", "H2AFX", "HIST1H2AB", "HIST2H2AC"],
    },
    {
        "query": "actin cytoskeleton",
        "expected": ["ACTB", "ACTG1", "ACTN1", "VCL", "FLNA"],
    },
]

TOP_K = 10  # how many results to show per query

# ---------------------------------------------------------------------------

DATA_DIR = Path("data/luad")
CHECKPOINT_PATH = DATA_DIR / "checkpoint.pt"
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

    # --- load models ---
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(device)

    protein_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    # --- load data ---
    with open(TEST_PATIENTS_PATH) as f:
        test_patient_ids = json.load(f)
    patient_id = test_patient_ids[0]
    print(f"Test patient: {patient_id}\n")

    abundance_df = pd.read_csv(
        DATA_DIR / "rank_normalized_proteomics.csv", header=[0, 1], index_col=0
    )

    with h5py.File(DATA_DIR / "esm2_embeddings.h5", "r") as h5f:
        esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
        esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
    esm2_id_to_idx = {pid: idx for idx, pid in enumerate(esm2_protein_ids)}

    # --- build protein index for this patient ---
    protein_vecs, protein_labels = build_protein_index(
        patient_id, abundance_df, esm2_matrix, esm2_id_to_idx, protein_head, device
    )
    n_proteins = len(protein_labels)
    print(f"Proteins in index: {n_proteins}\n")
    print("=" * 70)

    # --- run each query ---
    for entry in QUERIES:
        query = entry["query"]
        expected = entry["expected"]

        query_vec = encode_query(query, tokenizer, biobert_model, text_head, device)
        similarities = (protein_vecs @ query_vec.T).squeeze()
        ranked_indices = torch.argsort(similarities, descending=True).cpu().tolist()
        ranked_genes = [protein_labels[i] for i in ranked_indices]

        print(f"\nQuery: \"{query}\"")
        print(f"Expected proteins: {', '.join(expected)}")
        print(f"\n  Top-{TOP_K} results:")
        for rank, gene in enumerate(ranked_genes[:TOP_K], start=1):
            score = similarities[ranked_indices[rank - 1]].item()
            hit = "✓" if gene in expected else " "
            print(f"  {hit} {rank:2d}. {gene:<12s}  score: {score:.4f}")

        # how did the expected proteins actually rank?
        print(f"\n  Expected protein ranks:")
        for gene in expected:
            if gene in ranked_genes:
                rank = ranked_genes.index(gene) + 1
                score = similarities[ranked_indices[rank - 1]].item()
                top_pct = rank / n_proteins * 100
                print(f"    {gene:<12s}  rank {rank:5d} / {n_proteins}  ({top_pct:.1f}%)  score: {score:.4f}")
            else:
                print(f"    {gene:<12s}  not measured in this patient")

        print("-" * 70)


if __name__ == "__main__":
    run()
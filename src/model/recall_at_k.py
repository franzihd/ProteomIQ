from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import pandas as pd
import h5py
import torch
import numpy as np
from src.model.projection_head import ProjectionHead


def _load_heads(checkpoint_path: str, device):
    protein_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    text_head    = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    protein_head.load_state_dict(ckpt["protein_head"])
    text_head.load_state_dict(ckpt["text_head"])
    protein_head.eval()
    text_head.eval()
    return protein_head, text_head


def _ranks_for_cohort(cohort, test_ids, protein_head, text_head, device):
    data_dir = Path(f"data/{cohort}")
    abundance_df = pd.read_csv(data_dir / "rank_normalized_proteomics.csv",
                               header=[0, 1], index_col=0)
    with h5py.File(data_dir / "esm2_embeddings.h5") as f:
        esm2_matrix      = torch.tensor(f["embeddings"][:], dtype=torch.float32)
        esm2_protein_ids = [p.decode() for p in f["protein_ids"][:]]
    with h5py.File(data_dir / "biobert_embeddings.h5") as f:
        biobert_matrix       = torch.tensor(f["embeddings"][:], dtype=torch.float32)
        biobert_protein_ids  = [p.decode() for p in f["protein_ids"][:]]

    esm2_id_to_idx           = {p: i for i, p in enumerate(esm2_protein_ids)}
    biobert_protein_id_to_idx = {p: i for i, p in enumerate(biobert_protein_ids)}

    ranks = []
    for patient_id in test_ids:
        if patient_id not in abundance_df.index:
            continue
        patient_row = abundance_df.loc[patient_id]
        protein_embs, protein_ids_list = [], []
        for (_, protein_id), abundance in patient_row.items():
            if np.isnan(abundance): continue
            if protein_id not in esm2_id_to_idx: continue
            if protein_id not in biobert_protein_id_to_idx: continue
            esm2_vec = esm2_matrix[esm2_id_to_idx[protein_id]]
            protein_embs.append(torch.cat([esm2_vec, torch.tensor([abundance])]))
            protein_ids_list.append(protein_id)

        protein_embs = torch.stack(protein_embs).to(device)
        with torch.no_grad():
            protein_vecs = protein_head(protein_embs)

        text_embs = torch.stack([
            biobert_matrix[biobert_protein_id_to_idx[pid]]
            for pid in protein_ids_list
        ]).to(device)
        with torch.no_grad():
            text_vecs = text_head(text_embs)

        sim = text_vecs @ protein_vecs.T
        for i in range(len(text_vecs)):
            ranks.append((sim[i] > sim[i, i]).sum().item() + 1)

    return ranks


def compute_ranks_luad(checkpoint_path: str = "data/luad/checkpoint.pt"):
    """Evaluate on the 10 held-out LUAD test patients only (fixed anchor)."""
    with open("data/test_patients_per_cohort.json") as f:
        test_ids = json.load(f)["luad"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head, text_head = _load_heads(checkpoint_path, device)
    ranks = np.array(_ranks_for_cohort("luad", test_ids, protein_head, text_head, device))
    return {"r1": float((ranks <= 1).mean() * 100),
            "r10": float((ranks <= 10).mean() * 100),
            "r50": float((ranks <= 50).mean() * 100)}


def compute_ranks_all_cohorts(checkpoint_path: str, cohorts: list[str]):
    """Evaluate on 10 held-out patients from each cohort in the training set."""
    with open("data/test_patients_per_cohort.json") as f:
        per_cohort = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head, text_head = _load_heads(checkpoint_path, device)
    all_ranks = []
    for cohort in cohorts:
        test_ids = per_cohort.get(cohort, [])
        all_ranks.extend(_ranks_for_cohort(cohort, test_ids, protein_head, text_head, device))
    ranks = np.array(all_ranks)
    return {"ranks": ranks,
            "r1": float((ranks <= 1).mean() * 100),
            "r10": float((ranks <= 10).mean() * 100),
            "r50": float((ranks <= 50).mean() * 100)}


def compute_ranks(checkpoint_path: str = "data/luad/checkpoint.pt"):
    DATA_DIR = Path("data/luad")
    CHECKPOINT_PATH = Path(checkpoint_path)
    TEST_PATIENTS_PATH = DATA_DIR / "test_patients.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protein_projection_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    textual_projection_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    protein_projection_head.load_state_dict(checkpoint["protein_head"])
    textual_projection_head.load_state_dict(checkpoint["text_head"])

    protein_projection_head.eval()
    textual_projection_head.eval()


    # load the 15 held-out patient IDs that were never seen during training
    with open(TEST_PATIENTS_PATH) as f:
        test_patient_ids = json.load(f)

    # load the rank-normalized abundance matrix (rows=patients, cols=proteins)
    abundance_df = pd.read_csv(DATA_DIR / "rank_normalized_proteomics.csv", header=[0,1], index_col=0)

    # load ESM2 embeddings: one 640-dim vector per protein sequence
    with h5py.File(DATA_DIR / "esm2_embeddings.h5", "r") as h5f:
            esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype= torch.float32)
            esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

    # map ENSP ID → row index in esm2_matrix
    esm2_id_to_idx = {pid: idx for idx, pid in enumerate(esm2_protein_ids)}

    # load BioBERT embeddings: one 768-dim vector per protein (text no longer varies per patient)
    with h5py.File(DATA_DIR / "biobert_embeddings.h5", "r") as h5f:
        biobert_matrix = torch.tensor(h5f["embeddings"][:], dtype= torch.float32)
        biobert_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

    # map protein ID → index in biobert_matrix
    biobert_protein_id_to_idx = {pid: idx for idx, pid in enumerate(biobert_protein_ids)}


    all_ranks = []

    for test_patient_id in test_patient_ids:
        patient_row = abundance_df.loc[test_patient_id]

        protein_embs = []
        protein_ids_list = []

        # build protein embeddings: ESM2 (640-dim) + abundance scalar = 641-dim
        for (gene_name, protein_id), abundance in patient_row.items():
            if np.isnan(abundance):
                continue
            if protein_id not in esm2_id_to_idx:
                continue 
            if protein_id not in biobert_protein_id_to_idx:
                continue
            
            esm2_vec = esm2_matrix[esm2_id_to_idx[protein_id]]
            abundance_tensor = torch.tensor([abundance], dtype=torch.float32)
            protein_emb = torch.cat([esm2_vec, abundance_tensor], dim=0)

            protein_embs.append(protein_emb)
            protein_ids_list.append(protein_id)
        
        protein_embs = torch.stack(protein_embs).to(device) # shape (num_proteins, 641)

        # project protein embeddings into the shared 256-dim space
        with torch.no_grad():
            protein_vecs = protein_projection_head(protein_embs) # shape (num_proteins, 256)
        

        text_embs = []

        # look up the pre-computed BioBERT embedding for each protein
        for protein_id in protein_ids_list:
            q_idx = biobert_protein_id_to_idx[protein_id]
            text_embs.append(biobert_matrix[q_idx])
        
        text_embs = torch.stack(text_embs).to(device) # shape (num_proteins, 768)

        # project text embeddings into the shared 256-dim space
        with torch.no_grad():
            text_vecs = textual_projection_head(text_embs) # shape (num_proteins, 256)



        sim_matrix = text_vecs @ protein_vecs.T  # shape (num_proteins, num_proteins)


        for i in range(len(text_vecs)):
            sim = sim_matrix[i]

            # sim > sim[i] → creates a boolean tensor, e.g.: tensor([Fals,e True,... False  ])
            rank = (sim > sim[i]).sum().item() + 1  # rank of the correct protein for this text embedding

            all_ranks.append(rank)

        ranks = np.array(all_ranks)

    return {
        "ranks": ranks,
        "r1":  float((ranks <= 1).mean()  * 100),
        "r10": float((ranks <= 10).mean() * 100),
        "r50": float((ranks <= 50).mean() * 100),
    }






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/luad/checkpoint.pt")
    args = parser.parse_args()

    metrics = compute_ranks(checkpoint_path=args.checkpoint)
    ranks = metrics["ranks"]
    total = len(ranks)
    print(f"R@1:  {metrics['r1']:.2f}%")
    print(f"R@10: {metrics['r10']:.2f}%")
    print(f"R@50: {metrics['r50']:.2f}%")
    print(f"\nEvaluated {total:,} pairs across {total} held-out test pairs.")

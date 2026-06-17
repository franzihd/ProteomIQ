from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import pandas as pd
import h5py
import torch
import numpy as np
from src.model.projection_head import ProjectionHead


DATA_DIR = Path("data/luad")
CHECKPOINT_PATH = DATA_DIR / "checkpoint.pt"
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

# load BioBERT embeddings: one 768-dim vector per (patient, protein) pair
with h5py.File(DATA_DIR / "biobert_embeddings.h5", "r") as h5f:
    biobert_matrix = torch.tensor(h5f["embeddings"][:], dtype= torch.float32)
    biobert_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
    biobert_patient_ids = [pid.decode() for pid in h5f["patient_ids"][:]]

# map patient/protein ID → index in biobert_matrix
biobert_patient_id_to_idx = {pid: idx for idx, pid in enumerate(biobert_patient_ids)}
biobert_protein_id_to_idx = {pid: idx for idx, pid in enumerate(biobert_protein_ids)}   

hits_at_1 = 0
hits_at_10 = 0
hits_at_50 = 0
total_pairs = 0

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

    # look up the pre-computed BioBERT embedding for each (patient, protein) pair
    for protein_id in protein_ids_list:
         p_idx = biobert_patient_id_to_idx[test_patient_id]
         q_idx = biobert_protein_id_to_idx[protein_id]
         text_embs.append(biobert_matrix[p_idx, q_idx])
    
    text_embs = torch.stack(text_embs).to(device) # shape (num_proteins, 768)

    # project text embeddings into the shared 256-dim space
    with torch.no_grad():
        text_vecs = textual_projection_head(text_embs) # shape (num_proteins, 256)



    sim_matrix = text_vecs @ protein_vecs.T  # shape (num_proteins, num_proteins)


    for i in range(len(text_vecs)):
        sim = sim_matrix[i]

        # sim > sim[i] → creates a boolean tensor, e.g.: tensor([Fals,e True,... False  ])
        rank = (sim > sim[i]).sum().item() + 1  # rank of the correct protein for this text embedding

        if rank <= 1: hits_at_1 += 1
        if rank <=10: hits_at_10 += 1
        if rank <=50: hits_at_50 += 1

        total_pairs += 1

print(f"R@1:  {hits_at_1  / total_pairs * 100:.2f}%")
print(f"R@10: {hits_at_10 / total_pairs * 100:.2f}%")
print(f"R@50: {hits_at_50 / total_pairs * 100:.2f}%")
print(f"\nEvaluated {total_pairs:,} (protein, patient) pairs across {len(test_patient_ids)} held-out patients. "
      f"R@1 = the correct protein was ranked #1 out of ~9,500 candidates in {hits_at_1/total_pairs*100:.1f}% of queries. "
      f"Random baseline: R@1 ≈ 0.01%, R@10 ≈ 0.11%, R@50 ≈ 0.53%.")


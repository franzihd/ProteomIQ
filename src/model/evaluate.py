import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.embeddings.textual.embed_biobert import MAX_LENGTH, mean_pool
import torch
import h5py
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.model.projection_head import ProjectionHead

DATA_DIR = Path("data/luad")
CHECKPOINT_PATH = DATA_DIR / "checkpoint.pt"
TEST_PATIENTS_PATH = DATA_DIR / "test_patients.json"

def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(device)
    biobert_model.eval()

    protein_projection_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    textual_projection_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH)
    protein_projection_head.load_state_dict(checkpoint["protein_head"])
    textual_projection_head.load_state_dict(checkpoint["text_head"])

    protein_projection_head.eval()
    textual_projection_head.eval()

    with open(TEST_PATIENTS_PATH) as f:
        test_patient_ids = json.load(f)

    test_patient_id = test_patient_ids[0] # just evaluate on the first test patient for now
    print(f"Evaluating on patient {test_patient_id}...")

    abundance_df = pd.read_csv(DATA_DIR / "rank_normalized_proteomics.csv", header=[0,1], index_col=0)

    with h5py.File(DATA_DIR / "esm2_embeddings.h5", "r") as h5f:
        esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype= torch.float32)
        esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

    patient_row = abundance_df.loc[test_patient_id]
    esm2_id_to_idx = {pid: idx for idx, pid in enumerate(esm2_protein_ids)}

    protein_embs = []
    protein_labels = []  # store (gene_name, ENSP_ID) so we can display results later

    for (gene_name, protein_id), abundance in patient_row.items():
        if np.isnan(abundance):
            continue
        if protein_id not in esm2_id_to_idx:
            continue

        esm2_vec = esm2_matrix[esm2_id_to_idx[protein_id]]
        abundance_tensor = torch.tensor([abundance], dtype=torch.float32)
        protein_emb = torch.cat([esm2_vec, abundance_tensor], dim=0)

        protein_embs.append(protein_emb)
        protein_labels.append((gene_name, protein_id))

    protein_embs = torch.stack(protein_embs).to(device)
    print(f"Built embeddings for {len(protein_embs)} proteins")

    with torch.no_grad():
        protein_vecs = protein_projection_head(protein_embs)

    
    query1 = "receptor tyrosine kinase signaling in lung cancer"
    query2 = "ribosomal protein translation"

    encoded = tokenizer(query2, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = biobert_model(**encoded)
    
    textual_vec = mean_pool(outputs.last_hidden_state, encoded['attention_mask'])

    with torch.no_grad():
        textual_vec = textual_projection_head(textual_vec)

    similarities = (protein_vecs @ textual_vec.T).squeeze()  # shape (num_proteins,)

    ranked_indices = torch.argsort(similarities, descending=True)

    print(f"\nTop 15 proteins for query: '{query2}'\n")
    for rank, idx in enumerate(ranked_indices[:15], start=1):
        gene_name, protein_id = protein_labels[idx]
        score = similarities[idx].item()
        print(f"{rank:2d}. {gene_name:10s} ({protein_id})  similarity: {score:.4f}")

    print(f"\nLooking up specific lung-cancer oncogenes:\n")

    genes_of_interest = ["RPS3", "RPL11", "RPS6", "RPL7", "RPS14"]

    for gene in genes_of_interest:
        matches = [i for i, (gene_name, _) in enumerate(protein_labels) if gene_name == gene]
        if not matches:
            print(f"{gene}: not measured in this patient")
            continue

        idx = matches[0]
        rank = (ranked_indices == idx).nonzero(as_tuple=True)[0].item() + 1
        score = similarities[idx].item()
        print(f"{gene}: rank {rank} / {len(protein_labels)}   similarity: {score:.4f}")



if __name__ == "__main__":
    evaluate()
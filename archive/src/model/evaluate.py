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

    
    annotations = pd.read_csv(DATA_DIR / "protein_annotations.csv")
    annot_lookup = dict(zip(annotations["Gene"], annotations["Annotation"]))

    genes_of_interest = ["EGFR", "MET", "KRAS", "COL1A1", "RPS3", "RPL7", "CDK1", "ACTB", "PCNA", "TP53"]

    def get_rank(query_text):
        enc = tokenizer(query_text, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = biobert_model(**enc)
        vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
        with torch.no_grad():
            vec = textual_projection_head(vec)
        sims = (protein_vecs @ vec.T).squeeze()
        ranked = torch.argsort(sims, descending=True)
        return sims, ranked

    n_proteins = len(protein_labels)
    print(f"\n{'Gene':<10} {'Rank (with name)':>18} {'Rank (masked)':>15} {'Present?':>10}\n" + "-" * 58)

    for gene in genes_of_interest:
        gene_indices = [i for i, (g, _) in enumerate(protein_labels) if g == gene]
        if not gene_indices or gene not in annot_lookup:
            print(f"{gene:<10} {'—':>18} {'—':>15} {'not in sample':>10}")
            continue

        protein_idx = gene_indices[0]
        annotation  = annot_lookup[gene]
        masked      = annotation.split(": ", 1)[1] if ": " in annotation else annotation

        _, ranked_full   = get_rank(annotation)
        _, ranked_masked = get_rank(masked)

        rank_full   = (ranked_full   == protein_idx).nonzero(as_tuple=True)[0].item() + 1
        rank_masked = (ranked_masked == protein_idx).nonzero(as_tuple=True)[0].item() + 1

        print(f"{gene:<10} {rank_full:>12} / {n_proteins:<5} {rank_masked:>8} / {n_proteins:<5}")


if __name__ == "__main__":
    evaluate()
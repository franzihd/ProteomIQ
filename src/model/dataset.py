from pathlib import Path
import pandas as pd
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

DATA_DIR = Path("data/luad")
PROTEOMICS_EMB = DATA_DIR / "esm2_embeddings.h5"
RANK_NORM_PROTEOMICS = DATA_DIR / "rank_normalized_proteomics.csv"
BIOBERT_EMB = DATA_DIR / "biobert_embeddings.h5"
ANNOTATIONS_PATH = DATA_DIR / "protein_annotations.csv"

class ProteomicsDataset(Dataset):
    def __init__(self):
        # load the rank-normalized abundance matrix: rows = patients, columns = (protein_name, ENSP_ID)
        abundance_df = pd.read_csv(RANK_NORM_PROTEOMICS, header=[0,1], index_col=0)

        # load ESM2 protein embeddings: shape (11485, 640), one vector per protein sequence
        with h5py.File(PROTEOMICS_EMB, "r") as h5f:
            esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype= torch.float32)
            # IDs are stored as bytes in HDF5, decode to strings
            esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

        # load BioBERT text embeddings: shape (111, 11485, 768), one vector per (patient, protein) pair
        with h5py.File(BIOBERT_EMB, "r") as h5f:
            biobert_matrix = torch.tensor(h5f["embeddings"][:], dtype= torch.float32)
            biobert_patient_ids = [pid.decode() for pid in h5f["patient_ids"][:]]
            biobert_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

        # build lookup dicts: protein/patient ID string → integer index in the matrix
        # needed so we can go from a shared protein ID to the right row in each matrix
        esm2_id_to_idx = {pid: idx for idx, pid in enumerate(esm2_protein_ids)}
        biobert_protein_id_to_idx = {pid: i for i, pid in enumerate(biobert_protein_ids)}
        biobert_patient_id_to_idx = {pid: i for i, pid in enumerate(biobert_patient_ids)}

        # abundance columns are a MultiIndex (protein_name, ENSP_ID) — extract just the ENSP IDs
        abundance_protein_ids = [col[1] for col in abundance_df.columns]
        abundance_patient_ids = list(abundance_df.index)

        # keep only proteins that exist in all three sources (ESM2, BioBERT, abundance)
        # proteins missing from any one source cannot form a complete training pair
        shared_proteins = set(esm2_protein_ids) & set(biobert_protein_ids) & set(abundance_protein_ids)

        # find patients present in both BioBERT and abundance
        shared_patients = set(biobert_patient_ids) & set(abundance_patient_ids)

        # build a lookup for abundance: protein_id -> column position
        protein_id_to_abundance_col = {pid: i for i, pid in enumerate(abundance_protein_ids)}

        # convert abundance df to numpy for fast indexing
        abundance_array = abundance_df.values # shape (n_patients, n_proteins)
        abundance_patient_id_to_row = {patient_id: i for i, patient_id in enumerate(abundance_patient_ids)}

        # build flat list of (patient_id, protein_id, abundance_value) pairs where an abundance value is available - skip NaN values
        pairs = []
        for patient_id in shared_patients:
            patient_row = abundance_patient_id_to_row[patient_id]
            for protein_id in shared_proteins:
                p_col = protein_id_to_abundance_col[protein_id]
                val = abundance_array[patient_row, p_col]
                if not np.isnan(val):
                    pairs.append((patient_id, protein_id, float(val)))

        self.pairs = pairs
        self.esm2_matrix = esm2_matrix
        self.biobert_matrix = biobert_matrix
        self.esm2_id_to_idx = esm2_id_to_idx
        self.biobert_protein_id_to_idx = biobert_protein_id_to_idx
        self.biobert_patient_id_to_idx = biobert_patient_id_to_idx
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        patient_id, protein_id, abundance = self.pairs[idx]

        esm2_vec = self.esm2_matrix[self.esm2_id_to_idx[protein_id]] # shape (640,)
        abundance_tensor = torch.tensor([abundance], dtype=torch.float32) # shape (1,)
        protein_emb = torch.cat([esm2_vec, abundance_tensor], dim=0) # shape (641,)

        patient_idx = self.biobert_patient_id_to_idx[patient_id]
        protein_idx = self.biobert_protein_id_to_idx[protein_id]
        text_emb = self.biobert_matrix[patient_idx, protein_idx] # shape (768,) 

        return protein_emb, text_emb
    
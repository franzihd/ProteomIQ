from pathlib import Path
import pandas as pd
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class ProteomicsDataset(Dataset):
    def __init__(self, cohorts: list[str], test_patient_ids: set[str] | None = None):
        if test_patient_ids is None:
            test_patient_ids = set()

        self.pairs = []       # (cohort_idx, patient_id, protein_id, abundance)
        self.cohort_data = [] # per-cohort lookup structures

        for cohort_idx, cohort in enumerate(cohorts):
            data_dir = Path(f"data/{cohort}")

            abundance_df = pd.read_csv(
                data_dir / "rank_normalized_proteomics.csv",
                header=[0, 1], index_col=0
            )

            with h5py.File(data_dir / "esm2_embeddings.h5", "r") as h5f:
                esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
                esm2_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

            with h5py.File(data_dir / "biobert_embeddings.h5", "r") as h5f:
                biobert_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
                biobert_protein_ids = [pid.decode() for pid in h5f["protein_ids"][:]]

            esm2_id_to_idx = {pid: i for i, pid in enumerate(esm2_protein_ids)}
            biobert_protein_id_to_idx = {pid: i for i, pid in enumerate(biobert_protein_ids)}

            # Text no longer varies per patient (clinical description dropped), but we still
            # restrict training to the same patient pool as before (patients with a valid
            # clinical description) so this retrain isolates text content as the only variable.
            descriptions_path = data_dir / "descriptions.csv"
            if not descriptions_path.exists():
                descriptions_path = data_dir / "clinical_descriptions.csv"
            description_patient_ids = set(pd.read_csv(descriptions_path, index_col=0).index)

            abundance_protein_ids = [col[1] for col in abundance_df.columns]
            abundance_patient_ids = list(abundance_df.index)

            shared_proteins = (
                set(esm2_protein_ids) & set(biobert_protein_ids) & set(abundance_protein_ids)
            )
            shared_patients = (
                description_patient_ids & set(abundance_patient_ids)
            ) - test_patient_ids

            protein_id_to_col = {pid: i for i, pid in enumerate(abundance_protein_ids)}
            abundance_array = abundance_df.values
            patient_id_to_row = {pid: i for i, pid in enumerate(abundance_patient_ids)}

            for patient_id in shared_patients:
                patient_row = patient_id_to_row[patient_id]
                for protein_id in shared_proteins:
                    val = abundance_array[patient_row, protein_id_to_col[protein_id]]
                    if not np.isnan(val):
                        self.pairs.append((cohort_idx, patient_id, protein_id, float(val)))

            self.cohort_data.append({
                "esm2_matrix": esm2_matrix,
                "biobert_matrix": biobert_matrix,
                "esm2_id_to_idx": esm2_id_to_idx,
                "biobert_protein_id_to_idx": biobert_protein_id_to_idx,
            })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cohort_idx, patient_id, protein_id, abundance = self.pairs[idx]
        cd = self.cohort_data[cohort_idx]

        esm2_vec = cd["esm2_matrix"][cd["esm2_id_to_idx"][protein_id]]
        protein_emb = torch.cat([esm2_vec, torch.tensor([abundance], dtype=torch.float32)])

        protein_i = cd["biobert_protein_id_to_idx"][protein_id]
        text_emb = cd["biobert_matrix"][protein_i]

        return protein_emb, text_emb

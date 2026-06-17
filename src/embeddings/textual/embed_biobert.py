from pathlib import Path
import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse



MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
BATCH_SIZE = 256
MAX_LENGTH = 512


def mean_pool(token_embeddings, attention_mask):
    # Expand mask from (batch, seq_len) to (batch, seq_length, 768) so it matches token_embeddings.shape (BioBERT produces 768-dimensional embeddings)
    # Each 0/1 value is repeated 768 times to create a mask that can be applied to the token embeddings
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Zero out padding token embeddings and compute mean by dividing the sum of token embeddings by the number of non-padding tokens (sum of mask)
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def embed_batch (texts, tokenizer, model, device):
    # Tokenize the batch of texts, applying padding and truncation to ensure all sequences are the same length (max 512 tokens)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    # Move all tensors to the GPU (or CPU if GPU is not available) before passing them to the model
    encoded = {k: v.to(device) for k, v in encoded.items()}
   
    # Run BioBERT without computing gradients - we are not training, just extracting embeddings
    with torch.no_grad():
        outputs = model(**encoded)

    # output.last_hidden_state shape: (batch, seq_len, 768) — one vector per token.
    # mean_pool reduces it to (batch, 768) — one vector per sentence.
    # .cpu() moves result off GPU, .numpy() converts to numpy array.
    return mean_pool(outputs.last_hidden_state, encoded['attention_mask']).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()

    data_dir = Path(f"data/{args.cohort}")
    annotations_path = data_dir / "protein_annotations.csv"
    descriptions_path = data_dir / "clinical_descriptions.csv"
    output_path = data_dir / "biobert_embeddings.h5"
    # Load protein annotations and drop rows wehre Sequence is missing
    # reset_index is used to reset the DataFrame index after dropping rows, so it goes from 0 to n-1 instead of keeping the original indices
    annotations = pd.read_csv(annotations_path)
    annotations = annotations.dropna(subset=["Sequence", "Annotation"]).reset_index(drop=True)
    # Load patient descriptions, using the first column as the index
    descriptions = pd.read_csv(descriptions_path, index_col=0)

    n_proteins = len(annotations)
    n_patients = len(descriptions)
    print(f"Proteins: {n_proteins}, Patients: {n_patients}, Total pairs: {n_proteins * n_patients}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Download BioBERT weights from HuggingFace (cached after first run).
    # .to(device) moves model to GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # Open HDF5 file for writing. All datasets inside are written directly to disk.
    mode = "r+" if output_path.exists() else "w"
    with h5py.File(output_path, mode) as h5f:

        if mode == "w":
            emb_ds = h5f.create_dataset("embeddings", shape=(n_patients, n_proteins, 768), dtype="float32")
            h5f.create_dataset("patient_ids", data=np.array(descriptions.index.tolist(), dtype="S50"))
            h5f.create_dataset("protein_ids", data=np.array(annotations["ENSP_ID"].tolist(), dtype="S30"))
            completed = h5f.create_dataset("completed", data=np.zeros(n_patients, dtype=bool))
        else:
            emb_ds = h5f["embeddings"]
            completed = h5f["completed"]
            print(f"Resuming from patient {int(completed[:].sum())}/{n_patients}")

        for p_idx, patient_id in enumerate(tqdm(descriptions.index, desc="Patients")):

            if completed[p_idx]:
                continue

            clinical_text = descriptions.loc[patient_id, "description"]
            texts = (annotations["Annotation"] + " " + clinical_text).tolist()

            patient_embeddings = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch_emb = embed_batch(texts[i : i + BATCH_SIZE], tokenizer, model, device)
                patient_embeddings.append(batch_emb)

            emb_ds[p_idx] = np.vstack(patient_embeddings)
            completed[p_idx] = True

    print(f"Done. Saved to {output_path}")




if __name__ == "__main__":
    main()

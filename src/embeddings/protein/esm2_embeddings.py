import torch
import esm
from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cohort", required=True)
args = parser.parse_args()

DATA_DIR = Path(f"data/cptac/{args.cohort}")
ANNOTATIONS_PATH = DATA_DIR/"protein_annotations.csv"
OUTPUT_PATH = DATA_DIR/"esm2_embeddings.h5"
BATCH_SIZE = 32  # number of proteins processed at once — limited by GPU memory

# use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load ESM2 weights from disk (downloaded automatically on first run)
# model = the neural network; alphabet = the tokenizer (amino acid → integer mapping)
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
# batch_converter formats a list of (name, sequence) tuples into tensors the model can read
batch_converter = alphabet.get_batch_converter()
model.eval()    # disable dropout — we are not training, just extracting embeddings
model.to(device)  # move model weights to GPU

# load protein annotations, drop rows with missing sequence or annotation
annotations = pd.read_csv(ANNOTATIONS_PATH)
annotations = annotations.dropna(subset=["Sequence", "Annotation"]).reset_index(drop=True)
# build list of (protein_id, sequence) tuples; truncate to 1022 — ESM2's positional embedding limit
data = [(row.ENSP_ID, row.Sequence[:1022]) for row in annotations.itertuples()]

# open HDF5 in write mode on first run, read/write mode on resume
mode = "r+" if OUTPUT_PATH.exists() else "w"
with h5py.File(OUTPUT_PATH, mode) as h5f:

    if mode == "w":
        # pre-allocate the output matrix: one 640-dim vector per protein
        emb_ds = h5f.create_dataset("embeddings", shape=(len(data), 640), dtype="float32")
        # store protein IDs so we can look up which row belongs to which protein
        h5f.create_dataset("protein_ids", data=np.array(annotations["ENSP_ID"].tolist(), dtype="S30"))
        # boolean array tracking which proteins are done — allows safe stop/resume
        completed = h5f.create_dataset("completed", data=np.zeros(len(data), dtype=bool))
    else:
        # resume: reopen existing datasets
        emb_ds = h5f["embeddings"]
        completed = h5f["completed"]
        print(f"Resuming from protein {int(completed[:].sum())}/{len(data)}")

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Proteins"):

        if completed[i]:  # skip batches already written to disk
            continue

        chunk = data[i : i + BATCH_SIZE]  # take the next 32 proteins
        # tokenize: converts amino acid letters to integers and pads to equal length
        batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
        batch_tokens = batch_tokens.to(device)  # move tokens to GPU
        # count non-padding tokens per sequence (includes BOS and EOS special tokens)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():  # no gradient tracking needed — saves memory
            results = model(batch_tokens, repr_layers=[30], return_contacts=False)
            # shape: (batch_size, seq_len, 640) — one vector per token per protein
            token_representations = results["representations"][30]

        batch_representation = []
        for j, tokens_len in enumerate(batch_lens):
            # token 0 = BOS, token tokens_len-1 = EOS — skip both, average the rest
            # result: one 640-dim vector per protein
            batch_representation.append(token_representations[j, 1 : tokens_len - 1].mean(0))

        # convert list of tensors to numpy array and write directly to HDF5 on disk
        emb_ds[i : i + BATCH_SIZE] = np.array([r.cpu().numpy() for r in batch_representation])
        completed[i : i + BATCH_SIZE] = True  # mark batch as done

    print(f"Done. Saved to {OUTPUT_PATH}")

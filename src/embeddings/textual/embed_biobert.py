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


def strip_gene_name(gene, annotation):
    # Annotation is always built as "{gene}: {function text}" (fetch_uniprot.py).
    # Strips just that leading prefix -- the gene-name-masking ablation -- so the
    # rest of the functional description is untouched.
    prefix = f"{gene}: "
    if isinstance(gene, str) and annotation.startswith(prefix):
        return annotation[len(prefix):]
    return annotation


def append_go_terms(text, go_bp, go_cc, include_cc=True):
    # go_bp/go_cc are "; "-joined GO term label strings from fetch_uniprot.py
    # (already capped at MAX_GO_TERMS_PER_CATEGORY there), or NaN when a
    # protein has no terms in that category (~8% BP, ~3% CC) -- skip those.
    # include_cc=False drops Cellular Component terms entirely: they are far
    # more generic than BP terms (e.g. "cytosol"/"nucleus"/"cytoplasm" each
    # appear on 30%+ of all proteins -- verified against protein_annotations.csv),
    # which risks diluting the mean-pooled BioBERT embedding with boilerplate
    # shared across huge swaths of unrelated proteins rather than adding
    # distinguishing signal -- the suspected cause of the masked+GO pathway
    # AUROC/recall@K regression vs. masked-only (see CLAUDE.md).
    parts = [text]
    if isinstance(go_bp, str) and go_bp.strip():
        parts.append(f"Biological process: {go_bp}.")
    if include_cc and isinstance(go_cc, str) and go_cc.strip():
        parts.append(f"Cellular component: {go_cc}.")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--mask-gene-names", action="store_true",
                        help="strip the leading '{gene}: ' prefix from each annotation "
                             "before embedding; writes to biobert_embeddings_masked.h5 "
                             "instead of biobert_embeddings.h5")
    parser.add_argument("--include-go-terms", action="store_true",
                        help="append GO Biological Process + Cellular Component term "
                             "labels (from protein_annotations.csv) after the annotation "
                             "text; combinable with --mask-gene-names")
    parser.add_argument("--go-bp-only", action="store_true",
                        help="with --include-go-terms, append only Biological Process "
                             "terms and skip Cellular Component (which is highly generic "
                             "-- cytosol/nucleus/cytoplasm each cover 30%%+ of all proteins)")
    args = parser.parse_args()

    data_dir = Path(f"data/{args.cohort}")
    annotations_path = data_dir / "protein_annotations.csv"
    suffix = ""
    if args.mask_gene_names:
        suffix += "_masked"
    if args.include_go_terms:
        suffix += "_go_bp_only" if args.go_bp_only else "_go"
    output_filename = f"biobert_embeddings{suffix}.h5"
    output_path = data_dir / output_filename
    # Load protein annotations and drop rows where Sequence or Annotation is missing
    # reset_index is used to reset the DataFrame index after dropping rows, so it goes from 0 to n-1 instead of keeping the original indices
    annotations = pd.read_csv(annotations_path)
    annotations = annotations.dropna(subset=["Sequence", "Annotation"]).reset_index(drop=True)

    n_proteins = len(annotations)
    print(f"Proteins: {n_proteins}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Download BioBERT weights from HuggingFace (cached after first run).
    # .to(device) moves model to GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    if args.mask_gene_names:
        texts = [strip_gene_name(g, a) for g, a in zip(annotations["Gene"], annotations["Annotation"])]
        n_masked = sum(t != a for t, a in zip(texts, annotations["Annotation"]))
        print(f"Gene-name prefix stripped from {n_masked}/{n_proteins} annotations")
    else:
        texts = annotations["Annotation"].tolist()

    if args.include_go_terms:
        go_bp = annotations.get("GO_Biological_Process", pd.Series([None] * n_proteins))
        go_cc = annotations.get("GO_Cellular_Component", pd.Series([None] * n_proteins))
        texts = [append_go_terms(t, bp, cc, include_cc=not args.go_bp_only)
                 for t, bp, cc in zip(texts, go_bp, go_cc)]
        if args.go_bp_only:
            n_with_go = sum(isinstance(bp, str) for bp in go_bp)
            print(f"GO Biological Process terms appended to {n_with_go}/{n_proteins} annotations (CC skipped)")
        else:
            n_with_go = sum(isinstance(bp, str) or isinstance(cc, str) for bp, cc in zip(go_bp, go_cc))
            print(f"GO terms appended to {n_with_go}/{n_proteins} annotations")

    # Open HDF5 file for writing. All datasets inside are written directly to disk.
    mode = "r+" if output_path.exists() else "w"
    with h5py.File(output_path, mode) as h5f:

        if mode == "w":
            emb_ds = h5f.create_dataset("embeddings", shape=(n_proteins, 768), dtype="float32")
            h5f.create_dataset("protein_ids", data=np.array(annotations["ENSP_ID"].tolist(), dtype="S30"))
            completed = h5f.create_dataset("completed", data=np.zeros(n_proteins, dtype=bool))
        else:
            emb_ds = h5f["embeddings"]
            completed = h5f["completed"]
            print(f"Resuming from protein {int(completed[:].sum())}/{n_proteins}")

        for i in tqdm(range(0, n_proteins, BATCH_SIZE), desc="Proteins"):

            if completed[i]:
                continue

            batch_emb = embed_batch(texts[i : i + BATCH_SIZE], tokenizer, model, device)
            emb_ds[i : i + BATCH_SIZE] = batch_emb
            completed[i : i + BATCH_SIZE] = True

    print(f"Done. Saved to {output_path}")




if __name__ == "__main__":
    main()

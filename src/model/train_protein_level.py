"""
Train the no-abundance ablation: protein_emb = ESM2 only (640-dim), text_emb
= UniProt annotation only. No patient-level anything — abundance and clinical
description are both gone, so there is no patient-varying signal left at all.
Proteins are deduplicated globally across cohorts (src/model/dataset.py) and split
train/validation/test at the protein level (data/protein_splits.json), not
the patient level (see CLAUDE.md's "no-abundance ablation" notes for why).

Usage:
    uv run python src/model/train_protein_level.py \\
        --cohorts luad brca ccrcc gbm hnscc lscc ov pdac ucec \\
        --checkpoint data/runs/.../checkpoint_no_abundance.pt \\
        --epochs 300
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from transformers import get_cosine_schedule_with_warmup

BATCH_SIZE = 512
MAX_LR = 1e-5
MIN_RECOMMENDED_STEPS = 2000
SEED = 42


def infonce_loss(protein_vec, text_vec):
    logits = (protein_vec @ text_vec.T) * (1 / 0.07)
    labels = torch.arange(len(protein_vec), device=protein_vec.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train(cohorts: list[str], checkpoint_path: str, epochs: int,
          biobert_filename: str = "biobert_embeddings.h5"):
    # Fixed seed so repeated runs (e.g. comparing different --epochs values)
    # only differ because of epochs, not because of a different random weight
    # initialization or batch shuffle order each time.
    torch.manual_seed(SEED)

    with open("data/protein_splits.json") as f:
        protein_splits = json.load(f)
    val_protein_ids = set(protein_splits["validation"])
    held_out_protein_ids = val_protein_ids | set(protein_splits["test"])

    dataset = ProteomicsDataset(
        cohorts=cohorts, use_abundance=False, held_out_protein_ids=held_out_protein_ids,
        biobert_filename=biobert_filename,
    )
    print(f"Cohorts: {cohorts}  |  Training pairs (unique proteins): {len(dataset):,}")
    print(f"BioBERT source: {biobert_filename}")

    shuffle_generator = torch.Generator().manual_seed(SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            generator=shuffle_generator)

    val_dataset = ProteomicsDataset(
        cohorts=cohorts, use_abundance=False, restrict_to_protein_ids=val_protein_ids,
        biobert_filename=biobert_filename,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Validation pairs (proteins never trained on): {len(val_dataset):,}")

    total_steps = epochs * len(dataloader)
    if total_steps < MIN_RECOMMENDED_STEPS:
        print(f"[warning] only {total_steps:,} total optimizer steps "
              f"({len(dataloader)} steps/epoch x {epochs} epochs) — "
              f"consider a larger --epochs value")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head = ProjectionHead(input_dim=640, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)

    optimizer = torch.optim.AdamW(
        list(protein_head.parameters()) + list(text_head.parameters()), lr=MAX_LR
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps
    )

    out_path = Path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_name(out_path.stem + "_best" + out_path.suffix)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        protein_head.train()
        text_head.train()
        total_loss = 0
        for protein_emb, text_emb in dataloader:
            protein_emb, text_emb = protein_emb.to(device), text_emb.to(device)
            optimizer.zero_grad()

            protein_vec = protein_head(protein_emb)
            text_vec = text_head(text_emb)
            loss = infonce_loss(protein_vec, text_vec)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        protein_head.eval()
        text_head.eval()
        val_loss = 0
        with torch.no_grad():
            for protein_emb, text_emb in val_dataloader:
                protein_emb, text_emb = protein_emb.to(device), text_emb.to(device)
                protein_vec = protein_head(protein_emb)
                text_vec = text_head(text_emb)
                val_loss += infonce_loss(protein_vec, text_vec).item()
        val_loss /= len(val_dataloader)

        is_best = val_loss < best_val_loss
        print(f"Epoch {epoch + 1}/{epochs}  "
              f"loss: {total_loss / len(dataloader):.4f}  "
              f"val_loss: {val_loss:.4f}"
              + ("  <- best so far" if is_best else ""))

        checkpoint = {"protein_head": protein_head.state_dict(),
                      "text_head": text_head.state_dict(),
                      "val_loss": val_loss,
                      "epoch": epoch + 1,
                      "total_epochs": epochs}
        torch.save(checkpoint, out_path)
        if is_best:
            best_val_loss = val_loss
            torch.save(checkpoint, best_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", nargs="+", required=True,
                        help="cohorts to train on, e.g. luad brca")
    parser.add_argument("--checkpoint", required=True,
                        help="where to save checkpoint.pt")
    parser.add_argument("--epochs", type=int, required=True,
                        help="each epoch is only ~len(train_set)/512 steps (~20-160), "
                             "far fewer than the with-abundance model, so this needs to "
                             "be set deliberately — watch val_loss and stop once it plateaus")
    parser.add_argument("--biobert-filename", default="biobert_embeddings.h5",
                        help="e.g. biobert_embeddings_masked.h5 for the gene-name-masking ablation")
    args = parser.parse_args()
    train(cohorts=args.cohorts, checkpoint_path=args.checkpoint, epochs=args.epochs,
          biobert_filename=args.biobert_filename)

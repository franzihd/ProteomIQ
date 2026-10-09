import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from transformers import get_cosine_schedule_with_warmup
import json

BATCH_SIZE = 512
EPOCHS = 16
MAX_LR = 1e-5


def train(cohorts: list[str], checkpoint_path: str):
    with open("data/test_patients_per_cohort.json") as f:
        per_cohort = json.load(f)
    test_patient_ids = {pid for ids in per_cohort.values() for pid in ids}

    dataset = ProteomicsDataset(cohorts=cohorts, test_patient_ids=test_patient_ids)
    print(f"Cohorts: {cohorts}  |  Training pairs: {len(dataset):,}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)

    optimizer = torch.optim.AdamW(
        list(protein_head.parameters()) + list(text_head.parameters()), lr=MAX_LR
    )
    total_steps = EPOCHS * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps
    )

    out_path = Path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        total_loss = 0
        for protein_emb, text_emb in dataloader:
            protein_emb, text_emb = protein_emb.to(device), text_emb.to(device)
            optimizer.zero_grad()

            protein_vec = protein_head(protein_emb)
            text_vec = text_head(text_emb)

            logits = (protein_vec @ text_vec.T) * (1 / 0.07)
            labels = torch.arange(len(protein_vec), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {total_loss / len(dataloader):.4f}")
        torch.save({"protein_head": protein_head.state_dict(),
                    "text_head": text_head.state_dict()}, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", nargs="+", required=True,
                        help="cohorts to train on, e.g. luad brca")
    parser.add_argument("--checkpoint", required=True,
                        help="where to save checkpoint.pt, e.g. data/runs/luad_brca/checkpoint.pt")
    args = parser.parse_args()
    train(cohorts=args.cohorts, checkpoint_path=args.checkpoint)

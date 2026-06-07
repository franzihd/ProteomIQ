import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from transformers import get_cosine_schedule_with_warmup
import random
import json

protein_projection_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256)
textual_projection_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
EPOCHS = 16
MAX_LR = 1e-5

def train():
    dataset = ProteomicsDataset()

    # Step 1: get all unique patient IDs from the pairs (each pair = (patient_id, protein_id, abundance))
    all_patient_ids = sorted(set(patient_id for patient_id, _, _ in dataset.pairs))
    random.seed(42) # for reproducibility

    # randomly sample 10 patient IDs to hold out for testing
    test_patient_ids = set(random.sample(all_patient_ids, 10))

    # split the pairs based on whether their patient is in the test set
    train_pairs = [p for p in dataset.pairs if p[0] not in test_patient_ids]
    test_pairs = [p for p in dataset.pairs if p[0] in test_patient_ids]

    # dataset now only serves training pairs
    dataset.pairs = train_pairs

    with open("data/luad/test_patients.json", "w") as f:
        json.dump(sorted(test_patient_ids), f)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    protein_projection_head.to(device)
    textual_projection_head.to(device)

    optimizer = torch.optim.AdamW(list(protein_projection_head.parameters()) + list(textual_projection_head.parameters()), lr=MAX_LR)

    total_steps = EPOCHS * len(dataloader)
    warmup_steps = int(0.03 * total_steps) # 3% of total steps for warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        total_loss = 0
        for protein_emb, text_emb in dataloader:
            protein_emb = protein_emb.to(device)
            text_emb = text_emb.to(device)

            optimizer.zero_grad()

            protein_vec = protein_projection_head(protein_emb) # shape (batch_size, 256)
            text_vec = textual_projection_head(text_emb) # shape (batch_size, 256)

            logits = (protein_vec @ text_vec.T) * (1/0.07)   # shape (batch_size, batch_size)

            labels = torch.arange(len(protein_vec), device=device) # shape (batch_size,)

            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2 
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {total_loss / len(dataloader):.4f}")

        torch.save({
            "protein_head": protein_projection_head.state_dict(),
            "text_head": textual_projection_head.state_dict(),
        }, "data/luad/checkpoint.pt")


if __name__ == "__main__":    
    train()


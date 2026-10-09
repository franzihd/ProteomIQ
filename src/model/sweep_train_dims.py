"""
W&B sweep over hidden_dim / output_dim (lr, batch size, epochs fixed).

Usage:
    uv run wandb sweep configs/sweep_config_stage_b_dims.yaml
    uv run wandb agent <sweep-id> --count 14
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import wandb
from src.model.train_protein_level import train

FIXED = {
    "lr": 0.00011598207492707353,
    "batch_size": 512,
    "epochs": 100,
    "early_stopping_patience": 10,
    "biobert_filename": "biobert_embeddings_masked.h5",
    "splits_file": "data/splits/protein_splits_full_proteome.json",
}


def main():
    wandb.init(project="proteomiq-full-proteome-stage-b-dims")
    config = wandb.config
    hidden_dim, output_dim = (int(x) for x in config.dims.split("_"))

    checkpoint_path = f"data/runs/full_proteome/wandb_sweep_stage_b_dims/{wandb.run.id}.pt"

    def log_epoch(epoch, train_loss, val_loss):
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    best_path, best_val_loss = train(
        checkpoint_path=checkpoint_path,
        epochs=FIXED["epochs"],
        biobert_filename=FIXED["biobert_filename"],
        splits_file=FIXED["splits_file"],
        lr=FIXED["lr"],
        batch_size=FIXED["batch_size"],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        early_stopping_patience=FIXED["early_stopping_patience"],
        on_epoch_end=log_epoch,
    )

    wandb.log({
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "best_val_loss": best_val_loss,
    })
    wandb.finish()


if __name__ == "__main__":
    main()

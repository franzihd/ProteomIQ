"""
Stage B W&B sweep entry point: searches hidden_dim/output_dim, with lr,
batch_size, and epochs held fixed at the Stage A/v1 winner
(fancy-sweep-18 / run 11exn1fv: lr=1.16e-4, batch_size=512, epochs=100,
early_stopping_patience=10 -- see data/runs/full_proteome/final_v1_sweep_masked_best.pt).

Scored on val_loss, same as the v1 lr/batch_size/epochs sweep -- kept
deliberately independent of any evaluation-side metric (pathway AUROC,
recall@K, cross-modal). batch_size is fixed at 512 for every run in this
sweep (unlike v1, where it was itself swept), so there's no
ln(batch_size)-floor mismatch between candidates here -- plain val_loss is
already directly comparable across all dims combinations, no correction
needed.

Only valid (output_dim <= hidden_dim) combinations are swept: a Linear
layer's output can vary along at most min(hidden_dim, output_dim)
independent directions, so output_dim > hidden_dim is silently rank-capped,
not a real test of a larger embedding space (see CLAUDE.md 11f). Grid
search (not bayes) is used since the valid space is small and enumerable
(14 combinations) -- exhaustive coverage is both feasible and preferable to
a search heuristic here.

Not meant to be run directly -- launched by `wandb agent` once the sweep
has been created from sweep_config_stage_b_dims.yaml:

    uv run wandb sweep sweep_config_stage_b_dims.yaml
    uv run wandb agent <sweep-id-printed-above> --count 14
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import wandb
from src.model.train_protein_level import train

FIXED = {
    "cohorts": ["full_proteome"],
    "lr": 0.00011598207492707353,
    "batch_size": 512,
    "epochs": 100,
    "early_stopping_patience": 10,
    "biobert_filename": "biobert_embeddings_masked.h5",
    "splits_file": "data/protein_splits_full_proteome.json",
}


def main():
    wandb.init(project="proteomiq-full-proteome-stage-b-dims")
    config = wandb.config
    hidden_dim, output_dim = (int(x) for x in config.dims.split("_"))

    checkpoint_path = f"data/runs/full_proteome/wandb_sweep_stage_b_dims/{wandb.run.id}.pt"

    def log_epoch(epoch, train_loss, val_loss):
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    best_path, best_val_loss = train(
        cohorts=FIXED["cohorts"],
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

"""
v1 W&B sweep entry point: jointly searches lr, batch_size, and epochs
(projection dimensions stay fixed, see FIXED below -- that axis is left to
the separate, AUROC-based Stage B search, not tuned here).

Selects by `gap_below_random = ln(batch_size) - val_loss` -- NOT raw
val_loss, because val_loss's own random-guessing floor is ln(batch_size)
(the expected InfoNCE loss of a model that hasn't learned anything), which
is different for each swept batch_size value. Subtracting it out makes
val_loss fairly comparable across different batch_size candidates within
this sweep. Same correction already used once in this project's earlier
grid search (see CLAUDE.md section 11e) -- applied here from the start,
not discovered after the fact.

This sweep never touches pathway AUROC or the test split -- selection is
based entirely on the validation split's InfoNCE loss, so there is no
leakage risk (contrast with the leakage found in the earlier 11e/11f
search, fixed in 11j).

All W&B-specific code lives here -- train_protein_level.py's optional
early_stopping_patience param is used with patience=10 (stop once val_loss
hasn't improved for 10 consecutive epochs), so a run's sampled epochs value
is a ceiling, not a guarantee every run trains that long.

Not meant to be run directly -- launched by `wandb agent` once a sweep has
been created from sweep_config_v1_lr_only.yaml:

    uv run wandb sweep sweep_config_v1_lr_only.yaml
    uv run wandb agent <sweep-id-printed-above> --count 30
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import wandb
from src.model.train_protein_level import train

FIXED = {
    "hidden_dim": 512,
    "output_dim": 128,  # matches ProteinCLIP's dshared=128 (Lucas et al., 2024) -- not swept here
    "early_stopping_patience": 10,
    "biobert_filename": "biobert_embeddings_masked.h5",
    "splits_file": "data/protein_splits_full_proteome.json",
}
# temperature (0.07) is already hardcoded in train_protein_level.py's
# infonce_loss -- matches the spec exactly, nothing to configure here.


def main():
    wandb.init(project="proteomiq-full-proteome-v1-lr-sweep")
    config = wandb.config
    random_floor = math.log(config.batch_size)

    # Track the *maximum* gap_below_random seen during the run (i.e. the
    # best epoch) as the run's summary value -- this is what the sweep
    # optimizes (see sweep_config_v1_lr_only.yaml).
    wandb.define_metric("gap_below_random", summary="max")
    wandb.define_metric("val_loss", summary="min")

    checkpoint_path = f"data/runs/full_proteome/wandb_sweep_v1_lr_only/{wandb.run.id}.pt"

    def log_epoch(epoch, train_loss, val_loss):
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gap_below_random": random_floor - val_loss,
            "learning_rate": config.lr,
        }, step=epoch)

    train(
        checkpoint_path=checkpoint_path,
        epochs=config.epochs,
        biobert_filename=FIXED["biobert_filename"],
        splits_file=FIXED["splits_file"],
        lr=config.lr,
        batch_size=config.batch_size,
        hidden_dim=FIXED["hidden_dim"],
        output_dim=FIXED["output_dim"],
        early_stopping_patience=FIXED["early_stopping_patience"],
        on_epoch_end=log_epoch,
    )
    wandb.finish()


if __name__ == "__main__":
    main()

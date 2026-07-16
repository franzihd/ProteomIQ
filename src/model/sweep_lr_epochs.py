"""
Controlled grid sweep over learning rate x batch size x epoch count for the
masked, full-proteome, no-abundance model (see CLAUDE.md section 11).

Batch size and learning rate are swept jointly (not sequentially) because
they plausibly interact here: a smaller batch size gives more steps/epoch
(more frequent updates) but fewer InfoNCE negatives per step, while a larger
LR also targets "too few effective updates" from a different angle -- so
tuning one at a time, holding the other fixed at a guessed value, risks
missing the actual joint optimum. Epoch count is still swept alongside both,
since total_steps = epochs x steps_per_epoch determines the cosine schedule.

Calls train_protein_level.train() directly (not via subprocess) once per
(lr, batch_size, epochs) combination, all with the same fixed seed, so
results differ only because of the swept values -- not because of a
hand-edited module constant changing mid-sweep (the ambiguity that
motivated this script; see the lr/batch_size/biobert_filename/splits_file
fields now saved into every checkpoint for the same reason).

Usage:
    uv run python src/model/sweep_lr_epochs.py \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --splits-file data/protein_splits_full_proteome.json \\
        --out-dir data/runs/full_proteome \\
        --lrs 3e-6 1e-5 3e-5 1e-4 \\
        --batch-sizes 128 256 512 \\
        --epochs-list 250 500 1000
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import pandas as pd
from src.model.train_protein_level import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5")
    parser.add_argument("--splits-file", default="data/protein_splits_full_proteome.json")
    parser.add_argument("--out-dir", default="data/runs/full_proteome")
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-6, 1e-5, 3e-5, 1e-4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--epochs-list", type=int, nargs="+", default=[250, 500, 1000])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = []
    for lr in args.lrs:
        for batch_size in args.batch_sizes:
            for epochs in args.epochs_list:
                tag = f"sweep_lr{lr:.0e}_bs{batch_size}_e{epochs}"
                checkpoint_path = out_dir / f"{tag}.pt"
                print(f"\n=== {tag} ===")
                train(
                    cohorts=["full_proteome"],
                    checkpoint_path=str(checkpoint_path),
                    epochs=epochs,
                    biobert_filename=args.biobert_filename,
                    splits_file=args.splits_file,
                    lr=lr,
                    batch_size=batch_size,
                )
                best_path = checkpoint_path.with_name(checkpoint_path.stem + "_best.pt")
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                rows.append({
                    "lr": lr,
                    "batch_size": batch_size,
                    "epochs_budget": epochs,
                    "best_epoch": ckpt["epoch"],
                    "best_val_loss": ckpt["val_loss"],
                    "checkpoint": str(best_path),
                })
                # Write the running table after every config, not just at the end --
                # a crash or interrupt partway through the grid shouldn't lose
                # already-finished results.
                df = pd.DataFrame(rows).sort_values("best_val_loss")
                df.to_csv(out_dir / "lr_epoch_sweep_results.csv", index=False)

    print(f"\nSaved sweep summary to {out_dir / 'lr_epoch_sweep_results.csv'}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

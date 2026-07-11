"""
Generate a reproducible, GLOBAL protein split (train/validation/test) for the
no-abundance ablation. Unlike the patient-level split (test_patients_per_cohort.json),
these proteins are excluded from EVERY cohort during training, so the model
never sees them in any patient's data.

Validation proteins are used to pick a sane epoch count / watch training
progress (the deduped, no-abundance dataset needs far more epochs per
gradient-step budget than the original per-patient dataset, since each epoch
now iterates over ~11k unique proteins instead of ~1M-13M per-patient pairs).
Test proteins are touched only once, for the final reported numbers — using
them to pick the epoch count instead would leak test information into a
training decision.

Reads the same "shared_proteins" pool ProteomicsDataset uses (ESM2 ∩ BioBERT
∩ abundance columns) per cohort, takes the union across cohorts, and splits
80% train / 10% validation / 10% test with random.seed(42). Saves to
data/protein_splits.json as {"validation": [...], "test": [...]}; train is
implicitly everything else (dataset.py excludes validation ∪ test).

Usage:
    uv run python src/model/protein_train_validation_test.py
"""
import json
import random
import h5py
import pandas as pd
from pathlib import Path

COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
OUT = Path("data/protein_splits.json")

all_proteins = set()
for cohort in COHORTS:
    data_dir = Path(f"data/{cohort}")
    esm2_path = data_dir / "esm2_embeddings.h5"
    biobert_path = data_dir / "biobert_embeddings.h5"
    abundance_path = data_dir / "rank_normalized_proteomics.csv"
    if not (esm2_path.exists() and biobert_path.exists() and abundance_path.exists()):
        print(f"[skip] {cohort} — missing esm2/biobert/abundance file")
        continue

    with h5py.File(esm2_path) as f:
        esm2_ids = {p.decode() for p in f["protein_ids"][:]}
    with h5py.File(biobert_path) as f:
        biobert_ids = {p.decode() for p in f["protein_ids"][:]}
    abundance_columns = pd.read_csv(abundance_path, header=[0, 1], index_col=0).columns
    abundance_ids = {col[1] for col in abundance_columns}

    shared = esm2_ids & biobert_ids & abundance_ids
    print(f"{cohort}: {len(shared)} shared proteins")
    all_proteins |= shared

all_proteins = sorted(all_proteins)
shuffled = all_proteins.copy()
random.seed(42)
random.shuffle(shuffled)

n_val = round(len(shuffled) * VAL_FRACTION)
n_test = round(len(shuffled) * TEST_FRACTION)

validation = sorted(shuffled[:n_val])
test = sorted(shuffled[n_val:n_val + n_test])
train = sorted(shuffled[n_val + n_test:])

with open(OUT, "w") as f:
    json.dump({"validation": validation, "test": test}, f, indent=2)

print(f"\nTotal proteins across all cohorts: {len(all_proteins)}")
print(f"Train: {len(train)}  |  Validation: {len(validation)}  |  Test: {len(test)}")
print(f"Saved to {OUT}")

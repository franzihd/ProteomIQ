"""
Generate reproducible test splits: 10 patients per cohort.
Reads patient IDs from each cohort's biobert_embeddings.h5 and samples 10
with random.seed(42). Saves to data/test_patients_per_cohort.json.

Usage:
    uv run python src/model/generate_test_splits.py
"""
import sys
import json
import random
import h5py
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]
N_TEST  = 10
OUT     = Path("data/test_patients_per_cohort.json")

splits = {}
for cohort in COHORTS:
    h5_path  = Path(f"data/{cohort}/biobert_embeddings.h5")
    csv_path = Path(f"data/{cohort}/rank_normalized_proteomics.csv")
    if not h5_path.exists():
        print(f"[skip] {cohort} — biobert_embeddings.h5 not found")
        continue
    with h5py.File(h5_path) as f:
        biobert_ids = [p.decode() for p in f["patient_ids"][:]]
    import pandas as pd
    abundance_ids = set(pd.read_csv(csv_path, header=[0,1], index_col=0).index)
    patient_ids = [p for p in biobert_ids if p in abundance_ids]
    random.seed(42)
    test_ids = random.sample(patient_ids, N_TEST)
    splits[cohort] = test_ids
    print(f"{cohort}: {len(patient_ids)} valid patients → held out {test_ids}")

with open(OUT, "w") as f:
    json.dump(splits, f, indent=2)

print(f"\nSaved to {OUT}")
print(f"Total test patients: {sum(len(v) for v in splits.values())}")

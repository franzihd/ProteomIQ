"""
For each KEGG pathway, compute how many member genes are on average
measured (non-NaN) across all test patients. Saves a ranked CSV to
data/evaluation/pathway_coverage.csv — use this to pick pathways
worth evaluating.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
import numpy as np
import pandas as pd
import gseapy

TEST_PATIENT_PATH = Path("data/test_patients_per_cohort.json")
OUT_PATH = Path("data/evaluation/pathway/pathway_choice.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(TEST_PATIENT_PATH) as f:
    test_patients_per_cohort = json.load(f)

# collect the set of measured gene names per test patient
print("Loading measured genes per test patient ...")
patient_genes: dict[str, set] = {}

for cohort, patient_ids in test_patients_per_cohort.items():
    abundance_path = Path(f"data/cptac/{cohort}/rank_normalized_proteomics.csv")
    if not abundance_path.exists():
        print(f"  Skipping {cohort} — missing abundance file")
        continue

    df = pd.read_csv(abundance_path, header=[0, 1], index_col=0)

    for patient_id in patient_ids:
        if patient_id not in df.index:
            continue
        row = df.loc[patient_id]
        # gene name is the first level of the MultiIndex column
        genes = {gene for (gene, _), val in row.items() if not np.isnan(val)}
        patient_genes[patient_id] = genes

print(f"  {len(patient_genes)} test patients loaded\n")

print("Loading KEGG pathways ...")
gene_sets = gseapy.get_library("KEGG_2021_Human")
print(f"  {len(gene_sets)} pathways loaded\n")

results = []
for pathway_name, pathway_genes in gene_sets.items():
    pathway_genes = set(pathway_genes)
    counts = [len(pathway_genes & g) for g in patient_genes.values()]
    results.append({
        "pathway":           pathway_name,
        "kegg_total_genes":  len(pathway_genes),
        "mean_measured":     round(np.mean(counts), 1),
        "min_measured":      int(np.min(counts)),
        "max_measured":      int(np.max(counts)),
    })

df_out = pd.DataFrame(results).sort_values("mean_measured", ascending=False)
df_out.to_csv(OUT_PATH, index=False)

print(df_out.head(30).to_string(index=False))
print(f"\nSaved to {OUT_PATH}")

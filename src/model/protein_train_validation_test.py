"""
Generate a reproducible, GLOBAL, homology-aware protein split (train/
validation/test) over the full reviewed human UniProt proteome (see
CLAUDE.md section 11) -- not just the subset of proteins CPTAC happened to
measure. Supersedes protein_train_validation_test.py's original pure-random
design (data/protein_splits.json, kept untouched for comparability with the
existing baseline checkpoint).

Two changes from the original:
1. One flat protein universe (data/full_proteome/), not a union across 9
   cohorts -- Stage 1 doesn't need a protein to have been measured by CPTAC
   at all, only a sequence + UniProt text.
2. Split by CD-HIT sequence-identity CLUSTER (data/protein_clusters.json,
   90% identity), not by individual protein -- a near-identical paralog can
   no longer land on both sides of the split, which pure random assignment
   didn't guard against.

Validation proteins are used to pick a sane epoch count / watch training
progress. Test proteins are touched only once, for the final reported
numbers -- using them to pick the epoch count would leak test information
into a training decision.

Usage:
    uv run python src/model/protein_train_validation_test.py \\
        --biobert-filename biobert_embeddings_masked.h5
"""
import argparse
import json
import random
from pathlib import Path

import h5py

VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
FULL_PROTEOME_DIR = Path("data/full_proteome")
CLUSTERS_PATH = Path("data/protein_clusters.json")
OUT = Path("data/protein_splits_full_proteome.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5")
    args = parser.parse_args()

    with h5py.File(FULL_PROTEOME_DIR / "esm2_embeddings.h5") as f:
        esm2_ids = {p.decode() for p in f["protein_ids"][:]}
    with h5py.File(FULL_PROTEOME_DIR / args.biobert_filename) as f:
        biobert_ids = {p.decode() for p in f["protein_ids"][:]}

    usable_proteins = esm2_ids & biobert_ids
    print(f"Usable proteins (ESM2 ∩ BioBERT): {len(usable_proteins)}")

    with open(CLUSTERS_PATH) as f:
        protein_to_cluster = json.load(f)

    clusters = {}
    n_missing_cluster = 0
    next_singleton_id = max(protein_to_cluster.values(), default=-1) + 1
    for protein_id in usable_proteins:
        cluster_id = protein_to_cluster.get(protein_id)
        if cluster_id is None:
            # Wasn't in the CD-HIT input (e.g. dropped for missing sequence/
            # annotation there but present in ESM2+BioBERT some other way) --
            # treat as its own singleton cluster rather than dropping it.
            cluster_id = next_singleton_id
            next_singleton_id += 1
            n_missing_cluster += 1
        clusters.setdefault(cluster_id, []).append(protein_id)
    if n_missing_cluster:
        print(f"[note] {n_missing_cluster} usable proteins had no CD-HIT cluster; "
              f"assigned as singletons")

    cluster_ids = list(clusters.keys())
    random.seed(42)
    random.shuffle(cluster_ids)

    n_val_target = round(len(usable_proteins) * VAL_FRACTION)
    n_test_target = round(len(usable_proteins) * TEST_FRACTION)

    validation, test, train = [], [], []
    val_count = test_count = 0
    for cluster_id in cluster_ids:
        members = clusters[cluster_id]
        if val_count < n_val_target:
            validation.extend(members)
            val_count += len(members)
        elif test_count < n_test_target:
            test.extend(members)
            test_count += len(members)
        else:
            train.extend(members)

    validation = sorted(validation)
    test = sorted(test)
    train = sorted(train)

    # Sanity checks: no cluster should straddle a split boundary, and no
    # protein should appear in more than one split.
    val_clusters = {protein_to_cluster.get(p) for p in validation}
    test_clusters = {protein_to_cluster.get(p) for p in test}
    train_clusters = {protein_to_cluster.get(p) for p in train}
    assert not (val_clusters & test_clusters), "a cluster spans validation and test"
    assert not (val_clusters & train_clusters), "a cluster spans validation and train"
    assert not (test_clusters & train_clusters), "a cluster spans test and train"
    assert not (set(validation) & set(test) & set(train)), "a protein appears in multiple splits"

    with open(OUT, "w") as f:
        json.dump({"validation": validation, "test": test}, f, indent=2)

    print(f"\nTotal usable proteins: {len(usable_proteins)}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Train: {len(train)}  |  Validation: {len(validation)}  |  Test: {len(test)}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

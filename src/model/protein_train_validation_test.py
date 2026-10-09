"""
Homology-aware train/validation/test split (80/10/10) over CD-HIT clusters.

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
CLUSTERS_PATH = Path("data/splits/protein_clusters.json")
OUT = Path("data/splits/protein_splits_full_proteome.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5")
    parser.add_argument("--random-control", action="store_true",
                        help="diagnostic control split: shuffle individual proteins instead "
                             "of CD-HIT clusters, over the same full-proteome protein set. "
                             "Isolates whether the homology-aware split's effect on eval "
                             "metrics is really about paralog leakage, not a side effect of "
                             "some other change (protein universe, hyperparameters). Written "
                             "to data/splits/protein_splits_full_proteome_random.json instead of "
                             "overwriting the real homology-aware split.")
    args = parser.parse_args()

    with h5py.File(FULL_PROTEOME_DIR / "esm2_embeddings.h5") as f:
        esm2_ids = {p.decode() for p in f["protein_ids"][:]}
    with h5py.File(FULL_PROTEOME_DIR / args.biobert_filename) as f:
        biobert_ids = {p.decode() for p in f["protein_ids"][:]}

    usable_proteins = esm2_ids & biobert_ids
    print(f"Usable proteins (ESM2 ∩ BioBERT): {len(usable_proteins)}")

    if args.random_control:
        # Every protein is its own singleton "cluster" -- i.e. no homology
        # grouping at all, just a pure per-protein random split, over the
        # exact same protein set as the real homology-aware split.
        clusters = {i: [protein_id] for i, protein_id in enumerate(sorted(usable_proteins))}
        protein_to_cluster = {protein_id: i for i, members in clusters.items() for protein_id in members}
    else:
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

    out_path = Path("data/splits/protein_splits_full_proteome_random.json") if args.random_control else OUT
    with open(out_path, "w") as f:
        json.dump({"validation": validation, "test": test}, f, indent=2)

    print(f"\nTotal usable proteins: {len(usable_proteins)}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Train: {len(train)}  |  Validation: {len(validation)}  |  Test: {len(test)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

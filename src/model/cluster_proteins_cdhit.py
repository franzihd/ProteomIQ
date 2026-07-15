import json
import re
import subprocess
from pathlib import Path

import pandas as pd

CDHIT_BINARY = str(Path.home() / "tools" / "cdhit" / "cd-hit")
IDENTITY_THRESHOLD = 0.9  # strict: catches near-duplicate/isoform leakage
# without wiping small pathway families out of training entirely -- see
# CLAUDE.md section 11a.
WORD_LENGTH = 5  # CD-HIT's recommended word length for -c 0.9 (0.7-1.0 band)

FASTA_PATH = Path("data/full_proteome/proteins.fasta")
CDHIT_OUTPUT_PATH = Path("data/full_proteome/proteins_cdhit")
CLUSTERS_OUTPUT_PATH = Path("data/protein_clusters.json")


def write_fasta(annotations, fasta_path):
    with open(fasta_path, "w") as f:
        for _, row in annotations.iterrows():
            f.write(f">{row['ENSP_ID']}\n{row['Sequence']}\n")


def run_cdhit(fasta_path, output_path):
    subprocess.run(
        [
            CDHIT_BINARY,
            "-i", str(fasta_path),
            "-o", str(output_path),
            "-c", str(IDENTITY_THRESHOLD),
            "-n", str(WORD_LENGTH),
            "-M", "0",  # unlimited memory
            "-T", "0",  # use all available threads
        ],
        check=True,
    )


def parse_clstr_file(clstr_path):
    """CD-HIT .clstr format:
    >Cluster 0
    0    640aa, >ENSP00000123456.1... *
    1    620aa, >ENSP00000234567.2... at 95.00%
    >Cluster 1
    ...
    Returns {protein_id: cluster_id}."""
    protein_to_cluster = {}
    cluster_id = -1
    member_pattern = re.compile(r">(\S+)\.\.\.")
    with open(clstr_path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[-1])
            else:
                match = member_pattern.search(line)
                if match:
                    protein_to_cluster[match.group(1)] = cluster_id
    return protein_to_cluster


def main():
    annotations = pd.read_csv("data/full_proteome/protein_annotations.csv")
    annotations = annotations.dropna(subset=["Sequence", "Annotation"])
    print(f"Proteins to cluster: {len(annotations)}")

    FASTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(annotations, FASTA_PATH)

    print(f"Running CD-HIT at {IDENTITY_THRESHOLD*100:.0f}% identity...")
    run_cdhit(FASTA_PATH, CDHIT_OUTPUT_PATH)

    clstr_path = Path(str(CDHIT_OUTPUT_PATH) + ".clstr")
    protein_to_cluster = parse_clstr_file(clstr_path)

    n_clusters = len(set(protein_to_cluster.values()))
    print(f"Proteins clustered: {len(protein_to_cluster)}")
    print(f"Total clusters: {n_clusters}")
    print(f"Mean cluster size: {len(protein_to_cluster)/n_clusters:.2f}")

    cluster_sizes = pd.Series(list(protein_to_cluster.values())).value_counts()
    print(f"Clusters with >1 member (near-duplicates found): {(cluster_sizes > 1).sum()}")
    print(f"Largest cluster size: {cluster_sizes.max()}")

    with open(CLUSTERS_OUTPUT_PATH, "w") as f:
        json.dump(protein_to_cluster, f)
    print(f"Saved protein_id -> cluster_id map to {CLUSTERS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

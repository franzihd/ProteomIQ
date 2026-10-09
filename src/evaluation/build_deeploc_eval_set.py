"""
Builds the human-only DeepLoc test set for zero_shot_deeploc_eval.py.

Usage:
    uv run python src/evaluation/build_deeploc_eval_set.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests

DEEPLOC_URL = "https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta"
RAW_FASTA_PATH = Path("data/deeploc/deeploc_data.fasta")
OUTPUT_PATH = Path("data/evaluation/zero_shot_classification/deeploc/test_set.csv")
FULL_PROTEOME_ANNOTATIONS = Path("data/full_proteome/protein_annotations.csv")
SPLITS_FILE = Path("data/splits/protein_splits_full_proteome.json")

SOLUBILITY_SUFFIXES = {"M", "S", "U"}  # membrane / soluble / unknown -- not used here

# DeepLoc raw location label -> our class name. "Plastid" is kept as
# "Chloroplast" to match the 10 requested class names, even though it will
# have zero human members (chloroplasts don't exist in animals) -- included
# only for symmetry with the full class list, same as any other pathway/
# class that structurally can't score above baseline for this organism.
LOCATION_NAME_MAP = {
    "Nucleus": "Nucleus",
    "Cytoplasm": "Cytoplasm",
    "Extracellular": "Extracellular",
    "Mitochondrion": "Mitochondrion",
    "Cell.membrane": "Cell membrane",
    "Endoplasmic.reticulum": "Endoplasmic reticulum",
    "Plastid": "Chloroplast",
    "Golgi.apparatus": "Golgi apparatus",
    "Lysosome/Vacuole": "Lysosome/Vacuole",
    "Peroxisome": "Peroxisome",
}


def download_fasta(force: bool = False):
    if RAW_FASTA_PATH.exists() and not force:
        print(f"  {RAW_FASTA_PATH} already cached, skipping download (use --force to re-download)")
        return
    RAW_FASTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {DEEPLOC_URL} ...")
    response = requests.get(DEEPLOC_URL, timeout=60)
    response.raise_for_status()
    RAW_FASTA_PATH.write_bytes(response.content)
    print(f"  Saved to {RAW_FASTA_PATH}")


def parse_fasta_headers(fasta_path: Path):
    """Only the headers matter here (accession + label) -- sequences are
    already covered by our own full_proteome pull, so they're skipped."""
    entries = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line[1:].strip().split()
            accession = parts[0]
            raw_label = parts[1]
            deeploc_partition = parts[2] if len(parts) > 2 else "train"

            label_parts = raw_label.split("-")
            if label_parts[-1] in SOLUBILITY_SUFFIXES:
                location_raw = "-".join(label_parts[:-1])
            else:
                location_raw = raw_label

            entries.append({
                "uniprot_accession": accession,
                "location_raw": location_raw,
                "deeploc_partition": deeploc_partition,
            })
    return entries


def run(force_download: bool = False):
    print("Step 1: fetching DeepLoc-1.0 raw data ...")
    download_fasta(force=force_download)

    entries = parse_fasta_headers(RAW_FASTA_PATH)
    print(f"  {len(entries)} raw DeepLoc entries parsed\n")

    print("Step 2: keeping single-label entries in our 10 target classes ...")
    single_label = [e for e in entries if e["location_raw"] in LOCATION_NAME_MAP]
    dropped_multi = len(entries) - len(single_label)
    print(f"  {len(single_label)} single-label entries kept "
          f"({dropped_multi} dual/multi-location entries dropped, e.g. 'Cytoplasm-Nucleus')\n")

    print("Step 3: restricting to human proteins via our own full_proteome pull ...")
    proteome_df = pd.read_csv(FULL_PROTEOME_ANNOTATIONS)
    accession_to_ensp = dict(zip(proteome_df["UniProt_Accession"], proteome_df["ENSP_ID"]))
    accession_to_gene = dict(zip(proteome_df["UniProt_Accession"], proteome_df["Gene"]))
    human_ids = set(accession_to_ensp)

    human_entries = [e for e in single_label if e["uniprot_accession"] in human_ids]
    print(f"  {len(human_entries)} / {len(single_label)} entries are human "
          f"(present in our {len(human_ids):,}-protein reviewed human UniProt pull)\n")

    print("Step 4: assigning seen/unseen relative to our model's own train split ...")
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    # Split into unseen_validation / unseen_test rather than one combined
    # "unseen" bucket -- the final checkpoint's hyperparameters/architecture
    # were chosen via validation LOSS during contrastive training (11l/11p),
    # so validation-split proteins were "seen" by the model-selection
    # process, just never by this eval's own labels. Reporting unseen_test
    # separately gives a clean, once-touched figure, same discipline as
    # validation_only_auroc vs. test_only_auroc in pathway_auroc_protein_level.py
    #. "unseen" (validation+test combined) is still assigned
    # too, purely for backward compatibility with existing seen/unseen
    # reporting -- see split_bucket().
    validation_ensp_ids = set(splits["validation"])
    test_ensp_ids = set(splits["test"])

    def split_bucket(ensp_id: str) -> str:
        if ensp_id in validation_ensp_ids:
            return "unseen_validation"
        if ensp_id in test_ensp_ids:
            return "unseen_test"
        return "seen"

    rows = []
    for e in human_entries:
        accession = e["uniprot_accession"]
        ensp_id = accession_to_ensp[accession]
        rows.append({
            "uniprot_accession": accession,
            "ensp_id": ensp_id,
            "gene": accession_to_gene[accession],
            "true_class": LOCATION_NAME_MAP[e["location_raw"]],
            "split": split_bucket(ensp_id),
            "deeploc_partition": e["deeploc_partition"],  # informational only, see module docstring
        })

    result_df = pd.DataFrame(rows).sort_values(["true_class", "uniprot_accession"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    n_unseen = (result_df["split"] != "seen").sum()
    print(f"  seen: {(result_df['split'] == 'seen').sum()}, "
          f"unseen: {n_unseen} "
          f"(unseen_validation: {(result_df['split'] == 'unseen_validation').sum()}, "
          f"unseen_test: {(result_df['split'] == 'unseen_test').sum()})\n")
    print("Class distribution:")
    print(result_df.groupby(["true_class", "split"]).size().unstack(fill_value=0))
    print(f"\nSaved {len(result_df)} human proteins to {OUTPUT_PATH}")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-download the DeepLoc FASTA even if a cached copy exists")
    args = parser.parse_args()
    run(force_download=args.force)

"""
Builds the human-only DeepLoc subcellular-localization test set used by
zero_shot_deeploc_eval.py.

DeepLoc-1.0 (Almagro Armenteros et al. 2017) is a general eukaryote dataset
(~14,000 proteins across many species) -- it is NOT human-specific, so using
it as-is would let non-human proteins into the eval. We only care about
human proteins here (comparing against a model trained entirely on human
CPTAC data would otherwise not be a fair test), so this script filters to
the intersection with data/full_proteome/protein_annotations.csv -- our own
already-vetted, reviewed, organism_id:9606-only UniProt pull (see CLAUDE.md
section 11b). Intersecting by UniProt accession is a simple, safe way to
guarantee human-only: any accession that made it into our own human-only
pull is human by construction, no extra taxonomy lookup needed. The cost is
a small number of missed proteins where UniProt has since updated/merged the
accession since DeepLoc was published (2017) -- acceptable, not chased down
further (same spirit as other small, quantified data-loss footnotes in this
project, e.g. CLAUDE.md's "9 genes genuinely absent" note in section 11b).

DeepLoc's FASTA headers look like:
    >Q9H400 Cell.membrane-M test
    >P83456 Cell.membrane-M
The second token is "{Location}-{Membrane}" where Membrane in {M,S,U} is a
separate soluble/membrane-bound label DeepLoc also predicts -- irrelevant
here, stripped and discarded. Some proteins carry a compound location, e.g.
"Cytoplasm-Nucleus-U" (dual localization) -- excluded, since this eval is a
single-label argmax classification, not multi-label. The optional trailing
"test" token is DeepLoc's OWN train/test split, from a different task (they
used it to train their own classifier) -- irrelevant to us since we do zero-
shot classification (no training on DeepLoc labels at all), so it's kept
only as an informational column, not used for anything.

The seen/unseen split that matters here is defined relative to OUR model's
own training data: "seen" = protein was in the training portion of
data/protein_splits_full_proteome.json (i.e. the model's protein_head could
plausibly have learned this protein's representation during contrastive
training), "unseen" = protein is in that split's validation or test portion
(never trained on). This mirrors the test_only_auroc / validation_only_auroc
distinction already used throughout pathway_auroc_protein_level.py.

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
OUTPUT_PATH = Path("data/evaluation/deeploc_human_test_set.csv")
FULL_PROTEOME_ANNOTATIONS = Path("data/full_proteome/protein_annotations.csv")
SPLITS_FILE = Path("data/protein_splits_full_proteome.json")

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
    unseen_ensp_ids = set(splits["validation"]) | set(splits["test"])

    rows = []
    for e in human_entries:
        accession = e["uniprot_accession"]
        ensp_id = accession_to_ensp[accession]
        rows.append({
            "uniprot_accession": accession,
            "ensp_id": ensp_id,
            "gene": accession_to_gene[accession],
            "true_class": LOCATION_NAME_MAP[e["location_raw"]],
            "split": "unseen" if ensp_id in unseen_ensp_ids else "seen",
            "deeploc_partition": e["deeploc_partition"],  # informational only, see module docstring
        })

    result_df = pd.DataFrame(rows).sort_values(["true_class", "uniprot_accession"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"  seen: {(result_df['split'] == 'seen').sum()}, "
          f"unseen: {(result_df['split'] == 'unseen').sum()}\n")
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

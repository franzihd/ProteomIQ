"""
Builds the EC (Enzyme Commission) top-level-class test set used by
zero_shot_ec_eval.py.

Same overall recipe as build_deeploc_eval_set.py, applied to a different
label source. EC numbers aren't in data/full_proteome/protein_annotations.csv
(fetch_full_human_proteome.py never requested that field), so this script
re-queries the UniProt REST search API -- same organism_id:9606 AND
reviewed:true query as fetch_full_human_proteome.py, just asking for the
`ec` field instead of function/sequence/GO text -- and merges the result
back onto our own already-vetted human proteome pull by UniProt accession.
Merging (rather than using the fresh fetch directly) guarantees every
protein in the eval set already has a resolved ENSP_ID and passed the same
FUNCTION-annotation/Ensembl-xref filtering as the rest of the project.

A UniProt entry can carry more than one EC number (e.g. multi-domain or
multi-function enzymes) via proteinDescription.recommendedName.ecNumbers,
.alternativeNames[].ecNumbers, and per-domain names. We collect all of them
and take the EC top-level class (the first "N" in "N.n.n.n") -- exactly the
kind of thing that is ambiguous for a single-label task if a protein's EC
numbers span more than one top-level class, so we drop those (a handful),
same spirit as build_deeploc_eval_set.py dropping dual-localization entries.

Proteins with zero EC numbers are non-enzymes and dropped -- this eval only
covers enzymes, by construction.

Usage:
    uv run python src/evaluation/build_ec_eval_set.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests

FULL_PROTEOME_ANNOTATIONS = Path("data/full_proteome/protein_annotations.csv")
SPLITS_FILE = Path("data/protein_splits_full_proteome.json")
RAW_EC_PATH = Path("data/full_proteome/uniprot_ec_numbers.csv")
OUTPUT_PATH = Path("data/evaluation/ec_top_level_test_set.csv")

EC_TOP_LEVEL_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


def extract_ec_numbers(entry: dict) -> list[str]:
    """Walks proteinDescription looking for every "ecNumbers" list, anywhere
    in the structure (recommendedName, alternativeNames, per-domain names).
    A plain recursive walk is simpler and more robust than hardcoding the
    exact nesting UniProt uses, and doesn't break if a protein's EC numbers
    live somewhere other than the top-level recommendedName."""
    found = []
    seen = set()

    def walk(node):
        if isinstance(node, dict):
            if "ecNumbers" in node:
                for ec in node["ecNumbers"]:
                    value = ec.get("value")
                    if value and value not in seen:
                        seen.add(value)
                        found.append(value)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(entry.get("proteinDescription", {}))
    return found


def fetch_ec_numbers(force: bool = False) -> pd.DataFrame:
    """Same paginated organism_id:9606 AND reviewed:true query as
    fetch_full_human_proteome.py, requesting only accession + ec so each
    page is small and the run stays fast (a few minutes, same as the
    original full-proteome fetch)."""
    if RAW_EC_PATH.exists() and not force:
        print(f"  {RAW_EC_PATH} already cached, skipping fetch (use --force to re-fetch)")
        return pd.read_csv(RAW_EC_PATH)

    url = ("https://rest.uniprot.org/uniprotkb/search"
           "?query=organism_id:9606+AND+reviewed:true"
           "&fields=accession,ec&format=json&size=500")

    rows = []
    n_fetched = 0
    page = 0
    while url:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        page_data = response.json()
        results = page_data.get("results", [])
        n_fetched += len(results)
        page += 1

        for entry in results:
            accession = entry.get("primaryAccession", "")
            ec_numbers = extract_ec_numbers(entry)
            if ec_numbers:
                rows.append({"uniprot_accession": accession, "ec_numbers": ";".join(ec_numbers)})

        print(f"  page {page}: fetched {n_fetched} entries so far, {len(rows)} with an EC number")

        link = response.headers.get("Link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None

    ec_df = pd.DataFrame(rows)
    RAW_EC_PATH.parent.mkdir(parents=True, exist_ok=True)
    ec_df.to_csv(RAW_EC_PATH, index=False)
    return ec_df


def run(force_fetch: bool = False):
    print("Step 1: fetching EC numbers for all reviewed human UniProt entries ...")
    ec_df = fetch_ec_numbers(force=force_fetch)
    print(f"  {len(ec_df)} accessions have at least one EC number\n")

    print("Step 2: restricting to our own vetted human proteome pull ...")
    proteome_df = pd.read_csv(FULL_PROTEOME_ANNOTATIONS)
    accession_to_ensp = dict(zip(proteome_df["UniProt_Accession"], proteome_df["ENSP_ID"]))
    accession_to_gene = dict(zip(proteome_df["UniProt_Accession"], proteome_df["Gene"]))

    merged = ec_df[ec_df["uniprot_accession"].isin(accession_to_ensp)].reset_index(drop=True)
    print(f"  {len(merged)} / {len(ec_df)} EC-annotated accessions are in our "
          f"{len(accession_to_ensp):,}-protein reviewed human UniProt pull\n")

    print("Step 3: resolving each protein's EC top-level class ...")
    rows = []
    n_dropped_ambiguous = 0
    n_dropped_unrecognized = 0
    for _, row in merged.iterrows():
        ec_numbers = row["ec_numbers"].split(";")
        top_levels = {ec.split(".")[0] for ec in ec_numbers}
        # Drop anything that isn't a plain "1".."7" leading digit (defensive --
        # not expected for reviewed entries, but cheap to guard against).
        if not top_levels <= set(EC_TOP_LEVEL_NAMES):
            n_dropped_unrecognized += 1
            continue
        if len(top_levels) > 1:
            # Multi-domain/multi-function enzyme spanning more than one EC
            # top-level class -- ambiguous for single-label classification,
            # dropped (same spirit as DeepLoc's dual-localization drop).
            n_dropped_ambiguous += 1
            continue

        accession = row["uniprot_accession"]
        top_level = next(iter(top_levels))
        rows.append({
            "uniprot_accession": accession,
            "ensp_id": accession_to_ensp[accession],
            "gene": accession_to_gene[accession],
            "ec_numbers": row["ec_numbers"],
            "true_class": EC_TOP_LEVEL_NAMES[top_level],
        })

    print(f"  {len(rows)} single-EC-class proteins kept "
          f"({n_dropped_ambiguous} dropped as spanning multiple EC top-level classes, "
          f"{n_dropped_unrecognized} dropped as unrecognized EC formatting)\n")

    print("Step 4: assigning seen/unseen relative to our model's own train split ...")
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    unseen_ensp_ids = set(splits["validation"]) | set(splits["test"])

    for r in rows:
        r["split"] = "unseen" if r["ensp_id"] in unseen_ensp_ids else "seen"

    result_df = pd.DataFrame(rows).sort_values(["true_class", "uniprot_accession"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"  seen: {(result_df['split'] == 'seen').sum()}, "
          f"unseen: {(result_df['split'] == 'unseen').sum()}\n")
    print("Class distribution:")
    print(result_df.groupby(["true_class", "split"]).size().unstack(fill_value=0))
    print(f"\nSaved {len(result_df)} human enzymes to {OUTPUT_PATH}")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-fetch EC numbers from UniProt even if a cached copy exists")
    args = parser.parse_args()
    run(force_fetch=args.force)

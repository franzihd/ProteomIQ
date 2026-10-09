"""
Fetches all GO annotations (BP, MF, CC) from UniProt for the GO-term retrieval eval.

Usage:
    uv run python src/evaluation/fetch_go_annotations_uncapped.py
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

FULL_PROTEOME_ANNOTATIONS = Path("data/full_proteome/protein_annotations.csv")
OUTPUT_PATH = Path("data/full_proteome/uniprot_go_annotations_uncapped.csv")

ASPECT_LETTERS = {"P": "bp", "F": "mf", "C": "cc"}


def extract_all_aspects(uniprot_entry):
    """Returns {"bp": [(go_id, label), ...], "mf": [...], "cc": [...]} --
    all direct GO cross-references on this entry, uncapped, deduplicated
    within each aspect, in the order UniProt lists them (no re-sorting)."""
    buckets = {"bp": [], "mf": [], "cc": []}
    seen = {"bp": set(), "mf": set(), "cc": set()}
    for xref in uniprot_entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        go_id = xref.get("id")
        label = None
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoTerm":
                label = prop["value"]
                break
        if not label or len(label) < 3 or label[1] != ":":
            continue
        aspect_letter, term_name = label[0], label[2:]
        aspect = ASPECT_LETTERS.get(aspect_letter)
        if aspect is None:
            continue
        if go_id not in seen[aspect]:
            seen[aspect].add(go_id)
            buckets[aspect].append((go_id, term_name))
    return buckets


def fetch_go_annotations(force: bool = False) -> pd.DataFrame:
    if OUTPUT_PATH.exists() and not force:
        print(f"  {OUTPUT_PATH} already cached, skipping fetch (use --force to re-fetch)")
        return pd.read_csv(OUTPUT_PATH)

    url = ("https://rest.uniprot.org/uniprotkb/search"
           "?query=organism_id:9606+AND+reviewed:true"
           "&fields=accession,go_p&format=json&size=500")

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
            buckets = extract_all_aspects(entry)
            if any(buckets.values()):
                rows.append({
                    "uniprot_accession": accession,
                    "go_bp_ids": ";".join(t[0] for t in buckets["bp"]),
                    "go_bp_labels": ";".join(t[1] for t in buckets["bp"]),
                    "go_mf_ids": ";".join(t[0] for t in buckets["mf"]),
                    "go_mf_labels": ";".join(t[1] for t in buckets["mf"]),
                    "go_cc_ids": ";".join(t[0] for t in buckets["cc"]),
                    "go_cc_labels": ";".join(t[1] for t in buckets["cc"]),
                })

        print(f"  page {page}: fetched {n_fetched} entries so far, {len(rows)} with any GO annotation")

        link = response.headers.get("Link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None

    go_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    go_df.to_csv(OUTPUT_PATH, index=False)
    return go_df


def run(force: bool = False):
    print("Fetching uncapped GO annotations (BP + MF + CC) for all reviewed human UniProt entries ...")
    go_df = fetch_go_annotations(force=force)
    print(f"  {len(go_df)} accessions have at least one GO annotation (any aspect)\n")

    print("Restricting to our own vetted human proteome pool ...")
    proteome_df = pd.read_csv(FULL_PROTEOME_ANNOTATIONS)
    pool_accessions = set(proteome_df["UniProt_Accession"])
    merged = go_df[go_df["uniprot_accession"].isin(pool_accessions)].reset_index(drop=True)
    print(f"  {len(merged)} / {len(go_df)} GO-annotated accessions are in our "
          f"{len(pool_accessions):,}-protein reviewed human UniProt pull\n")

    for aspect, col in [("BP", "go_bp_ids"), ("MF", "go_mf_ids"), ("CC", "go_cc_ids")]:
        has_aspect = (merged[col].fillna("") != "").sum()
        max_terms = merged[col].fillna("").apply(lambda s: len(s.split(";")) if s else 0).max()
        print(f"  {aspect}: {has_aspect} / {len(pool_accessions):,} pool proteins have >=1 direct "
              f"{aspect} annotation (max {max_terms} terms for one protein, uncapped)")

    print(f"\nSaved to {OUTPUT_PATH}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-fetch from UniProt even if a cached copy exists")
    args = parser.parse_args()
    run(force=args.force)

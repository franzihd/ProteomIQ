"""
Shared ground-truth loading for the GO-term retrieval eval.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SPLITS_PATH = Path("data/splits/protein_splits_full_proteome.json")
PROTEOME_ANNOTATIONS = Path("data/full_proteome/protein_annotations.csv")
GO_ANNOTATIONS_PATH = Path("data/full_proteome/uniprot_go_annotations_uncapped.csv")

BRANCHES = {
    "mf": ("go_mf_ids", "go_mf_labels"),
    "bp": ("go_bp_ids", "go_bp_labels"),
    "cc": ("go_cc_ids", "go_cc_labels"),
}


def load_pool_ensp_ids(split_key: str = "validation") -> set[str]:
    if split_key not in ("validation", "test"):
        raise ValueError(f"split_key must be 'validation' or 'test', got {split_key!r}")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    return set(splits[split_key])


def load_validation_ensp_ids() -> set[str]:
    """Kept as its own function (rather than just calling
    load_pool_ensp_ids("validation") at every call site) so
    build_go_retrieval_eligible_terms.py's Step 1 -- which must ALWAYS use
    validation, never test, to choose --min-positives -- can't accidentally
    be switched to test by a future edit that changes a shared default."""
    return load_pool_ensp_ids("validation")


def load_ensp_accession_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (ensp_to_accession, accession_to_ensp). protein_annotations.csv
    is one row per ENSP_ID (the full-proteome dedup already resolved the
    multi-isoform-per-accession problem), so this mapping is
    effectively 1:1 in each direction; on the rare case of a collision when
    inverting, the last row encountered wins (not chased further -- same
    negligible-edge-case tolerance already accepted elsewhere for this data,
    e.g. the 9-gene proteome-rebuild gap in 11b)."""
    df = pd.read_csv(PROTEOME_ANNOTATIONS)
    ensp_to_accession = dict(zip(df["ENSP_ID"], df["UniProt_Accession"]))
    accession_to_ensp = dict(zip(df["UniProt_Accession"], df["ENSP_ID"]))
    return ensp_to_accession, accession_to_ensp


def _parse_terms(cell) -> list[str]:
    if not isinstance(cell, str) or not cell:
        return []
    return cell.split(";")


def load_branch_ground_truth(branch: str, accessions_of_interest: set[str]):
    """Returns (positives, term_names):
      positives   -- {go_id: set(uniprot_accession)}, restricted to
                      accessions_of_interest, DIRECT annotations only.
      term_names  -- {go_id: uniprot_label} (the GO term name as UniProt's
                      own cross-reference lists it -- kept only as a
                      human-readable cross-check; the *official* GO name/
                      definition used as query text comes from go_obo_utils
                      instead, not from this label)."""
    id_col, label_col = BRANCHES[branch]
    go_df = pd.read_csv(GO_ANNOTATIONS_PATH)
    go_df = go_df[go_df["uniprot_accession"].isin(accessions_of_interest)].reset_index(drop=True)

    positives: dict[str, set[str]] = defaultdict(set)
    term_names: dict[str, str] = {}

    for _, row in go_df.iterrows():
        accession = row["uniprot_accession"]
        ids = _parse_terms(row[id_col])
        labels = _parse_terms(row[label_col])
        for go_id, label in zip(ids, labels):
            positives[go_id].add(accession)
            term_names.setdefault(go_id, label)

    return dict(positives), term_names

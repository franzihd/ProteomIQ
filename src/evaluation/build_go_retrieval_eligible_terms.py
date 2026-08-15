"""
Zero-shot text-to-protein RETRIEVAL evaluation -- Step 1: report GO-term
positive-count distributions per branch (MF/BP/CC), restricted to the
VALIDATION split only. Does NOT choose or apply a minimum-positive-count
threshold -- per the explicit methodological rule for this evaluation, a
threshold may only be chosen after seeing these distributions, never after
seeing retrieval performance. This script's job ends at reporting; the
threshold itself is a required (no default) CLI argument on the actual
retrieval-eval script (zero_shot_go_retrieval_eval.py), applied there, not
frozen here.

Inspired by ProtST's zero-shot text-to-protein retrieval idea -- NOT a
reproduction of ProtST's protocol. See CLAUDE.md for the full scientific
documentation of this evaluation.

What this script does:
  1. Loads the validation-split protein IDs (data/protein_splits_full_
     proteome.json, key "validation" -- test is never touched here).
  2. Loads ground-truth GO annotations via go_retrieval_utils
     (data/full_proteome/uniprot_go_annotations_uncapped.csv -- DIRECT
     UniProt cross-references, uncapped, not GO-DAG-propagated; see
     fetch_go_annotations_uncapped.py's docstring for the full
     direct-vs-propagated discussion).
  3. Loads official GO term names + definitions from data/go/go-basic.obo
     via go_obo_utils.load_go_term_info().
  4. For each of MF/BP/CC separately: counts, for every GO term that
     appears on >=1 validation-split protein, how many validation-split
     proteins carry that term (the "positive count" for that term within
     the validation candidate pool -- NOT the term's global positive count
     across the whole proteome).
  5. Reports how many terms would remain at >=1, >=5, >=10, >=20, >=50
     positives, per branch, and saves the full per-term positive-count
     table (not filtered) for reuse by the actual eval script.

Obsolete-term filter (added 2026-08-15, after a manual dataset sanity check
at min_positives=10 surfaced this): a GO term is only counted as "eligible"
at a given threshold if it (a) meets the positive-count threshold AND
(b) has a non-obsolete entry in go-basic.obo with a real name -- i.e. can
actually supply query text (name + definition). Two IDs in the >=10-
positive set were found to be officially obsolete despite still appearing
in UniProt's own GO cross-references: GO:0005615 ("extracellular space",
CC, 131 validation positives -- superseded by GO:0005576 "extracellular
region", which is separately present and unaffected) and GO:0051082
("unfolded protein binding", MF, 11 positives). go_obo_utils.
load_go_term_info() already drops obsolete terms, so these silently had no
name/definition to query with; they are now excluded from the eligible
count and table entirely, rather than left in with blank text (which would
have made "n_eligible_terms" overstate how many terms the retrieval script
can actually query).

Ground truth uses UniProt accessions; validation-split protein IDs are
ENSP IDs -- data/full_proteome/protein_annotations.csv links the two
(same join key used throughout this project).

Usage:
    uv run python src/evaluation/build_go_retrieval_eligible_terms.py
"""
from pathlib import Path

import pandas as pd

from go_obo_utils import load_go_term_info
from go_retrieval_utils import (
    BRANCHES,
    load_branch_ground_truth,
    load_ensp_accession_maps,
    load_validation_ensp_ids,
)

OUTPUT_DIR = Path("data/evaluation/go_retrieval")
THRESHOLD_CHECKPOINTS = [1, 5, 10, 20, 50]


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_ensp_ids = load_validation_ensp_ids()
    print(f"Validation split: {len(validation_ensp_ids)} proteins (candidate pool for retrieval)")

    ensp_to_accession, _ = load_ensp_accession_maps()
    validation_accessions = {
        ensp_to_accession[e] for e in validation_ensp_ids if e in ensp_to_accession
    }
    print(f"  -> {len(validation_accessions)} of those resolved to a UniProt accession\n")

    term_info = load_go_term_info()

    import json
    summary = {}

    for branch in BRANCHES:
        positives, term_names = load_branch_ground_truth(branch, validation_accessions)

        n_terms_total = len(positives)
        print(f"=== GO-{branch.upper()} ===")
        print(f"  {n_terms_total} distinct GO-{branch.upper()} terms appear on >=1 validation protein")

        # A term only has usable query text if it has a non-obsolete
        # go-basic.obo entry (see module docstring, "Obsolete-term filter").
        has_text = {go_id: bool(term_info.get(go_id, {}).get("name")) for go_id in positives}
        n_obsolete_or_unmapped = sum(1 for v in has_text.values() if not v)
        if n_obsolete_or_unmapped:
            dropped = [go_id for go_id, ok in has_text.items() if not ok]
            print(f"  {n_obsolete_or_unmapped} term(s) with >=1 positive have no usable name/definition "
                  f"(obsolete or unmapped in go-basic.obo) and are excluded from eligibility: {dropped}")

        checkpoint_counts = {}
        for thresh in THRESHOLD_CHECKPOINTS:
            n = sum(1 for go_id, accs in positives.items() if len(accs) >= thresh and has_text[go_id])
            checkpoint_counts[thresh] = n
            print(f"    terms with >= {thresh:>2} positives (and usable text): {n}")
        print()

        rows = []
        for go_id, accs in sorted(positives.items(), key=lambda kv: -len(kv[1])):
            info = term_info.get(go_id, {})
            rows.append({
                "go_id": go_id,
                "uniprot_label": term_names.get(go_id, ""),
                "official_name": info.get("name", ""),
                "official_def": info.get("def", ""),
                "n_validation_positives": len(accs),
                "has_usable_text": has_text[go_id],
            })
        table = pd.DataFrame(rows)
        out_path = OUTPUT_DIR / f"go_{branch}_validation_term_counts.csv"
        table.to_csv(out_path, index=False)
        print(f"  Saved full per-term table ({len(table)} terms) to {out_path}\n")

        summary[branch] = {
            "n_terms_total": n_terms_total,
            "n_obsolete_or_unmapped": n_obsolete_or_unmapped,
            "threshold_checkpoints": checkpoint_counts,
        }

    summary_path = OUTPUT_DIR / "go_retrieval_eligible_terms_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "validation_pool_size": len(validation_ensp_ids),
            "validation_pool_size_with_accession": len(validation_accessions),
            "per_branch": summary,
            "note": (
                "This report does NOT choose a minimum-positive-count threshold. "
                "The threshold is a required argument on zero_shot_go_retrieval_eval.py, "
                "chosen only after reviewing these distributions, never after reviewing "
                "retrieval performance."
            ),
        }, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    run()

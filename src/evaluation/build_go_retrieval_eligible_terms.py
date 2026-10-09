"""
GO-term retrieval, step 1: positive counts per GO term on the validation split
(used to choose --min-positives).

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

OUTPUT_DIR = Path("data/evaluation/go_retrieval/eligible_terms")
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
        out_path = OUTPUT_DIR / f"validation_term_counts_{branch}.csv"
        table.to_csv(out_path, index=False)
        print(f"  Saved full per-term table ({len(table)} terms) to {out_path}\n")

        summary[branch] = {
            "n_terms_total": n_terms_total,
            "n_obsolete_or_unmapped": n_obsolete_or_unmapped,
            "threshold_checkpoints": checkpoint_counts,
        }

    summary_path = OUTPUT_DIR / "eligible_terms_summary.json"
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

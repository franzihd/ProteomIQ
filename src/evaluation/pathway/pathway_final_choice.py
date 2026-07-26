"""
Applies objective, performance-blind inclusion criteria to all KEGG pathways
to select which ones are fair/valid to evaluate the model on. This replaces
hand-picking pathways (which introduces selection bias) with a pre-specified
filter, so the resulting list is defensible independent of how the model
happens to score on it.

Criteria (decided before looking at any AUROC/AP results):
1. Coverage filter (hard cutoff): >= 50% of a pathway's KEGG member genes
   must be measured (non-NaN), on average across test patients.
2. Power filter (hard cutoff): every test patient must contribute at least
   MIN_POSITIVES measured pathway genes (matches the MIN_POSITIVES=5 cutoff
   already used in pathway_auroc_evaluation.py / pathway_ap_evaluation.py),
   so no patient is silently dropped from the pathway's evaluation.
3. Annotation coverage (covariate, NOT a filter): fraction of a pathway's
   genes that have a real UniProt annotation (i.e. were part of the trained
   vocabulary, not the ~1,421 unannotated TrEMBL entries). Reported alongside
   AUROC so a reader can judge whether a low score reflects a model failure
   or a UniProt annotation gap.

Cohort-matched cancer-type pathways are flagged separately as a pre-specified
secondary analysis (not a selection criterion) -- motivated by "these are the
cancer types the model was trained on", decided in advance either way.

Usage:
    uv run python src/evaluation/pathway/pathway_final_choice.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

COVERAGE_PATH = Path("data/evaluation/pathway/pathway_choice.csv")
OUT_PATH = Path("data/evaluation/pathway/pathway_final_choice.csv")

COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]

# Hard cutoffs
MIN_COVERAGE_FRACTION = 0.5
MIN_POSITIVES = 5  # must match pathway_auroc_evaluation.py / pathway_ap_evaluation.py

# Pre-specified, decided independent of results: KEGG has no exact match for
# hnscc or ov, which is noted as a limitation rather than substituted.
COHORT_TO_KEGG_PATHWAY = {
    "luad": "Non-small cell lung cancer",
    "lscc": "Non-small cell lung cancer",
    "brca": "Breast cancer",
    "ccrcc": "Renal cell carcinoma",
    "gbm": "Glioma",
    "pdac": "Pancreatic cancer",
    "ucec": "Endometrial cancer",
    # "hnscc": no direct KEGG match
    # "ov": no direct KEGG match
}


def load_annotated_genes() -> set[str]:
    """Union, across all cohorts, of gene symbols with a real UniProt annotation."""
    annotated = set()
    for cohort in COHORTS:
        path = Path(f"data/{cohort}/protein_annotations.csv")
        if not path.exists():
            print(f"  Skipping {cohort} -- protein_annotations.csv missing")
            continue
        df = pd.read_csv(path)
        has_annotation = df["Annotation"].notna() & (df["Annotation"].str.strip() != "")
        annotated.update(df.loc[has_annotation, "Gene"].dropna().unique())
    return annotated


def main():
    print("Loading pathway coverage stats ...")
    df = pd.read_csv(COVERAGE_PATH)

    df["coverage_fraction"] = df["mean_measured"] / df["kegg_total_genes"]
    df["passes_coverage_filter"] = df["coverage_fraction"] >= MIN_COVERAGE_FRACTION
    df["passes_power_filter"] = df["min_measured"] >= MIN_POSITIVES
    df["passes_all_filters"] = df["passes_coverage_filter"] & df["passes_power_filter"]

    print("Loading KEGG pathway -> gene membership (for annotation coverage) ...")
    import gseapy
    gene_sets = gseapy.get_library("KEGG_2021_Human")

    print("Building union of annotated genes across all cohorts ...")
    annotated_genes = load_annotated_genes()
    print(f"  {len(annotated_genes)} unique annotated genes\n")

    annotation_coverage = {}
    for pathway_name, genes in gene_sets.items():
        genes = set(genes)
        if not genes:
            continue
        annotation_coverage[pathway_name] = len(genes & annotated_genes) / len(genes)

    df["annotation_coverage_fraction"] = df["pathway"].map(annotation_coverage)

    matched_cohorts_by_pathway: dict[str, list[str]] = {}
    for cohort, pathway_name in COHORT_TO_KEGG_PATHWAY.items():
        matched_cohorts_by_pathway.setdefault(pathway_name, []).append(cohort)

    df["is_cohort_matched"] = df["pathway"].isin(matched_cohorts_by_pathway)
    df["matched_cohorts"] = df["pathway"].map(
        lambda p: ",".join(matched_cohorts_by_pathway.get(p, []))
    )

    df = df.sort_values("coverage_fraction", ascending=False)
    df.to_csv(OUT_PATH, index=False)

    n_total = len(df)
    n_pass = int(df["passes_all_filters"].sum())
    print("=" * 72)
    print(f"Pathways evaluated        : {n_total}")
    print(f"Pass coverage filter (>= {MIN_COVERAGE_FRACTION:.0%})   : {int(df['passes_coverage_filter'].sum())}")
    print(f"Pass power filter (>= {MIN_POSITIVES} measured/patient) : {int(df['passes_power_filter'].sum())}")
    print(f"Pass BOTH filters         : {n_pass}")
    print(f"Cohort-matched pathways   : {int(df['is_cohort_matched'].sum())} "
          f"({', '.join(sorted(set(COHORT_TO_KEGG_PATHWAY.values())))})")
    print(f"Cohorts with no KEGG match: hnscc, ov (noted as limitation)")
    print("=" * 72)
    print(f"\nSaved full annotated table to {OUT_PATH}")

    print(f"\nPathways passing all filters (primary evaluation set, n={n_pass}):")
    print(
        df[df["passes_all_filters"]][
            ["pathway", "kegg_total_genes", "coverage_fraction",
             "annotation_coverage_fraction", "is_cohort_matched"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

"""
Builds the GO Molecular Function (GO-Slim-MF) zero-shot classification test
set used by zero_shot_go_mf_eval.py.

Same overall recipe as build_ec_eval_set.py / build_deeploc_eval_set.py, with
one structural difference: DeepLoc's 10 locations and EC's 7 top-level
numbers are FIXED, external class lists. Here there is no fixed class list --
raw GO Molecular Function terms are far too fine-grained and long-tailed to
use directly (thousands of distinct terms, most with a handful of members),
so this script (a) maps every raw GO-MF term to the coarser GO Slim
(molecular_function branch) vocabulary, then (b) derives the class list itself
from the resulting frequency distribution, keeping just enough top classes to
cover ~75-80% of annotated proteins.

Because the class list is DATA-DERIVED (unlike DeepLoc/EC), the frequency
distribution that decides it must be computed over the FULL protein pool
(every seen + unseen protein together) -- never restricted to only the
proteins that will later be scored as "unseen"/test. Doing otherwise would
tailor the benchmark's own class list to the held-out split, which is a
softer version of the same leakage CLAUDE.md section 11j flagged for
model/hyperparameter selection (there: picking a checkpoint by test
performance; here: picking a class list by test-subset composition). Seen/
unseen is assigned only AFTER the class list and eligible-protein pool are
fixed, exactly like DeepLoc/EC -- it's a label attached at the end, not an
input to class selection.

Pipeline:
  1. Fetch raw GO Molecular Function (aspect "F") terms per protein from
     UniProt -- not already in data/full_proteome/protein_annotations.csv
     (only GO_Biological_Process/GO_Cellular_Component were fetched there,
     see CLAUDE.md 11b), so this re-queries UniProt the same way
     build_ec_eval_set.py re-queries for EC numbers.
     Two distinct, explicitly-counted drop reasons apply before class
     selection: (a) a protein has zero GO-MF terms in UniProt at all --
     GO annotation coverage is a separate, incomplete curation effort from
     the FUNCTION-text annotation used elsewhere in this project -- and
     (b) a protein has a GO-MF term but that term's ancestor chain never
     crosses a GO-Slim category (see mapslim_primary()). Both are reported
     in the printed output and saved to CLASS_SELECTION_PATH.
  2. Map each protein's PRIMARY (first-listed) raw GO-MF term to GO Slim via
     goatools' mapslim() against go-basic.obo + goslim_generic.obo. A GO term
     can have multiple GO-Slim ancestors (multiple parent paths); mapslim's
     "direct ancestors" are the closest/most specific slim term per path --
     ties are broken by taking the alphabetically-first GO-Slim ID, a simple,
     deterministic, and clearly-stated simplification (see mapslim_primary()).
  3. Frequency distribution of GO-Slim-MF classes across ALL annotated
     proteins (not test-only, see above), printed with cumulative coverage.
  4. Select the smallest number of top classes reaching ~75-80% cumulative
     coverage; drop proteins whose slim class falls outside that set.
  4b. Most proteins have more than one GO-MF term, and different terms often
     map to DIFFERENT selected classes (e.g. many transcription factors hit
     both "DNA binding" and "transcription regulator activity" -- measured
     at ~33% of the eval set). Scoring a zero-shot prediction only against
     the single primary-term label would mark a biologically correct but
     non-primary prediction as wrong. So every protein's FULL set of valid
     selected classes (from ALL its raw terms, not just the primary one) is
     also computed and saved as "all_valid_classes" -- zero_shot_go_mf_eval.py
     uses it to report a lenient accuracy (any valid class counts) alongside
     the strict primary-label accuracy (needed anyway for macro-F1/confusion
     matrix, which require exactly one label per row).
  5. Assign seen/unseen relative to our model's own train split (same as
     DeepLoc/EC), then report, as a diagnostic (not a filter), the seen/
     unseen breakdown of the selected classes -- including how many unseen
     proteins survive filtering and a flag if that count is too low (<30-40)
     for the unseen-subset result to be statistically meaningful.

Usage:
    uv run python src/evaluation/build_go_mf_eval_set.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from goatools.obo_parser import GODag
from goatools.mapslim import mapslim

FULL_PROTEOME_ANNOTATIONS = Path("data/full_proteome/protein_annotations.csv")
SPLITS_FILE = Path("data/protein_splits_full_proteome.json")
RAW_GO_MF_PATH = Path("data/full_proteome/uniprot_go_mf_terms.csv")
GO_OBO_PATH = Path("data/go/go-basic.obo")
GOSLIM_OBO_PATH = Path("data/go/goslim_generic.obo")
OUTPUT_PATH = Path("data/evaluation/go_mf_test_set.csv")
CLASS_SELECTION_PATH = Path("data/evaluation/go_mf_class_selection.json")

GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
GOSLIM_OBO_URL = "https://current.geneontology.org/ontology/subsets/goslim_generic.obo"

TARGET_COVERAGE = 0.80  # stop adding classes once cumulative coverage reaches this
MIN_UNSEEN_FOR_SIGNIFICANCE = 30  # below this, flag the unseen-subset result as underpowered


def download_obo_files(force: bool = False):
    for path, url in [(GO_OBO_PATH, GO_OBO_URL), (GOSLIM_OBO_PATH, GOSLIM_OBO_URL)]:
        if path.exists() and not force:
            print(f"  {path} already cached, skipping download (use --force to re-download)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {url} ...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
        print(f"  Saved to {path}")


def extract_mf_terms(uniprot_entry):
    """Same GO-xref-walking pattern as fetch_full_human_proteome.py's
    extract_go_terms, but keeps BOTH the GO ID (needed for slim mapping) and
    the label (for readability), filtered to aspect "F" (Molecular Function),
    uncapped (unlike the 10-term cap used for embedding text -- here we only
    ever use the first term, so capping would be pointless)."""
    terms = []
    seen = set()
    for xref in uniprot_entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        go_id = xref.get("id")
        label = None
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoTerm":
                label = prop["value"]
                break
        if not label or not label.startswith("F:"):
            continue
        if go_id not in seen:
            seen.add(go_id)
            terms.append((go_id, label[2:]))
    return terms


def fetch_go_mf_terms(force: bool = False) -> pd.DataFrame:
    """Same paginated organism_id:9606 AND reviewed:true UniProt query as
    fetch_full_human_proteome.py / build_ec_eval_set.py, requesting the go_f
    field. Note: like go_p/go_c in fetch_full_human_proteome.py, requesting
    "go_f" still returns ALL GO cross-references per entry (all 3 aspects) --
    the aspect filtering happens client-side via the "F:" prefix, not
    server-side (verified directly against the live API)."""
    if RAW_GO_MF_PATH.exists() and not force:
        print(f"  {RAW_GO_MF_PATH} already cached, skipping fetch (use --force to re-fetch)")
        return pd.read_csv(RAW_GO_MF_PATH)

    url = ("https://rest.uniprot.org/uniprotkb/search"
           "?query=organism_id:9606+AND+reviewed:true"
           "&fields=accession,go_f&format=json&size=500")

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
            mf_terms = extract_mf_terms(entry)
            if mf_terms:
                rows.append({
                    "uniprot_accession": accession,
                    "go_mf_ids": ";".join(t[0] for t in mf_terms),
                    "go_mf_labels": ";".join(t[1] for t in mf_terms),
                })

        print(f"  page {page}: fetched {n_fetched} entries so far, {len(rows)} with a GO-MF term")

        link = response.headers.get("Link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None

    go_mf_df = pd.DataFrame(rows)
    RAW_GO_MF_PATH.parent.mkdir(parents=True, exist_ok=True)
    go_mf_df.to_csv(RAW_GO_MF_PATH, index=False)
    return go_mf_df


def mapslim_primary(go_id: str, go_dag: GODag, goslim_dag: GODag) -> str | None:
    """Maps one raw GO-MF ID to a single canonical GO-Slim-MF ID.
    mapslim() returns the DIRECT ancestors (closest slim term per parent
    path) -- a term can have more than one if it has multiple parent paths.
    Ties are broken by taking the alphabetically-first GO ID, a simple,
    deterministic choice (documented here rather than silently picked).
    Returns None if the term isn't in go_dag at all (rare -- obsolete/
    unrecognized ID) or has no slim ancestor."""
    if go_id not in go_dag:
        return None
    direct_ancestors, _all_ancestors = mapslim(go_id, go_dag, goslim_dag)
    if not direct_ancestors:
        return None
    return sorted(direct_ancestors)[0]


def run(force_fetch: bool = False, force_obo: bool = False, target_coverage: float = TARGET_COVERAGE):
    print("Step 1: fetching GO Molecular Function terms for all reviewed human UniProt entries ...")
    go_mf_df = fetch_go_mf_terms(force=force_fetch)
    print(f"  {len(go_mf_df)} accessions have at least one GO-MF term\n")

    print("Step 2: restricting to our own vetted human proteome pull ...")
    proteome_df = pd.read_csv(FULL_PROTEOME_ANNOTATIONS)
    accession_to_ensp = dict(zip(proteome_df["UniProt_Accession"], proteome_df["ENSP_ID"]))
    accession_to_gene = dict(zip(proteome_df["UniProt_Accession"], proteome_df["Gene"]))

    merged = go_mf_df[go_mf_df["uniprot_accession"].isin(accession_to_ensp)].reset_index(drop=True)
    print(f"  {len(merged)} / {len(go_mf_df)} GO-MF-annotated accessions are in our "
          f"{len(accession_to_ensp):,}-protein reviewed human UniProt pull")

    # Proteins in our own proteome pool that never got a row in go_mf_df at
    # all -- i.e. UniProt has zero GO Molecular Function terms for them
    # (distinct from n_no_slim_mapping below, which counts proteins that DO
    # have a term but whose term doesn't map to any GO-Slim category). GO
    # annotation coverage is a separate, incomplete curation effort from the
    # FUNCTION-text annotation used everywhere else in this project, so a
    # real gap here is expected, not a bug -- but it was previously silent
    # (no row is ever written for these proteins), so it's counted and
    # reported explicitly here.
    proteins_with_mf_term = set(go_mf_df["uniprot_accession"]) & set(accession_to_ensp)
    n_no_mf_term_at_all = len(accession_to_ensp) - len(proteins_with_mf_term)
    print(f"  {n_no_mf_term_at_all} / {len(accession_to_ensp):,} proteins in our pool have "
          f"ZERO GO-MF terms in UniProt at all (silently excluded, not a mapping failure)\n")

    print("Step 3: downloading GO OBO files and mapping each protein's primary "
          "GO-MF term to GO Slim ...")
    download_obo_files(force=force_obo)
    go_dag = GODag(str(GO_OBO_PATH))
    goslim_dag = GODag(str(GOSLIM_OBO_PATH))

    rows = []
    n_no_slim_mapping = 0
    for _, row in merged.iterrows():
        primary_go_id = row["go_mf_ids"].split(";")[0]
        primary_go_label = row["go_mf_labels"].split(";")[0]
        slim_id = mapslim_primary(primary_go_id, go_dag, goslim_dag)
        if slim_id is None:
            n_no_slim_mapping += 1
            continue
        accession = row["uniprot_accession"]
        rows.append({
            "uniprot_accession": accession,
            "ensp_id": accession_to_ensp[accession],
            "gene": accession_to_gene[accession],
            "primary_go_mf_id": primary_go_id,
            "primary_go_mf_label": primary_go_label,
            "go_slim_mf_id": slim_id,
            "go_slim_mf_name": goslim_dag[slim_id].name,
        })
    print(f"  {len(rows)} proteins mapped to a GO-Slim-MF class "
          f"({n_no_slim_mapping} dropped -- primary term had no GO-Slim ancestor)\n")

    all_df = pd.DataFrame(rows)

    print("Step 4: assigning seen/unseen relative to our model's own train split ...")
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    unseen_ensp_ids = set(splits["validation"]) | set(splits["test"])
    all_df["split"] = all_df["ensp_id"].apply(lambda e: "unseen" if e in unseen_ensp_ids else "seen")
    print(f"  {(all_df['split'] == 'seen').sum()} seen, {(all_df['split'] == 'unseen').sum()} unseen\n")

    print("Step 5: class frequency distribution over the FULL seen+unseen pool "
          "(NOT test-only -- see module docstring for why) ...")
    freq = all_df["go_slim_mf_name"].value_counts()
    cum_frac = (freq.cumsum() / freq.sum())
    freq_table = pd.DataFrame({
        "class": freq.index,
        "count": freq.values,
        "frac_of_all": (freq.values / freq.sum()),
        "cumulative_frac": cum_frac.values,
    })
    print(f"  {len(freq_table)} distinct GO-Slim-MF classes present, "
          f"{freq.sum()} total proteins\n")
    print(freq_table.to_string(index=False, formatters={
        "frac_of_all": "{:.2%}".format, "cumulative_frac": "{:.2%}".format,
    }))
    print()

    print(f"Step 6: selecting the smallest class set reaching >= {target_coverage:.0%} "
          f"cumulative coverage ...")
    n_selected = int((cum_frac < target_coverage).sum()) + 1
    n_selected = min(n_selected, len(freq_table))
    selected_classes = freq_table["class"].iloc[:n_selected].tolist()
    achieved_coverage = float(freq_table["cumulative_frac"].iloc[n_selected - 1])
    print(f"  Selected {n_selected} classes, achieving {achieved_coverage:.2%} cumulative "
          f"coverage of all {freq.sum()} GO-MF-annotated proteins\n")

    filtered_df = all_df[all_df["go_slim_mf_name"].isin(selected_classes)].reset_index(drop=True)
    filtered_df = filtered_df.rename(columns={"go_slim_mf_name": "true_class"})
    filtered_df = filtered_df.sort_values(["true_class", "uniprot_accession"]).reset_index(drop=True)

    print("Step 6b: computing each protein's FULL set of valid selected classes ...")
    print("  Most proteins have more than one GO-MF term, but 'true_class' above only used")
    print("  the first-listed one. A protein whose OTHER terms also map to a different")
    print("  selected class is genuinely, biologically described by both classes -- scoring")
    print("  a zero-shot prediction only against the single primary label would wrongly")
    print("  mark a correct-but-non-primary prediction as an error. This computes, for every")
    print("  protein, every selected class reachable from ANY of its raw GO-MF terms (not")
    print("  just the primary one), so zero_shot_go_mf_eval.py can additionally report a")
    print("  lenient accuracy (\"was the prediction any of the protein's valid classes\")")
    print("  alongside the strict primary-label accuracy.")
    accession_to_go_ids = dict(zip(merged["uniprot_accession"], merged["go_mf_ids"]))
    slim_name_cache: dict[str, frozenset] = {}

    def slim_names_for_go_id(go_id: str) -> frozenset:
        if go_id in slim_name_cache:
            return slim_name_cache[go_id]
        if go_id not in go_dag:
            slim_name_cache[go_id] = frozenset()
            return slim_name_cache[go_id]
        direct_ancestors, _ = mapslim(go_id, go_dag, goslim_dag)
        names = frozenset(goslim_dag[i].name for i in direct_ancestors)
        slim_name_cache[go_id] = names
        return names

    def all_valid_classes_for(accession: str) -> str:
        hit = set()
        for gid in accession_to_go_ids[accession].split(";"):
            hit |= slim_names_for_go_id(gid)
        return ";".join(sorted(hit & set(selected_classes)))

    filtered_df["all_valid_classes"] = filtered_df["uniprot_accession"].apply(all_valid_classes_for)
    n_multi_valid = int((filtered_df["all_valid_classes"].str.count(";") > 0).sum())
    print(f"  {n_multi_valid} / {len(filtered_df)} proteins "
          f"({n_multi_valid / len(filtered_df):.1%}) have MORE THAN ONE valid selected class\n")

    print("Step 7: seen/unseen breakdown of the SELECTED classes (diagnostic, not a filter) ...")
    breakdown = filtered_df.groupby(["true_class", "split"]).size().unstack(fill_value=0)
    print(breakdown)
    n_unseen_after_filter = int((filtered_df["split"] == "unseen").sum())
    unseen_total_before_filter = int((all_df["split"] == "unseen").sum())
    unseen_coverage_pct = n_unseen_after_filter / unseen_total_before_filter if unseen_total_before_filter else 0.0
    print(f"\n  Unseen proteins remaining after class filtering: {n_unseen_after_filter} "
          f"({unseen_coverage_pct:.2%} of the {unseen_total_before_filter} unseen proteins "
          f"that had a GO-Slim-MF mapping before filtering)")
    unseen_warning = None
    if n_unseen_after_filter < MIN_UNSEEN_FOR_SIGNIFICANCE:
        unseen_warning = (
            f"WARNING: only {n_unseen_after_filter} unseen proteins survive class filtering "
            f"(< {MIN_UNSEEN_FOR_SIGNIFICANCE}) -- the unseen-subset accuracy/F1 in "
            f"zero_shot_go_mf_eval.py may not be statistically meaningful."
        )
        print(f"\n  {unseen_warning}")
    print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(OUTPUT_PATH, index=False)

    selection_info = {
        "target_coverage": target_coverage,
        "n_classes_selected": n_selected,
        "achieved_coverage": achieved_coverage,
        "selected_classes": selected_classes,
        "full_frequency_table": freq_table.to_dict(orient="records"),
        "n_proteins_in_our_proteome_pool": len(accession_to_ensp),
        "n_proteins_with_zero_go_mf_terms": n_no_mf_term_at_all,
        "n_proteins_with_go_mf_term_but_no_slim_mapping": n_no_slim_mapping,
        "n_proteins_before_filter": int(len(all_df)),
        "n_proteins_after_filter": int(len(filtered_df)),
        "seen_after_filter": int((filtered_df["split"] == "seen").sum()),
        "n_proteins_with_multiple_valid_classes": n_multi_valid,
        "frac_proteins_with_multiple_valid_classes": n_multi_valid / len(filtered_df),
        "unseen_after_filter": n_unseen_after_filter,
        "unseen_coverage_pct_of_unseen_pool": unseen_coverage_pct,
        "unseen_warning": unseen_warning,
    }
    with open(CLASS_SELECTION_PATH, "w") as f:
        json.dump(selection_info, f, indent=2)

    print(f"Saved {len(filtered_df)} proteins ({n_selected} classes) to {OUTPUT_PATH}")
    print(f"Saved class selection details to {CLASS_SELECTION_PATH}")

    return filtered_df, selection_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="re-fetch GO-MF terms from UniProt even if a cached copy exists")
    parser.add_argument("--force-obo", action="store_true",
                        help="re-download the GO / GO-Slim OBO files even if cached copies exist")
    parser.add_argument("--target-coverage", type=float, default=TARGET_COVERAGE,
                        help="cumulative coverage fraction to reach when selecting classes (default 0.80)")
    args = parser.parse_args()
    run(force_fetch=args.force, force_obo=args.force_obo, target_coverage=args.target_coverage)

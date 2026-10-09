"""
Fuses each protein's masked UniProt annotation + GO Biological Process/
Cellular Component terms into ONE coherent natural-language paragraph via
the Claude API, instead of raw concatenation (embed_biobert.py's
--include-go-terms, see CLAUDE.md 10n: raw concatenation regressed pathway
AUROC/recall@K in both BP+CC and BP-only variants -- diagnosed as the added
term-list text acting as a self-identification shortcut rather than genuine
signal, plausibly because a semicolon-separated term list is a completely
different register from both the base annotation prose and every eval query
style tried so far).

Step 1 (this script, always run): builds ONE global deduped protein table
across all 9 cohorts -- not per-cohort like fetch_uniprot.py/embed_biobert.py
-- taking whichever cohort's row is encountered first for a given protein_id
(same convention as the existing cross-cohort dedup in dataset.py/10b).
Doing the dedup before generation, rather than after (as the rest of the
pipeline does), avoids ever generating two different fused paragraphs for
the same protein from two cohorts' slightly-diverged annotation text.

Step 2 (--sample N, default mode): generates fused text for a small random
sample only, for manual review, before committing to the full ~15,700-call
run. Fixed seed (42) for reproducibility. Always includes EGFR and ARF5 by
name (already-discussed examples elsewhere in this file/CLAUDE.md) alongside
the random sample, so before/after comparison is easy.

Full run: pass --full instead of --sample to process every protein and save
data/protein_fused_text.csv (protein_id -> fused_text). NOT run by default --
review the sample output first.

One fixed prompt template, applied identically to every protein, decided
before generating anything -- same no-hand-tuning discipline as
pathway_nl_queries.py.

Usage:
    uv run python src/embeddings/textual/fuse_protein_text_llm.py --sample 20
    uv run python src/embeddings/textual/fuse_protein_text_llm.py --full
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import random
import pandas as pd
import anthropic

from embed_biobert import strip_gene_name

COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]
OUT_PATH = Path("data/protein_fused_text.csv")
SEED = 42
MODEL = "claude-sonnet-5"

PROMPT_TEMPLATE = (
    "You are helping prepare training data for a protein-text embedding model. "
    "Below are factual annotations for a single human protein, extracted from "
    "UniProt and Gene Ontology (GO) databases. Combine them into ONE coherent "
    "paragraph (2-4 sentences) in natural, flowing prose describing the "
    "protein's function and cellular localization.\n\n"
    "Rules:\n"
    "- Use ONLY the facts given below. Do not add any outside knowledge, do "
    "not invent additional functions, processes, or locations not listed.\n"
    "- Do not mention the gene name or any protein name/symbol.\n"
    "- Do not use bullet points, semicolons as list separators, or labels "
    "like \"Biological process:\" -- write natural sentences.\n"
    "- If a fact category is \"(none provided)\", simply omit it from the "
    "paragraph -- do not say information is missing.\n\n"
    "Function (from UniProt):\n{annotation}\n\n"
    "Biological process terms (GO):\n{go_bp}\n\n"
    "Cellular component terms (GO):\n{go_cc}\n\n"
    "Respond with only the fused paragraph, nothing else."
)


def build_global_protein_table(cohorts):
    """One row per unique protein across all cohorts -- first-encountered
    cohort wins, matching the existing cross-cohort dedup convention."""
    seen = {}
    for cohort in cohorts:
        path = Path(f"data/cptac/{cohort}/protein_annotations.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.dropna(subset=["Sequence", "Annotation"])
        for _, row in df.iterrows():
            pid = row["ENSP_ID"]
            if pid in seen:
                continue
            seen[pid] = {
                "protein_id": pid,
                "gene": row["Gene"],
                "annotation": row["Annotation"],
                "go_bp": row.get("GO_Biological_Process"),
                "go_cc": row.get("GO_Cellular_Component"),
            }
    return pd.DataFrame(seen.values())


def fuse_one(client, gene, annotation, go_bp, go_cc):
    masked_annotation = strip_gene_name(gene, annotation)
    prompt = PROMPT_TEMPLATE.format(
        annotation=masked_annotation,
        go_bp=go_bp if isinstance(go_bp, str) and go_bp.strip() else "(none provided)",
        go_cc=go_cc if isinstance(go_cc, str) and go_cc.strip() else "(none provided)",
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int, help="generate a small random sample for review")
    group.add_argument("--full", action="store_true", help="generate for every protein, save to disk")
    args = parser.parse_args()

    print("Building global deduped protein table ...")
    table = build_global_protein_table(COHORTS)
    print(f"  {len(table):,} unique proteins across {len(COHORTS)} cohorts\n")

    client = anthropic.Anthropic()

    if args.sample:
        random.seed(SEED)
        highlight_genes = {"EGFR", "ARF5"}
        highlighted = table[table["gene"].isin(highlight_genes)]
        rest = table[~table["gene"].isin(highlight_genes)]
        sample_rows = pd.concat([
            highlighted,
            rest.sample(n=max(0, args.sample - len(highlighted)), random_state=SEED),
        ])

        print(f"Generating fused text for {len(sample_rows)} sample proteins ...\n")
        sample_results = []
        for _, row in sample_rows.iterrows():
            fused = fuse_one(client, row["gene"], row["annotation"], row["go_bp"], row["go_cc"])
            masked_annotation = strip_gene_name(row["gene"], row["annotation"])
            print("=" * 88)
            print(f"Gene: {row['gene']}  (protein_id={row['protein_id']})")
            print(f"-- ORIGINAL masked annotation --\n{masked_annotation}")
            print(f"-- GO_BP -- {row['go_bp']}")
            print(f"-- GO_CC -- {row['go_cc']}")
            print(f"-- FUSED --\n{fused}")
            sample_results.append({
                "protein_id": row["protein_id"],
                "gene": row["gene"],
                "masked_annotation": masked_annotation,
                "go_bp": row["go_bp"],
                "go_cc": row["go_cc"],
                "fused_text": fused,
            })
        print("=" * 88)

        sample_out_path = Path("data/protein_fused_text_sample.csv")
        pd.DataFrame(sample_results).to_csv(sample_out_path, index=False)
        print(f"\nSample saved to {sample_out_path} -- review before running --full "
              f"(~{len(table):,} API calls).")
    else:
        print(f"Generating fused text for all {len(table):,} proteins ...")
        fused_texts = []
        for i, row in table.iterrows():
            fused = fuse_one(client, row["gene"], row["annotation"], row["go_bp"], row["go_cc"])
            fused_texts.append(fused)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(table)} done")
        table["fused_text"] = fused_texts
        table[["protein_id", "gene", "fused_text"]].to_csv(OUT_PATH, index=False)
        print(f"\nSaved {len(table):,} fused texts to {OUT_PATH}")


if __name__ == "__main__":
    main()

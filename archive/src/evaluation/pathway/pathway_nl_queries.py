"""
Generates natural-language rewrites of KEGG pathway names, for the pathways
that passed the objective selection criteria in pathway_final_choice.py.

Purpose: test whether the low AUROC on cohort-matched cancer pathways (mean
0.473, see CLAUDE.md Timeline 8d) is a query-phrasing problem -- raw KEGG
names are database jargon -- or a genuine generalization limit. See CLAUDE.md
"Next steps (as of 2026-07-06)" -> "Decision: natural-language queries before
more training data".

Methodology note: one fixed prompt template is applied identically to every
pathway, generated before any AUROC results are computed. This mirrors the
objective-criteria approach used for pathway selection -- hand-tuning
individual query wordings after seeing scores would reintroduce the same
selection bias that pathway_final_choice.py was built to avoid.

Uses direct (non-batch) API calls, one per pathway. The Message Batches API
would be ~50% cheaper, but at this volume (245 short requests, well under $1
either way) the cost difference is negligible -- a stuck/slow batch queue
isn't worth trading for it.

Usage:
    uv run python src/evaluation/pathway/pathway_nl_queries.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import anthropic

SELECTION_PATH = Path("data/evaluation/pathway/pathway_final_choice.csv")
OUT_PATH = Path("data/evaluation/pathway/pathway_nl_queries.csv")

MODEL = "claude-opus-4-8"

PROMPT_TEMPLATE = (
    "Rewrite this KEGG biological pathway name as a short, natural-language "
    "description of its biological function -- the way a cancer researcher "
    "unfamiliar with KEGG's database naming would type it into a search box. "
    "One sentence. Do not include any specific gene or protein names. Do not "
    "include the word 'KEGG' or 'pathway' unless it reads naturally. "
    "Respond with only the rewritten query text, nothing else.\n\n"
    "Pathway name: {pathway_name}"
)


def main():
    # --- Load pathways and keep only the objectively-selected 245 ---
    df = pd.read_csv(SELECTION_PATH)
    df = df[df["passes_all_filters"]].reset_index(drop=True)
    print(f"Generating natural-language queries for {len(df)} pathways ...")

    client = anthropic.Anthropic()

    # --- Call the API once per pathway, synchronously, in order ---
    nl_queries = []
    for i, pathway_name in enumerate(df["pathway"]):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=200,
                output_config={"effort": "low"},
                messages=[{
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(pathway_name=pathway_name),
                }],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            nl_queries.append(text.strip())
        except anthropic.APIError as e:
            print(f"  [{i}] {pathway_name!r} failed: {e}")
            nl_queries.append(None)

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            print(f"  {i + 1}/{len(df)} done")

    # --- Save pathway -> natural-language query mapping to CSV ---
    df["nl_query"] = nl_queries
    out_df = df[["pathway", "nl_query"]].rename(columns={"pathway": "raw_name"})
    out_df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(out_df)} natural-language queries to {OUT_PATH}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

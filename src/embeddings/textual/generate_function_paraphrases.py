"""
Generates N alternative phrasings of each protein's masked UniProt Function
annotation via the Claude API -- paraphrase-augmentation, NOT field-augmentation
(contrast with fetch_family_location.py / embed_biobert.py's --include-family/
--include-location, which add NEW content and were found to hurt open-vocabulary
generalization via a shortcut mechanism, see CLAUDE.md's field-augmentation
ablation). Here the underlying facts stay exactly the same per protein --
only the wording/sentence structure varies -- aimed at two things at once:
(1) pair scarcity (13,515 unique training pairs after global dedup is small
by CLIP/ProteinCLIP standards -- more independent phrasings per protein is
more independent training signal without adding new content to memorize),
and (2) discouraging the same identity-shortcut mechanism diagnosed in the
field-augmentation ablation, since a protein no longer has one fixed,
memorizable string it's always matched to during training.

Each protein's masked Function annotation (gene name already stripped, same
convention as embed_biobert.py's strip_gene_name) is sent once to Claude
Sonnet 5, which returns N alternative phrasings in a single call (cheaper
and more internally-consistent than N separate calls). Each phrasing must:
- describe exactly the same facts as the original -- no new information,
  nothing omitted, nothing invented
- use different wording/sentence structure from the original and from each
  other
- never mention the gene name or any protein symbol (masking discipline
  preserved)

Step 1 (--sample N, default mode): generates paraphrases for a small random
sample only, for manual review + a real per-call cost estimate, before
committing to the full ~16,900-call run. Fixed seed (42) for reproducibility.
Always includes EGFR and ARF5 by name (existing spot-check proteins used
elsewhere in this project, e.g. fuse_protein_text_llm.py) alongside the
random sample.

Step 2 (--full): generates for every protein in
data/full_proteome/protein_annotations.csv with both Sequence and Annotation,
saves data/full_proteome/protein_function_paraphrases.csv
(protein_id, gene, masked_original, paraphrase_1 .. paraphrase_N).

One fixed prompt template, applied identically to every protein, no
per-protein hand-tuning -- same discipline as pathway_nl_queries.py /
fuse_protein_text_llm.py.

Usage:
    uv run python src/embeddings/textual/generate_function_paraphrases.py --sample 8
    uv run python src/embeddings/textual/generate_function_paraphrases.py --full
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import anthropic

from embed_biobert import strip_gene_name

DATA_PATH = Path("data/full_proteome/protein_annotations.csv")
FULL_OUT_PATH = Path("data/full_proteome/protein_function_paraphrases.csv")
SEED = 42
MODEL = "claude-sonnet-5"
N_PARAPHRASES = 4

# $/1M tokens (in, out) -- intro pricing where applicable (through 2026-08-31 for Sonnet 5)
PRICING = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

PROMPT_TEMPLATE = (
    "You are helping prepare training data for a protein-text embedding model. "
    "Below is a factual functional description of a human protein (the gene "
    "name has already been removed).\n\n"
    "Function:\n{annotation}\n\n"
    "Write {n} alternative phrasings of this description. Each phrasing must:\n"
    "- describe exactly the same facts as the original -- do not add any new "
    "information, do not omit any fact, do not invent anything not stated above\n"
    "- use different wording and sentence structure, both from the original "
    "and from each other (vary vocabulary, sentence order, active/passive voice)\n"
    "- never mention a gene name, protein name, or symbol\n"
    "- read as natural, flowing prose -- no bullet points, no field labels, and "
    "no internal numbered or lettered lists even when describing multiple items "
    "(e.g. write \"X, Y, and Z\" rather than \"1) X 2) Y 3) Z\")\n\n"
    "Return exactly {n} phrasings as a JSON array of strings."
)

# Structured output schema -- avoids the ambiguity of parsing a numbered-list
# text response, which broke on proteins whose own facts include a number
# (e.g. EGFR's "4 major downstream signaling cascades" got enumerated inline
# and was mistaken for new top-level list items by a regex-based parser).
PARAPHRASES_SCHEMA = {
    "type": "object",
    "properties": {
        "paraphrases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["paraphrases"],
    "additionalProperties": False,
}


def parse_paraphrases(text, n):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    paraphrases = data.get("paraphrases") if isinstance(data, dict) else None
    if not isinstance(paraphrases, list):
        return []
    return [p.strip() for p in paraphrases if isinstance(p, str) and p.strip()]


def paraphrase_one(client, gene, annotation, n=N_PARAPHRASES, model=MODEL):
    masked = strip_gene_name(gene, annotation)
    prompt = PROMPT_TEMPLATE.format(annotation=masked, n=n)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": PARAPHRASES_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    paraphrases = parse_paraphrases(text, n)
    return masked, paraphrases, response.usage


def load_protein_table():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Sequence", "Annotation"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int, help="generate for a small random sample, for review")
    group.add_argument("--full", action="store_true", help="generate for every protein, save to disk")
    parser.add_argument("--n-paraphrases", type=int, default=N_PARAPHRASES,
                        help=f"paraphrases per protein (default {N_PARAPHRASES})")
    parser.add_argument("--model", default=MODEL, choices=list(PRICING),
                        help=f"default {MODEL}")
    args = parser.parse_args()

    table = load_protein_table()
    print(f"{len(table):,} proteins with both Sequence and Annotation\n")

    client = anthropic.Anthropic()
    n = args.n_paraphrases
    model = args.model
    price_in, price_out = PRICING[model]

    if args.sample:
        random.seed(SEED)
        highlight_genes = {"EGFR", "ARF5"}
        highlighted = table[table["Gene"].isin(highlight_genes)]
        rest = table[~table["Gene"].isin(highlight_genes)]
        sample_rows = pd.concat([
            highlighted,
            rest.sample(n=max(0, args.sample - len(highlighted)), random_state=SEED),
        ])

        print(f"Generating {n} paraphrases each for {len(sample_rows)} sample proteins with {model} ...\n")
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for _, row in sample_rows.iterrows():
            masked, paraphrases, usage = paraphrase_one(client, row["Gene"], row["Annotation"], n, model)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            print("=" * 88)
            print(f"Gene: {row['Gene']}  (protein_id={row['ENSP_ID']})")
            print(f"-- ORIGINAL masked --\n{masked}")
            for i, p in enumerate(paraphrases, 1):
                print(f"-- PARAPHRASE {i} --\n{p}")
            if len(paraphrases) != n:
                print(f"[WARNING] expected {n} paraphrases, parsed {len(paraphrases)}")
            results.append({
                "protein_id": row["ENSP_ID"], "gene": row["Gene"], "masked_original": masked,
                **{f"paraphrase_{i+1}": p for i, p in enumerate(paraphrases)},
            })
        print("=" * 88)

        sample_out_path = Path(f"data/full_proteome/protein_function_paraphrases_sample_{model}.csv")
        pd.DataFrame(results).to_csv(sample_out_path, index=False)
        n_calls = len(sample_rows)
        avg_in = total_input_tokens / n_calls
        avg_out = total_output_tokens / n_calls
        cost_per_call = (avg_in / 1e6) * price_in + (avg_out / 1e6) * price_out
        print(f"\nSample saved to {sample_out_path}")
        print(f"Avg tokens/call: {avg_in:.0f} in / {avg_out:.0f} out")
        print(f"Estimated cost for --full (~{len(table):,} proteins) with {model}: "
              f"${cost_per_call * len(table):.2f} direct, "
              f"${cost_per_call * len(table) * 0.5:.2f} via Batch API")
    else:
        print(f"Generating {n} paraphrases each for {len(table):,} proteins with {model} ...")
        results = []
        for i, row in table.iterrows():
            masked, paraphrases, _ = paraphrase_one(client, row["Gene"], row["Annotation"], n, model)
            if len(paraphrases) != n:
                print(f"[WARNING] {row['ENSP_ID']}: expected {n}, got {len(paraphrases)}")
            results.append({
                "protein_id": row["ENSP_ID"], "gene": row["Gene"], "masked_original": masked,
                **{f"paraphrase_{j+1}": p for j, p in enumerate(paraphrases)},
            })
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(table)} done")
        pd.DataFrame(results).to_csv(FULL_OUT_PATH, index=False)
        print(f"\nSaved {len(table):,} rows to {FULL_OUT_PATH}")


if __name__ == "__main__":
    main()

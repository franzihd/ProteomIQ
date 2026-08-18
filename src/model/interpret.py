"""
Stage 1+2 (sample-restricted) + Stage 3: retrieve relevant proteins from
WITHIN one specific patient/uploaded sample, then ask Claude to interpret
the result in plain language.

Unlike retrieve_and_annotate.py's stage1_retrieve (which searches the WHOLE
human proteome, ~16,893 proteins, then separately labels each candidate's
abundance -- built for fair pathway-ground-truth spot-checking, where a
protein can't be excluded just because it wasn't detected in one sample, see
CLAUDE.md 10j), this script restricts the candidate pool itself to only the
proteins actually measured in the given sample. Every result is therefore
guaranteed to have a real abundance value -- no "not detected" placeholders --
at the cost of never surfacing a protein the instrument failed to detect, even
if it would otherwise be the best answer to the query. This trade-off was a
deliberate choice for the interactive researcher-facing tool (as opposed to
the evaluation scripts, which keep the wide-pool design unchanged).

Because retrieval is already restricted to the sample, Stage 1 (semantic
relevance) and Stage 2 (abundance annotation) collapse into one pass: every
candidate already has both a cosine similarity and an abundance percentile.
Ranking/selection of the top-K stays on cosine similarity alone (semantic
relevance to the query) -- abundance percentile is attached to each result as
context, not blended into the ranking. (An earlier draft of this script
multiplied cos_sim * percentile into one "activity_score" -- that's the exact
mechanism CLAUDE.md section 10j already tried and deliberately moved away
from, since it demotes truly relevant proteins just for being low-abundance.
Reintroducing it here was a mistake, corrected before this script was ever
run.)

Stage 3 then sends the top-cosine-similarity proteins, their percentiles, and
their (unmasked -- see build_annotation_lookup) UniProt annotations to Claude,
with one fixed prompt template asking for a plain-language, hypothesis-only
interpretation (never a diagnosis).

Usage:
    uv run python src/model/interpret.py \\
        --query "receptor tyrosine kinase signaling" \\
        --patients 11LU013 --cohort luad

    uv run python src/model/interpret.py \\
        --query "receptor tyrosine kinase signaling" \\
        --uploaded-csv path/to/sample.csv
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pandas as pd
import anthropic

from src.model.retrieve_and_annotate import (
    build_gene_lookup,
    load_esm2_lookup,
    load_model,
    encode_query_vector,
)
from src.preprocessing.normalize_uploaded_sample import normalize_uploaded_sample

MODEL = "claude-sonnet-5"
DEFAULT_TOP_K = 20

PROMPT_TEMPLATE = (
    "You are helping a researcher explore a proteomics sample. Below is a list "
    "of proteins retrieved from this specific patient's own measured sample as "
    "semantically relevant to their query, together with each protein's actual "
    "measured abundance percentile (0-100, higher = more highly expressed "
    "relative to everything else measured in this sample) and its UniProt "
    "functional annotation.\n\n"
    "Researcher's query: \"{query}\"\n\n"
    "Retrieved proteins (sorted by relevance to the query):\n"
    "{protein_list}\n\n"
    "Write a short (2-4 sentence) natural-language interpretation of what "
    "this pattern might suggest biologically. Rules:\n"
    "- Use ONLY the facts given above. Do not invent additional proteins, "
    "functions, or clinical facts not listed.\n"
    "- Refer to proteins by their gene name.\n"
    "- Frame this as hypothesis generation for further investigation, not a "
    "clinical diagnosis -- do not claim certainty about disease state or "
    "make treatment recommendations.\n"
    "- If the retrieved proteins don't obviously relate to the query, say so "
    "honestly rather than forcing a connection.\n\n"
    "Respond with only the interpretation, nothing else."
)


def build_annotation_lookup():
    """ENSP_ID -> unmasked UniProt annotation (gene name included). Unlike the
    training-time text (which masks the gene name to avoid the name-shortcut,
    CLAUDE.md 10m), Stage 3's output is read by a human, not used as a
    contrastive-training target -- so gene identity should stay in the text."""
    df = pd.read_csv("data/full_proteome/protein_annotations.csv",
                      usecols=["ENSP_ID", "Annotation"])
    return dict(zip(df["ENSP_ID"], df["Annotation"]))


def get_patient_measured_proteins(patient_id: str, cohort: str) -> dict[str, float]:
    """ENSP_ID -> abundance percentile for every protein actually measured
    (non-NaN) in one real CPTAC patient."""
    abundance_df = pd.read_csv(f"data/cptac/{cohort}/rank_normalized_proteomics.csv",
                                header=[0, 1], index_col=0)
    if patient_id not in abundance_df.index:
        raise ValueError(f"Patient {patient_id} not found in data/cptac/{cohort}/rank_normalized_proteomics.csv")
    patient_row = abundance_df.loc[patient_id]
    return {ensp_id: float(val) for (_, ensp_id), val in patient_row.items() if not pd.isna(val)}


def get_uploaded_measured_proteins(uploaded_csv: str, gene_col: str, abundance_col: str) -> dict[str, float]:
    """Same job as get_patient_measured_proteins, but for a researcher's own
    freshly uploaded sample. normalize_uploaded_sample.py (CLAUDE.md 11q)
    handles rank-normalization and gene -> ENSP_ID matching."""
    result_df, unmatched, n_ambiguous = normalize_uploaded_sample(uploaded_csv, gene_col, abundance_col)
    print(f"  Uploaded sample: {len(result_df)} proteins usable "
          f"({n_ambiguous} dropped as ambiguous, {len(unmatched)} not in our reference set)")
    return dict(zip(result_df["ENSP_ID"], result_df["percentile"]))


def retrieve_within_sample(query: str, measured_proteins: dict[str, float], esm2_lookup,
                            protein_head, text_head, tokenizer, biobert_model, device):
    """Rank ONLY the proteins actually measured in this sample, by cosine
    similarity to the query (semantic relevance) alone -- abundance
    percentile is attached per candidate as context, not used to rank or
    filter (see this file's module docstring for why the earlier
    activity_score blend was rejected). Returns a list of
    (protein_id, cos_sim, percentile), sorted by cos_sim descending."""
    protein_ids = [pid for pid in measured_proteins if pid in esm2_lookup]
    esm2_matrix = torch.stack([esm2_lookup[pid] for pid in protein_ids]).to(device)

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix)
        query_vec = encode_query_vector(query, tokenizer, biobert_model, text_head, device)
        cos_sims = (protein_vecs @ query_vec.T).squeeze(-1)

    scored = [
        (pid, cos_sim, measured_proteins[pid])
        for pid, cos_sim in zip(protein_ids, cos_sims.tolist())
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def build_prompt(query, selected, gene_lookup, annotation_lookup):
    lines = []
    for protein_id, cos_sim, percentile in selected:
        gene = gene_lookup.get(protein_id, protein_id)
        annotation = annotation_lookup.get(protein_id, "(no annotation available)")
        lines.append(
            f"- {gene}: {percentile * 100:.0f}th percentile abundance, "
            f"relevance score {cos_sim:.3f}. Function: {annotation}"
        )
    protein_list = "\n".join(lines)
    return PROMPT_TEMPLATE.format(query=query, protein_list=protein_list)


def stage3_interpret(client, query, scored_candidates, gene_lookup, annotation_lookup, top_k):
    selected = scored_candidates[:top_k]
    if not selected:
        return "No proteins in this sample matched anything usable for interpretation."

    prompt = build_prompt(query, selected, gene_lookup, annotation_lookup)
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--patients", nargs="+", default=[],
                         help="existing CPTAC patient IDs, looked up in --cohort's abundance matrix")
    parser.add_argument("--cohort", default="luad")
    parser.add_argument("--uploaded-csv", default=None,
                         help="path to a researcher's own uploaded sample (gene name + raw abundance)")
    parser.add_argument("--gene-col", default="gene", help="only used with --uploaded-csv")
    parser.add_argument("--abundance-col", default="abundance", help="only used with --uploaded-csv")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                         help="how many top-cosine-similarity proteins to display and send to the LLM")
    args = parser.parse_args()

    if not args.patients and not args.uploaded_csv:
        parser.error("provide at least one of --patients or --uploaded-csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_lookup = build_gene_lookup()
    annotation_lookup = build_annotation_lookup()
    esm2_lookup = load_esm2_lookup()
    protein_head, text_head, tokenizer, biobert_model = load_model(device)
    client = anthropic.Anthropic()

    print(f"Query: \"{args.query}\"")

    samples = {}
    for patient_id in args.patients:
        samples[patient_id] = get_patient_measured_proteins(patient_id, args.cohort)
    if args.uploaded_csv:
        samples["uploaded"] = get_uploaded_measured_proteins(args.uploaded_csv, args.gene_col, args.abundance_col)

    for label, measured in samples.items():
        print(f"\n{'=' * 88}\nSample: {label}  ({len(measured)} proteins measured)\n{'=' * 88}")

        scored = retrieve_within_sample(
            args.query, measured, esm2_lookup, protein_head, text_head, tokenizer, biobert_model, device
        )

        print(f"\nTop {args.top_k} by cosine similarity (relevance to query):")
        for pid, cos_sim, pct in scored[:args.top_k]:
            gene = gene_lookup.get(pid, pid)
            print(f"  {gene:<15} cos_sim={cos_sim:6.3f}  percentile={pct * 100:5.1f}")

        text = stage3_interpret(client, args.query, scored, gene_lookup, annotation_lookup, top_k=args.top_k)
        print("\nInterpretation:\n" + text)


if __name__ == "__main__":
    main()

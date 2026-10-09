"""
GO-term retrieval, step 2: text-to-protein retrieval per GO branch (mAP, P@K, R@K).

Usage:
    uv run python src/evaluation/zero_shot_go_retrieval_eval.py \\
        --checkpoint data/runs/full_proteome/multiview_experiment/multiview_best.pt \\
        --branch all --min-positives 10 --k-values 10,50,100
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

from go_obo_utils import load_go_term_info
from go_retrieval_utils import (
    BRANCHES,
    load_branch_ground_truth,
    load_ensp_accession_maps,
    load_pool_ensp_ids,
)

BIOBERT_PATH = "dmis-lab/biobert-base-cased-v1.2"
OUTPUT_DIR = Path("data/evaluation/go_retrieval/results")

QUERY_TYPES = ("name", "definition")


def load_context(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5",
                  split_key: str = "validation") -> dict:
    """Loads everything that does NOT depend on --branch/--min-positives:
    checkpoint, projection heads, BioBERT, and -- since the protein side of
    this eval never depends on which GO term is being queried -- the
    protein tower's embeddings for the full resolved pool, computed once
    here. Mirrors the load_context()/run_condition() split already used in
    the other zero-shot eval scripts (avoids reloading the checkpoint/
    BioBERT per branch).

    split_key: "validation" (default) for repeat checkpoint comparisons;
    "test" is reserved -- spend it exactly once, on the single final chosen
    checkpoint (see module/go_retrieval_utils.py docstrings)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}\n")

    pool_ensp_ids = load_pool_ensp_ids(split_key)
    print(f"{split_key.capitalize()} split (candidate pool): {len(pool_ensp_ids)} proteins")

    print("Loading checkpoint and building protein/text towers ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_dim = checkpoint.get("hidden_dim", 512)
    output_dim = checkpoint.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, revision="67c9c25b46986521ca33df05d8540da1210b3256")
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, revision="67c9c25b46986521ca33df05d8540da1210b3256").to(device)
    biobert_model.eval()

    print(f"Embedding the resolved {split_key} pool with the protein tower ...")
    dataset = ProteomicsDataset(
        cohorts=[], use_abundance=False,
        biobert_filename=biobert_filename,
        restrict_to_protein_ids=pool_ensp_ids,
    )
    ensp_ids = dataset.pairs
    missing = pool_ensp_ids - set(ensp_ids)
    if missing:
        print(f"  WARNING: {len(missing)}/{len(pool_ensp_ids)} {split_key} proteins have no "
              f"'{biobert_filename}' embedding and are dropped from the candidate pool this run")
    print(f"  Resolved candidate pool: {len(ensp_ids)} proteins\n")

    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in ensp_ids])
    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))  # (N, output_dim), L2-normalized

    ensp_to_accession, accession_to_ensp = load_ensp_accession_maps()
    resolved_accessions = {ensp_to_accession[e] for e in ensp_ids if e in ensp_to_accession}
    ensp_idx = {pid: i for i, pid in enumerate(ensp_ids)}

    print("Loading official GO term names/definitions from go-basic.obo ...")
    term_info = load_go_term_info()

    return {
        "device": device,
        "checkpoint_path": checkpoint_path,
        "biobert_filename": biobert_filename,
        "split_key": split_key,
        "ensp_ids": ensp_ids,
        "ensp_idx": ensp_idx,
        "protein_vecs": protein_vecs,
        "resolved_accessions": resolved_accessions,
        "accession_to_ensp": accession_to_ensp,
        "term_info": term_info,
        "tokenizer": tokenizer,
        "biobert_model": biobert_model,
        "text_head": text_head,
    }


def compute_ranking_metrics(sims: np.ndarray, positive_mask: np.ndarray, k_values: list[int]) -> dict:
    """sims, positive_mask: both shape (N,), aligned by candidate index.
    Returns AP + P@K/R@K for every k in k_values."""
    n_pos = int(positive_mask.sum())
    order = np.argsort(-sims)
    ranked_positive = positive_mask[order]

    cum_hits = np.cumsum(ranked_positive)
    ranks = np.arange(1, len(sims) + 1)
    hit_positions = ranked_positive.astype(bool)
    ap = float((cum_hits[hit_positions] / ranks[hit_positions]).mean()) if n_pos > 0 else None

    result = {"ap": ap, "n_positives": n_pos}
    for k in k_values:
        kk = min(k, len(sims))
        hits_at_k = int(ranked_positive[:kk].sum())
        result[f"p_at_{k}"] = hits_at_k / kk
        result[f"r_at_{k}"] = hits_at_k / n_pos if n_pos > 0 else None
        # Normalized Recall@K -- see module docstring. Ceiling is
        # min(K, n_pos): a term with fewer positives than K can in
        # principle reach 100% raw recall already, so the two coincide
        # there; only matters when n_pos > K.
        ceiling = min(kk, n_pos)
        result[f"r_at_{k}_normalized"] = hits_at_k / ceiling if ceiling > 0 else None
    return result


def random_baseline_metrics(n_pos: int, n_total: int, k_values: list[int]) -> dict:
    """Analytical expectation under a uniformly random ranking -- see module
    docstring for the derivation. E[AP] = E[P@K] = n_pos/N; E[R@K] = K/N;
    E[normalized R@K] = E[hits@K] / min(K, n_pos)."""
    prevalence = n_pos / n_total
    result = {"ap": prevalence}
    for k in k_values:
        kk = min(k, n_total)
        result[f"p_at_{k}"] = prevalence
        result[f"r_at_{k}"] = kk / n_total
        ceiling = min(kk, n_pos)
        expected_hits = kk * prevalence
        result[f"r_at_{k}_normalized"] = expected_hits / ceiling if ceiling > 0 else None
    return result


def run_branch(ctx: dict, branch: str, min_positives: int, k_values: list[int]) -> dict:
    print("=" * 88)
    print(f"GO-{branch.upper()} zero-shot text-to-protein retrieval "
          f"(min_positives={min_positives}, candidate pool={len(ctx['ensp_ids'])})")
    print("=" * 88)

    positives, uniprot_labels = load_branch_ground_truth(branch, ctx["resolved_accessions"])
    term_info = ctx["term_info"]
    # Obsolete-term filter (build_go_retrieval_eligible_terms.py,
    # 2026-08-15 dataset sanity check): a term is only eligible if it also
    # has a non-obsolete go-basic.obo entry with a real name, i.e. can
    # actually supply query text. Two IDs (GO:0005615, GO:0051082) were
    # found this way and excluded -- see that script's docstring.
    eligible = {
        go_id: accs for go_id, accs in positives.items()
        if len(accs) >= min_positives and term_info.get(go_id, {}).get("name")
    }
    n_dropped_obsolete = sum(
        1 for go_id, accs in positives.items()
        if len(accs) >= min_positives and not term_info.get(go_id, {}).get("name")
    )
    print(f"{len(positives)} GO-{branch.upper()} terms appear on >=1 candidate-pool protein; "
          f"{len(eligible)} pass the >= {min_positives}-positive threshold with usable text "
          f"({n_dropped_obsolete} more met the threshold but were obsolete/unmapped and excluded)\n")

    n_total = len(ctx["ensp_ids"])
    accession_to_ensp = ctx["accession_to_ensp"]
    ensp_idx = ctx["ensp_idx"]
    term_info = ctx["term_info"]

    per_term_rows = []
    per_query_type_terms = {qt: [] for qt in QUERY_TYPES}
    skipped_no_text = {qt: 0 for qt in QUERY_TYPES}

    for i, (go_id, accs) in enumerate(sorted(eligible.items())):
        if (i + 1) % 100 == 0 or (i + 1) == len(eligible):
            print(f"  ... {i + 1}/{len(eligible)} terms scored")

        positive_ensp = {accession_to_ensp[a] for a in accs if a in accession_to_ensp}
        positive_idx = [ensp_idx[e] for e in positive_ensp if e in ensp_idx]
        if not positive_idx:
            continue
        positive_mask = np.zeros(n_total, dtype=bool)
        positive_mask[positive_idx] = True

        info = term_info.get(go_id, {})
        query_texts = {"name": info.get("name", ""), "definition": info.get("def", "")}

        row = {
            "go_id": go_id,
            "official_name": info.get("name", ""),
            "n_positives": len(positive_idx),
        }
        for qt in QUERY_TYPES:
            text = query_texts[qt]
            if not text:
                skipped_no_text[qt] += 1
                row[f"{qt}_ap"] = None
                continue
            with torch.no_grad():
                query_vec = encode_query(text, ctx["tokenizer"], ctx["biobert_model"],
                                          ctx["text_head"], ctx["device"])
            sims = (ctx["protein_vecs"] @ query_vec.T).squeeze(-1).cpu().numpy()
            m = compute_ranking_metrics(sims, positive_mask, k_values)
            row[f"{qt}_ap"] = m["ap"]
            for k in k_values:
                row[f"{qt}_p_at_{k}"] = m[f"p_at_{k}"]
                row[f"{qt}_r_at_{k}"] = m[f"r_at_{k}"]
                row[f"{qt}_r_at_{k}_normalized"] = m[f"r_at_{k}_normalized"]
            per_query_type_terms[qt].append({"go_id": go_id, "n_positives": len(positive_idx), **m})

        per_term_rows.append(row)

    results_df = pd.DataFrame(per_term_rows)

    summary = {
        "branch": branch,
        "min_positives": min_positives,
        "k_values": k_values,
        "candidate_pool_size": n_total,
        "n_terms_with_any_positive": len(positives),
        "n_eligible_terms": len(eligible),
        "n_eligible_terms_with_resolved_positives": len(results_df),
        "checkpoint": str(ctx["checkpoint_path"]),
        "biobert_filename": ctx["biobert_filename"],
        "query_types": {},
    }

    print()
    for qt in QUERY_TYPES:
        terms = per_query_type_terms[qt]
        n_skipped = skipped_no_text[qt]
        aps = [t["ap"] for t in terms if t["ap"] is not None]
        mean_map = float(np.mean(aps)) if aps else None

        mean_p_at_k = {k: float(np.mean([t[f"p_at_{k}"] for t in terms])) for k in k_values}
        mean_r_at_k = {
            k: float(np.mean([t[f"r_at_{k}"] for t in terms if t[f"r_at_{k}"] is not None]))
            for k in k_values
        }
        mean_r_at_k_normalized = {
            k: float(np.mean([t[f"r_at_{k}_normalized"] for t in terms if t[f"r_at_{k}_normalized"] is not None]))
            for k in k_values
        }

        baselines = [random_baseline_metrics(t["n_positives"], n_total, k_values) for t in terms]
        baseline_map = float(np.mean([b["ap"] for b in baselines])) if baselines else None
        baseline_p_at_k = {k: float(np.mean([b[f"p_at_{k}"] for b in baselines])) for k in k_values} if baselines else {}
        baseline_r_at_k = {k: float(np.mean([b[f"r_at_{k}"] for b in baselines])) for k in k_values} if baselines else {}
        baseline_r_at_k_normalized = (
            {k: float(np.mean([b[f"r_at_{k}_normalized"] for b in baselines])) for k in k_values}
            if baselines else {}
        )

        summary["query_types"][qt] = {
            "n_terms_scored": len(terms),
            "n_terms_skipped_no_text": n_skipped,
            "mAP": mean_map,
            "mean_P_at_K": mean_p_at_k,
            "mean_R_at_K": mean_r_at_k,
            "mean_R_at_K_normalized": mean_r_at_k_normalized,
            "random_baseline_mAP": baseline_map,
            "random_baseline_P_at_K": baseline_p_at_k,
            "random_baseline_R_at_K": baseline_r_at_k,
            "random_baseline_R_at_K_normalized": baseline_r_at_k_normalized,
        }

        print(f"[{branch.upper()} / {qt}]  n_terms={len(terms)}  (skipped, no {qt} text: {n_skipped})")
        print(f"  mAP:     {mean_map:.4f}   (random baseline: {baseline_map:.4f})" if mean_map is not None
              else "  mAP:     n/a")
        for k in k_values:
            print(f"  P@{k:<3}:  {mean_p_at_k[k]:.4f}   (random: {baseline_p_at_k[k]:.4f})   "
                  f"R@{k:<3}: {mean_r_at_k[k]:.4f}   (random: {baseline_r_at_k[k]:.4f})   "
                  f"R@{k:<3} (normalized): {mean_r_at_k_normalized[k]:.4f}   "
                  f"(random: {baseline_r_at_k_normalized[k]:.4f})")
        print()

    checkpoint_stem = ctx["checkpoint_path"].stem
    # "validation" keeps the existing, un-suffixed filenames used by every
    # comparison run so far; "test" gets its own _test-suffixed filename so
    # it can never silently overwrite a validation result (or vice versa).
    split_suffix = "" if ctx["split_key"] == "validation" else f"_{ctx['split_key']}"
    predictions_path = OUTPUT_DIR / f"predictions_{branch}_{checkpoint_stem}_min{min_positives}{split_suffix}.csv"
    summary_path = OUTPUT_DIR / f"summary_{branch}_{checkpoint_stem}_min{min_positives}{split_suffix}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Per-term results saved to {predictions_path}")
    print(f"Summary saved to {summary_path}\n")

    return summary


def run(checkpoint_path: Path, branch: str, min_positives: int, k_values: list[int],
        biobert_filename: str = "biobert_embeddings_masked.h5",
        split_key: str = "validation") -> dict:
    ctx = load_context(checkpoint_path, biobert_filename, split_key)
    branches = list(BRANCHES.keys()) if branch == "all" else [branch]
    return {b: run_branch(ctx, b, min_positives, k_values) for b in branches}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/full_proteome/final_v2_dims_masked_best.pt")
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5",
                        help="only needs to exist for data/full_proteome -- GO-term queries are "
                             "encoded live, same as the other zero-shot eval scripts")
    parser.add_argument("--branch", required=True, choices=["mf", "bp", "cc", "all"],
                        help="which GO aspect to evaluate. 'all' runs mf+bp+cc in one process, "
                             "reusing the once-computed protein embeddings (branch-independent).")
    parser.add_argument("--min-positives", type=int, required=True,
                        help="minimum number of validation-pool positives a GO term must have to "
                             "be included. REQUIRED, no default -- see "
                             "build_go_retrieval_eligible_terms.py for the distribution report "
                             "this choice must be based on. Must be chosen from the reported data "
                             "distributions, never from retrieval performance. Always chosen from "
                             "the VALIDATION distribution (build_go_retrieval_eligible_terms.py), "
                             "even when --split-key test -- the threshold itself must never be "
                             "re-picked using test data.")
    parser.add_argument("--k-values", default="10,50,100",
                        help="comma-separated list of K values for Precision@K/Recall@K")
    parser.add_argument("--split-key", choices=["validation", "test"], default="validation",
                        help="candidate pool to evaluate against. 'validation' (default) for "
                             "repeat checkpoint comparisons; 'test' is reserved -- spend it "
                             "exactly once, on the single final chosen checkpoint.")
    args = parser.parse_args()
    k_values = [int(k) for k in args.k_values.split(",")]
    run(Path(args.checkpoint), args.branch, args.min_positives, k_values,
        biobert_filename=args.biobert_filename, split_key=args.split_key)

"""
Protein-level pathway AUROC using natural-language pathway query rewrites,
instead of raw KEGG pathway names, against the no-abundance ablation
checkpoints.

Motivation: masked+GO's pathway AUROC regressed vs. masked-only despite
improving self-ID R@K (see CLAUDE.md). One candidate explanation is a
train/eval format mismatch -- text_head is trained only on long, GO-term-
augmented protein annotation text, then queried at eval with a bare 1-4 word
raw pathway name (e.g. "Ribosome"), a much shorter and differently-structured
input than anything seen in training. A natural-language query (a full
sentence or short phrase, closer in register to the training text) tests
whether narrowing that format gap changes the picture for masked-only vs.
masked+GO.

Two independently-motivated NL rewrite styles are available, both reused
from the original 8e ablation (see CLAUDE.md) via git history (they were
generated once, before any AUROC results existed, and never hand-tuned):
  - "explanatory" (pathway_nl_queries.csv): a full sentence describing the
    pathway's biological function.
  - "compact" (pathway_nl_queries_compact.csv): a short 3-8 word search
    phrase, no explanation -- closer to how a researcher actually types a
    query into a search box.

Ground truth (KEGG gene set membership) is unchanged -- only the query TEXT
fed to encode_query differs. This otherwise exactly mirrors
pathway_auroc_protein_level.py (same candidate pool, same checkpoint
loading, same AUROC computation).

Usage:
    uv run python src/evaluation/pathway/pathway_nl_auroc_protein_level.py \\
        --checkpoint data/runs/.../checkpoint_no_abundance_masked_e2000_best.pt \\
        --style explanatory
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import gseapy
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

SELECTION_PATH = Path("data/evaluation/pathway/pathway_final_choice.csv")
NL_QUERY_FILES = {
    "explanatory": ("data/evaluation/pathway/pathway_nl_queries.csv", "nl_query"),
    "compact": ("data/evaluation/pathway/pathway_nl_queries_compact.csv", "nl_query_compact"),
}
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"
COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]
MIN_POSITIVES = 5  # must match pathway_final_choice.py's power filter


def build_gene_name_map(cohorts):
    """protein_id (ENSP) -> gene symbol, needed to match KEGG gene sets (which
    are keyed by gene symbol, not ENSP id)."""
    gene_name = {}
    for cohort in cohorts:
        path = Path(f"data/{cohort}/rank_normalized_proteomics.csv")
        if not path.exists():
            continue
        columns = pd.read_csv(path, header=[0, 1], index_col=0, nrows=0).columns
        for gene, protein_id in columns:
            gene_name.setdefault(protein_id, gene)
    return gene_name


def run(checkpoint_path: Path, style: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"NL query style: {style}\n")

    print("Loading pre-selected pathway evaluation set + NL query rewrites ...")
    selection_df = pd.read_csv(SELECTION_PATH)
    selection_df = selection_df[selection_df["passes_all_filters"]].copy()
    pathway_names = set(selection_df["pathway"])

    nl_path, nl_column = NL_QUERY_FILES[style]
    nl_df = pd.read_csv(nl_path)
    nl_query_map = dict(zip(nl_df["raw_name"], nl_df[nl_column]))
    missing = pathway_names - set(nl_query_map)
    if missing:
        raise ValueError(f"{len(missing)} pathways have no NL query rewrite: {missing}")
    print(f"  {len(pathway_names)} pathways passed the objective filters\n")

    with open("data/protein_splits.json") as f:
        protein_splits = json.load(f)
    test_protein_ids = set(protein_splits["test"])

    print("Building global candidate pool ...")
    dataset = ProteomicsDataset(cohorts=COHORTS, use_abundance=False)
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])
    print(f"  {len(protein_ids):,} proteins "
          f"({len(test_protein_ids):,} held-out test proteins)\n")

    gene_name_map = build_gene_name_map(COHORTS)
    protein_labels = [gene_name_map.get(pid, pid) for pid in protein_ids]
    is_test = np.array([pid in test_protein_ids for pid in protein_ids])

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, local_files_only=True)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, local_files_only=True).to(device)
    biobert_model.eval()

    protein_head = ProjectionHead(input_dim=640, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))

    print("Loading KEGG pathways from gseapy ...")
    gene_sets = gseapy.get_library("KEGG_2021_Human")

    print(f"Computing AUROC for {len(pathway_names)} pre-selected pathways "
          f"(natural-language queries, style={style}) ...\n")
    results = []
    for pathway_name in pathway_names:
        pathway_genes = set(gene_sets[pathway_name])
        labels = np.array([1 if g in pathway_genes else 0 for g in protein_labels])
        if labels.sum() < MIN_POSITIVES:
            continue

        nl_query = nl_query_map[pathway_name]
        query_vec = encode_query(nl_query, tokenizer, biobert_model, text_head, device)
        with torch.no_grad():
            sims = (protein_vecs @ query_vec.T).squeeze().cpu().numpy()

        overall_auroc = roc_auc_score(labels, sims)

        test_positive_mask = (labels == 1) & is_test
        n_test_positives = int(test_positive_mask.sum())
        test_only_auroc = None
        if n_test_positives >= MIN_POSITIVES:
            eval_mask = test_positive_mask | (labels == 0)
            test_only_auroc = roc_auc_score(labels[eval_mask], sims[eval_mask])

        results.append({
            "pathway": pathway_name,
            "nl_query": nl_query,
            "overall_auroc": overall_auroc,
            "n_positives": int(labels.sum()),
            "n_test_positives": n_test_positives,
            "test_only_auroc": test_only_auroc,
        })

    results_df = pd.DataFrame(results).sort_values("overall_auroc", ascending=False)
    out_path = Path(f"data/evaluation/pathway/nl_query_rewrites_results/pathway_nl_auroc_protein_level_results_{style}_{checkpoint_path.stem}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print("=" * 88)
    print(f"Protein-level pathway AUROC, natural-language queries ({style}) (no-abundance ablation)")
    print("=" * 88)
    print(results_df.drop(columns=["nl_query"]).to_string(index=False))

    print(f"\nOverall mean AUROC (n={len(results_df)} pathways): "
          f"{results_df['overall_auroc'].mean():.4f}")

    with_test = results_df[results_df["test_only_auroc"].notna()]
    print(f"\nPathways with enough test-only positives to score separately "
          f"(n={len(with_test)}): mean test-only AUROC = "
          f"{with_test['test_only_auroc'].mean():.4f}")

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/.../checkpoint_no_abundance_masked_e2000_best.pt")
    parser.add_argument("--style", choices=["explanatory", "compact"], required=True,
                        help="'explanatory' = full sentence, 'compact' = short 3-8 word "
                             "search phrase")
    args = parser.parse_args()
    run(Path(args.checkpoint), style=args.style)

"""
Stage 1 + Stage 2: wide semantic retrieval, then patient-specific abundance annotation.

Stage 1 (semantic retrieval, patient-invariant): embed a free-text query with
BioBERT + the trained text head, rank all proteins by cosine similarity against
the trained protein head's embeddings (checkpoint_no_abundance_best.pt), take the
top-N candidates. Candidate membership never depends on abundance, so a relevant
protein can never be excluded for being lowly expressed in one patient.

Stage 2 (patient-specific annotation): for a chosen real patient, look up each
candidate's actual rank-normalized abundance percentile and attach it. This never
adds or removes candidates from Stage 1's list -- it only labels them.

Usage:
    uv run python src/model/retrieve_and_annotate.py \\
        --query "receptor tyrosine kinase signaling" \\
        --patients 11LU013 11LU016 --cohort luad --top-n 50
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.embeddings.textual.embed_biobert import MAX_LENGTH, mean_pool
from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead

CHECKPOINT_PATH = "data/runs/luad_brca_ccrcc_gbm_hnscc_lscc_ov_pdac_ucec/checkpoint_no_abundance_best.pt"
COHORTS = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]


def build_gene_lookup():
    """ENSP_ID -> Gene name, merged across all cohorts (first-seen wins, same
    convention as the global protein dedup in dataset.py)."""
    lookup = {}
    for cohort in COHORTS:
        path = Path(f"data/{cohort}/protein_annotations.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["ENSP_ID", "Gene"])
        for ensp_id, gene in zip(df["ENSP_ID"], df["Gene"]):
            lookup.setdefault(ensp_id, gene)
    return lookup


def stage1_retrieve(query: str, top_n: int, device):
    dataset = ProteomicsDataset(cohorts=COHORTS, use_abundance=False)
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])

    protein_head = ProjectionHead(input_dim=640, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    protein_head.load_state_dict(ckpt["protein_head"])
    text_head.load_state_dict(ckpt["text_head"])
    protein_head.eval()
    text_head.eval()

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(device)
    biobert_model.eval()

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))

        enc = tokenizer(query, padding=True, truncation=True,
                         max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = biobert_model(**enc)
        query_vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
        query_vec = text_head(query_vec)

        sims = (protein_vecs @ query_vec.T).squeeze()
        ranked_idx = torch.argsort(sims, descending=True)[:top_n]

    candidates = [(protein_ids[i], sims[i].item()) for i in ranked_idx.tolist()]
    return candidates


def get_kegg_members(query: str):
    """If `query` is a literal KEGG pathway name, return its member gene set.
    Otherwise return None (query is free text, no ground truth available)."""
    import gseapy
    gene_sets = gseapy.get_library("KEGG_2021_Human")
    return set(gene_sets[query]) if query in gene_sets else None


def stage2_annotate(candidates, patient_id: str, cohort: str):
    abundance_df = pd.read_csv(f"data/{cohort}/rank_normalized_proteomics.csv",
                                header=[0, 1], index_col=0)
    protein_id_to_col = {ensp_id: (gene, ensp_id) for gene, ensp_id in abundance_df.columns}

    if patient_id not in abundance_df.index:
        raise ValueError(f"Patient {patient_id} not found in data/{cohort}/rank_normalized_proteomics.csv")

    patient_row = abundance_df.loc[patient_id]

    annotated = []
    for protein_id, cos_sim in candidates:
        col = protein_id_to_col.get(protein_id)
        if col is None:
            percentile = None  # protein not present in this cohort's abundance matrix at all
        else:
            val = patient_row[col]
            percentile = None if pd.isna(val) else float(val)
        annotated.append((protein_id, cos_sim, percentile))
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--cohort", default="luad")
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_lookup = build_gene_lookup()

    print(f"Query: \"{args.query}\"")
    print(f"Stage 1: retrieving top {args.top_n} candidates by semantic relevance...")
    candidates = stage1_retrieve(args.query, args.top_n, device)

    per_patient = {}
    for patient_id in args.patients:
        per_patient[patient_id] = stage2_annotate(candidates, patient_id, args.cohort)

    genes = [gene_lookup.get(pid, pid) for pid, _ in candidates]
    gene_counts = {g: genes.count(g) for g in set(genes)}
    display_names = [
        f"{g} ({pid})" if gene_counts[g] > 1 else g
        for g, (pid, _) in zip(genes, candidates)
    ]

    kegg_members = get_kegg_members(args.query)
    if kegg_members is not None:
        found = sorted(set(genes) & kegg_members)
        print(f"KEGG ground truth: {len(found)} of {len(kegg_members)} known pathway "
              f"members found in top {args.top_n} ({', '.join(found) if found else 'none'})")
    else:
        print("Query is not a literal KEGG pathway name — no ground truth to check against.")

    member_col = kegg_members is not None
    header = f"{'Gene':<22} {'cos_sim':>8}"
    if member_col:
        header += f"  {'KEGG?':>6}"
    for patient_id in args.patients:
        header += f"  {patient_id:>14}"
    print("\n" + header)
    print("-" * len(header))

    for i, (_, cos_sim) in enumerate(candidates):
        row = f"{display_names[i]:<22} {cos_sim:>8.4f}"
        if member_col:
            row += f"  {'✓' if genes[i] in kegg_members else '':>6}"
        for patient_id in args.patients:
            _, _, pct = per_patient[patient_id][i]
            cell = "not detected" if pct is None else f"{pct * 100:5.1f}th pctile"
            row += f"  {cell:>14}"
        print(row)


if __name__ == "__main__":
    main()

"""
k-NN embedding-space-quality check (Wu, Xiong, Yu, Lin, CVPR 2018,
"Unsupervised Feature Learning via Non-Parametric Instance Discrimination",
arXiv:1805.01978 -- the paper that introduced weighted-kNN-on-frozen-features
as a representation-quality check; also used as the primary evaluation in
Caron et al. 2021, "Emerging Properties in Self-Supervised Vision
Transformers" / DINO, arXiv:2104.14294).

Genuinely different from every other eval in this project's suite: uses
PROTEIN-SIDE EMBEDDINGS ONLY (ESM2 -> protein_head), no text encoder, no
zero-shot query, no true (protein, text) pair needed at all -- just a
protein's embedding and an already-known fact about it (its true DeepLoc
location / EC class / GO-MF class, reused directly from the existing
zero-shot eval sets' ground truth). See CLAUDE.md for the full discussion
of why this complements (not replaces) alignment_uniformity_eval.py, which
needs a true text pair, and the topic-specific zero-shot evals, which need
a text-side query.

Mechanics, per label set (run independently -- DeepLoc/EC/GO-MF are
unrelated categorization systems, not merged):
  1. Embed every eligible protein: ESM2 -> protein_head (L2-normalized,
     same as everywhere else in this project).
  2. For each protein, find its K nearest OTHER proteins by cosine
     similarity (excluding itself).
  3. Check what fraction of those K neighbors share the protein's own true
     class. GO-MF is multi-label (lenient scoring, same convention as
     zero_shot_go_mf_eval.py) -- a neighbor counts as agreeing if ANY of
     its valid classes overlaps with the query protein's valid classes,
     not exact single-label match.
  4. Average over all eligible proteins -> one number per K, per label set.
  5. Compare against the random-chance baseline for that label set
     (1/n_classes for DeepLoc/EC; mean per-protein valid-set-size / n_classes
     for GO-MF's lenient rule, same baseline logic as
     zero_shot_go_mf_eval.py) -- same "never trust a raw number without a
     baseline" discipline as every other eval here.

VALIDATION-STYLE POOL BY DEFAULT: `unseen_test` rows are excluded from the
neighbor pool by default (--include-test to override), matching the
standing rule elsewhere in this project (CLAUDE.md 11ab) that test stays
reserved for a single final confirmatory check, not repeated
checkpoint-comparison use -- this script is meant for comparing
checkpoints (multiview vs. multi-positive vs. M0), not a final report.

Usage:
    uv run python src/evaluation/knn_embedding_quality_eval.py \\
        --checkpoint data/runs/full_proteome/multiview_experiment/multiview_best.pt \\
        --eval all --k-values 5,10,20
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
from src.model.projection_head import ProjectionHead

OUTPUT_DIR = Path("data/evaluation/knn_embedding_quality")

EVAL_SETS = {
    "deeploc": {
        "path": "data/evaluation/zero_shot_classification/deeploc/test_set.csv",
        "multi_label": False,
    },
    "ec": {
        "path": "data/evaluation/zero_shot_classification/ec_top_level/test_set.csv",
        "multi_label": False,
    },
    "go_mf": {
        "path": "data/evaluation/zero_shot_classification/go_mf/test_set.csv",
        "multi_label": True,
    },
}


def load_protein_head(checkpoint_path: str, device: torch.device) -> ProjectionHead:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 512)
    output_dim = ckpt.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(ckpt["protein_head"])
    protein_head.eval()
    return protein_head


def knn_agreement(vecs: torch.Tensor, labels: list, multi_label: bool, k_values: list[int]) -> dict:
    """vecs: (n, d), already L2-normalized. Returns {k: mean_agreement_fraction}."""
    n = vecs.shape[0]
    sim = vecs @ vecs.T
    sim.fill_diagonal_(-2.0)  # exclude self from its own neighbor search

    results = {}
    for k in k_values:
        if k >= n:
            continue
        topk_idx = sim.topk(k, dim=1).indices.cpu()
        agreements = []
        for i in range(n):
            if multi_label:
                agree = sum(1 for j in topk_idx[i].tolist() if labels[j] & labels[i])
            else:
                agree = sum(1 for j in topk_idx[i].tolist() if labels[j] == labels[i])
            agreements.append(agree / k)
        results[k] = sum(agreements) / len(agreements)
    return results


def run_one_eval_set(name: str, cfg: dict, protein_head: ProjectionHead,
                      esm2_matrix: torch.Tensor, esm2_id_to_idx: dict,
                      device: torch.device, k_values: list[int], include_test: bool):
    df = pd.read_csv(cfg["path"])
    if not include_test:
        df = df[df["split"] != "unseen_test"].reset_index(drop=True)
    df = df[df["ensp_id"].isin(esm2_id_to_idx)].reset_index(drop=True)

    if cfg["multi_label"]:
        labels = [set(s.split(";")) for s in df["all_valid_classes"]]
        n_classes = len(set().union(*labels))
        mean_valid_per_protein = sum(len(s) for s in labels) / len(labels)
        random_baseline = mean_valid_per_protein / n_classes
    else:
        labels = df["true_class"].tolist()
        n_classes = df["true_class"].nunique()
        random_baseline = 1.0 / n_classes

    idx = [esm2_id_to_idx[pid] for pid in df["ensp_id"]]
    esm2_subset = esm2_matrix[idx].to(device)

    with torch.no_grad():
        trained_vecs = protein_head(esm2_subset)               # (n, d), L2-normalized by ProjectionHead
        raw_esm2_vecs = F.normalize(esm2_subset, dim=1)         # baseline: no trained head at all

    trained_results = knn_agreement(trained_vecs, labels, cfg["multi_label"], k_values)
    raw_esm2_results = knn_agreement(raw_esm2_vecs, labels, cfg["multi_label"], k_values)

    return {
        "eval_set": name,
        "n_proteins": len(df),
        "n_classes": n_classes,
        "random_baseline": random_baseline,
        "agreement_at_k": trained_results,
        "raw_esm2_agreement_at_k": raw_esm2_results,
    }


def main(checkpoint_path: str, which: str, k_values: list[int], include_test: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head = load_protein_head(checkpoint_path, device)

    with h5py.File("data/full_proteome/esm2_embeddings.h5", "r") as h5f:
        esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
        esm2_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
    esm2_id_to_idx = {pid: i for i, pid in enumerate(esm2_ids)}

    names = list(EVAL_SETS.keys()) if which == "all" else [which]
    all_results = {}
    for name in names:
        res = run_one_eval_set(name, EVAL_SETS[name], protein_head, esm2_matrix,
                                esm2_id_to_idx, device, k_values, include_test)
        all_results[name] = res

        print(f"\n=== {name} ===")
        print(f"n_proteins={res['n_proteins']:,}  n_classes={res['n_classes']}  "
              f"random_baseline={res['random_baseline']*100:.2f}%")
        for k, v in res["agreement_at_k"].items():
            raw = res["raw_esm2_agreement_at_k"][k]
            ratio = v / res["random_baseline"] if res["random_baseline"] > 0 else float("nan")
            print(f"  k={k:>3}: trained={v*100:.2f}%  raw_esm2={raw*100:.2f}%  "
                  f"random_baseline={res['random_baseline']*100:.2f}%  "
                  f"({ratio:.2f}x random, +{(v-raw)*100:.2f}pp over raw ESM2)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = Path(checkpoint_path).stem
    out_path = OUTPUT_DIR / f"summary_{checkpoint_stem}.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(checkpoint_path), "include_test": include_test,
                   "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval", default="all", choices=["all", "deeploc", "ec", "go_mf"])
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--include-test", action="store_true",
                        help="include unseen_test rows in the neighbor pool -- default off, "
                             "test stays reserved (same standing rule as GO-retrieval, CLAUDE.md 11ab)")
    args = parser.parse_args()
    k_values = [int(k) for k in args.k_values.split(",")]
    main(args.checkpoint, args.eval, k_values, args.include_test)

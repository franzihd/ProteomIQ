"""
Visualize the trained protein-embedding space via UMAP -- a qualitative,
illustrative RESULTS figure, scoped narrowly to two claims a UMAP projection
can actually support:

  1. Do proteins sharing a true label (DeepLoc/EC/GO-MF class) form any
     visible local structure at all? (Panels 1-3.)
  2. Do held-out (unseen) proteins land interleaved among trained (seen)
     proteins, rather than forming their own separate island? (Panel 4 --
     the more load-bearing panel: this is the kind of local-neighborhood
     question UMAP is actually suited to answer.)

An earlier version of this script also produced a 3-way grid comparing
UMAP layouts across checkpoints (M0 / multiview / multi-positive) to argue
"multiview's space is visually better organized." That was deliberately
removed: each checkpoint's embeddings get their own separate UMAP fit, and
UMAP does not reliably preserve global distances or cluster tightness
across separate fits -- so "does this cluster look tighter than that one"
is not a controlled measurement, just an impression that happens to agree
with the real evidence (linear_probe_eval.py / alignment_uniformity_eval.py
/ knn_embedding_quality_eval.py) exactly when it agrees, and gets waved
away as "well, UMAP just doesn't show it" when it doesn't -- confirmation,
not evidence. This script makes no cross-checkpoint claim.

Computes no new numbers itself (beyond picking which 3 classes to
highlight per label set) -- the actual evidence for "the space is
well-organized" is the three task-agnostic evals; this figure is for a
reader's intuition, not additional proof.

Reuses the exact same DeepLoc/EC/GO-MF eval-set CSVs and their `split`
column (seen / unseen_validation / unseen_test, derived from
data/protein_splits_full_proteome.json) as linear_probe_eval.py and
knn_embedding_quality_eval.py -- no new label logic, same standing
test-reserve rule (CLAUDE.md 11ab): meant to be run on the one final
chosen checkpoint, not per-ablation-candidate, since it does touch
`unseen_test` points for the split panel (visualization only, not used to
pick a winner).

Category-coloring design note: a categorical palette cannot be made
colorblind-safe for *all pairs* once a scatter plot shows more than ~3
simultaneous categories (a hard limit of categorical color perception, not
a shortcut -- see the dataviz skill's palette.md, "past three, fold to
Other or facet"). DeepLoc (9 classes), EC (7), and GO-MF (10, and
genuinely multi-label, see build_go_mf_eval_set.py) all exceed that.
Rather than ship an all-pairs-unsafe rainbow that would also just be
unreadable at that cardinality, each panel highlights only its top-3 most
frequent classes in the dataviz skill's first 3 (all-pairs-validated)
categorical slots, and folds every other class into a neutral "Other"
gray. The seen/unseen panel has only 2-3 classes, so it needs no such
folding.

Usage:
    uv run python src/evaluation/visualize_embedding_space.py \\
        --checkpoint data/runs/full_proteome/multiview_experiment/multiview_best.pt
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from src.model.projection_head import ProjectionHead

ESM2_PATH = "data/full_proteome/esm2_embeddings.h5"
OUTPUT_DIR = Path("data/plots/embedding_space")

EVAL_SETS = {
    "DeepLoc (subcellular location)": "data/evaluation/zero_shot_classification/deeploc/test_set.csv",
    "EC top-level (enzyme class)": "data/evaluation/zero_shot_classification/ec_top_level/test_set.csv",
    "GO-Slim-MF (primary class)": "data/evaluation/zero_shot_classification/go_mf/test_set.csv",
}

# dataviz skill categorical palette, slots 1-3 -- the only slots validated
# all-pairs colorblind-safe for a scatter/bubble chart (palette.md). Slot 4+
# is not usable here for that reason; everything beyond the top 3 classes
# per panel folds into COLOR_OTHER instead of extending the palette.
COLOR_1 = "#2a78d6"  # blue
COLOR_2 = "#eb6834"  # orange
COLOR_3 = "#1baf7a"  # aqua
COLOR_OTHER = "#c7c6c1"  # neutral gray -- "every other class" / "not in this task"

SPLIT_ORDER = ["seen", "unseen_validation", "unseen_test"]
SPLIT_COLORS = {"seen": COLOR_1, "unseen_validation": COLOR_2, "unseen_test": COLOR_3}
SPLIT_LABELS = {
    "seen": "Seen (train)",
    "unseen_validation": "Unseen (validation)",
    "unseen_test": "Unseen (test)",
}


def load_protein_head(checkpoint_path: str, device: torch.device) -> ProjectionHead:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 512)
    output_dim = ckpt.get("output_dim", 256)
    head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    head.load_state_dict(ckpt["protein_head"])
    head.eval()
    return head


def build_pool(esm2_id_to_idx: dict) -> tuple[dict, dict]:
    """Union of proteins across all 3 eval sets. Returns (pool, dfs) where
    pool[ensp_id] = {"split": ..., task_name: true_class, ...} (task_name key
    present only for tasks that eval set actually covers this protein)."""
    dfs = {}
    pool = {}
    for task_name, path in EVAL_SETS.items():
        df = pd.read_csv(path)
        df = df[df["ensp_id"].isin(esm2_id_to_idx)]
        dfs[task_name] = df
        for ensp_id, split, true_class in zip(df["ensp_id"], df["split"], df["true_class"]):
            entry = pool.setdefault(ensp_id, {"split": split})
            entry[task_name] = true_class
    return pool, dfs


def top3_colors_for_task(pool: dict, ensp_ids: list, task_name: str) -> dict:
    labels = pd.Series([pool[pid].get(task_name) for pid in ensp_ids]).dropna()
    top3 = labels.value_counts().head(3).index.tolist()
    return {cls: color for cls, color in zip(top3, [COLOR_1, COLOR_2, COLOR_3])}


def plot_class_panel(ax, coords: np.ndarray, ensp_ids: list, pool: dict, task_name: str):
    labels = [pool[pid].get(task_name) for pid in ensp_ids]
    colors_by_class = top3_colors_for_task(pool, ensp_ids, task_name)

    other_mask = np.array([lbl not in colors_by_class for lbl in labels])
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1], s=4, c=COLOR_OTHER,
               alpha=0.35, linewidths=0, label="Other / not in this task")
    for cls, color in colors_by_class.items():
        m = np.array([lbl == cls for lbl in labels])
        ax.scatter(coords[m, 0], coords[m, 1], s=6, c=color, alpha=0.75, linewidths=0, label=cls)

    ax.set_title(task_name, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=2, loc="best", framealpha=0.9)


def plot_split_panel(ax, coords: np.ndarray, ensp_ids: list, pool: dict):
    splits = [pool[pid]["split"] for pid in ensp_ids]
    for split_key in SPLIT_ORDER:
        m = np.array([s == split_key for s in splits])
        if m.sum() == 0:
            continue
        # plot seen first (background), unseen splits on top so they're not buried
        ax.scatter(coords[m, 0], coords[m, 1], s=5, c=SPLIT_COLORS[split_key],
                   alpha=0.55, linewidths=0, label=f"{SPLIT_LABELS[split_key]} (n={m.sum():,})",
                   zorder=2 if split_key != "seen" else 1)
    ax.set_title("Train vs. held-out split", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.9)


def compute_pool_and_coords(checkpoint_path: str, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head = load_protein_head(checkpoint_path, device)

    with h5py.File(ESM2_PATH, "r") as h5f:
        esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
        esm2_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
    esm2_id_to_idx = {pid: i for i, pid in enumerate(esm2_ids)}

    pool, dfs = build_pool(esm2_id_to_idx)
    ensp_ids = list(pool.keys())
    counts = ", ".join(f"{name.split(' (')[0]} {len(df):,}" for name, df in dfs.items())
    print(f"Unified pool: {len(ensp_ids):,} proteins ({counts})")

    idx = [esm2_id_to_idx[pid] for pid in ensp_ids]
    esm2_subset = esm2_matrix[idx].to(device)
    with torch.no_grad():
        embeddings = protein_head(esm2_subset).cpu().numpy()

    print(f"Running UMAP on {embeddings.shape[0]:,} x {embeddings.shape[1]}-dim vectors...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
    coords = reducer.fit_transform(embeddings)
    return pool, ensp_ids, coords


def main(checkpoint_path: str, seed: int):
    pool, ensp_ids, coords = compute_pool_and_coords(checkpoint_path, seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    checkpoint_stem = Path(checkpoint_path).stem
    fig.suptitle(f"Protein-embedding space (UMAP) -- checkpoint: {checkpoint_stem}", fontsize=13)

    for ax, task_name in zip(axes.flat[:3], EVAL_SETS):
        plot_class_panel(ax, coords, ensp_ids, pool, task_name)
    plot_split_panel(axes.flat[3], coords, ensp_ids, pool)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"umap_{checkpoint_stem}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main_split_only(checkpoint_path: str, seed: int):
    """Standalone version of just the seen/unseen split panel -- the one
    panel actually worth citing in the thesis (see module docstring); a
    single, thesis-ready figure rather than a crop out of the 2x2 grid."""
    pool, ensp_ids, coords = compute_pool_and_coords(checkpoint_path, seed)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    checkpoint_stem = Path(checkpoint_path).stem
    plot_split_panel(ax, coords, ensp_ids, pool)
    ax.set_title("")  # title goes in the thesis caption instead
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"umap_split_{checkpoint_stem}.png"
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/runs/full_proteome/multiview_experiment/multiview_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-only", action="store_true",
                         help="Save just the seen/unseen split panel as its own figure "
                              "(the one recommended for the thesis) instead of the 2x2 grid.")
    args = parser.parse_args()
    if args.split_only:
        main_split_only(args.checkpoint, args.seed)
    else:
        main(args.checkpoint, args.seed)

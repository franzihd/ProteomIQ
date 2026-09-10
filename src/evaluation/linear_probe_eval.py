"""
Linear probe embedding-space-quality check -- standard representation-learning
diagnostic (used as the headline eval in SimCLR, MoCo, DINO, and others): freeze
the encoder completely and fit the smallest possible classifier (one linear
layer, no hidden layers/nonlinearities) on top of its embeddings, then measure
how accurately that classifier recovers a known label.

Complements (does not replace) knn_embedding_quality_eval.py and
alignment_uniformity_eval.py -- same "protein-side embeddings only, no text
encoder, no zero-shot query" design, reusing the exact same DeepLoc/EC/GO-MF
label sets and seen/unseen split as every other eval in this project. The
difference from k-NN: k-NN asks "do nearby points already agree on a label,
with zero fitting at all"; a linear probe asks the stricter question "can a
*trained* minimal classifier decode the label directly from the embedding" --
a more standard, more citable form of evidence that the space is not just
locally clustered but genuinely linearly decodable. See CLAUDE.md for the
full discussion of how this fits alongside k-NN/alignment-uniformity.

Mechanics, per label set (run independently -- DeepLoc/EC/GO-MF are unrelated
categorization systems, not merged):
  1. Embed every eligible protein: ESM2 -> protein_head (L2-normalized, same
     as everywhere else in this project). Also compute the raw-untrained-ESM2
     version, same "trained vs. untrained" comparison as the k-NN eval.
  2. Split by the eval set's own `split` column: `seen` rows train the probe,
     `unseen_validation` rows are what accuracy is reported on. `unseen_test`
     is never touched here -- same standing rule as everywhere else in this
     project (CLAUDE.md 11ab) that test stays reserved for a single final
     confirmatory check, not repeated checkpoint-comparison use.
  3. Train: nn.Linear(output_dim, n_classes), full-batch Adam. DeepLoc/EC are
     single-label (cross-entropy). GO-MF is genuinely multi-label (11o) --
     trained with a multi-hot BCE target (a protein's full `all_valid_classes`
     set, not just its arbitrary first-listed term), scored the same lenient
     way as every other GO-MF eval in this suite: a prediction (argmax of the
     probe's output) counts as correct if it lands in ANY of the protein's
     valid classes, not just the single primary one.
  4. Evaluate accuracy on `unseen_validation`, for both the trained-checkpoint
     embeddings and the raw-ESM2 baseline.
  5. Report against a random baseline and a majority-class baseline (the most
     common class among the *training* (`seen`) rows, evaluated on
     `unseen_validation`) -- same "never trust a raw number without a
     baseline" discipline as every other eval here.

Usage:
    uv run python src/evaluation/linear_probe_eval.py \\
        --checkpoint data/runs/full_proteome/multiview_experiment/multiview_best.pt \\
        --eval all
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from src.model.projection_head import ProjectionHead

OUTPUT_DIR = Path("data/evaluation/linear_probe")

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

SEED = 42
EPOCHS = 300
LR = 1e-2
WEIGHT_DECAY = 1e-3


def load_protein_head(checkpoint_path: str, device: torch.device) -> ProjectionHead:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 512)
    output_dim = ckpt.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(ckpt["protein_head"])
    protein_head.eval()
    return protein_head


def train_and_eval_probe(train_vecs: torch.Tensor, train_labels, val_vecs: torch.Tensor, val_labels,
                          n_classes: int, class_to_idx: dict, multi_label: bool, device: torch.device) -> float:
    """Fits a single nn.Linear(dim, n_classes) probe on train_vecs/labels, returns accuracy on val_vecs/labels.
    labels are either a list of single class names (single-label) or a list of sets of class names (multi-label,
    the protein's full `all_valid_classes`)."""
    torch.manual_seed(SEED)
    dim = train_vecs.shape[1]
    probe = nn.Linear(dim, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if multi_label:
        target = torch.zeros(len(train_labels), n_classes, device=device)
        for i, valid_set in enumerate(train_labels):
            for c in valid_set:
                target[i, class_to_idx[c]] = 1.0
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        target = torch.tensor([class_to_idx[c] for c in train_labels], device=device)
        loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = probe(train_vecs)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_preds = probe(val_vecs).argmax(dim=1).cpu().tolist()

    if multi_label:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        correct = sum(1 for pred, valid_set in zip(val_preds, val_labels) if idx_to_class[pred] in valid_set)
    else:
        correct = sum(1 for pred, true_c in zip(val_preds, val_labels) if class_to_idx[true_c] == pred)
    return correct / len(val_labels)


def run_one_eval_set(name: str, cfg: dict, protein_head: ProjectionHead,
                      esm2_matrix: torch.Tensor, esm2_id_to_idx: dict, device: torch.device) -> dict:
    df = pd.read_csv(cfg["path"])
    df = df[df["ensp_id"].isin(esm2_id_to_idx)].reset_index(drop=True)

    train_df = df[df["split"] == "seen"].reset_index(drop=True)
    val_df = df[df["split"] == "unseen_validation"].reset_index(drop=True)

    if cfg["multi_label"]:
        train_labels = [set(s.split(";")) for s in train_df["all_valid_classes"]]
        val_labels = [set(s.split(";")) for s in val_df["all_valid_classes"]]
        all_classes = sorted(set().union(*train_labels, *val_labels))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        n_classes = len(all_classes)
        mean_valid_per_protein = sum(len(s) for s in val_labels) / len(val_labels)
        random_baseline = mean_valid_per_protein / n_classes
        # majority baseline: single most-common class (by membership frequency) among training rows
        from collections import Counter
        counts = Counter(c for s in train_labels for c in s)
        majority_class = counts.most_common(1)[0][0]
        majority_baseline = sum(1 for s in val_labels if majority_class in s) / len(val_labels)
    else:
        train_labels = train_df["true_class"].tolist()
        val_labels = val_df["true_class"].tolist()
        all_classes = sorted(set(train_labels) | set(val_labels))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        n_classes = len(all_classes)
        random_baseline = 1.0 / n_classes
        majority_class = pd.Series(train_labels).value_counts().idxmax()
        majority_baseline = sum(1 for c in val_labels if c == majority_class) / len(val_labels)

    def embed(sub_df):
        idx = [esm2_id_to_idx[pid] for pid in sub_df["ensp_id"]]
        esm2_subset = esm2_matrix[idx].to(device)
        with torch.no_grad():
            trained = protein_head(esm2_subset)
            raw = F.normalize(esm2_subset, dim=1)
        return trained, raw

    train_trained, train_raw = embed(train_df)
    val_trained, val_raw = embed(val_df)

    trained_acc = train_and_eval_probe(train_trained, train_labels, val_trained, val_labels,
                                        n_classes, class_to_idx, cfg["multi_label"], device)
    raw_esm2_acc = train_and_eval_probe(train_raw, train_labels, val_raw, val_labels,
                                         n_classes, class_to_idx, cfg["multi_label"], device)

    return {
        "eval_set": name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_classes": n_classes,
        "random_baseline": random_baseline,
        "majority_baseline": majority_baseline,
        "majority_class": majority_class,
        "probe_accuracy": trained_acc,
        "raw_esm2_probe_accuracy": raw_esm2_acc,
    }


def main(checkpoint_path: str, which: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_head = load_protein_head(checkpoint_path, device)

    with h5py.File("data/full_proteome/esm2_embeddings.h5", "r") as h5f:
        esm2_matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
        esm2_ids = [pid.decode() for pid in h5f["protein_ids"][:]]
    esm2_id_to_idx = {pid: i for i, pid in enumerate(esm2_ids)}

    names = list(EVAL_SETS.keys()) if which == "all" else [which]
    all_results = {}
    for name in names:
        res = run_one_eval_set(name, EVAL_SETS[name], protein_head, esm2_matrix, esm2_id_to_idx, device)
        all_results[name] = res

        print(f"\n=== {name} ===")
        print(f"n_train={res['n_train']:,}  n_val={res['n_val']:,}  n_classes={res['n_classes']}")
        print(f"  random_baseline={res['random_baseline']*100:.2f}%  "
              f"majority_baseline={res['majority_baseline']*100:.2f}% ({res['majority_class']})")
        print(f"  probe (trained checkpoint): {res['probe_accuracy']*100:.2f}%")
        print(f"  probe (raw ESM2, untrained): {res['raw_esm2_probe_accuracy']*100:.2f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = Path(checkpoint_path).stem
    out_path = OUTPUT_DIR / f"summary_{checkpoint_stem}.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(checkpoint_path), "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval", default="all", choices=["all", "deeploc", "ec", "go_mf"])
    args = parser.parse_args()
    main(args.checkpoint, args.eval)

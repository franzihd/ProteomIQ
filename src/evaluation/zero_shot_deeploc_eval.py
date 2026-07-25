"""
Zero-shot subcellular-localization classification via DeepLoc labels.

Standard CLIP-style zero-shot classification (Radford et al. 2021, section
3.1.4): the model is never trained on DeepLoc at all. Each of the 10 DeepLoc
classes is turned into a handful of natural-language prompts, embedded with
our text tower, and averaged into one class embedding ("prompt ensembling" --
same idea as CLIP's "a photo of a {label}" / "a photo of many {label}"
ensemble: different phrasings of the same class reduce sensitivity to any
one prompt's exact wording). Each test protein is embedded with our protein
tower and assigned to whichever class embedding it is most cosine-similar to.

This is a genuinely different evaluation from pathway_auroc_protein_level.py:
pathway AUROC is a ranking metric (does the model rank pathway members above
non-members), computed one pathway at a time. This is a forced-choice
classification over a small, fixed, mutually-exclusive label set (a protein
has exactly one primary DeepLoc location) -- closer to conventional zero-shot
image classification benchmarks, and a useful sanity check that doesn't
depend on gseapy/KEGG at all.

Run src/evaluation/build_deeploc_eval_set.py first to build the human-only,
seen/unseen-labeled test set this script reads.

Usage:
    uv run python src/evaluation/zero_shot_deeploc_eval.py \\
        --checkpoint data/runs/full_proteome/final_hparams_masked_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

EVAL_SET_PATH = Path("data/evaluation/deeploc_human_test_set.csv")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"

# Fixed class order -- reused everywhere (prompt building, argmax, confusion
# matrix axes) so results are directly comparable across runs. "Chloroplast"
# has no human members (see build_deeploc_eval_set.py): kept by default for
# the full 10-way task, but --drop-chloroplast removes it from the candidate
# set for a fairer human-only 9-way run -- every "Chloroplast" prediction is
# a guaranteed miss (no human protein can ever have that true label), so
# leaving it in structurally deflates accuracy for a class that can't score.
CLASSES_10WAY = [
    "Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion", "Cell membrane",
    "Endoplasmic reticulum", "Chloroplast", "Golgi apparatus",
    "Lysosome/Vacuole", "Peroxisome",
]
CLASSES_9WAY_HUMAN = [c for c in CLASSES_10WAY if c != "Chloroplast"]

# A natural-language phrase per class, filled into the 3 templates below.
LOCATION_PHRASES = {
    "Nucleus": "the nucleus",
    "Cytoplasm": "the cytoplasm",
    "Extracellular": "the extracellular space",
    "Mitochondrion": "the mitochondrion",
    "Cell membrane": "the cell membrane",
    "Endoplasmic reticulum": "the endoplasmic reticulum",
    "Chloroplast": "the chloroplast",
    "Golgi apparatus": "the Golgi apparatus",
    "Lysosome/Vacuole": "the lysosome or vacuole",
    "Peroxisome": "the peroxisome",
}

# 3 independent phrasings per class ("prompt ensembling"), applied uniformly
# across all classes via {phrase}.
TEMPLATES = [
    "A protein located in {phrase}.",
    "This protein is found in {phrase} and carries out its function there.",
    "Subcellular localization: {phrase}.",
]


def build_class_embeddings(classes, tokenizer, biobert_model, text_head, device):
    """One embedding per class: encode each of its 3 template prompts (each
    already L2-normalized by ProjectionHead), average, then re-normalize --
    the standard CLIP prompt-ensembling recipe. Averaging without the final
    re-normalization would let classes whose 3 templates happen to agree
    more end up with larger-norm embeddings, which would silently bias the
    argmax toward them (cosine similarity via dot product scales with norm).
    """
    class_vecs = {}
    for cls in classes:
        phrase = LOCATION_PHRASES[cls]
        template_vecs = [
            encode_query(t.format(phrase=phrase), tokenizer, biobert_model, text_head, device)
            for t in TEMPLATES
        ]
        averaged = torch.stack(template_vecs).mean(dim=0)
        class_vecs[cls] = F.normalize(averaged, dim=-1)
    return torch.cat([class_vecs[cls] for cls in classes], dim=0)  # (len(classes), output_dim)


def compute_subset_metrics(df: pd.DataFrame, classes, pred_col: str = "predicted_class") -> dict:
    y_true = df["true_class"]
    y_pred = df[pred_col]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    majority_class = y_true.value_counts().idxmax()
    majority_baseline_accuracy = (y_true == majority_class).mean()
    random_baseline_accuracy = 1.0 / len(classes)

    return {
        "n": len(df),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        "majority_class": majority_class,
        "majority_baseline_accuracy": float(majority_baseline_accuracy),
        "random_baseline_accuracy": float(random_baseline_accuracy),
    }


def run(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5",
        drop_chloroplast: bool = False, calibrate: bool = False):
    classes = CLASSES_9WAY_HUMAN if drop_chloroplast else CLASSES_10WAY
    variant_suffix = "_9way_human" if drop_chloroplast else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Classes ({len(classes)}-way): {classes}\n")

    print("Loading DeepLoc human-only test set ...")
    eval_df = pd.read_csv(EVAL_SET_PATH)
    print(f"  {len(eval_df)} proteins "
          f"({(eval_df['split'] == 'seen').sum()} seen, {(eval_df['split'] == 'unseen').sum()} unseen)\n")

    print("Loading checkpoint and building protein/text towers ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Older checkpoints predate saving these keys -- fall back to the original defaults.
    hidden_dim = checkpoint.get("hidden_dim", 512)
    output_dim = checkpoint.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, local_files_only=True)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, local_files_only=True).to(device)
    biobert_model.eval()

    print(f"Embedding the {len(classes)} class prompts (3 templates each, ensembled) ...")
    class_vecs = build_class_embeddings(classes, tokenizer, biobert_model, text_head, device)

    print("Embedding DeepLoc test proteins with the protein tower ...")
    eval_ensp_ids = set(eval_df["ensp_id"])
    dataset = ProteomicsDataset(
        cohorts=[], use_abundance=False,
        biobert_filename=biobert_filename,
        restrict_to_protein_ids=eval_ensp_ids,
    )
    resolved_ids = dataset.pairs
    missing = eval_ensp_ids - set(resolved_ids)
    if missing:
        print(f"  WARNING: {len(missing)}/{len(eval_ensp_ids)} DeepLoc proteins have no "
              f"'{biobert_filename}' embedding and are dropped from this run")
    eval_df = eval_df[eval_df["ensp_id"].isin(resolved_ids)].reset_index(drop=True)
    print(f"  {len(eval_df)} proteins resolved\n")

    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in eval_df["ensp_id"]])
    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))
        sims = (protein_vecs @ class_vecs.T).cpu().numpy()  # (n_proteins, len(classes))

    predicted_idx = sims.argmax(axis=1)
    eval_df["predicted_class"] = [classes[i] for i in predicted_idx]
    eval_df["correct"] = eval_df["true_class"] == eval_df["predicted_class"]

    if calibrate:
        # Class-wise centering: some class prompts sit in a region of
        # embedding space that is generically close to most proteins,
        # regardless of whether that protein actually belongs there --
        # diagnosed directly from the uncalibrated run (Nucleus, the most
        # common true class, was drastically under-predicted; rare classes
        # like Lysosome/Vacuole and Peroxisome were over-predicted 10-13x).
        # Subtracting each class's own mean similarity (computed across all
        # test proteins in this run) before argmax removes that constant,
        # class-specific offset so a class only wins when a protein is
        # relatively closer to it than to other proteins, not just close in
        # absolute terms. Standard zero-shot CLIP calibration trick.
        class_mean_sim = sims.mean(axis=0)
        centered_sims = sims - class_mean_sim
        predicted_idx_cal = centered_sims.argmax(axis=1)
        eval_df["predicted_class_calibrated"] = [classes[i] for i in predicted_idx_cal]
        eval_df["correct_calibrated"] = eval_df["true_class"] == eval_df["predicted_class_calibrated"]

    print("Computing metrics for seen / unseen / all ...\n")
    subsets = {
        "all": eval_df,
        "seen": eval_df[eval_df["split"] == "seen"],
        "unseen": eval_df[eval_df["split"] == "unseen"],
    }
    metrics = {name: compute_subset_metrics(subset_df, classes) for name, subset_df in subsets.items()}
    metrics["generalization_gap"] = metrics["seen"]["accuracy"] - metrics["unseen"]["accuracy"]
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["biobert_filename"] = biobert_filename
    metrics["classes"] = classes

    if calibrate:
        metrics["calibrated"] = {
            name: compute_subset_metrics(subset_df, classes, pred_col="predicted_class_calibrated")
            for name, subset_df in subsets.items()
        }
        metrics["calibrated"]["generalization_gap"] = (
            metrics["calibrated"]["seen"]["accuracy"] - metrics["calibrated"]["unseen"]["accuracy"]
        )
        metrics["class_mean_similarity"] = dict(zip(classes, class_mean_sim.tolist()))

    predictions_path = Path(f"data/evaluation/deeploc_zero_shot_predictions_{checkpoint_path.stem}{variant_suffix}.csv")
    summary_path = Path(f"data/evaluation/deeploc_zero_shot_summary_{checkpoint_path.stem}{variant_suffix}.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 88)
    print(f"Zero-shot DeepLoc subcellular localization classification ({len(classes)}-way"
          f"{', human-only, no Chloroplast' if drop_chloroplast else ''})")
    print("=" * 88)
    for name in ["all", "seen", "unseen"]:
        m = metrics[name]
        print(f"\n[{name}]  n={m['n']}")
        print(f"  uncalibrated accuracy:   {m['accuracy']*100:.2f}%  "
              f"(random baseline: {m['random_baseline_accuracy']*100:.2f}%, "
              f"majority-class baseline ['{m['majority_class']}']: {m['majority_baseline_accuracy']*100:.2f}%)")
        print(f"  uncalibrated macro-F1:   {m['macro_f1']:.4f}")
        if calibrate:
            mc = metrics["calibrated"][name]
            print(f"  calibrated accuracy:     {mc['accuracy']*100:.2f}%")
            print(f"  calibrated macro-F1:     {mc['macro_f1']:.4f}")
    print(f"\nGeneralization gap, uncalibrated (seen accuracy - unseen accuracy): "
          f"{metrics['generalization_gap']*100:.2f}pp")
    if calibrate:
        print(f"Generalization gap, calibrated: {metrics['calibrated']['generalization_gap']*100:.2f}pp")
        print("\nPer-class mean similarity (before centering) -- classes with a high mean here "
              "were winning the argmax more often just by being generically close to most proteins:")
        for cls, mean_sim in sorted(metrics["class_mean_similarity"].items(), key=lambda kv: -kv[1]):
            print(f"  {cls:<24s} {mean_sim:.4f}")
    print(f"\nPer-protein predictions saved to {predictions_path}")
    print(f"Summary + confusion matrices saved to {summary_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/full_proteome/final_hparams_masked_best.pt")
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5",
                        help="only needs to exist for data/full_proteome -- class prompts are "
                             "encoded live, so this doesn't need to exactly match the "
                             "checkpoint's trained text variant, same as pathway_auroc_protein_level.py")
    parser.add_argument("--drop-chloroplast", action="store_true",
                        help="9-way human-only variant: removes 'Chloroplast' from the candidate "
                             "classes, since no human protein can ever have that true label -- "
                             "every Chloroplast prediction in the 10-way run is a guaranteed miss "
                             "that structurally deflates accuracy. Output files get a '_9way_human' suffix.")
    parser.add_argument("--calibrate", action="store_true",
                        help="also report class-wise-centered predictions (subtract each class's "
                             "mean similarity across all test proteins before argmax) alongside the "
                             "uncalibrated ones, to correct for classes that are generically close "
                             "to most proteins regardless of true label. Does not replace the "
                             "uncalibrated results -- both are reported side by side.")
    args = parser.parse_args()
    run(Path(args.checkpoint), biobert_filename=args.biobert_filename,
        drop_chloroplast=args.drop_chloroplast, calibrate=args.calibrate)

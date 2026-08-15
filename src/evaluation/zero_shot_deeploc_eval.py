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

Three prompt conditions (--condition / --all-conditions), run and compared
separately -- mirrors the ablation CLIP's own paper uses to isolate how much
prompt engineering and ensembling each contribute individually on top of a
bare-label baseline: label_only (N=0, bare class name), engineered (N=1,
the first template from the frozen pool below, fixed a priori -- not
validation-selected, unlike CLIP's own single-prompt choice), ensemble
(N=5, default, all 5 templates averaged). Templates come from the frozen
5-prompt pool in data/evaluation/zero_shot_prompt_templates.json (see
generate_zero_shot_prompt_templates.py), generated once via a fixed LLM
meta-prompt before any evaluation was run. {label} is filled with
DEEPLOC_SENTENCE_LABELS, a dedicated sentence-ready phrase per class (not
just CLASSES_10WAY.lower()) -- see that dict's comment for why a few classes
need real rewording rather than de-capitalization.

This is a genuinely different evaluation from pathway_auroc_protein_level.py:
pathway AUROC is a ranking metric (does the model rank pathway members above
non-members), computed one pathway at a time. This is a forced-choice
classification over a small, fixed, mutually-exclusive label set (a protein
has exactly one primary DeepLoc location) -- closer to conventional zero-shot
image classification benchmarks, and a useful sanity check that doesn't
depend on gseapy/KEGG at all.

Loading the checkpoint, BioBERT, and embedding every test protein with the
protein tower is the expensive part of this script -- and none of it
depends on which prompt condition is being scored (only the class/text side
does). --all-conditions therefore loads that context ONCE and runs all
three conditions against it, instead of the ~3x slower approach of invoking
this script three times with --condition and reloading everything each time.

Run src/evaluation/build_deeploc_eval_set.py first to build the human-only,
seen/unseen-labeled test set this script reads.

Usage:
    uv run python src/evaluation/zero_shot_deeploc_eval.py \\
        --checkpoint data/runs/full_proteome/final_hparams_masked_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --drop-chloroplast --all-conditions
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
PROMPT_TEMPLATES_PATH = Path("data/evaluation/zero_shot_prompt_templates.json")
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

# Sentence-ready form of each class name, used to fill {label} in the
# templates loaded from PROMPT_TEMPLATES_PATH. Not just CLASSES_10WAY.lower()
# -- three classes need real rewording, not just de-capitalization:
# "Extracellular" -> "extracellular space" (bare lowercasing reads as an
# adjective with no noun), "Lysosome/Vacuole" -> "lysosome or vacuole" (the
# slash isn't natural language), and "Golgi apparatus" keeps its capital G
# (a proper noun -- Camillo Golgi -- correct mid-sentence too, unlike the
# other, genuinely generic compartment names). All 5 templates already
# supply their own leading "the" (e.g. "The protein is located in the
# {label}."), so these values deliberately exclude the article.
DEEPLOC_SENTENCE_LABELS = {
    "Nucleus": "nucleus",
    "Cytoplasm": "cytoplasm",
    "Extracellular": "extracellular space",
    "Mitochondrion": "mitochondrion",
    "Cell membrane": "cell membrane",
    "Endoplasmic reticulum": "endoplasmic reticulum",
    "Chloroplast": "chloroplast",
    "Golgi apparatus": "Golgi apparatus",
    "Lysosome/Vacuole": "lysosome or vacuole",
    "Peroxisome": "peroxisome",
}

CONDITION_SUFFIXES = {"label_only": "_label_only", "engineered": "_engineered", "ensemble": ""}
ALL_CONDITIONS = ("label_only", "engineered", "ensemble")


def build_class_embeddings(classes, templates, tokenizer, biobert_model, text_head, device, label_only=False):
    """One embedding per class. Default: encode each of the given template
    prompts (each already L2-normalized by ProjectionHead), average, then
    re-normalize -- the standard CLIP prompt-ensembling recipe (Radford et
    al. 2021, section 3.1.4). Averaging without the final re-normalization
    would let classes whose templates happen to agree more end up with
    larger-norm embeddings, which would silently bias the argmax toward them
    (cosine similarity via dot product scales with norm). label_only=True
    instead encodes just the bare class name as a single prompt (the
    "Label-only" / N=0 condition, run_condition() passes an empty
    `templates` list in that case) -- same recipe as zero_shot_ec_eval.py's
    label_only, for direct comparability across the three zero-shot evals.
    Whether `templates` has 1 entry ("Engineered") or all 5 ("Ensemble") is
    decided by the caller, not here.
    """
    class_vecs = {}
    for cls in classes:
        if label_only:
            prompt_vecs = [encode_query(cls, tokenizer, biobert_model, text_head, device)]
        else:
            label = DEEPLOC_SENTENCE_LABELS[cls]
            prompt_vecs = [
                encode_query(t.format(label=label), tokenizer, biobert_model, text_head, device)
                for t in templates
            ]
        averaged = torch.stack(prompt_vecs).mean(dim=0)
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


def load_context(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5",
                  drop_chloroplast: bool = False) -> dict:
    """Loads everything that does NOT depend on --condition: the checkpoint,
    projection heads, BioBERT tokenizer/model, the frozen template pool, and
    -- since the protein side of this eval never depends on the class
    prompt at all -- the protein tower's embeddings for every DeepLoc test
    protein, computed once here. This is the expensive part of the script
    (model loading + embedding thousands of proteins); run_all_conditions()
    calls this once and reuses it for all three conditions instead of
    reloading it 3 times over. drop_chloroplast (which classes to test
    against) is also fixed per context, not per condition."""
    classes = CLASSES_9WAY_HUMAN if drop_chloroplast else CLASSES_10WAY

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

    with open(PROMPT_TEMPLATES_PATH) as f:
        all_templates = json.load(f)["deeploc"]["templates"]

    print("Embedding DeepLoc test proteins with the protein tower (once, shared across all conditions) ...")
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

    return {
        "device": device,
        "checkpoint_path": checkpoint_path,
        "biobert_filename": biobert_filename,
        "classes": classes,
        "drop_chloroplast": drop_chloroplast,
        "eval_df": eval_df,
        "protein_vecs": protein_vecs,
        "tokenizer": tokenizer,
        "biobert_model": biobert_model,
        "text_head": text_head,
        "all_templates": all_templates,
    }


def run_condition(ctx: dict, condition: str, calibrate: bool = False) -> dict:
    """Runs one prompt condition against an already-loaded context (see
    load_context) -- only builds the class-side text embeddings and scores;
    the expensive part (BioBERT/ESM2/checkpoint loading, protein tower
    embeddings) is assumed already done."""
    classes = ctx["classes"]
    drop_chloroplast = ctx["drop_chloroplast"]
    variant_suffix = ("_9way_human" if drop_chloroplast else "") + CONDITION_SUFFIXES[condition]

    all_templates = ctx["all_templates"]
    # "engineered" (N=1) uses template[0] from the frozen pool -- fixed a
    # priori, not validation-selected (unlike CLIP's own single-prompt
    # choice, which they picked by iterating against a held-out set).
    # Deliberately different from CLIP here: performance-based template
    # selection is exactly the kind of leakage this project avoids
    # elsewhere (see CLAUDE.md 11j).
    if condition == "label_only":
        templates_used = []
    elif condition == "engineered":
        templates_used = all_templates[:1]
    else:
        templates_used = all_templates

    prompt_desc = {"label_only": "bare keyword", "engineered": "1 template (engineered, N=1)",
                   "ensemble": f"{len(all_templates)} templates, ensembled"}[condition]
    print(f"\n[{condition}] Embedding the {len(classes)} class prompts ({prompt_desc}) ...")
    class_vecs = build_class_embeddings(classes, templates_used, ctx["tokenizer"], ctx["biobert_model"],
                                         ctx["text_head"], ctx["device"], label_only=(condition == "label_only"))

    eval_df = ctx["eval_df"].copy()
    with torch.no_grad():
        sims = (ctx["protein_vecs"] @ class_vecs.T).cpu().numpy()  # (n_proteins, len(classes))

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

    # build_deeploc_eval_set.py (2026-08-15 revision) now assigns "split" as
    # one of seen / unseen_validation / unseen_test rather than a single
    # combined "unseen" bucket. "unseen" is kept here (validation+test
    # combined) for backward compatibility with earlier reported numbers;
    # "unseen_test" is the new, strict, once-touched figure -- never seen by
    # anything, including the validation-loss-based hyperparameter/
    # architecture selection in 11l/11p -- and is the one to prefer when a
    # clean test-only claim matters (same discipline as validation_only_auroc
    # / test_only_auroc in pathway_auroc_protein_level.py, CLAUDE.md 11j).
    subsets = {
        "all": eval_df,
        "seen": eval_df[eval_df["split"] == "seen"],
        "unseen": eval_df[eval_df["split"] != "seen"],
        "unseen_validation": eval_df[eval_df["split"] == "unseen_validation"],
        "unseen_test": eval_df[eval_df["split"] == "unseen_test"],
    }
    metrics = {name: compute_subset_metrics(subset_df, classes) for name, subset_df in subsets.items()}
    metrics["generalization_gap"] = metrics["seen"]["accuracy"] - metrics["unseen"]["accuracy"]
    metrics["generalization_gap_test_only"] = metrics["seen"]["accuracy"] - metrics["unseen_test"]["accuracy"]
    metrics["checkpoint"] = str(ctx["checkpoint_path"])
    metrics["biobert_filename"] = ctx["biobert_filename"]
    metrics["classes"] = classes
    metrics["condition"] = condition

    if calibrate:
        metrics["calibrated"] = {
            name: compute_subset_metrics(subset_df, classes, pred_col="predicted_class_calibrated")
            for name, subset_df in subsets.items()
        }
        metrics["calibrated"]["generalization_gap"] = (
            metrics["calibrated"]["seen"]["accuracy"] - metrics["calibrated"]["unseen"]["accuracy"]
        )
        metrics["class_mean_similarity"] = dict(zip(classes, class_mean_sim.tolist()))

    checkpoint_stem = ctx["checkpoint_path"].stem
    predictions_path = Path(f"data/evaluation/deeploc_zero_shot_predictions_{checkpoint_stem}{variant_suffix}.csv")
    summary_path = Path(f"data/evaluation/deeploc_zero_shot_summary_{checkpoint_stem}{variant_suffix}.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 88)
    print(f"Zero-shot DeepLoc subcellular localization classification ({len(classes)}-way, "
          f"condition={condition}{', human-only, no Chloroplast' if drop_chloroplast else ''})")
    print("=" * 88)
    for name in ["all", "seen", "unseen", "unseen_validation", "unseen_test"]:
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
    print(f"\nGeneralization gap, uncalibrated, test-only (seen accuracy - unseen_test accuracy): "
          f"{metrics['generalization_gap_test_only']*100:.2f}pp")
    print(f"Generalization gap, uncalibrated (seen accuracy - unseen accuracy): "
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


def run(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5",
        drop_chloroplast: bool = False, calibrate: bool = False, condition: str = "ensemble") -> dict:
    """Single-condition entry point (backward compatible). For all three
    conditions, prefer run_all_conditions() -- it loads the checkpoint/
    BioBERT/protein-tower context once instead of reloading it per call."""
    ctx = load_context(checkpoint_path, biobert_filename, drop_chloroplast)
    return run_condition(ctx, condition, calibrate=calibrate)


def run_all_conditions(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5",
                        drop_chloroplast: bool = False, calibrate: bool = False,
                        conditions=ALL_CONDITIONS) -> dict:
    """Loads the checkpoint/BioBERT/protein-tower context ONCE, then runs
    every given condition against it -- the expensive setup (model loading,
    embedding every test protein) only happens once instead of once per
    condition, since none of it depends on which prompt condition is being
    scored (only the class/text side does)."""
    ctx = load_context(checkpoint_path, biobert_filename, drop_chloroplast)
    return {condition: run_condition(ctx, condition, calibrate=calibrate) for condition in conditions}


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
    parser.add_argument("--condition", choices=["label_only", "engineered", "ensemble"], default="ensemble",
                        help="'label_only' (N=0): bare class name only, e.g. 'Nucleus'. 'engineered' "
                             "(N=1): the first template from the frozen 5-template pool, fixed a priori "
                             "(not validation-selected, unlike CLIP's own single-prompt choice). "
                             "'ensemble' (N=5, default): all 5 templates, embedding-space-averaged -- "
                             "same recipe as zero_shot_ec_eval.py. Ignored if --all-conditions is set.")
    parser.add_argument("--all-conditions", action="store_true",
                        help="run label_only, engineered, and ensemble in one process, loading the "
                             "checkpoint/BioBERT/protein embeddings only once instead of three times "
                             "-- much faster than invoking this script three times with --condition. "
                             "Output files get a matching suffix ('_label_only' / '_engineered' / none).")
    args = parser.parse_args()
    if args.all_conditions:
        run_all_conditions(Path(args.checkpoint), biobert_filename=args.biobert_filename,
                            drop_chloroplast=args.drop_chloroplast, calibrate=args.calibrate)
    else:
        run(Path(args.checkpoint), biobert_filename=args.biobert_filename,
            drop_chloroplast=args.drop_chloroplast, calibrate=args.calibrate, condition=args.condition)

"""
Zero-shot molecular function classification (GO-Slim MF, multi-label scoring).

Usage:
    uv run python src/evaluation/zero_shot_go_mf_eval.py \\
        --checkpoint data/runs/full_proteome/multiview_experiment/multiview_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --all-conditions
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

EVAL_SET_PATH = Path("data/evaluation/zero_shot_classification/go_mf/test_set.csv")
CLASS_SELECTION_PATH = Path("data/evaluation/zero_shot_classification/go_mf/class_selection.json")
PROMPT_TEMPLATES_PATH = Path("data/evaluation/zero_shot_classification/zero_shot_prompt_templates.json")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"

# Templates come from the frozen 5-prompt pool in PROMPT_TEMPLATES_PATH (see
# generate_zero_shot_prompt_templates.py), generated once via a fixed LLM
# meta-prompt before any evaluation was run. Unlike DeepLoc/EC, GO-Slim-MF
# class names ("hydrolase activity", "DNA binding") need no sentence-fitting
# transformation at all -- they are already lowercase, sentence-ready noun
# phrases and are filled into {label} unchanged, checked directly by filling
# all 10 classes into all 5 templates before use.

CONDITION_SUFFIXES = {"label_only": "_label_only", "engineered": "_engineered", "ensemble": ""}
ALL_CONDITIONS = ("label_only", "engineered", "ensemble")


def build_class_embeddings(classes, templates, tokenizer, biobert_model, text_head, device, label_only=False):
    """One embedding per class. Default: encode each of the given template
    prompts (each already L2-normalized by ProjectionHead), average, then
    re-normalize -- the standard CLIP prompt-ensembling recipe (Radford et
    al. 2021, section 3.1.4). Re-normalizing after averaging matters:
    without it, classes whose templates happen to agree more end up with a
    larger-norm embedding, which would silently bias the argmax toward them
    (cosine similarity via dot product scales with norm). label_only=True
    instead encodes just the bare class name as a single prompt (the
    "Label-only" / N=0 condition, run_condition() passes an empty
    `templates` list in that case) -- no averaging needed for a single
    vector, but still re-normalized for consistency. Whether `templates`
    has 1 entry ("Engineered") or all 5 ("Ensemble") is decided by the
    caller, not here."""
    class_vecs = {}
    for cls in classes:
        if label_only:
            prompt_vecs = [encode_query(cls, tokenizer, biobert_model, text_head, device)]
        else:
            prompt_vecs = [
                encode_query(t.format(label=cls), tokenizer, biobert_model, text_head, device)
                for t in templates
            ]
        averaged = torch.stack(prompt_vecs).mean(dim=0)
        class_vecs[cls] = F.normalize(averaged, dim=-1)
    return torch.cat([class_vecs[cls] for cls in classes], dim=0)  # (len(classes), output_dim)


def compute_subset_metrics(df: pd.DataFrame, classes) -> dict:
    """Only lenient accuracy -- a prediction is correct if it is ANY of the
    protein's valid selected classes (all_valid_classes), not just the
    single first-listed one. See module docstring for why there is no
    single-label ("strict") variant here.

    Baselines are chosen to match this scoring rule, not a single-label one:
      - random_baseline_accuracy: a uniformly random guess among len(classes)
        options is expected to land in a protein's valid set with
        probability (that protein's valid-set size) / len(classes) --
        averaged over the subset, this is mean_valid_classes / len(classes).
      - majority_baseline_accuracy: the best possible constant-guess
        strategy under lenient scoring is "always guess whichever single
        class is valid for the most proteins" -- computed directly (not
        assumed), by checking membership frequency for every class and
        taking the max, rather than reusing the primary-term frequency
        (which optimizes a different, single-label notion of "majority").
    """
    y_pred = df["predicted_class"]
    valid_sets = df["all_valid_classes"].fillna("").apply(
        lambda s: set(s.split(";")) if s else set()
    )

    correct = [pred in valid for pred, valid in zip(y_pred, valid_sets)]
    accuracy = sum(correct) / len(df) if len(df) else 0.0
    mean_valid_classes = valid_sets.apply(len).mean() if len(df) else 0.0
    random_baseline_accuracy = mean_valid_classes / len(classes)

    if len(df):
        membership_frac = {cls: valid_sets.apply(lambda v: cls in v).mean() for cls in classes}
        majority_class = max(membership_frac, key=membership_frac.get)
        majority_baseline_accuracy = membership_frac[majority_class]
    else:
        majority_class = None
        majority_baseline_accuracy = 0.0

    return {
        "n": len(df),
        "accuracy": float(accuracy),
        "majority_class": majority_class,
        "majority_baseline_accuracy": float(majority_baseline_accuracy),
        "random_baseline_accuracy": float(random_baseline_accuracy),
        "mean_valid_classes_per_protein": float(mean_valid_classes),
    }


def load_context(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5") -> dict:
    """Loads everything that does NOT depend on --condition: the class
    selection, checkpoint, projection heads, BioBERT tokenizer/model, the
    frozen template pool, and -- since the protein side of this eval never
    depends on the class prompt at all -- the protein tower's embeddings
    for every GO-MF test protein, computed once here. This is the expensive
    part of the script (model loading + embedding thousands of proteins);
    run_all_conditions() calls this once and reuses it for all three
    conditions instead of reloading it 3 times over."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}\n")

    print("Loading GO-Slim-MF class selection ...")
    with open(CLASS_SELECTION_PATH) as f:
        class_selection = json.load(f)
    classes = class_selection["selected_classes"]
    print(f"  {len(classes)} classes, {class_selection['achieved_coverage']:.2%} coverage "
          f"of the full annotated pool: {classes}\n")
    if class_selection.get("unseen_warning"):
        print(f"  {class_selection['unseen_warning']}\n")

    print("Loading GO-Slim-MF test set ...")
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
        all_templates = json.load(f)["go_mf"]["templates"]

    print("Embedding GO-MF test proteins with the protein tower (once, shared across all conditions) ...")
    eval_ensp_ids = set(eval_df["ensp_id"])
    dataset = ProteomicsDataset(
        cohorts=[], use_abundance=False,
        biobert_filename=biobert_filename,
        restrict_to_protein_ids=eval_ensp_ids,
    )
    resolved_ids = dataset.pairs
    missing = eval_ensp_ids - set(resolved_ids)
    if missing:
        print(f"  WARNING: {len(missing)}/{len(eval_ensp_ids)} GO-MF proteins have no "
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
        "class_selection": class_selection,
        "eval_df": eval_df,
        "protein_vecs": protein_vecs,
        "tokenizer": tokenizer,
        "biobert_model": biobert_model,
        "text_head": text_head,
        "all_templates": all_templates,
    }


def run_condition(ctx: dict, condition: str) -> dict:
    """Runs one prompt condition against an already-loaded context (see
    load_context) -- only builds the class-side text embeddings and scores;
    the expensive part (BioBERT/ESM2/checkpoint loading, protein tower
    embeddings) is assumed already done."""
    classes = ctx["classes"]
    class_selection = ctx["class_selection"]
    variant_suffix = CONDITION_SUFFIXES[condition]

    all_templates = ctx["all_templates"]
    # "engineered" (N=1) uses template[0] from the frozen pool -- fixed a
    # priori, not validation-selected (unlike CLIP's own single-prompt
    # choice, which they picked by iterating against a held-out set).
    # Deliberately different from CLIP here: performance-based template
    # selection is exactly the kind of leakage this project avoids
    # elsewhere.
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
    eval_df["correct"] = [
        pred in set(str(valid).split(";"))
        for pred, valid in zip(eval_df["predicted_class"], eval_df["all_valid_classes"])
    ]

    # build_go_mf_eval_set.py (2026-08-15 revision) now assigns "split" as
    # one of seen / unseen_validation / unseen_test rather than a single
    # combined "unseen" bucket. "unseen" is kept here (validation+test
    # combined) for backward compatibility with earlier reported numbers;
    # "unseen_test" is the new, strict, once-touched figure -- never seen by
    # anything, including the validation-loss-based hyperparameter/
    # architecture selection in 11l/11p -- and is the one to prefer when a
    # clean test-only claim matters (same discipline as validation_only_auroc
    # / test_only_auroc in pathway_auroc_protein_level.py).
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
    metrics["achieved_coverage"] = class_selection["achieved_coverage"]
    metrics["unseen_warning"] = class_selection.get("unseen_warning")
    metrics["unseen_test_warning"] = class_selection.get("unseen_test_warning")

    checkpoint_stem = ctx["checkpoint_path"].stem
    predictions_path = Path(f"data/evaluation/zero_shot_classification/go_mf/predictions_{checkpoint_stem}{variant_suffix}.csv")
    summary_path = Path(f"data/evaluation/zero_shot_classification/go_mf/summary_{checkpoint_stem}{variant_suffix}.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 88)
    print(f"Zero-shot GO-Slim-MF classification ({len(classes)}-way, condition={condition}, {prompt_desc})")
    print("Correct = prediction is ANY of the protein's genuinely valid classes")
    print("(not single-label exact match -- see module docstring for why)")
    print("=" * 88)
    for name in ["all", "seen", "unseen", "unseen_validation", "unseen_test"]:
        m = metrics[name]
        print(f"\n[{name}]  n={m['n']}")
        print(f"  accuracy:   {m['accuracy']*100:.2f}%  "
              f"(random baseline: {m['random_baseline_accuracy']*100:.2f}%, "
              f"majority-class baseline ['{m['majority_class']}']: {m['majority_baseline_accuracy']*100:.2f}%)  "
              f"[mean {m['mean_valid_classes_per_protein']:.2f} valid classes/protein]")
    print(f"\nGeneralization gap, test-only (seen accuracy - unseen_test accuracy): "
          f"{metrics['generalization_gap_test_only']*100:.2f}pp")
    print(f"Generalization gap (seen accuracy - unseen accuracy): {metrics['generalization_gap']*100:.2f}pp")
    if metrics["unseen_warning"]:
        print(f"\n{metrics['unseen_warning']}")
    if metrics["unseen_test_warning"]:
        print(f"\n{metrics['unseen_test_warning']}")
    print(f"\nPer-protein predictions saved to {predictions_path}")
    print(f"Summary saved to {summary_path}")

    return metrics


def run(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5", condition: str = "ensemble") -> dict:
    """Single-condition entry point (backward compatible). For all three
    conditions, prefer run_all_conditions() -- it loads the checkpoint/
    BioBERT/protein-tower context once instead of reloading it per call."""
    ctx = load_context(checkpoint_path, biobert_filename)
    return run_condition(ctx, condition)


def run_all_conditions(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5",
                        conditions=ALL_CONDITIONS) -> dict:
    """Loads the checkpoint/BioBERT/protein-tower context ONCE, then runs
    every given condition against it -- the expensive setup (model loading,
    embedding every test protein) only happens once instead of once per
    condition, since none of it depends on which prompt condition is being
    scored (only the class/text side does)."""
    ctx = load_context(checkpoint_path, biobert_filename)
    return {condition: run_condition(ctx, condition) for condition in conditions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/full_proteome/final_hparams_masked_best.pt")
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5",
                        help="only needs to exist for data/full_proteome -- class prompts are "
                             "encoded live, so this doesn't need to exactly match the "
                             "checkpoint's trained text variant, same as pathway_auroc_protein_level.py")
    parser.add_argument("--condition", choices=["label_only", "engineered", "ensemble"], default="ensemble",
                        help="'label_only' (N=0): bare class name only, e.g. 'hydrolase activity'. "
                             "'engineered' (N=1): the first template from the frozen 5-template pool, "
                             "fixed a priori (not validation-selected, unlike CLIP's own single-prompt "
                             "choice). 'ensemble' (N=5, default): all 5 templates, "
                             "embedding-space-averaged -- same recipe as zero_shot_ec_eval.py. Ignored "
                             "if --all-conditions is set.")
    parser.add_argument("--all-conditions", action="store_true",
                        help="run label_only, engineered, and ensemble in one process, loading the "
                             "checkpoint/BioBERT/protein embeddings only once instead of three times "
                             "-- much faster than invoking this script three times with --condition. "
                             "Output files get a matching suffix ('_label_only' / '_engineered' / none).")
    args = parser.parse_args()
    if args.all_conditions:
        run_all_conditions(Path(args.checkpoint), biobert_filename=args.biobert_filename)
    else:
        run(Path(args.checkpoint), biobert_filename=args.biobert_filename, condition=args.condition)

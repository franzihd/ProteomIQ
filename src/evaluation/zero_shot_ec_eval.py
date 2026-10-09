"""
Zero-shot enzyme class classification (EC top level).

Usage:
    uv run python src/evaluation/zero_shot_ec_eval.py \\
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

EVAL_SET_PATH = Path("data/evaluation/zero_shot_classification/ec_top_level/test_set.csv")
PROMPT_TEMPLATES_PATH = Path("data/evaluation/zero_shot_classification/zero_shot_prompt_templates.json")
BIOBERT_PATH = "dmis-lab/biobert-base-cased-v1.2"

# Fixed class order, reused everywhere (prompt building, argmax, confusion
# matrix axes) so results are directly comparable across runs.
CLASSES = [
    "Oxidoreductases", "Transferases", "Hydrolases", "Lyases",
    "Isomerases", "Ligases", "Translocases",
]

CONDITION_SUFFIXES = {"label_only": "_label_only", "engineered": "_engineered", "ensemble": ""}
ALL_CONDITIONS = ("label_only", "engineered", "ensemble")


def ec_sentence_label(cls: str) -> str:
    """Sentence-ready form of a canonical CLASSES entry, used to fill {label}
    in the templates loaded from PROMPT_TEMPLATES_PATH. Lowercasing is the
    only transformation EC needs -- CLASSES' plural forms ("Hydrolases") are
    already correct English nouns once de-capitalized, unlike DeepLoc (see
    DEEPLOC_SENTENCE_LABELS in zero_shot_deeploc_eval.py), which needs real
    per-class rewording. Not used for --condition label_only, which encodes
    the raw canonical CLASSES entry as-is."""
    return cls.lower()


def build_class_embeddings(classes, templates, tokenizer, biobert_model, text_head, device, label_only=False):
    """One embedding per class. Default: encode each of the given template
    prompts (each already L2-normalized by ProjectionHead), average, then
    re-normalize -- the standard CLIP prompt-ensembling recipe (Radford et
    al. 2021, section 3.1.4: "we construct the ensemble over the embedding
    space instead of probability space"). Re-normalizing after averaging
    matters: without it, classes whose templates happen to agree more end up
    with a larger-norm embedding, which would silently bias the argmax
    toward them (cosine similarity via dot product scales with norm).
    label_only=True instead encodes just the bare class name as a single
    prompt (the "Label-only" / N=0 condition, run_condition() passes an
    empty `templates` list in that case) -- no averaging needed for a
    single vector, but still re-normalized for consistency. Whether
    `templates` has 1 entry (the "Engineered" condition) or all 5 (the
    "Ensemble" condition) is decided by the caller, not here -- this
    function just embeds and averages whatever it's given."""
    class_vecs = {}
    for cls in classes:
        if label_only:
            prompt_vecs = [encode_query(cls, tokenizer, biobert_model, text_head, device)]
        else:
            label = ec_sentence_label(cls)
            prompt_vecs = [
                encode_query(t.format(label=label), tokenizer, biobert_model, text_head, device)
                for t in templates
            ]
        averaged = torch.stack(prompt_vecs).mean(dim=0)
        class_vecs[cls] = F.normalize(averaged, dim=-1)
    return torch.cat([class_vecs[cls] for cls in classes], dim=0)  # (len(classes), output_dim)


def compute_subset_metrics(df: pd.DataFrame, classes) -> dict:
    y_true = df["true_class"]
    y_pred = df["predicted_class"]

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


def load_context(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5") -> dict:
    """Loads everything that does NOT depend on --condition: the checkpoint,
    projection heads, BioBERT tokenizer/model, the frozen template pool, and
    -- since the protein side of this eval never depends on the class
    prompt at all -- the protein tower's embeddings for every EC test
    protein, computed once here. This is the expensive part of the script
    (model loading + embedding ~thousands of proteins); run_all_conditions()
    calls this once and reuses it for all three conditions instead of
    reloading it 3 times over."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Classes (7-way EC top-level): {CLASSES}\n")

    print("Loading EC top-level test set ...")
    eval_df = pd.read_csv(EVAL_SET_PATH)
    print(f"  {len(eval_df)} enzymes "
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

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, revision="67c9c25b46986521ca33df05d8540da1210b3256")
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, revision="67c9c25b46986521ca33df05d8540da1210b3256").to(device)
    biobert_model.eval()

    with open(PROMPT_TEMPLATES_PATH) as f:
        all_templates = json.load(f)["ec"]["templates"]

    print("Embedding EC test proteins with the protein tower (once, shared across all conditions) ...")
    eval_ensp_ids = set(eval_df["ensp_id"])
    dataset = ProteomicsDataset(
        cohorts=[], use_abundance=False,
        biobert_filename=biobert_filename,
        restrict_to_protein_ids=eval_ensp_ids,
    )
    resolved_ids = dataset.pairs
    missing = eval_ensp_ids - set(resolved_ids)
    if missing:
        print(f"  WARNING: {len(missing)}/{len(eval_ensp_ids)} EC proteins have no "
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
    variant_suffix = CONDITION_SUFFIXES[condition]
    all_templates = ctx["all_templates"]
    # "engineered" (N=1) uses template[0] from the frozen pool -- fixed a
    # priori, not validation-selected (unlike CLIP's own single-prompt
    # choice for e.g. ImageNet, which they picked by iterating against a
    # held-out set). Deliberately different from CLIP here: performance-based
    # template selection is exactly the kind of leakage this project has
    # avoided elsewhere.
    if condition == "label_only":
        templates_used = []
    elif condition == "engineered":
        templates_used = all_templates[:1]
    else:
        templates_used = all_templates

    prompt_desc = {"label_only": "bare keyword", "engineered": "1 template (engineered, N=1)",
                   "ensemble": f"{len(all_templates)} templates, ensembled"}[condition]
    print(f"\n[{condition}] Embedding the {len(CLASSES)} class prompts ({prompt_desc}) ...")
    class_vecs = build_class_embeddings(CLASSES, templates_used, ctx["tokenizer"], ctx["biobert_model"],
                                         ctx["text_head"], ctx["device"], label_only=(condition == "label_only"))

    eval_df = ctx["eval_df"].copy()
    with torch.no_grad():
        sims = (ctx["protein_vecs"] @ class_vecs.T).cpu().numpy()  # (n_proteins, len(CLASSES))

    predicted_idx = sims.argmax(axis=1)
    eval_df["predicted_class"] = [CLASSES[i] for i in predicted_idx]
    eval_df["correct"] = eval_df["true_class"] == eval_df["predicted_class"]

    # build_ec_eval_set.py (2026-08-15 revision) now assigns "split" as one
    # of seen / unseen_validation / unseen_test rather than a single combined
    # "unseen" bucket. "unseen" is kept here (validation+test combined) for
    # backward compatibility with earlier reported numbers; "unseen_test" is
    # the new, strict, once-touched figure -- never seen by anything,
    # including the validation-loss-based hyperparameter/architecture
    # selection in 11l/11p -- and is the one to prefer when a clean
    # test-only claim matters (same discipline as validation_only_auroc /
    # test_only_auroc in pathway_auroc_protein_level.py).
    subsets = {
        "all": eval_df,
        "seen": eval_df[eval_df["split"] == "seen"],
        "unseen": eval_df[eval_df["split"] != "seen"],
        "unseen_validation": eval_df[eval_df["split"] == "unseen_validation"],
        "unseen_test": eval_df[eval_df["split"] == "unseen_test"],
    }
    metrics = {name: compute_subset_metrics(subset_df, CLASSES) for name, subset_df in subsets.items()}
    metrics["generalization_gap"] = metrics["seen"]["accuracy"] - metrics["unseen"]["accuracy"]
    metrics["generalization_gap_test_only"] = metrics["seen"]["accuracy"] - metrics["unseen_test"]["accuracy"]
    metrics["checkpoint"] = str(ctx["checkpoint_path"])
    metrics["biobert_filename"] = ctx["biobert_filename"]
    metrics["classes"] = CLASSES
    metrics["condition"] = condition

    checkpoint_stem = ctx["checkpoint_path"].stem
    predictions_path = Path(f"data/evaluation/zero_shot_classification/ec_top_level/predictions_{checkpoint_stem}{variant_suffix}.csv")
    summary_path = Path(f"data/evaluation/zero_shot_classification/ec_top_level/summary_{checkpoint_stem}{variant_suffix}.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 88)
    print(f"Zero-shot EC top-level classification (7-way, condition={condition}, {prompt_desc})")
    print("=" * 88)
    for name in ["all", "seen", "unseen", "unseen_validation", "unseen_test"]:
        m = metrics[name]
        print(f"\n[{name}]  n={m['n']}")
        print(f"  accuracy:   {m['accuracy']*100:.2f}%  "
              f"(random baseline: {m['random_baseline_accuracy']*100:.2f}%, "
              f"majority-class baseline ['{m['majority_class']}']: {m['majority_baseline_accuracy']*100:.2f}%)")
        print(f"  macro-F1:   {m['macro_f1']:.4f}")
    print(f"\nGeneralization gap, test-only (seen accuracy - unseen_test accuracy): "
          f"{metrics['generalization_gap_test_only']*100:.2f}pp")
    print(f"Generalization gap (seen accuracy - unseen accuracy): {metrics['generalization_gap']*100:.2f}pp")
    print(f"\nPer-protein predictions saved to {predictions_path}")
    print(f"Summary + confusion matrices saved to {summary_path}")

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
                        help="'label_only' (N=0): bare class name only, e.g. 'Oxidoreductases' -- risks "
                             "inflating accuracy via lexical shortcut-matching against UniProt's own "
                             "annotation vocabulary. 'engineered' (N=1): the first template from the "
                             "frozen 5-template pool, fixed a priori (not validation-selected, unlike "
                             "CLIP's own single-prompt choice). 'ensemble' (N=5, default): all 5 "
                             "templates, embedding-space-averaged. Ignored if --all-conditions is set.")
    parser.add_argument("--all-conditions", action="store_true",
                        help="run label_only, engineered, and ensemble in one process, loading the "
                             "checkpoint/BioBERT/protein embeddings only once instead of three times "
                             "-- much faster than invoking this script three times with --condition. "
                             "Output files get a matching suffix ('_label_only' / '_engineered' / none) "
                             "so all three are kept side by side.")
    args = parser.parse_args()
    if args.all_conditions:
        run_all_conditions(Path(args.checkpoint), biobert_filename=args.biobert_filename)
    else:
        run(Path(args.checkpoint), biobert_filename=args.biobert_filename, condition=args.condition)

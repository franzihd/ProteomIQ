"""
Zero-shot GO Molecular Function (GO-Slim-MF) classification.

Same recipe as zero_shot_deeploc_eval.py / zero_shot_ec_eval.py, applied to a
label set that -- unlike DeepLoc's 10 fixed locations or EC's 7 fixed
top-level numbers -- was itself derived from the data (see
build_go_mf_eval_set.py): the smallest set of GO-Slim-MF classes covering
~75-80% of GO-MF-annotated proteins, selected over the FULL seen+unseen pool
to avoid tailoring the class list to the held-out split. The class list and
per-class training-pool frequency are loaded from
data/evaluation/go_mf_class_selection.json (not hardcoded), since they are a
function of that data-derived selection, not a fixed external taxonomy.

Scoring is deliberately NOT single-label exact-match. Most proteins genuinely
have more than one molecular function (measured directly: ~33% of the test
set has more than one of the 10 selected classes among their real GO-MF
terms, see build_go_mf_eval_set.py's Step 6b) -- unlike DeepLoc's
subcellular location (largely single-valued biologically) or EC's top-level
class (multi-class enzymes are explicitly dropped as ambiguous at eval-set
build time). Forcing a single "correct" answer here would penalize
biologically valid predictions just because they weren't UniProt's
first-listed term for that protein. So a prediction is scored correct if it
falls anywhere in the protein's full set of valid selected classes
(all_valid_classes, built in build_go_mf_eval_set.py). There is no separate
"strict" single-label accuracy/macro-F1/confusion-matrix in this script --
they were tried and dropped as not meaningful for this label set.

Two prompt variants, run and compared separately rather than merged into one
ensemble (same pattern as zero_shot_ec_eval.py's --keyword-only):
  - default: 3 independently-phrased template prompts per class, averaged +
    re-normalized ("prompt ensembling")
  - --keyword-only: a single bare class-name prompt per class (e.g.
    "hydrolase activity"), no sentence wrapping

Run src/evaluation/build_go_mf_eval_set.py first to build the test set and
class selection file this script reads.

Usage:
    uv run python src/evaluation/zero_shot_go_mf_eval.py \\
        --checkpoint data/runs/full_proteome/final_hparams_masked_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5
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

EVAL_SET_PATH = Path("data/evaluation/go_mf_test_set.csv")
CLASS_SELECTION_PATH = Path("data/evaluation/go_mf_class_selection.json")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"

# 3 independent phrasings per class ("prompt ensembling"), same idea as
# zero_shot_deeploc_eval.py / zero_shot_ec_eval.py's TEMPLATES. Deliberately
# NOT "... has {c} activity." as in the original spec -- most GO-Slim-MF
# class names already end in "activity" or "binding" (e.g. "hydrolase
# activity", "DNA binding"), so appending "activity" would literally double
# it ("hydrolase activity activity."). These three avoid that while keeping
# the same spirit (a factual sentence, an "involved in" phrasing, and a bare
# labeled form). Only used when keyword_only=False.
TEMPLATES = [
    "This protein has {c}.",
    "This protein is involved in {c}.",
    "Molecular function: {c}.",
]


def build_class_embeddings(classes, tokenizer, biobert_model, text_head, device, keyword_only=False):
    """One embedding per class. Default: encode the 3 template prompts (each
    already L2-normalized by ProjectionHead), average, then re-normalize --
    the standard CLIP prompt-ensembling recipe. Re-normalizing after
    averaging matters: without it, classes whose templates happen to agree
    more end up with a larger-norm embedding, which would silently bias the
    argmax toward them (cosine similarity via dot product scales with norm).
    keyword_only=True instead encodes just the bare class name as a single
    prompt -- no averaging needed for a single vector, but still
    re-normalized for consistency."""
    class_vecs = {}
    for cls in classes:
        if keyword_only:
            prompt_vecs = [encode_query(cls, tokenizer, biobert_model, text_head, device)]
        else:
            prompt_vecs = [
                encode_query(t.format(c=cls), tokenizer, biobert_model, text_head, device)
                for t in TEMPLATES
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


def run(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5", keyword_only: bool = False):
    variant_suffix = "_keyword_only" if keyword_only else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt variant: {'bare keyword only' if keyword_only else '3-template explanatory ensemble'}\n")

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

    prompt_desc = "bare keyword" if keyword_only else "3 templates each, ensembled"
    print(f"Embedding the {len(classes)} class prompts ({prompt_desc}) ...")
    class_vecs = build_class_embeddings(classes, tokenizer, biobert_model, text_head, device, keyword_only=keyword_only)

    print("Embedding GO-MF test proteins with the protein tower ...")
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
        sims = (protein_vecs @ class_vecs.T).cpu().numpy()  # (n_proteins, len(classes))

    predicted_idx = sims.argmax(axis=1)
    eval_df["predicted_class"] = [classes[i] for i in predicted_idx]
    eval_df["correct"] = [
        pred in set(str(valid).split(";"))
        for pred, valid in zip(eval_df["predicted_class"], eval_df["all_valid_classes"])
    ]

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
    metrics["keyword_only"] = keyword_only
    metrics["achieved_coverage"] = class_selection["achieved_coverage"]
    metrics["unseen_warning"] = class_selection.get("unseen_warning")

    predictions_path = Path(f"data/evaluation/go_mf_zero_shot_predictions_{checkpoint_path.stem}{variant_suffix}.csv")
    summary_path = Path(f"data/evaluation/go_mf_zero_shot_summary_{checkpoint_path.stem}{variant_suffix}.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 88)
    print(f"Zero-shot GO-Slim-MF classification ({len(classes)}-way, {prompt_desc})")
    print("Correct = prediction is ANY of the protein's genuinely valid classes")
    print("(not single-label exact match -- see module docstring for why)")
    print("=" * 88)
    for name in ["all", "seen", "unseen"]:
        m = metrics[name]
        print(f"\n[{name}]  n={m['n']}")
        print(f"  accuracy:   {m['accuracy']*100:.2f}%  "
              f"(random baseline: {m['random_baseline_accuracy']*100:.2f}%, "
              f"majority-class baseline ['{m['majority_class']}']: {m['majority_baseline_accuracy']*100:.2f}%)  "
              f"[mean {m['mean_valid_classes_per_protein']:.2f} valid classes/protein]")
    print(f"\nGeneralization gap (seen accuracy - unseen accuracy): {metrics['generalization_gap']*100:.2f}pp")
    if metrics["unseen_warning"]:
        print(f"\n{metrics['unseen_warning']}")
    print(f"\nPer-protein predictions saved to {predictions_path}")
    print(f"Summary saved to {summary_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/full_proteome/final_hparams_masked_best.pt")
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5",
                        help="only needs to exist for data/full_proteome -- class prompts are "
                             "encoded live, so this doesn't need to exactly match the "
                             "checkpoint's trained text variant, same as pathway_auroc_protein_level.py")
    parser.add_argument("--keyword-only", action="store_true",
                        help="use a single bare class-name prompt per class (e.g. 'hydrolase "
                             "activity') instead of the default 3-template explanatory ensemble. "
                             "Run both and compare -- output files get a '_keyword_only' suffix "
                             "so both variants' results are kept side by side, same as "
                             "zero_shot_ec_eval.py.")
    args = parser.parse_args()
    run(Path(args.checkpoint), biobert_filename=args.biobert_filename, keyword_only=args.keyword_only)

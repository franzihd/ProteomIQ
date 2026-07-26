"""
Zero-shot Enzyme Commission (EC) top-level-class classification.

Same recipe as zero_shot_deeploc_eval.py, applied to a different label set:
the model is never trained on EC labels. Each of the 7 EC top-level classes
(Oxidoreductases, Transferases, Hydrolases, Lyases, Isomerases, Ligases,
Translocases) is turned into a text prompt, embedded with our text tower,
into one class embedding. Each test protein (already filtered to enzymes
only -- see build_ec_eval_set.py) is embedded with our protein tower and
assigned to whichever class embedding it is most cosine-similar to.

Two prompt variants, run and compared separately rather than merged into
one ensemble:
  - default: 3 explanatory natural-language prompts per class, averaged +
    re-normalized ("prompt ensembling", same as zero_shot_deeploc_eval.py)
  - --keyword-only: a single bare class-name prompt per class (e.g.
    "Oxidoreductases"), no sentence wrapping

Comparing the two matters here more than usual: EC class names double as
the exact vocabulary UniProt's own FUNCTION annotations use to describe
catalytic activity (e.g. "...has serine hydrolase activity"), so
keyword-only risks scoring well via lexical shortcut-matching rather than
genuine understanding -- the "name shortcut" pattern documented throughout
CLAUDE.md for pathway/gene names (see e.g. section 8e's raw-name vs.
explanatory-NL pathway query comparison). Running both variants, rather
than picking one, is how this project has consistently handled that risk.

Run src/evaluation/build_ec_eval_set.py first to build the test set this
script reads.

Usage:
    uv run python src/evaluation/zero_shot_ec_eval.py \\
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel

from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead
from src.model.query_evaluation import encode_query

EVAL_SET_PATH = Path("data/evaluation/ec_top_level_test_set.csv")
BIOBERT_PATH = "/home/franziska/.cache/huggingface/hub/models--dmis-lab--biobert-base-cased-v1.2/snapshots/67c9c25b46986521ca33df05d8540da1210b3256"

# Fixed class order, reused everywhere (prompt building, argmax, confusion
# matrix axes) so results are directly comparable across runs.
CLASSES = [
    "Oxidoreductases", "Transferases", "Hydrolases", "Lyases",
    "Isomerases", "Ligases", "Translocases",
]

# A short description of what each EC top-level class does, filled into the
# 3 templates below -- these are the standard IUBMB definitions, kept short
# and gene/example-free so the model can't shortcut on protein names.
EC_CLASS_PHRASES = {
    "Oxidoreductases": "catalyzes oxidation-reduction reactions by transferring electrons, hydrogen, or oxygen between molecules",
    "Transferases": "catalyzes the transfer of a functional group, such as a methyl or phosphate group, from one molecule to another",
    "Hydrolases": "catalyzes the hydrolysis of a chemical bond by adding a water molecule",
    "Lyases": "catalyzes the breaking of a chemical bond by a means other than hydrolysis or oxidation, often forming a double bond or a ring",
    "Isomerases": "catalyzes the rearrangement of atoms within a single molecule to form a structural isomer",
    "Ligases": "catalyzes the joining of two molecules together, typically coupled to the hydrolysis of ATP",
    "Translocases": "catalyzes the movement of ions or molecules across a membrane, or their separation within a membrane",
}

# 3 independent phrasings per class ("prompt ensembling"), applied uniformly
# across all classes via {phrase} -- same idea and template shape as
# zero_shot_deeploc_eval.py's TEMPLATES. Only used when keyword_only=False.
TEMPLATES = [
    "An enzyme that {phrase}.",
    "This protein is an enzyme; it {phrase}.",
    "Enzymatic activity: {phrase}.",
]


def build_class_embeddings(classes, tokenizer, biobert_model, text_head, device, keyword_only=False):
    """One embedding per class. Default: encode the 3 explanatory template
    prompts (each already L2-normalized by ProjectionHead), average, then
    re-normalize -- the standard CLIP prompt-ensembling recipe.
    Re-normalizing after averaging matters: without it, classes whose
    templates happen to agree more end up with a larger-norm embedding,
    which would silently bias the argmax toward them (cosine similarity via
    dot product scales with norm). keyword_only=True instead encodes just
    the bare class name as a single prompt -- no averaging needed for a
    single vector, but still re-normalized for consistency."""
    class_vecs = {}
    for cls in classes:
        if keyword_only:
            prompt_vecs = [encode_query(cls, tokenizer, biobert_model, text_head, device)]
        else:
            phrase = EC_CLASS_PHRASES[cls]
            prompt_vecs = [
                encode_query(t.format(phrase=phrase), tokenizer, biobert_model, text_head, device)
                for t in TEMPLATES
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


def run(checkpoint_path: Path, biobert_filename: str = "biobert_embeddings_masked.h5", keyword_only: bool = False):
    variant_suffix = "_keyword_only" if keyword_only else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompt variant: {'bare keyword only' if keyword_only else '3-template explanatory ensemble'}")
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

    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, local_files_only=True)
    biobert_model = AutoModel.from_pretrained(BIOBERT_PATH, local_files_only=True).to(device)
    biobert_model.eval()

    prompt_desc = "bare keyword" if keyword_only else "3 templates each, ensembled"
    print(f"Embedding the {len(CLASSES)} class prompts ({prompt_desc}) ...")
    class_vecs = build_class_embeddings(CLASSES, tokenizer, biobert_model, text_head, device, keyword_only=keyword_only)

    print("Embedding EC test proteins with the protein tower ...")
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
        sims = (protein_vecs @ class_vecs.T).cpu().numpy()  # (n_proteins, len(CLASSES))

    predicted_idx = sims.argmax(axis=1)
    eval_df["predicted_class"] = [CLASSES[i] for i in predicted_idx]
    eval_df["correct"] = eval_df["true_class"] == eval_df["predicted_class"]

    print("Computing metrics for seen / unseen / all ...\n")
    subsets = {
        "all": eval_df,
        "seen": eval_df[eval_df["split"] == "seen"],
        "unseen": eval_df[eval_df["split"] == "unseen"],
    }
    metrics = {name: compute_subset_metrics(subset_df, CLASSES) for name, subset_df in subsets.items()}
    metrics["generalization_gap"] = metrics["seen"]["accuracy"] - metrics["unseen"]["accuracy"]
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["biobert_filename"] = biobert_filename
    metrics["classes"] = CLASSES
    metrics["keyword_only"] = keyword_only

    predictions_path = Path(f"data/evaluation/ec_zero_shot_predictions_{checkpoint_path.stem}{variant_suffix}.csv")
    summary_path = Path(f"data/evaluation/ec_zero_shot_summary_{checkpoint_path.stem}{variant_suffix}.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 88)
    print(f"Zero-shot EC top-level classification (7-way, {prompt_desc})")
    print("=" * 88)
    for name in ["all", "seen", "unseen"]:
        m = metrics[name]
        print(f"\n[{name}]  n={m['n']}")
        print(f"  accuracy:   {m['accuracy']*100:.2f}%  "
              f"(random baseline: {m['random_baseline_accuracy']*100:.2f}%, "
              f"majority-class baseline ['{m['majority_class']}']: {m['majority_baseline_accuracy']*100:.2f}%)")
        print(f"  macro-F1:   {m['macro_f1']:.4f}")
    print(f"\nGeneralization gap (seen accuracy - unseen accuracy): {metrics['generalization_gap']*100:.2f}pp")
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
    parser.add_argument("--keyword-only", action="store_true",
                        help="use a single bare class-name prompt per class (e.g. 'Oxidoreductases') "
                             "instead of the default 3-template explanatory ensemble. Run both and "
                             "compare -- keyword-only risks inflating accuracy via lexical shortcut-"
                             "matching against UniProt's own annotation vocabulary. Output files get "
                             "a '_keyword_only' suffix so both variants' results are kept side by side.")
    args = parser.parse_args()
    run(Path(args.checkpoint), biobert_filename=args.biobert_filename, keyword_only=args.keyword_only)

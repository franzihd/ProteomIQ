"""
Alignment & Uniformity (Wang & Isola, ICML 2020, "Understanding Contrastive
Representation Learning through Alignment and Uniformity on the Hypersphere")
-- a task-agnostic pair of embedding-space-quality metrics, computed from
embeddings alone, with NO labeled downstream task and NO topic-specific
ground truth at all. Complementary to, not a replacement for, the five
topic-specific zero-shot evals (cross-modal, DeepLoc, EC, GO-MF, GO-retrieval)
-- those measure "can this space answer THIS specific kind of question,"
this measures "is the space itself well-formed" independent of any one
question. See CLAUDE.md for the full discussion of why this was added and
how it differs from (and partly overlaps with) cross_modal_eval.py's R@K.

Both projection heads L2-normalize their output (ProjectionHead.forward),
so every embedding already lives on the unit hypersphere -- "alignment" and
"uniformity" describe how points are arranged on that sphere.

Alignment: average squared distance between TRUE (protein, matching text)
pairs. For unit vectors, ||x-y||^2 = 2 - 2*cos_sim(x,y) -- this is exactly
the same true-pair cosine similarity infonce_loss already tries to push up,
reported on its own as one plain number, with no batch/negative machinery
around it. LOWER is better (true matches sit close together).

Uniformity: log(mean over all i != j pairs of exp(-t * ||x_i - x_j||^2)),
t=2 (the paper's standard choice). Catches a failure mode alignment alone
cannot: if every embedding collapsed to one point, alignment would look
PERFECT (distance 0 for every true pair) while the space became useless
(nothing distinguishable from anything else). Uniformity measures how
evenly points are spread across the whole sphere -- LOWER (more negative)
is better (points spread apart, not clumped).

FIXED TEXT SOURCE FOR ALL CHECKPOINTS, deliberately: computed using the
single fixed --biobert-filename (default: masked Function-only text) for
every checkpoint being compared, regardless of whether that checkpoint was
itself trained with view_biobert_filenames/multi_positive. Same principle
train_protein_level.py already applies to validation loss -- comparability
across different training setups requires holding the evaluation input
fixed, otherwise a checkpoint trained with more text per protein could look
"better" simply from having more/different data at eval time, not from
having a better space.

VALIDATION SPLIT, not test: this is a comparison tool for choosing between
checkpoints (like validation_only_auroc in pathway_auroc_protein_level.py),
not the final, one-time confirmatory check -- test stays reserved per the
standing rule (see CLAUDE.md 11ab) until a final checkpoint is locked in.

Usage:
    uv run python src/evaluation/alignment_uniformity_eval.py \\
        --checkpoint data/runs/full_proteome/multiview_experiment/multiview_best.pt \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --splits-file data/protein_splits_full_proteome.json
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.model.dataset import ProteomicsDataset
from src.model.projection_head import ProjectionHead

UNIFORMITY_T = 2.0  # paper's standard choice, fixed here (same treatment as
                     # infonce_loss's hardcoded temperature=0.07 elsewhere)
ALIGNMENT_ALPHA = 2  # paper's standard choice (squared distance)


def alignment_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = ALIGNMENT_ALPHA) -> float:
    """x, y: (n, d) matched pairs (x[i] should be close to y[i]). Lower is better."""
    return (x - y).norm(p=2, dim=1).pow(alpha).mean().item()


def uniformity_loss(x: torch.Tensor, t: float = UNIFORMITY_T) -> float:
    """x: (n, d), any set of embeddings (not necessarily paired with anything).
    All i != j pairs via torch.pdist (upper-triangle only, but that's fine --
    the mean over i<j equals the mean over i!=j by symmetry of distance).
    Lower (more negative) is better."""
    sq_dists = torch.pdist(x, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log().item()


def compute_alignment_uniformity(checkpoint_path: str, biobert_filename: str,
                                  splits_file: str = "data/protein_splits_full_proteome.json"):
    with open(splits_file) as f:
        protein_splits = json.load(f)
    validation_protein_ids = set(protein_splits["validation"])

    # Single fixed text source for every checkpoint -- see module docstring.
    # `cohorts` is unused on the use_abundance=False path (always reads
    # data/full_proteome/), kept only for call-site consistency with the
    # ablation-baseline path, same as cross_modal_eval.py.
    dataset = ProteomicsDataset(
        cohorts=[], use_abundance=False,
        biobert_filename=biobert_filename,
        restrict_to_protein_ids=validation_protein_ids,
    )
    protein_ids = dataset.pairs
    esm2_matrix = torch.stack([dataset.protein_esm2[pid] for pid in protein_ids])
    text_matrix = torch.stack([dataset.protein_text[pid] for pid in protein_ids])
    print(f"Validation pool: {len(protein_ids):,} proteins "
          f"(expected {len(validation_protein_ids):,} minus any with no usable text/ESM2)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 512)
    output_dim = ckpt.get("output_dim", 256)
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    protein_head.load_state_dict(ckpt["protein_head"])
    text_head.load_state_dict(ckpt["text_head"])
    protein_head.eval()
    text_head.eval()

    with torch.no_grad():
        protein_vecs = protein_head(esm2_matrix.to(device))
        text_vecs = text_head(text_matrix.to(device))

        align = alignment_loss(protein_vecs, text_vecs)
        uniform_protein = uniformity_loss(protein_vecs)
        uniform_text = uniformity_loss(text_vecs)
        # Pooled: proteins and texts together, on the SAME shared sphere --
        # checks whether the two modalities occupy the space jointly in a
        # spread-out way, not just each internally.
        uniform_pooled = uniformity_loss(torch.cat([protein_vecs, text_vecs], dim=0))

        # --- Reference points, same "don't trust a raw number, compare to a
        # baseline" discipline used for every other eval in this project. ---
        # Best-case uniformity: N genuinely random points, uniformly
        # distributed on the SAME d-dimensional hypersphere (normalized
        # Gaussian samples -- a standard, exact way to sample uniformly on a
        # hypersphere). Averaged over several draws for a stable estimate,
        # since a single random draw is itself noisy.
        n, d = protein_vecs.shape
        random_uniformities = []
        for _ in range(5):
            random_points = torch.nn.functional.normalize(torch.randn(n, d, device=device), dim=1)
            random_uniformities.append(uniformity_loss(random_points))
        uniform_random_reference = sum(random_uniformities) / len(random_uniformities)

        # Worst-case alignment reference: shuffle the true text_vecs so each
        # protein is paired with a RANDOM (wrong) protein's text -- the
        # "how far apart do MISMATCHED pairs sit" baseline. True-pair
        # alignment should be well below this for the space to be doing
        # anything at all (same role as random_baseline_accuracy elsewhere).
        shuffled_idx = torch.randperm(n, device=device)
        align_mismatched_reference = alignment_loss(protein_vecs, text_vecs[shuffled_idx])

    return {
        "pool_size": len(protein_ids),
        "alignment": align,
        "alignment_mismatched_reference": align_mismatched_reference,
        "uniformity_protein": uniform_protein,
        "uniformity_text": uniform_text,
        "uniformity_pooled": uniform_pooled,
        "uniformity_random_reference": uniform_random_reference,
        "checkpoint": str(checkpoint_path),
        "biobert_filename": biobert_filename,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. data/runs/full_proteome/multiview_experiment/multiview_best.pt")
    parser.add_argument("--biobert-filename", default="biobert_embeddings_masked.h5",
                        help="FIXED for every checkpoint compared -- default is the masked "
                             "Function-only text, same convention validation loss already uses. "
                             "Do not vary this per checkpoint when comparing results.")
    parser.add_argument("--splits-file", default="data/protein_splits_full_proteome.json")
    args = parser.parse_args()

    metrics = compute_alignment_uniformity(args.checkpoint, args.biobert_filename, args.splits_file)

    print(f"\nPool size: {metrics['pool_size']:,}")
    print(f"Alignment (lower = better, true pairs closer):        {metrics['alignment']:.4f}")
    print(f"  -- reference, mismatched (wrong) pairs:              {metrics['alignment_mismatched_reference']:.4f}")
    print(f"Uniformity, protein-side (lower = better, more spread): {metrics['uniformity_protein']:.4f}")
    print(f"Uniformity, text-side (lower = better, more spread):    {metrics['uniformity_text']:.4f}")
    print(f"Uniformity, pooled protein+text (lower = better):       {metrics['uniformity_pooled']:.4f}")
    print(f"  -- reference, genuinely random points (best case):   {metrics['uniformity_random_reference']:.4f}")

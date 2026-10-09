"""
Training-data-size ablation for the final multiview configuration (validation only).

Question: does validation performance keep improving as the number of training
proteins grows, or does it already saturate within the ~13,500-protein human
training set? Either outcome is allowed -- nothing here assumes more data helps.

Design (see the methodology notes printed by --plan, and CLAUDE.md):
  - Fractions 25/50/75/100% of the EXISTING training split. Validation and test
    are untouched. Subsets are nested (25 ⊂ 50 ⊂ 75 ⊂ 100): the sorted training
    pool is shuffled ONCE with --subset-seed and each fraction takes a prefix.
  - Subsetting reuses train()'s existing held_out_protein_ids argument: the
    unused part of the training pool is simply passed as additional held-out
    proteins, so train_protein_level.py itself is not modified.
  - Training randomness (weight init, DataLoader shuffle, multiview view draws)
    is varied per run via --seeds, by setting train_protein_level.SEED and
    dataset.MULTI_VIEW_SEED before each call. Both are read at call/construction
    time, so this changes nothing for any other caller. The subset seed is a
    separate, fixed value -- repeated runs at one fraction see the same proteins.
  - Constant optimizer-step budget: the final multiview run used epochs=100 at
    27 steps/epoch (13,515 proteins / batch 512) = 2,700 steps. Each fraction
    gets epochs = round(target_steps / steps_per_epoch), so every run sees
    ~2,700 gradient updates and the identical warmup/cosine-decay shape
    (scheduler length is defined in steps).
  - Early stopping is disabled (the final run stopped at 92/100 epochs, i.e.
    almost the full schedule): patience is counted in epochs, which would mean
    a 4x shorter patience in steps at 25%. Every run trains the full step
    budget; the evaluated checkpoint is the lowest-val_loss one (*_best.pt),
    the same selection rule as every other checkpoint in this project.
  - Evaluation: cross-modal retrieval on the validation pool (both directions,
    R@1/5/10/50 + median rank), GO-term retrieval (all 3 branches,
    min_positives=10, validation pool) and zero-shot DeepLoc (9-way human) / EC /
    GO-Slim-MF on unseen_validation only, ensemble condition. The test split is
    never read for scoring.

Outputs (kept OUT of data/evaluation/{cross_modal_retrieval,zero_shot_classification}
so aggregate_results.py / all_results.csv are not affected):
  data/runs/full_proteome/datasize_experiment/          checkpoints + subsets.json + train logs
  data/evaluation/datasize_experiment/raw_runs.csv       one row per (fraction, seed)
  data/evaluation/datasize_experiment/aggregated.csv     mean/std per fraction
  data/evaluation/datasize_experiment/zero_shot/<task>/  per-run zero-shot predictions/summaries
  data/evaluation/datasize_experiment/go_retrieval/      per-run GO-retrieval predictions/summaries

Usage:
    uv run python src/model/datasize_ablation.py                 # train missing runs + evaluate all
    uv run python src/model/datasize_ablation.py --plan          # print subset sizes / epochs, do nothing
"""
import sys
import json
import math
import random
import shutil
import argparse
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import pandas as pd

import src.model.dataset as dataset_module
import src.model.train_protein_level as train_module

SPLITS_FILE = "data/protein_splits_full_proteome.json"
VAL_BIOBERT = "biobert_embeddings_masked.h5"
VIEW_BIOBERTS = ["biobert_embeddings_masked.h5",
                 "biobert_embeddings_family_view.h5",
                 "biobert_embeddings_location_view.h5"]
# Final multiview configuration (read from multiview_experiment/multiview_best.pt's metadata).
FINAL_CONFIG = dict(lr=0.00011598207492707353, batch_size=512, hidden_dim=2048, output_dim=1024)
FINAL_EPOCHS = 100

RUN_DIR = Path("data/runs/full_proteome/datasize_experiment")
EVAL_DIR = Path("data/evaluation/datasize_experiment")
ZERO_SHOT_SRC = Path("data/evaluation/zero_shot_classification")
GO_RETRIEVAL_SRC = Path("data/evaluation/go_retrieval/results")
GO_RETRIEVAL_MIN_POSITIVES = 10  # same threshold as every other go_retrieval run


def training_pool() -> list[str]:
    """Same protein set ProteomicsDataset's multiview branch trains on:
    ESM2 ∩ union(view files) − validation − test, sorted."""
    data_dir = Path("data/full_proteome")
    with h5py.File(data_dir / "esm2_embeddings.h5", "r") as f:
        esm2_ids = {p.decode() for p in f["protein_ids"][:]}
    view_ids = set()
    for fname in VIEW_BIOBERTS:
        with h5py.File(data_dir / fname, "r") as f:
            view_ids |= {p.decode() for p in f["protein_ids"][:]}
    splits = json.load(open(SPLITS_FILE))
    held_out = set(splits["validation"]) | set(splits["test"])
    return sorted((esm2_ids & view_ids) - held_out), held_out


def nested_subsets(pool: list[str], fractions: list[float], subset_seed: int) -> dict:
    order = list(pool)
    random.Random(subset_seed).shuffle(order)
    subsets = {f: sorted(order[:round(f * len(order))]) for f in fractions}
    ordered = sorted(fractions)
    for small, large in zip(ordered, ordered[1:]):
        assert set(subsets[small]) <= set(subsets[large]), "subsets must be nested"
    return subsets


def run_name(fraction: float, seed: int) -> str:
    return f"datasize_f{round(fraction * 100):03d}_seed{seed}"


def train_one(fraction, seed, subset, pool, base_held_out, epochs):
    ckpt = RUN_DIR / f"{run_name(fraction, seed)}.pt"
    best = ckpt.with_name(ckpt.stem + "_best.pt")
    if best.exists() and ckpt.exists():
        print(f"[skip] {best} already exists")
        return best
    held_out = base_held_out | (set(pool) - set(subset))
    train_module.SEED = seed
    dataset_module.MULTI_VIEW_SEED = seed
    log_path = RUN_DIR / f"{run_name(fraction, seed)}_train_log.txt"
    print(f"[train] fraction={fraction} seed={seed} proteins={len(subset):,} epochs={epochs} -> {log_path}")
    with open(log_path, "w") as log, contextlib.redirect_stdout(log):
        _train_with_held_out(str(ckpt), epochs, held_out)
    return best


def _train_with_held_out(ckpt_path: str, epochs: int, held_out: set[str]):
    """train() builds held_out_protein_ids from the splits file itself (val ∪ test).
    To subsample without touching train_protein_level.py, wrap ProteomicsDataset
    so the TRAINING dataset (the one built with held_out_protein_ids) also holds
    out the unused part of the training pool. The validation dataset (built with
    restrict_to_protein_ids) is passed through unchanged."""
    original_cls = train_module.ProteomicsDataset

    def subset_dataset(*args, **kwargs):
        if kwargs.get("held_out_protein_ids") is not None:
            kwargs["held_out_protein_ids"] = held_out
        return original_cls(*args, **kwargs)

    train_module.ProteomicsDataset = subset_dataset
    try:
        return train_module.train(
            checkpoint_path=ckpt_path, epochs=epochs,
            biobert_filename=VAL_BIOBERT, splits_file=SPLITS_FILE,
            view_biobert_filenames=VIEW_BIOBERTS, early_stopping_patience=None,
            **FINAL_CONFIG,
        )
    finally:
        train_module.ProteomicsDataset = original_cls


def evaluate_one(best_ckpt: Path, skip_zero_shot: bool) -> dict:
    import torch
    from src.evaluation.cross_modal_eval import compute_cross_modal_eval
    meta = torch.load(best_ckpt, map_location="cpu")
    row = {"best_epoch": meta["epoch"], "total_epochs": meta["total_epochs"],
           "best_val_loss": meta["val_loss"]}

    cm = compute_cross_modal_eval(str(best_ckpt), VAL_BIOBERT, SPLITS_FILE, split_key="validation")
    for direction, prefix in [("text_to_protein", "t2p"), ("protein_to_text", "p2t")]:
        for k, v in cm[direction].items():
            row[f"{prefix}_{k}"] = v
    row["val_pool_size"] = cm["pool_size"]

    if skip_zero_shot:
        return row

    row.update(evaluate_go_retrieval(best_ckpt))

    from src.evaluation import zero_shot_deeploc_eval, zero_shot_ec_eval, zero_shot_go_mf_eval
    tasks = [
        ("deeploc", zero_shot_deeploc_eval, dict(drop_chloroplast=True), "_9way_human"),
        ("ec_top_level", zero_shot_ec_eval, {}, ""),
        ("go_mf", zero_shot_go_mf_eval, {}, ""),
    ]
    for task, module, ctx_kwargs, suffix in tasks:
        ctx = module.load_context(best_ckpt, VAL_BIOBERT, **ctx_kwargs)
        stem = best_ckpt.stem
        try:
            metrics = module.run_condition(ctx, "ensemble")
        finally:
            # run_condition writes into the main eval dirs; move this run's files
            # out immediately so aggregate_results.py never sees them.
            dest = EVAL_DIR / "zero_shot" / task
            dest.mkdir(parents=True, exist_ok=True)
            for kind, ext in [("predictions", "csv"), ("summary", "json")]:
                src = ZERO_SHOT_SRC / task / f"{kind}_{stem}{suffix}.{ext}"
                if src.exists():
                    shutil.move(str(src), dest / src.name)
        uv = metrics["unseen_validation"]
        row[f"{task}_acc"] = uv["accuracy"] * 100
        row[f"{task}_n"] = uv["n"]
        if "macro_f1" in uv:
            row[f"{task}_macro_f1"] = uv["macro_f1"]
        del ctx
        torch.cuda.empty_cache()
    return row


def evaluate_go_retrieval(best_ckpt: Path) -> dict:
    """GO-term zero-shot retrieval on the validation pool, all 3 branches x both
    query styles. go_mean_map = mean of the 6 mAPs, the same headline number
    aggregate_results.py reports as go_retrieval_mean_mAP."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # its sibling imports are flat
    import zero_shot_go_retrieval_eval as go_eval
    try:
        summaries = go_eval.run(best_ckpt, "all", GO_RETRIEVAL_MIN_POSITIVES, [10, 50, 100],
                                biobert_filename=VAL_BIOBERT, split_key="validation")
    finally:
        # Same as the zero-shot evals: move this run's files out of the main
        # results dir so aggregate_results.py never sees them.
        dest = EVAL_DIR / "go_retrieval"
        dest.mkdir(parents=True, exist_ok=True)
        for src in GO_RETRIEVAL_SRC.glob(f"*_{best_ckpt.stem}_min{GO_RETRIEVAL_MIN_POSITIVES}.*"):
            shutil.move(str(src), dest / src.name)
    row, maps = {}, []
    for branch, summary in summaries.items():
        for qt, qd in summary["query_types"].items():
            row[f"go_{branch}_{qt}_map"] = qd["mAP"]
            maps.append(qd["mAP"])
    row["go_mean_map"] = sum(maps) / len(maps)
    torch.cuda.empty_cache()
    return row


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in raw.columns if c not in ("fraction", "seed", "n_train_proteins", "run")]
    grouped = raw.groupby(["fraction", "n_train_proteins"])[metric_cols]
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg.insert(0, "n_seeds", raw.groupby(["fraction", "n_train_proteins"]).size())
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="training seeds (model init, shuffle, view draws); 42 = the original final run's seed")
    parser.add_argument("--subset-seed", type=int, default=42,
                        help="fixed seed for building the nested protein subsets (independent of --seeds)")
    parser.add_argument("--target-steps", type=int, default=None,
                        help="optimizer-step budget per run (default: final run's epochs x steps/epoch at 100%%)")
    parser.add_argument("--plan", action="store_true", help="print subset sizes / epochs and exit")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="evaluate existing checkpoints, train nothing")
    parser.add_argument("--go-retrieval-only", action="store_true",
                        help="add GO-retrieval columns to the existing raw_runs.csv, nothing else")
    args = parser.parse_args()

    pool, base_held_out = training_pool()
    subsets = nested_subsets(pool, args.fractions, args.subset_seed)
    bs = FINAL_CONFIG["batch_size"]
    target_steps = args.target_steps or FINAL_EPOCHS * math.ceil(len(pool) / bs)
    plan = {}
    for f in args.fractions:
        steps_per_epoch = math.ceil(len(subsets[f]) / bs)
        epochs = round(target_steps / steps_per_epoch)
        plan[f] = dict(n_proteins=len(subsets[f]), steps_per_epoch=steps_per_epoch,
                       epochs=epochs, total_steps=epochs * steps_per_epoch)
    print(f"Training pool: {len(pool):,} proteins | target steps: {target_steps:,}")
    for f, p in plan.items():
        print(f"  {f:>4.0%}: {p['n_proteins']:>6,} proteins  {p['steps_per_epoch']:>3} steps/epoch  "
              f"{p['epochs']:>4} epochs  {p['total_steps']:>5,} steps")
    if args.plan:
        return

    if args.go_retrieval_only:
        raw = pd.read_csv(EVAL_DIR / "raw_runs.csv")
        go_rows = [evaluate_go_retrieval(RUN_DIR / f"{r}_best.pt") for r in raw["run"]]
        go_df = pd.DataFrame(go_rows)
        raw = pd.concat([raw.drop(columns=[c for c in go_df.columns if c in raw.columns]), go_df], axis=1)
        raw.to_csv(EVAL_DIR / "raw_runs.csv", index=False)
        aggregate(raw).to_csv(EVAL_DIR / "aggregated.csv", index=False)
        print(f"\nAdded GO-retrieval to {EVAL_DIR / 'raw_runs.csv'} and {EVAL_DIR / 'aggregated.csv'}")
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "subsets.json", "w") as fh:
        json.dump({"subset_seed": args.subset_seed, "target_steps": target_steps,
                   "plan": {str(f): p for f, p in plan.items()},
                   "subsets": {str(f): s for f, s in subsets.items()}}, fh, indent=1)

    rows = []
    for f in args.fractions:
        for seed in args.seeds:
            best = RUN_DIR / f"{run_name(f, seed)}_best.pt"
            if not args.eval_only:
                best = train_one(f, seed, subsets[f], pool, base_held_out, plan[f]["epochs"])
            print(f"[eval] {best}")
            row = {"run": run_name(f, seed), "fraction": f, "seed": seed,
                   "n_train_proteins": len(subsets[f])}
            row.update(evaluate_one(best, args.skip_zero_shot))
            rows.append(row)
            pd.DataFrame(rows).to_csv(EVAL_DIR / "raw_runs.csv", index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(EVAL_DIR / "raw_runs.csv", index=False)
    aggregate(raw).to_csv(EVAL_DIR / "aggregated.csv", index=False)
    print(f"\nSaved {EVAL_DIR / 'raw_runs.csv'} and {EVAL_DIR / 'aggregated.csv'}")


if __name__ == "__main__":
    main()

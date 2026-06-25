"""
Incremental scaling experiment: train on growing sets of cohorts, 
evaluate R@K on fixed LUAD test patients after each, save results to JSON.

Usage:
    uv run python src/model/run_experiments.py
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.train import train
from src.model.recall_at_k import compute_ranks_luad, compute_ranks_all_cohorts

# Add cohorts in the order you want to add them.
# Only include cohorts that have all 5 preprocessing steps done.
COHORT_ORDER = [
    "luad",
    "brca",
    "ccrcc",
    "gbm",
    "hnscc",
    "lscc",
    "ov",
    "pdac",
    "ucec"
]

RESULTS_PATH = Path("data/scaling_results_v2.json")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# load existing results so we can resume if interrupted
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
else:
    results = {}

for i in range(1, len(COHORT_ORDER) + 1):
    cohorts = COHORT_ORDER[:i]
    run_name = "_".join(cohorts)

    if run_name in results:
        print(f"[skip] {run_name} already done")
        continue

    checkpoint_path = f"data/runs/{run_name}/checkpoint.pt"
    print(f"\n{'='*60}")
    print(f"Experiment: {run_name}")
    print(f"{'='*60}")

    train(cohorts=cohorts, checkpoint_path=checkpoint_path)

    luad    = compute_ranks_luad(checkpoint_path=checkpoint_path)
    all_c   = compute_ranks_all_cohorts(checkpoint_path=checkpoint_path, cohorts=cohorts)

    results[run_name] = {
        "cohorts":   cohorts,
        "n_cohorts": i,
        "luad_r1":   luad["r1"],   "luad_r10":  luad["r10"],  "luad_r50":  luad["r50"],
        "all_r1":    all_c["r1"],  "all_r10":   all_c["r10"], "all_r50":   all_c["r50"],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"LUAD  R@1={luad['r1']:.2f}%  R@10={luad['r10']:.2f}%  R@50={luad['r50']:.2f}%")
    print(f"All   R@1={all_c['r1']:.2f}%  R@10={all_c['r10']:.2f}%  R@50={all_c['r50']:.2f}%")

print("\nAll experiments done. Results saved to", RESULTS_PATH)

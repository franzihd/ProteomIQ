#!/usr/bin/env bash
# Re-runs every thesis evaluation from scratch and rebuilds the consolidated results files.
#
# What it does, in order:
#   1. BACKUP   -- moves every existing eval output (summary_*/predictions_* files) plus
#                  all_results.csv/results_summary.csv into data/evaluation_backup_<timestamp>/.
#                  Nothing is deleted; restoring = moving the files back.
#   2. VALIDATION -- all 5 topic evals on all 10 ablation checkpoints, plus the 3 task-agnostic
#                  checks on the same checkpoints they were originally run for.
#   3. TEST     -- multiview ONLY (the one final checkpoint). Never add another checkpoint here.
#                  Note: the DeepLoc/EC/GO-MF scripts always compute a test-pool row for every
#                  checkpoint as a by-product -- those rows exist in all_results.csv, but only
#                  multiview's may be cited.
#   4. SORT     -- moves ablation outputs back into unfreeze_experiment/ / text_experiment/
#                  subfolders (same layout as before).
#   5. AGGREGATE -- rebuilds all_results.csv, results_summary.csv (validation) and
#                  results_final_test.csv (test, multiview only).
#   6. COMPARE  -- prints every number that differs between the backup and the fresh run.
#
# Usage (from the repo root):
#   bash scripts/rerun_all_evals.sh 2>&1 | tee rerun_evals.log
#
# Superseded checkpoints (final_v2_dims_masked_best = pre-fix M0) are NOT re-run, so their rows
# disappear from all_results.csv -- intended, they must not be cited anyway. Their old files stay
# in the backup folder.

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=data/runs/full_proteome
SPLITS=data/protein_splits_full_proteome.json
TEXT=biobert_embeddings_masked.h5   # fixed Function-only text for every checkpoint (verified to reproduce the saved numbers)

# checkpoint paths, grouped the same way as the ablation study
UNFREEZE=(
  $RUNS/unfreeze_experiment/biobert_unfreeze_0_m0_best.pt   # = M0
  $RUNS/unfreeze_experiment/biobert_unfreeze_1_m0_best.pt
  $RUNS/unfreeze_experiment/biobert_unfreeze_2_m0_best.pt
  $RUNS/unfreeze_experiment/biobert_unfreeze_3_m0_best.pt
)
TEXTVARIANTS=(
  $RUNS/text_experiment/m1_family_masked_best.pt
  $RUNS/text_experiment/location_only_masked_best.pt        # = Function+Location
  $RUNS/text_experiment/m2_family_location_masked_best.pt
  $RUNS/text_experiment/m3_name_family_location_best.pt
)
M0=$RUNS/unfreeze_experiment/biobert_unfreeze_0_m0_best.pt
MULTIPOS=$RUNS/multi_positive_experiment/multi_positive_best.pt
MULTIVIEW=$RUNS/multiview_experiment/multiview_best.pt

ALL=("${UNFREEZE[@]}" "${TEXTVARIANTS[@]}" "$MULTIPOS" "$MULTIVIEW")
KNN_SET=("$M0" "${TEXTVARIANTS[@]}" "$MULTIPOS" "$MULTIVIEW")
PROBE_SET=("$M0" "$MULTIPOS" "$MULTIVIEW")

OUTPUT_DIRS=(
  data/evaluation/cross_modal_retrieval
  data/evaluation/go_retrieval/results
  data/evaluation/zero_shot_classification/deeploc
  data/evaluation/zero_shot_classification/ec_top_level
  data/evaluation/zero_shot_classification/go_mf
  data/evaluation/knn_embedding_quality
  data/evaluation/linear_probe
  data/evaluation/alignment_uniformity
)

step() { echo; echo "================ $(date +%H:%M:%S)  $*"; }

for c in "${ALL[@]}"; do [[ -f $c ]] || { echo "missing checkpoint: $c"; exit 1; }; done

# ---------------------------------------------------------------- 1. backup
BACKUP=data/evaluation_backup_$(date +%Y%m%d_%H%M%S)
step "1. Backup of existing outputs -> $BACKUP"
mkdir -p "$BACKUP/data/evaluation"
cp data/evaluation/all_results.csv data/evaluation/results_summary.csv data/evaluation/results_final_test.csv "$BACKUP/data/evaluation/"
find "${OUTPUT_DIRS[@]}" -type f \( -name 'summary_*' -o -name 'predictions_*' \) -print0 |
  while IFS= read -r -d '' f; do
    mkdir -p "$BACKUP/$(dirname "$f")"
    mv "$f" "$BACKUP/$f"
  done
echo "moved $(find "$BACKUP" -type f | wc -l) files"

# ---------------------------------------------------------------- 2. validation
for c in "${ALL[@]}"; do
  step "2. VALIDATION  $(basename "$c")"
  uv run python src/evaluation/cross_modal_eval.py --checkpoint "$c" \
      --biobert-filename $TEXT --splits-file $SPLITS --split-key validation
  uv run python src/evaluation/zero_shot_deeploc_eval.py --checkpoint "$c" --drop-chloroplast --all-conditions
  uv run python src/evaluation/zero_shot_ec_eval.py      --checkpoint "$c" --all-conditions
  uv run python src/evaluation/zero_shot_go_mf_eval.py   --checkpoint "$c" --all-conditions
  uv run python src/evaluation/zero_shot_go_retrieval_eval.py --checkpoint "$c" \
      --branch all --min-positives 10 --split-key validation
done

for c in "${KNN_SET[@]}"; do
  step "2. k-NN (validation)  $(basename "$c")"
  uv run python src/evaluation/knn_embedding_quality_eval.py --checkpoint "$c"
done

for c in "${PROBE_SET[@]}"; do
  step "2. linear probe + alignment/uniformity (validation)  $(basename "$c")"
  uv run python src/evaluation/linear_probe_eval.py --checkpoint "$c"
  uv run python src/evaluation/alignment_uniformity_eval.py --checkpoint "$c" --splits-file $SPLITS
done

# ---------------------------------------------------------------- 3. test (multiview only!)
step "3. TEST  multiview only"
uv run python src/evaluation/cross_modal_eval.py --checkpoint "$MULTIVIEW" \
    --biobert-filename $TEXT --splits-file $SPLITS --split-key test
uv run python src/evaluation/zero_shot_go_retrieval_eval.py --checkpoint "$MULTIVIEW" \
    --branch all --min-positives 10 --split-key test
# DeepLoc/EC/GO-MF test numbers (unseen_test) were already produced in step 2.

# ---------------------------------------------------------------- 4. sort into experiment subfolders
step "4. Sorting ablation outputs into experiment subfolders"
for d in data/evaluation/cross_modal_retrieval data/evaluation/go_retrieval/results \
         data/evaluation/zero_shot_classification/{deeploc,ec_top_level,go_mf}; do
  mkdir -p "$d/unfreeze_experiment" "$d/text_experiment"
  find "$d" -maxdepth 1 -type f -name '*biobert_unfreeze_*' -exec mv {} "$d/unfreeze_experiment/" \;
  find "$d" -maxdepth 1 -type f \( -name '*m1_family_masked_best*' -o -name '*location_only_masked_best*' \
       -o -name '*m2_family_location_masked_best*' -o -name '*m3_name_family_location_best*' \) \
       -exec mv {} "$d/text_experiment/" \;
done

# ---------------------------------------------------------------- 5. aggregate
step "5. Rebuilding all_results.csv, results_summary.csv, results_final_test.csv"
uv run python src/evaluation/aggregate_results.py

# ---------------------------------------------------------------- 6. compare
step "6. Comparing fresh run against backup"
uv run python - "$BACKUP/data/evaluation/all_results.csv" data/evaluation/all_results.csv <<'EOF'
import sys, pandas as pd
old, new = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
key = ["eval", "checkpoint", "pool", "variant", "metric"]
for df in (old, new):
    df["variant"] = df["variant"].fillna("n/a")
m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
both = m[m._merge == "both"].copy()
both["diff"] = (both.value_new - both.value_old).abs()
changed = both[both["diff"] > 1e-6].sort_values("diff", ascending=False)
print(f"rows compared: {len(both):,}   identical: {len(both) - len(changed):,}   different: {len(changed):,}")
if len(changed):
    print("\nlargest differences:")
    print(changed[key + ["value_old", "value_new", "diff"]].head(40).to_string(index=False))
only_old = m[m._merge == "left_only"]
only_new = m[m._merge == "right_only"]
if len(only_old):
    print(f"\nonly in backup ({len(only_old)} rows), by checkpoint:")
    print(only_old.groupby(['eval', 'checkpoint']).size().to_string())
if len(only_new):
    print(f"\nonly in fresh run ({len(only_new)} rows), by checkpoint:")
    print(only_new.groupby(['eval', 'checkpoint']).size().to_string())
EOF

step "Done. Backup of the previous results: $BACKUP"

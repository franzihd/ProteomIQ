#!/usr/bin/env bash
# Mirrors the files the Streamlit app needs into the standalone demo repo
# (deployed on Streamlit Community Cloud), plus the deploy-specific files in
# deploy/streamlit_cloud/. The target's .git is left alone, so after running
# this you just commit + push in the target.
#
# Usage: deploy/sync_space.sh [target_dir]   (default: ../ProteomIQ-demo)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$REPO/../ProteomIQ-demo}"

FILES=(
  src/app.py
  src/model/__init__.py
  src/model/interpret.py
  src/model/retrieve_and_annotate.py
  src/model/projection_head.py
  src/model/dataset.py
  src/embeddings/textual/embed_biobert.py
  src/preprocessing/normalize_uploaded_sample.py
  src/assets/icon.png
  src/assets/favicon.png
  src/assets/logo.jpg
  src/assets/final_logo.png
  src/assets/final_logo_header.png
  .streamlit/config.toml
  data/full_proteome/esm2_embeddings.h5
  data/full_proteome/protein_annotations.csv
  data/runs/full_proteome/multiview_experiment/multiview_best.pt
  data/test_abundance_matrix/11LU013.csv
  data/test_abundance_matrix/11LU016.csv
)
# All example samples (2 per CPTAC cohort) -- picked up by glob so new
# example files don't need to be listed here one by one.
FILES+=(data/test_abundance_matrix/*.csv)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cd "$REPO"
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
  cp --parents "$f" "$STAGE/"
done
cp -r deploy/streamlit_cloud/. "$STAGE/"

mkdir -p "$TARGET"
# --delete removes files dropped from FILES; .git is excluded so history survives.
rsync -a --delete --exclude .git "$STAGE/" "$TARGET/"

echo "Demo repo updated: $TARGET ($(du -sh --exclude .git "$TARGET" | cut -f1))"

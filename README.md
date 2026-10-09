# ProteomIQ

Natural-language search over the proteins of a proteomics sample. A CLIP-style model maps
protein sequences (ESM2, frozen) and UniProt text (BioBERT, frozen) into a shared embedding
space via two trained projection heads (InfoNCE loss).

This repository contains the code of the thesis: data processing, training, evaluation and
figures. The app is provided separately:

- App: [proteomiq.streamlit.app](https://proteomiq.streamlit.app/)
- App repository: [ProteomIQ-demo](https://github.com/franzihd/ProteomIQ-demo)

Final model: `data/runs/full_proteome/multiview_experiment/multiview_best.pt`

## Structure

```
src/
  app.py               Streamlit tool
  data_acquisition/    CPTAC download, UniProt fetching
  preprocessing/       abundance rank normalization
  embeddings/          ESM2 and BioBERT embeddings
  model/               dataset, projection head, split, training, sweeps
  evaluation/          all evaluations
  plotting/            all figures
configs/               W&B sweep configs
scripts/               re-run all evaluations, package the app
deploy/                Streamlit Cloud demo
data/
  full_proteome/       UniProt annotations and sequences
  cptac/               CPTAC abundance data
  splits/              CD-HIT clusters and train/validation/test split
  runs/                trained checkpoints
  evaluation/          evaluation results (all_results.csv, results_final_test.csv)
  plots/               figures (overview/ = Results figures)
  test_abundance_matrix/  example samples for the app
archive/               earlier, superseded code
```

## Reproducing the results

The `.h5` embedding files are not in the repository and are created in step 2.

```bash
uv sync

# 1. Data and split
uv run python src/data_acquisition/fetch_full_human_proteome.py
uv run python src/data_acquisition/fetch_family_location.py
uv run python src/model/cluster_proteins_cdhit.py
uv run python src/model/protein_train_validation_test.py

# 2. Embeddings
uv run python src/embeddings/protein/esm2_embeddings.py --cohort full_proteome
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --mask-gene-names
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --field family
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --field location

# 3. Training (multiview)
uv run python src/model/train_protein_level.py \
    --checkpoint data/runs/full_proteome/multiview_experiment/multiview.pt \
    --epochs 100 --lr 1.16e-4 --batch-size 512 --hidden-dim 2048 --output-dim 1024 \
    --early-stopping-patience 10 \
    --biobert-filename biobert_embeddings_masked.h5 \
    --view-biobert-filenames biobert_embeddings_masked.h5 biobert_embeddings_family_view.h5 \
                             biobert_embeddings_location_view.h5 \
    --splits-file data/splits/protein_splits_full_proteome.json

# 4. Evaluation and figures
bash scripts/rerun_all_evals.sh
uv run python src/plotting/plot_ablation_comparison.py
uv run python src/plotting/plot_task_agnostic_comparison.py
uv run python src/plotting/plot_final_test_results.py
```

## Running the app

The app is online at [proteomiq.streamlit.app](https://proteomiq.streamlit.app/). To run it
locally without reproducing the pipeline, use
[ProteomIQ-demo](https://github.com/franzihd/ProteomIQ-demo), which includes all required
data (built from this repository by `deploy/sync_space.sh`).

From this repository, the app requires `data/full_proteome/esm2_embeddings.h5` (step 2 above).
For the optional Claude interpretation, set `ANTHROPIC_API_KEY`.

```bash
uv sync
uv run streamlit run src/app.py
```

Upload a CSV with one row per protein, or use the built-in example sample:

```
gene,abundance
EGFR,1.23
KRAS,0.45
```

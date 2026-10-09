# ProteomIQ

Code for the bachelor thesis *A Contrastive Multimodal Embedding Approach For Natural Language
Retrieval in Clinical Proteomics*, Franziska Hald, Eberhard Karls University of Tübingen, 2026.
Supervisor: Prof. Dr. Sven Nahnsen.

This repository contains data processing, training, evaluation and figures. The app is
provided separately:

- App: [proteomiq.streamlit.app](https://proteomiq.streamlit.app/)
- App repository: [ProteomIQ-demo](https://github.com/franzihd/ProteomIQ-demo)

Final model: `data/runs/full_proteome/multiview_experiment/multiview_best.pt`

## Requirements

Linux (x86_64), Python ≥ 3.11, [uv](https://docs.astral.sh/uv/). Some pinned packages
(PyTorch CUDA 11.8 build, NVIDIA libraries) are only available for Linux, so installation fails
on macOS and Windows. A GPU speeds up retraining but is not required.

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
scripts/               re-run all evaluations
deploy/                Streamlit Cloud demo
data/
  full_proteome/       UniProt annotations, sequences and embeddings
  cptac/               CPTAC abundance data
  splits/              CD-HIT clusters and train/validation/test split
  runs/                trained checkpoints
  evaluation/          evaluation results (all_results.csv, results_final_test.csv)
  plots/               figures (overview/ = Results figures)
  test_abundance_matrix/  example samples for the app
```

## Reproducing the results

UniProt data, CPTAC data, CD-HIT clusters, the split, the trained checkpoints and the
embeddings needed for evaluation are included in `data/`. Re-downloading the CPTAC data
(`src/data_acquisition/import_cptac.py`) requires `uv sync --extra data`.

Test results of the final model:

```bash
uv sync

CKPT=data/runs/full_proteome/multiview_experiment/multiview_best.pt
uv run python src/evaluation/cross_modal_eval.py --checkpoint $CKPT \
    --biobert-filename biobert_embeddings_masked.h5 --split-key test
uv run python src/evaluation/zero_shot_go_retrieval_eval.py --checkpoint $CKPT \
    --branch all --min-positives 10 --split-key test
uv run python src/evaluation/zero_shot_deeploc_eval.py --checkpoint $CKPT --drop-chloroplast --all-conditions
uv run python src/evaluation/zero_shot_ec_eval.py --checkpoint $CKPT --all-conditions
uv run python src/evaluation/zero_shot_go_mf_eval.py --checkpoint $CKPT --all-conditions
uv run python src/evaluation/aggregate_results.py      # -> data/evaluation/results_final_test.csv
uv run python src/plotting/plot_final_test_results.py
```

Retraining the final model (seeded, but not bit-identical on GPU). This first computes the
Family and Location embeddings:

```bash
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --field family
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --field location

uv run python src/model/train_protein_level.py \
    --checkpoint data/runs/retrain/multiview.pt \
    --epochs 100 --lr 1.16e-4 --batch-size 512 --hidden-dim 2048 --output-dim 1024 \
    --early-stopping-patience 10 \
    --biobert-filename biobert_embeddings_masked.h5 \
    --view-biobert-filenames biobert_embeddings_masked.h5 biobert_embeddings_family_view.h5 \
                             biobert_embeddings_location_view.h5 \
    --splits-file data/splits/protein_splits_full_proteome.json
```

## Running the app

**Online:** [proteomiq.streamlit.app](https://proteomiq.streamlit.app/). The optional
interpretation by Claude uses my own API key with a limited budget. If it stops
responding, it's used up.

**Locally:**

```bash
uv sync
# optional, for the Claude interpretation: export ANTHROPIC_API_KEY=<your key>
uv run streamlit run src/app.py
```

[ProteomIQ-demo](https://github.com/franzihd/ProteomIQ-demo) contains only the app and is the
source of the online app.

Upload a CSV with one row per protein, or use the built-in example sample:

```
gene,abundance
EGFR,1.23
KRAS,0.45
```

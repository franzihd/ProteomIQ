# ProteomIQ

Natural-language search over the proteins of a proteomics sample. A CLIP-style model maps
protein sequences (ESM2, frozen) and UniProt text (BioBERT, frozen) into a shared embedding
space via two trained projection heads (InfoNCE loss).

Code for the bachelor thesis *A Contrastive Multimodal Embedding Approach For Natural Language
Retrieval in Clinical Proteomics*, Franziska Hald, Eberhard Karls University of Tübingen, 2026.
Supervisor: Prof. Dr. Sven Nahnsen.

This repository contains data processing, training, evaluation and figures. The app is
provided separately:

- App: [proteomiq.streamlit.app](https://proteomiq.streamlit.app/)
- App repository: [ProteomIQ-demo](https://github.com/franzihd/ProteomIQ-demo)

Final model: `data/runs/full_proteome/multiview_experiment/multiview_best.pt`

## Requirements

Linux, Python ≥ 3.11, [uv](https://docs.astral.sh/uv/). PyTorch is pinned to a CUDA 11.8 build,
so installation fails on macOS. A GPU is recommended for computing the embeddings but not
required. To only try the app, use the online version or ProteomIQ-demo.

## Data sources

CPTAC (proteomics), UniProt (sequences, annotations), DeepLoc 1.0 (localization labels),
Gene Ontology (GO terms).

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
  full_proteome/       UniProt annotations and sequences
  cptac/               CPTAC abundance data
  splits/              CD-HIT clusters and train/validation/test split
  runs/                trained checkpoints
  evaluation/          evaluation results (all_results.csv, results_final_test.csv)
  plots/               figures (overview/ = Results figures)
  test_abundance_matrix/  example samples for the app
```

## Reproducing the results

UniProt data, CD-HIT clusters, the split and the trained checkpoints are included in `data/`.
The embedding files (`.h5`) are not and are created first (GPU recommended).

Test results of the final model:

```bash
uv sync

uv run python src/embeddings/protein/esm2_embeddings.py --cohort full_proteome
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --mask-gene-names

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

Retraining the final model (seeded, but not bit-identical on GPU):

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

The app is online at [proteomiq.streamlit.app](https://proteomiq.streamlit.app/). To run it
locally without reproducing the pipeline, use
[ProteomIQ-demo](https://github.com/franzihd/ProteomIQ-demo), which includes all required
data (built from this repository by `deploy/sync_space.sh`).

From this repository, the app requires `data/full_proteome/esm2_embeddings.h5`
(see Reproducing the results).
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

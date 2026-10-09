# ProteomIQ

A protein-level joint embedding space for proteomics and natural language. Upload a proteomics
sample, ask a question in plain language (e.g. *"receptor tyrosine kinase signaling"*), and see
which of that sample's measured proteins are most relevant — with each result's relative
abundance in the sample shown alongside it, and an optional AI-generated interpretation.

This repository accompanies the bachelor's thesis and contains the full pipeline — data
acquisition, embedding pre-computation, training, evaluation, figures — plus the interactive
Streamlit tool. The methodology and results are described in the thesis; this README explains
how the repository is organized and how to run things.

## Overview

A CLIP-style model aligns two frozen encoders through small trainable projection heads:

```
protein sequence → ESM2 (esm2_t30_150M, frozen) → MLP projection head ─┐
                                                                         ├─ shared 1024-dim space, InfoNCE loss
UniProt text     → BioBERT (frozen)             → MLP projection head ─┘
```

- **Protein universe:** the reviewed human UniProt proteome (16,893 proteins), split 80/10/10
  into train/validation/test by CD-HIT clusters (90 % identity), so near-identical paralogs never
  cross the split.
- **Final model ("multiview"):** each training step pairs a protein with **one** randomly drawn
  text view — Function, Family, or Subcellular Location (gene names masked). Checkpoint:
  `data/runs/full_proteome/multiview_experiment/multiview_best.pt`.
- **Evaluation:** five topic-specific evals (cross-modal retrieval, zero-shot DeepLoc / EC /
  GO-Slim-MF classification, GO-term retrieval) and three task-agnostic checks
  (alignment/uniformity, k-NN, linear probe). Model selection used the validation split only;
  the test split was evaluated exactly once, on the final checkpoint.
- **Tool:** a researcher uploads one sample (`gene,abundance`), asks a question in plain
  language, and gets the sample's most relevant proteins (Stage 1, retrieval), their
  within-sample abundance percentile (Stage 2), and an optional Claude-based interpretation
  (Stage 3).

## Repository structure

```
src/
  app.py                     Streamlit tool (Stages 1–3)
  data_acquisition/          download CPTAC data; fetch UniProt annotations (full human proteome,
                             Family/Location fields)
  preprocessing/             within-sample rank normalization; uploaded-sample normalization
  embeddings/                pre-compute ESM2 (protein) and BioBERT (text) embeddings
  model/                     dataset, projection head, CD-HIT split, training, W&B sweeps,
                             data-size ablation, retrieval + interpretation used by the tool
  evaluation/                all evaluations + aggregate_results.py (→ all_results.csv)
  plotting/                  every thesis figure
configs/                     W&B hyperparameter-sweep definitions
scripts/
  rerun_all_evals.sh         re-run every evaluation and rebuild the results tables
  build_app_release.py       package the tool for distribution
deploy/                      Streamlit Community Cloud demo deployment
data/
  full_proteome/             UniProt annotations, sequences, CD-HIT clusters (Stage 1 data)
  cptac/<cohort>/            CPTAC abundance matrices (used by Stage 2 and example samples)
  splits/                    CD-HIT clusters + the homology-aware train/validation/test split
  runs/full_proteome/        trained checkpoints (final model + every ablation)
  evaluation/                eval outputs; all_results.csv, results_summary.csv (validation),
                             results_final_test.csv (test, final model only)
  plots/                     every figure, each with a .csv of exactly the plotted numbers;
                             overview/ holds the Results figures
  test_abundance_matrix/     example patient samples in upload format
archive/                     superseded code from the exploratory phase (see archive/README.md)
```

## Reproducing the results

Training and embedding pre-computation need a GPU (developed on an RTX A6000); evaluations and
plotting run on CPU. The large `.h5` embedding files are not in the repository — regenerate them
with the embedding scripts below.

```bash
# 1. Data
uv run python src/data_acquisition/fetch_full_human_proteome.py
uv run python src/data_acquisition/fetch_family_location.py
uv run python src/model/cluster_proteins_cdhit.py
uv run python src/model/protein_train_validation_test.py

# 2. Frozen-encoder embeddings (ESM2 + the three BioBERT text views)
uv run python src/embeddings/protein/esm2_embeddings.py --cohort full_proteome
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --mask-gene-names
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --field family
uv run python src/embeddings/textual/embed_biobert.py --cohort full_proteome --field location

# 3. Train the final multiview model
uv run python src/model/train_protein_level.py \
    --checkpoint data/runs/full_proteome/multiview_experiment/multiview.pt \
    --epochs 100 --lr 1.16e-4 --batch-size 512 --hidden-dim 2048 --output-dim 1024 \
    --early-stopping-patience 10 \
    --biobert-filename biobert_embeddings_masked.h5 \
    --view-biobert-filenames biobert_embeddings_masked.h5 biobert_embeddings_family_view.h5 \
                             biobert_embeddings_location_view.h5 \
    --splits-file data/splits/protein_splits_full_proteome.json

# 4. All evaluations + consolidated tables, then the Results figures
bash scripts/rerun_all_evals.sh
uv run python src/plotting/plot_ablation_comparison.py        # Results 2.1
uv run python src/plotting/plot_task_agnostic_comparison.py   # Results 2.2
uv run python src/plotting/plot_final_test_results.py         # Results 2.3
uv run python src/plotting/plot_per_eval.py                   # per-eval detail figures
```

Hyperparameters were selected with Weights & Biases sweeps on validation loss
(`configs/sweep_config_v1_lr_only.yaml`, `configs/sweep_config_stage_b_dims.yaml`); the exported sweep results
are in `data/evaluation/hyperparameter_sweeps/`. Every checkpoint stores its own
hyperparameters, so `torch.load(...)` shows exactly how it was trained.

## Running the interactive tool

**What this tool is, and isn't:** ProteomIQ explores one sample at a time, interactively — closer
to how a clinician reviews a single patient's profile than to a population-level study. It is
**not** a substitute for group-level differential expression analysis (tools like GSEA or limma,
which compare many samples across conditions with proper statistical testing). Treat its output
as a starting point for hypothesis generation, not a validated finding.

### Setup

#### 1. Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (Python package/environment manager) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A GPU is **not** required to run the interactive tool — it works on a normal laptop CPU,
  just a bit slower per query (a few seconds instead of under a second).

#### 2. Get the project

Clone or copy the `ProteomIQ` project folder onto your machine, then from inside it:

```bash
uv sync
```

This installs all dependencies into a local virtual environment (`.venv/`) — no need to activate
it manually, `uv run` handles that automatically for every command below.

#### 3. Get the missing data file

One required file (~42 MB) is too large to be stored in the code repository and needs to be
copied in separately: **`data/full_proteome/esm2_embeddings.h5`**

Ask whoever gave you this project for that file, and place it at exactly that path inside the
project folder (i.e. `ProteomIQ/data/full_proteome/esm2_embeddings.h5`). Everything else the tool
needs (the trained model checkpoint, protein annotations) is already included.

#### 4. (Optional) Enable AI interpretation

The tool's third step ("Interpret with Claude") uses the Claude API to explain results in plain
language. This step is optional — retrieval and abundance results work without it. To enable it,
set an environment variable with your own Anthropic API key before starting the tool:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

(Get a key at [console.anthropic.com](https://console.anthropic.com) if you don't have one — this
uses your own account and incurs a small per-request cost, a few cents per interpretation.)

### Running the tool

From the project folder:

```bash
uv run streamlit run src/app.py
```

This opens the tool in your browser automatically (usually at `http://localhost:8501`). The
first run takes a little longer while the model loads; it's cached after that.

### Using the tool

1. **Sample** — upload your own proteomics CSV, or click **"Try with example data"** to explore a
   real (already-included) patient sample without preparing your own file first.

   Expected CSV format — one row per protein, gene name + a raw abundance value:
   ```
   gene,abundance
   EGFR,1.23
   KRAS,0.45
   MET,2.67
   ```
2. **Question** — type what you're interested in, in plain language (e.g. *"immune evasion"*,
   *"cell cycle regulation"*, *"proteins associated with poor prognosis in lung cancer"*).
3. **Results** — the most relevant measured proteins, ranked by relevance to your question, each
   showing how abundant it was in this specific sample. Click a protein's "What is `<gene>`?" to
   see its known biological function.
4. **Interpretation** (optional, needs an API key — see Setup step 4) — click **"Interpret with
   Claude"** for a natural-language summary of what the result set might mean biologically.

### Troubleshooting

- **"Required data files are missing"** on startup — see Setup step 3; the tool tells you exactly
  which file path is missing.
- **"No Claude API key configured"** — Stage 3 interpretation is disabled but everything else
  still works; see Setup step 4 to enable it.
- **Slow on first query** — normal; the model and reference data are being loaded and cached, and
  only need to happen once per session.

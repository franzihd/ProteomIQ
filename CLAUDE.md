# ProteomIQ

A bioinformatics tool that creates a joint embedding space for proteomics data and natural language, enabling semantic search over protein expression data via natural-language queries.

## Project goal

Build a CLIP-style model where:
- Proteomics samples (protein abundance vectors) are encoded into an embedding space
- Text descriptions / queries are encoded into the same embedding space
- Nearest-neighbor search over embeddings answers questions like "Are there proteins related to lung cancer?"

The closest prior work is [CellWhisperer](https://github.com/theislab/CellWhisperer), which does the same for single-cell transcriptomics (scRNA-seq). ProteomIQ targets mass-spectrometry-based proteomics instead.

## Domain glossary

- **Proteomics**: large-scale study of all proteins expressed in a cell, tissue, or organism — measured with mass spectrometry
- **CPTAC**: Clinical Proteomic Tumor Analysis Consortium — the main data source (`cptac` Python package)
- **LUAD**: Lung Adenocarcinoma — the cancer type currently used as the first dataset
- **Protein abundance matrix**: rows = samples (patients), columns = proteins (identified by gene name + database ID, stored as a two-level MultiIndex), values = measured expression levels
- **NaN**: Not a Number — missing value, common in proteomics when a protein was not detected in a sample
- **CLIP**: Contrastive Language-Image Pretraining — trains two encoders (one per modality) with a contrastive loss so matching pairs end up close in a shared embedding space
- **Contrastive loss**: loss function that pulls embeddings of matching pairs together and pushes non-matching pairs apart
- **Nearest-neighbor search**: given a query embedding, find the k most similar embeddings by cosine or Euclidean distance (e.g. via FAISS or `sklearn`)

## Tech stack

- Python ≥ 3.11, managed with `uv`
- `cptac` — proteomics dataset access (CPTAC cohorts)
- `jupyter` — interactive data exploration in VS Code notebooks
- Deep learning framework: TBD (likely PyTorch)
- Embedding / retrieval: TBD (likely FAISS)

## Project layout

```
main.py                                        # entry point (stub)
pyproject.toml                                 # dependencies
src/
  import_cptac.py                              # downloads proteomics + clinical data for all 10 CPTAC cohorts
  text_generation.py                           # converts clinical metadata rows into natural-language patient descriptions
data/
  luad/
    proteomics_raw.csv                         # 213 samples × 13,302 proteins (umich source)
    clinical.csv                               # 111 samples × 124 clinical variables (mssm source)
  brca/
    proteomics_raw.csv                         # 125 samples × 12,922 proteins
    clinical.csv                               # 134 samples × 124 clinical variables
  ccrcc/
    proteomics_raw.csv                         # 194 samples × 11,889 proteins
    clinical.csv                               # 110 samples × 124 clinical variables
  coad/                                        # skipped — all sources broken in cptac library
  hnscc/
    proteomics_raw.csv                         # 182 samples × 12,224 proteins
    clinical.csv                               # 112 samples × 124 clinical variables
  lscc/
    proteomics_raw.csv                         # 212 samples × 13,715 proteins
    clinical.csv                               # 110 samples × 124 clinical variables
  ucec/
    proteomics_raw.csv                         # 152 samples × 12,662 proteins
    clinical.csv                               # 103 samples × 124 clinical variables
  gbm/
    proteomics_raw.csv                         # 110 samples × 13,002 proteins
    clinical.csv                               # 100 samples × 124 clinical variables
  pdac/
    proteomics_raw.csv                         # 238 samples × 12,017 proteins
    clinical.csv                               # 140 samples × 124 clinical variables
  ov/
    proteomics_raw.csv                         # 103 samples × 11,099 proteins
    clinical.csv                               # 110 samples × 124 clinical variables
```

## Data notes

- Proteomics CSV has **two header rows** (protein name + database ID) — read with `header=[0, 1]`
- Proteomics source: `umich` (bcm source is broken in the cptac library for all cohorts)
- Clinical source: `mssm` for all cohorts
- COAD is missing — both available sources fail internally in the cptac library
- Large CSVs cannot be opened in VS Code's text editor — use Jupyter notebooks for exploration

## Text generation

`src/text_generation.py` converts a clinical metadata row into a natural-language patient description for BioBERT embedding. Each description is a flexible sentence — only fields with valid values are included, so length varies per patient.

**Columns used** (in order):
- `age`, `sex` — patient demographics
- `ethnicity_race_ancestry_identified` — ancestry
- `tumor_stage_pathological` — Stage I/II/III/IV
- `histologic_type` — adenocarcinoma subtype
- `histologic_grade` — G1/G2/G3
- `tumor_size_cm` — tumor size in cm
- `tumor_laterality` — left or right lung
- `tumor_site` — upper/lower/middle lobe
- `tumor_focality` — unifocal or multifocal
- `tobacco_smoking_history` — smoking status (shortened from raw strings)
- `alcohol_consumption` — drinking status (shortened from raw strings)
- `bmi` — body mass index

**Key design decisions:**
- Values in `SKIP_VALUES` (e.g. "Unknown", "Not Reported", "not specified") are silently omitted
- Long smoking/alcohol strings are mapped to short phrases
- `cancer_type` is passed as a parameter, not read from the CSV
- BioBERT handles variable-length sentences — no fixed template needed

## Development conventions

- Use `uv` to manage the virtual environment and add dependencies (`uv add <pkg>`)
- Run scripts with `uv run python <script>`
- Use Jupyter notebooks inside VS Code (Microsoft Jupyter extension) for data exploration
- Explain domain-specific proteomics or ML concepts when they come up — the developer is new to both research and proteomics

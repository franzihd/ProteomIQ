# ProteomIQ

A protein-level joint embedding space for proteomics and natural language. Upload a proteomics
sample, ask a question in plain language (e.g. *"receptor tyrosine kinase signaling"*), and see
which of that sample's measured proteins are most relevant — with each result's relative
abundance in the sample shown alongside it, and an optional AI-generated interpretation.

For the full methodology, training pipeline, and evaluation results, see the accompanying
thesis. This document only covers running the interactive tool.

**What this tool is, and isn't:** ProteomIQ explores one sample at a time, interactively — closer
to how a clinician reviews a single patient's profile than to a population-level study. It is
**not** a substitute for group-level differential expression analysis (tools like GSEA or limma,
which compare many samples across conditions with proper statistical testing). Treat its output
as a starting point for hypothesis generation, not a validated finding.

## Setup

### 1. Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (Python package/environment manager) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A GPU is **not** required to run the interactive tool — it works on a normal laptop CPU,
  just a bit slower per query (a few seconds instead of under a second).

### 2. Get the project

Clone or copy the `ProteomIQ` project folder onto your machine, then from inside it:

```bash
uv sync
```

This installs all dependencies into a local virtual environment (`.venv/`) — no need to activate
it manually, `uv run` handles that automatically for every command below.

### 3. Get the missing data file

One required file (~42 MB) is too large to be stored in the code repository and needs to be
copied in separately: **`data/full_proteome/esm2_embeddings.h5`**

Ask whoever gave you this project for that file, and place it at exactly that path inside the
project folder (i.e. `ProteomIQ/data/full_proteome/esm2_embeddings.h5`). Everything else the tool
needs (the trained model checkpoint, protein annotations) is already included.

### 4. (Optional) Enable AI interpretation

The tool's third step ("Interpret with Claude") uses the Claude API to explain results in plain
language. This step is optional — retrieval and abundance results work without it. To enable it,
set an environment variable with your own Anthropic API key before starting the tool:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

(Get a key at [console.anthropic.com](https://console.anthropic.com) if you don't have one — this
uses your own account and incurs a small per-request cost, a few cents per interpretation.)

## Running the tool

From the project folder:

```bash
uv run streamlit run src/app.py
```

This opens the tool in your browser automatically (usually at `http://localhost:8501`). The
first run takes a little longer while the model loads; it's cached after that.

## Using the tool

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

## Troubleshooting

- **"Required data files are missing"** on startup — see Setup step 3; the tool tells you exactly
  which file path is missing.
- **"No Claude API key configured"** — Stage 3 interpretation is disabled but everything else
  still works; see Setup step 4 to enable it.
- **Slow on first query** — normal; the model and reference data are being loaded and cached, and
  only need to happen once per session.

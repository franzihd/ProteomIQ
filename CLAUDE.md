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
- **Protein abundance matrix**: rows = samples (patients/conditions), columns = proteins (identified by gene name or UniProt ID), values = measured expression levels
- **CLIP**: Contrastive Language-Image Pretraining — trains two encoders (one per modality) with a contrastive loss so matching pairs end up close in a shared embedding space
- **Contrastive loss**: loss function that pulls embeddings of matching pairs together and pushes non-matching pairs apart
- **Nearest-neighbor search**: given a query embedding, find the k most similar embeddings by cosine or Euclidean distance (e.g. via FAISS or `sklearn`)

## Tech stack

- Python ≥ 3.11, managed with `uv`
- `cptac` — proteomics dataset access (CPTAC cohorts)
- Deep learning framework: TBD (likely PyTorch)
- Embedding / retrieval: TBD (likely FAISS)

## Project layout

```
main.py          # entry point (stub)
pyproject.toml   # dependencies
```

## Development conventions

- Use `uv` to manage the virtual environment and add dependencies (`uv add <pkg>`)
- Run the project with `uv run python main.py`
- Explain domain-specific proteomics or ML concepts when they come up — the developer is new to both research and proteomics

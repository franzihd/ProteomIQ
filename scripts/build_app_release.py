"""
Assembles a standalone, professor-facing release of just the ProteomIQ tool
(src/app.py and its runtime dependencies), separate from the thesis's model
training/evaluation code.

This is a GENERATED artifact, not something to hand-edit -- regenerate it any
time src/app.py or one of the files it imports changes. That's the point:
copying the actual source files (rather than hand-maintaining a second,
rewritten copy) means the release can never silently drift out of sync with
the real code.

CODE_FILES below is the exact import chain src/app.py touches at runtime,
traced by hand: app.py -> retrieve_and_annotate.py / interpret.py ->
dataset.py / projection_head.py / embed_biobert.py / normalize_uploaded_sample.py.
Nothing under src/evaluation/, src/model/train*.py, src/model/sweep*.py,
src/Textual/, or src/preprocessing/rank_normalize.py is imported by the app,
directly or transitively, so none of it is included.

DATA_FILES is the checkpoint, ESM2 embeddings, protein annotations, and one
example patient sample -- everything the app needs to run out of the box.
biobert_embeddings*.h5 (the ~4GB training-text cube) is deliberately NOT
included: the app only ever encodes query text live (encode_query_vector in
retrieve_and_annotate.py), it never reads a precomputed biobert_embeddings*.h5
file.

Usage:
    uv run python scripts/build_app_release.py [--zip]

Output: dist/proteomiq-app/ -- a self-contained folder (code + a trimmed
pyproject.toml + ~94MB of data files + its own README, no thesis code
visible). With --zip, also writes dist/proteomiq-app.zip for easy sharing.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "proteomiq-app"

CODE_FILES = [
    "src/app.py",
    "src/model/__init__.py",
    "src/model/retrieve_and_annotate.py",
    "src/model/interpret.py",
    "src/model/dataset.py",
    "src/model/projection_head.py",
    "src/embeddings/textual/embed_biobert.py",
    "src/preprocessing/normalize_uploaded_sample.py",
]

# Branding assets (favicon + header wordmark) + the Streamlit theme config
# (recolors native widgets -- buttons, slider, sidebar -- to match the brand,
# and hides the local-dev "Deploy" toolbar). All optional: app.py falls back
# to a plain emoji title if the images are missing, and Streamlit just uses
# its own defaults if config.toml is missing -- neither crashes the app --
# but a professor-facing release should obviously ship them.
ASSET_FILES = [
    "src/assets/icon.png",
    "src/assets/favicon.png",
    "src/assets/logo.jpg",
    ".streamlit/config.toml",
]

DATA_FILES = [
    "data/full_proteome/esm2_embeddings.h5",
    "data/full_proteome/protein_annotations.csv",
    "data/runs/full_proteome/multiview_experiment/multiview_best.pt",
    "data/test_abundance_matrix/11LU013.csv",
]

# Deliberately NOT the main project's pyproject.toml: that one pins torch to
# a Linux+CUDA-specific custom index (cu118) and hard-depends on nvidia-*-cu12
# packages, which would break `uv sync` outright on a Mac or Windows machine.
# Plain PyPI torch ships CPU-capable wheels for every OS, which is all the
# app needs (README: "a GPU is not required").
RELEASE_PYPROJECT = """\
[project]
name = "proteomiq-app"
version = "0.1.0"
description = "ProteomIQ -- natural-language proteomics search tool"
readme = "README.md"
requires-python = ">=3.11,<3.14"
dependencies = [
    "adjusttext>=1.4.0",
    "anthropic>=0.111.0",
    "h5py>=3.16.0",
    "matplotlib>=3.11.0",
    "pandas>=3.0.2",
    "pillow>=11.0.0",
    "streamlit>=1.58.0",
    "torch>=2.6.0",
    "tqdm>=4.67.3",
    "transformers>=5.8.0",
]
"""

RELEASE_README = """\
# ProteomIQ

A tool for exploring one proteomics sample with natural-language questions --
upload your measured proteins, ask something like *"receptor tyrosine kinase
signaling"*, and see which of your sample's own proteins are most relevant,
alongside how abundant each one is.

This is a self-contained, ready-to-run package -- everything needed is
already included (the trained model, the reference protein set, and an
example sample), nothing else to download.

## Setup

1. Install [`uv`](https://docs.astral.sh/uv/) (Python package manager), if
   you don't already have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. From inside this folder:
   ```bash
   uv sync
   ```
   (A GPU is not required -- this runs fine on a normal laptop CPU, just a
   few seconds slower per query.)

## Running it

```bash
uv run streamlit run src/app.py
```

Opens automatically in your browser (usually http://localhost:8501). The
first run takes a little longer while the model loads.

## Using it

1. **Sample** -- upload your own CSV (`gene,abundance`, one row per protein),
   or click "Try with example data" to explore an already-included real
   patient sample.
2. **Question** -- ask something in plain language, e.g. *"immune evasion"*
   or *"proteins associated with poor prognosis in lung cancer"*.
3. **Results** -- the most relevant measured proteins, each shown with its
   relevance to your question and its abundance in this sample, plus a rank
   plot showing where they sit in the sample's overall abundance profile.
4. **Interpretation** (optional) -- click "Interpret with Claude" for a
   natural-language summary. Needs an Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```
   (get one at [console.anthropic.com](https://console.anthropic.com); a
   small per-request cost applies, a few cents per interpretation). Retrieval
   and abundance results work fine without this.

## What this tool is, and isn't

ProteomIQ explores *one sample at a time* -- closer to how a clinician
reviews a single patient's profile than to a population-level study. It is
**not** a substitute for group-level differential expression analysis (tools
like GSEA or limma, which compare many samples across conditions with proper
statistical testing). Treat its output as a starting point for hypothesis
generation, not a validated finding.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true", help="also write dist/proteomiq-app.zip")
    args = parser.parse_args()

    missing = [f for f in CODE_FILES + DATA_FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(
            "Can't build the release -- missing from this checkout:\n"
            + "\n".join(f"  {m}" for m in missing)
        )
    # Asset files are optional -- app.py itself degrades gracefully (falls
    # back to a plain emoji title) if branding isn't present, so a missing
    # logo shouldn't fail the whole build the way a missing checkpoint does.
    present_assets = [f for f in ASSET_FILES if (ROOT / f).exists()]

    if OUT.exists():
        shutil.rmtree(OUT)

    all_files = CODE_FILES + present_assets + DATA_FILES
    for rel_path in all_files:
        dst = OUT / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel_path, dst)

    (OUT / "pyproject.toml").write_text(RELEASE_PYPROJECT)
    (OUT / "README.md").write_text(RELEASE_README)

    total_size = sum((OUT / f).stat().st_size for f in all_files)
    print(f"Built {OUT} ({total_size / 1e6:.0f} MB across "
          f"{len(CODE_FILES)} code files + {len(present_assets)}/{len(ASSET_FILES)} "
          f"branding assets + {len(DATA_FILES)} data files)")

    if args.zip:
        zip_path = shutil.make_archive(
            str(ROOT / "dist" / "proteomiq-app"), "zip",
            root_dir=OUT.parent, base_dir=OUT.name,
        )
        print(f"Zipped to {zip_path}")


if __name__ == "__main__":
    main()

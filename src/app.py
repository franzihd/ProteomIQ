import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import torch
import pandas as pd
import anthropic

from src.model.retrieve_and_annotate import load_model, load_esm2_lookup, build_gene_lookup
from src.model.interpret import (
    build_annotation_lookup,
    retrieve_within_sample,
    stage3_interpret,
    DEFAULT_TOP_K,
)
from src.preprocessing.normalize_uploaded_sample import normalize_uploaded_sample

EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "test_abundance_matrix" / "11LU013.csv"
EXAMPLE_LABEL = "LUAD patient 11LU013 (from the CPTAC study data)"
REQUIRED_FILES = [
    Path("data/full_proteome/esm2_embeddings.h5"),
    Path("data/full_proteome/protein_annotations.csv"),
    Path("data/runs/full_proteome/multiview_experiment/multiview_best.pt"),
]

st.set_page_config(page_title="ProteomIQ", page_icon="🧬", layout="wide")

# ---------------------------------------------------------------------------
# Styling. Streamlit's default look is intentionally neutral -- this gives
# the tool its own identity (a calm science/biology palette, card-based
# results instead of a raw dataframe, numbered steps) without needing a
# different framework.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --teal-900: #0b3d3a;
    --teal-700: #146b64;
    --teal-500: #1f9188;
    --teal-100: #e3f3f1;
    --ink: #1a2421;
    --muted: #5b6b67;
    --card-bg: #ffffff;
    --card-border: #e2e8e6;
}

.hero {
    background: linear-gradient(135deg, var(--teal-900), var(--teal-700));
    border-radius: 16px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { margin: 0 0 0.35rem 0; font-size: 2.1rem; font-weight: 700; }
.hero p { margin: 0; font-size: 1.02rem; opacity: 0.92; max-width: 46rem; line-height: 1.5; }

.step-header {
    display: flex; align-items: center; gap: 0.65rem;
    margin: 1.6rem 0 0.6rem 0;
}
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.8rem; height: 1.8rem; border-radius: 50%;
    background: var(--teal-700); color: white; font-weight: 700; font-size: 0.95rem;
    flex-shrink: 0;
}
.step-title { font-size: 1.15rem; font-weight: 600; color: var(--ink); }
.step-sub { color: var(--muted); font-size: 0.92rem; margin: 0 0 0.9rem 2.45rem; }

.result-card {
    background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px;
    padding: 0.85rem 1.1rem; margin-bottom: 0.6rem;
}
.result-row { display: flex; align-items: center; gap: 0.9rem; }
.rank-badge {
    width: 1.6rem; height: 1.6rem; border-radius: 50%; background: var(--teal-100);
    color: var(--teal-900); font-weight: 700; font-size: 0.8rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.gene-name { font-weight: 700; font-size: 1.05rem; color: var(--ink); min-width: 5.5rem; }
.metric-block { flex: 1; min-width: 9rem; }
.metric-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
.metric-bar-track { background: #eef2f1; border-radius: 6px; height: 0.5rem; overflow: hidden; margin-top: 0.15rem; }
.metric-bar-fill { height: 100%; border-radius: 6px; }
.metric-value { font-size: 0.78rem; color: var(--ink); font-weight: 600; margin-top: 0.1rem; }

.callout {
    background: var(--teal-100); border-left: 4px solid var(--teal-500);
    border-radius: 8px; padding: 1rem 1.2rem; color: var(--ink); line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)


def step_header(number: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="step-header"><span class="step-badge">{number}</span>'
        f'<span class="step-title">{title}</span></div>'
        f'<div class="step-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def render_result_card(rank: int, gene: str, cos_sim: float, percentile: float, annotation: str | None):
    # cos_sim is a cosine similarity in roughly [-1, 1]; clamp to [0, 1] for
    # the bar width so a negative (irrelevant) score doesn't render as a
    # negative-width bar.
    relevance_pct = max(0.0, min(1.0, (cos_sim + 1) / 2)) * 100
    abundance_pct = percentile * 100
    with st.container():
        st.markdown(f"""
        <div class="result-card">
          <div class="result-row">
            <span class="rank-badge">{rank}</span>
            <span class="gene-name">{gene}</span>
            <div class="metric-block">
              <div class="metric-label">Semantic relevance to query</div>
              <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{relevance_pct:.1f}%; background:var(--teal-500);"></div></div>
              <div class="metric-value">cosine similarity {cos_sim:.3f}</div>
            </div>
            <div class="metric-block">
              <div class="metric-label">Abundance in this sample</div>
              <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{abundance_pct:.1f}%; background:#3b6fa0;"></div></div>
              <div class="metric-value">{abundance_pct:.0f}th percentile</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if annotation:
            with st.expander(f"What is {gene}?", expanded=False):
                st.caption(annotation)


# ---------------------------------------------------------------------------
# Startup checks -- fail with a clear, actionable message instead of a raw
# traceback if the large data files this tool needs aren't present (they're
# not shipped via git; see the setup note in the sidebar).
# ---------------------------------------------------------------------------
missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.markdown('<div class="hero"><h1>🧬 ProteomIQ</h1>'
                '<p>Query your proteomics sample with natural language.</p></div>',
                unsafe_allow_html=True)
    st.error(
        "Required data files are missing, so the tool can't start:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\nThese are large files not stored in the git repository. "
          "Ask whoever set up this project to share them with you (they should be copied "
          "into the same relative paths shown above, inside the ProteomIQ project folder)."
    )
    st.stop()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model(_device):
    """Trained projection heads (checkpoint) + frozen BioBERT for live query
    encoding -- reused from retrieve_and_annotate.py, not reimplemented."""
    return load_model(_device)


@st.cache_resource
def get_esm2_lookup():
    return load_esm2_lookup()


@st.cache_resource
def get_gene_lookup():
    return build_gene_lookup()


@st.cache_resource
def get_annotation_lookup():
    return build_annotation_lookup()


@st.cache_resource
def get_claude_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic()


with st.spinner("Loading model..."):
    protein_head, text_head, tokenizer, biobert_model = get_model(DEVICE)
    esm2_lookup = get_esm2_lookup()
    gene_lookup = get_gene_lookup()
    annotation_lookup = get_annotation_lookup()
    client = get_claude_client()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🧬 ProteomIQ</h1>
  <p>Upload a patient's proteomics sample, then ask in plain language which proteins are
  relevant to a biological question — e.g. "receptor tyrosine kinase signaling" or
  "proteins associated with poor prognosis in lung cancer". ProteomIQ searches only the
  proteins actually measured in this sample, and shows both how relevant each one is to
  your question and how abundant it is in this patient.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How does this work, and what do the numbers mean?"):
    st.markdown("""
- **Semantic relevance** (teal bar) — how closely a protein's known biological function matches
  your question, measured by a model trained to connect protein sequences with natural-language
  descriptions of protein function. Score range: **cosine similarity**, roughly -1 (unrelated) to 1
  (closely related, this model was seen to be around 0.3–0.4 for a strong match).
- **Abundance** (blue bar) — how highly this protein was measured relative to every other protein
  in *this specific sample*, as a percentile. This is relative within the sample, not an absolute
  concentration, and not a statement about whether the protein is "overexpressed" — a naturally
  abundant protein (e.g. albumin) will always rank high regardless of regulation.
- **Ranking** is by semantic relevance only — abundance is shown for context, not used to reorder
  or filter results, so a highly relevant but low-abundance protein still shows up.
- Only proteins that were actually measured in the uploaded sample are searched — not the whole
  human proteome — so every result is something a researcher can actually follow up on in this
  patient.
    """)

# ---------------------------------------------------------------------------
# Sidebar: sample input
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Sample")
    uploaded = st.file_uploader("Upload proteomics CSV", type="csv")
    use_example = st.button(f"Try with example data\n({EXAMPLE_LABEL})", use_container_width=True)
    top_k = st.slider("Number of results to show", min_value=5, max_value=50, value=DEFAULT_TOP_K)
    st.markdown("---")
    st.markdown(
        "**Expected CSV format:**\n"
        "```\n"
        "gene,abundance\n"
        "EGFR,1.23\n"
        "KRAS,0.45\n"
        "MET,2.67\n"
        "```\n"
        "One row per protein: gene name + raw abundance value from your own measurement."
    )
    if client is None:
        st.markdown("---")
        st.warning(
            "No Claude API key configured — Stage 3 interpretation is disabled. "
            "Set the `ANTHROPIC_API_KEY` environment variable and restart to enable it. "
            "Retrieval and abundance results still work without it."
        )

# resolve which file to process: an upload takes priority; otherwise the
# example button (sticky via session_state so it survives the rerun the
# button click itself triggers)
if use_example:
    st.session_state["use_example"] = True
if uploaded is not None:
    st.session_state["use_example"] = False

active_source = None
if uploaded is not None:
    active_source = uploaded
    source_id = (uploaded.name, uploaded.size)
elif st.session_state.get("use_example"):
    active_source = EXAMPLE_CSV
    source_id = ("__example__", EXAMPLE_LABEL)

if active_source is None:
    st.info("👈 Upload a proteomics CSV in the sidebar, or click **Try with example data**, to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Step 1: process the sample
# ---------------------------------------------------------------------------
step_header("1", "Sample", "The uploaded (or example) sample, matched against our reference proteome.")

if st.session_state.get("file_id") != source_id:
    with st.spinner("Processing sample..."):
        try:
            result_df, unmatched, n_ambiguous = normalize_uploaded_sample(active_source, "gene", "abundance")
        except Exception as e:
            st.error(
                f"Couldn't read this file as a proteomics sample: {e}\n\n"
                "Make sure it's a CSV with a `gene` column and an `abundance` column, one row per protein."
            )
            st.stop()
        measured_proteins = dict(zip(result_df["ENSP_ID"], result_df["percentile"]))

    st.session_state["file_id"] = source_id
    st.session_state["measured_proteins"] = measured_proteins
    st.session_state["match_info"] = (len(result_df), n_ambiguous, len(unmatched))
    st.session_state["last_query"] = None
    st.session_state.pop("scored", None)
    st.session_state.pop("interpretation", None)

measured_proteins = st.session_state["measured_proteins"]
n_matched, n_ambiguous, n_unmatched = st.session_state["match_info"]

label = source_id[1] if isinstance(active_source, Path) else source_id[0]
cols = st.columns(3)
cols[0].metric("Sample", label if isinstance(active_source, Path) else f"📄 {label}")
cols[1].metric("Proteins matched", f"{n_matched:,}")
cols[2].metric("Dropped (ambiguous / unmatched)", f"{n_ambiguous} / {n_unmatched}")

# ---------------------------------------------------------------------------
# Step 2: query
# ---------------------------------------------------------------------------
step_header("2", "Ask a question", "Describe the biology you're interested in, in your own words.")
query = st.text_input(
    "Question", label_visibility="collapsed",
    placeholder="e.g. receptor tyrosine kinase signaling, or immune evasion in lung cancer",
)

if not query:
    st.stop()

if st.session_state.get("last_query") != query:
    with st.spinner("Searching this sample's proteins..."):
        scored = retrieve_within_sample(
            query, measured_proteins, esm2_lookup,
            protein_head, text_head, tokenizer, biobert_model, DEVICE,
        )
    st.session_state["scored"] = scored
    st.session_state["last_query"] = query
    st.session_state.pop("interpretation", None)
else:
    scored = st.session_state["scored"]

# ---------------------------------------------------------------------------
# Step 3: results
# ---------------------------------------------------------------------------
step_header("3", "Most relevant proteins", f"Top {top_k} of {len(scored):,} measured proteins, ranked by relevance to your question.")

top = scored[:top_k]
for i, (pid, cos_sim, pct) in enumerate(top, start=1):
    gene = gene_lookup.get(pid, pid)
    annotation = annotation_lookup.get(pid)
    render_result_card(i, gene, cos_sim, pct, annotation)

# ---------------------------------------------------------------------------
# Step 4: interpretation
# ---------------------------------------------------------------------------
step_header("4", "Interpretation", "Have Claude explain what this result set might mean biologically.")

if client is None:
    st.caption("Unavailable — no Claude API key configured (see sidebar).")
else:
    if st.button("🔍 Interpret with Claude", type="primary"):
        with st.spinner("Asking Claude..."):
            try:
                text = stage3_interpret(client, query, scored, gene_lookup, annotation_lookup, top_k=top_k)
                st.session_state["interpretation"] = text
            except Exception as e:
                st.error(f"Claude request failed: {e}")

    if "interpretation" in st.session_state:
        st.markdown(f'<div class="callout">{st.session_state["interpretation"]}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "ProteomIQ is a research prototype for hypothesis generation, not a diagnostic tool. "
    "See the accompanying thesis for methodology, evaluation, and known limitations."
)

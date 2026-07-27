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


st.set_page_config(page_title="ProteomIQ", layout="wide")
st.title("ProteomIQ")
st.caption("Query your proteomics sample with natural language.")

with st.sidebar:
    st.header("Sample")
    uploaded = st.file_uploader("Upload proteomics CSV", type="csv")
    top_k = st.slider("Top K results", min_value=5, max_value=50, value=DEFAULT_TOP_K)
    st.markdown("---")
    st.markdown(
        "**Expected CSV format:**\n"
        "```\n"
        "gene,abundance\n"
        "EGFR,1.23\n"
        "KRAS,0.45\n"
        "MET,2.67\n"
        "```\n"
        "One row per protein: gene name + raw abundance value."
    )

if uploaded is None:
    st.info("Upload a proteomics CSV in the sidebar to get started")


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
    return anthropic.Anthropic()


protein_head, text_head, tokenizer, biobert_model = get_model(DEVICE)
esm2_lookup = get_esm2_lookup()
gene_lookup = get_gene_lookup()
annotation_lookup = get_annotation_lookup()
client = get_claude_client()


if uploaded is not None:
    # process the sample only when a new file is uploaded -- reruns on every
    # widget interaction otherwise, so this must be guarded
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Processing sample..."):
            result_df, unmatched, n_ambiguous = normalize_uploaded_sample(uploaded, "gene", "abundance")
            measured_proteins = dict(zip(result_df["ENSP_ID"], result_df["percentile"]))

        st.session_state["file_id"] = file_id
        st.session_state["measured_proteins"] = measured_proteins
        st.session_state["match_info"] = (len(result_df), n_ambiguous, len(unmatched))
        st.session_state["last_query"] = None
        st.session_state.pop("scored", None)
        st.session_state.pop("interpretation", None)

    measured_proteins = st.session_state["measured_proteins"]
    n_matched, n_ambiguous, n_unmatched = st.session_state["match_info"]
    st.sidebar.success(
        f"{n_matched} proteins matched"
        f" ({n_ambiguous} dropped as ambiguous, {n_unmatched} not in our reference set)"
    )

    query = st.text_input("Ask your question about the sample",
                           placeholder="e.g. receptor tyrosine kinase signaling")

    if query:
        # only rerun retrieval when the query text itself changed -- not on
        # every rerun caused by an unrelated widget (e.g. the top-K slider)
        if st.session_state.get("last_query") != query:
            with st.spinner("Retrieving..."):
                scored = retrieve_within_sample(
                    query, measured_proteins, esm2_lookup,
                    protein_head, text_head, tokenizer, biobert_model, DEVICE,
                )
            st.session_state["scored"] = scored
            st.session_state["last_query"] = query
            st.session_state.pop("interpretation", None)
        else:
            scored = st.session_state["scored"]

        top = scored[:top_k]
        results = pd.DataFrame({
            "Rank": range(1, len(top) + 1),
            "Gene": [gene_lookup.get(pid, pid) for pid, _, _ in top],
            "Cosine similarity": [round(cos_sim, 4) for _, cos_sim, _ in top],
            "Abundance percentile": [round(pct * 100, 1) for _, _, pct in top],
        })
        st.dataframe(results, use_container_width=True, hide_index=True)

        if st.button("Interpret with Claude"):
            with st.spinner("Asking Claude..."):
                text = stage3_interpret(client, query, scored, gene_lookup, annotation_lookup, top_k=top_k)
            st.session_state["interpretation"] = text

        if "interpretation" in st.session_state:
            st.markdown("### Interpretation")
            st.write(st.session_state["interpretation"])

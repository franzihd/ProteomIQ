import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import torch
import h5py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.model.projection_head import ProjectionHead
from src.embeddings.textual.embed_biobert import MAX_LENGTH, mean_pool


st.set_page_config(page_title="ProteomIQ", layout="wide")
st.title("ProteomIQ")
st.caption("Query your proteomics saple with natural language.")

with st.sidebar:
    st.header("Sample")
    uploaded = st.file_uploader("Upload proteomics CSV", type="csv")
    top_k = st.slider("Top K results", min_value=5, max_value=50, value=15)
    st.markdown("---")
    st.markdown(
        "**Expected CSV format:**\n"
        "```\n"
        "EGFR,KRAS,MET,...\n"
        "sample_001,1.23,0.45,2.67,...\n"
        "```\n"
        "First row: gene names. Second row: sample ID + abundance values."
    )

if uploaded is None:
    st.info("Upload a proteomics CSV in the sidebar to get started")


CHECKPOINT_PATH = Path("data/runs/luad_brca_ccrcc_gbm_hnscc_lscc_ov_pdac_ucec/checkpoint.pt")
ESM2_PATH = Path("data/luad/esm2_embeddings.h5")
PROTEIN_ANNOTATIONS_PATH = Path("data/luad/protein_annotations.csv")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(device)
    biobert.eval()
    protein_head = ProjectionHead(input_dim=641, hidden_dim=512, output_dim=256).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=512, output_dim=256).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    protein_head.load_state_dict(checkpoint["protein_head"])
    text_head.load_state_dict(checkpoint["text_head"])
    protein_head.eval()
    text_head.eval()
    return device, tokenizer, biobert, protein_head, text_head

@st.cache_resource
def load_esm2():
    with h5py.File(ESM2_PATH, "r") as h5f:
        matrix = torch.tensor(h5f["embeddings"][:], dtype=torch.float32)
        ids = [pid.decode() for pid in h5f["protein_ids"][:]]
        
    return matrix, {pid: i for i, pid in enumerate(ids)}

@st.cache_data
def load_gene_to_ensp():
    df = pd.read_csv(PROTEIN_ANNOTATIONS_PATH)
    return dict(zip(df["Gene"], df["ENSP_ID"]))
                                   

device, tokenizer, biobert, protein_head, text_head = load_models()
esm2_matrix, esm2_id_to_idx = load_esm2()
gene_to_ensp = load_gene_to_ensp()

def encode_sample(df, esm2_matrix, esm2_id_to_idx, gene_to_ensp, protein_head, device):
    row = df.iloc[0]

    # rank-normalize abundances within the sample to percentiles [0, 1]
    ranked = row.rank(pct=True)

    protein_embs, gene_names = [], []
    for gene_name, abundance in ranked.items():
        # skip missing values
        if pd.isna(abundance):
            continue
        # translate gene name to ENSP ID, skip if no match
        ensp_id = gene_to_ensp.get(gene_name)
        if ensp_id is None or ensp_id not in esm2_id_to_idx:
            continue
        # look up 640-dim ESM2 vector and append abundance scalar → 641-dim
        esm2_vec= esm2_matrix[esm2_id_to_idx[ensp_id]]
        protein_emb = torch.cat([esm2_vec, torch.tensor([abundance], dtype=torch.float32)])
        protein_embs.append(protein_emb)
        gene_names.append(gene_name)

    # run all protein embeddings through the projection head → 256-dim vectors
    stacked = torch.stack(protein_embs).to(device)
    with torch.no_grad():
        protein_vecs = protein_head(stacked)
    return protein_vecs.cpu(), gene_names

def encode_query(query, tokenizer, biobert, text_head, device):
    # tokenize the query text
    encoded = tokenizer(query, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    # move to GPU/CPU
    encoded = {k: v.to(device) for k, v in encoded.items()}
    # run through BioBERT
    with torch.no_grad():
        outputs = biobert(**encoded)
    # mean-pool token embeddings -> singel 768 dim vector
    text_emb = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    # project to shared 356-dim space
    with torch.no_grad():
        vec = text_head(text_emb)
    return vec.cpu()

if uploaded is not None:
    # process the saple only when a new file is uploaded
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("file_id") != file_id:
        df = pd.read_csv(uploaded, index_col=0)
        with st.spinner("Processing sample..."):
            protein_vecs, gene_names = encode_sample(df, esm2_matrix, esm2_id_to_idx, gene_to_ensp, protein_head, device)
        
        st.session_state["file_id"] = file_id
        st.session_state["protein_vecs"] = protein_vecs
        st.session_state["gene_names"] = gene_names

    # retrieve stored protein vectors and gene names from session state
    protein_vecs = st.session_state["protein_vecs"]
    gene_names = st.session_state["gene_names"]

    # show how many proteins were successfully matched in the sidebar
    st.sidebar.success(f"{len(gene_names)} proteins matched")

    # text input for the query
    query = st.text_input("Ask your question about the sample", placeholder="e.g. receptor tyorsine kinase signaling in lung cancer")

    if query:
        with st.spinner("Retrieving..."):
            # encode query text → 256-dim vector via BioBERT + text projection head
            query_vec = encode_query(query, tokenizer, biobert, text_head, device)
            # cosine similiarity between query and all proteins vectors
            sims = (protein_vecs @ query_vec.T).squeeze().numpy()
            top_indices = np.argsort(sims)[::-1][:top_k]

        # build results table
        results = pd.DataFrame({
            "Rank": range(1, top_k + 1),
            "Gene": [gene_names[i] for i in top_indices],
            "Similarity": [round(float(sims[i]), 4) for i in top_indices],
        })
        # display as interactive table
        st.dataframe(results, use_container_width=True, hide_index=True)
        
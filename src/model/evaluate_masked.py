"""
For patient 11LU035, take 100 proteins present in their sample.
For each protein, query with full UniProt annotation (with gene name) and
masked annotation (gene name stripped). Compare ranks side by side.

Produces a scatter plot: x = rank with name, y = rank without name.
Points on the diagonal = masking made no difference.

Usage:
    uv run python src/model/evaluate_masked.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from src.embeddings.textual.embed_biobert import MAX_LENGTH, mean_pool
from src.model.projection_head import ProjectionHead

DATA_DIR        = Path("data/luad")
CHECKPOINT_PATH = Path("data/runs/luad_brca_ccrcc_gbm_hnscc_lscc_ov_pdac_ucec/checkpoint.pt")
N_PROTEINS      = None  # None = use all proteins in the patient's sample
PATIENT_ID      = "11LU035"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
biobert   = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2").to(device).eval()

protein_head = ProjectionHead(641, 512, 256).to(device)
text_head    = ProjectionHead(768, 512, 256).to(device)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
protein_head.load_state_dict(ckpt["protein_head"])
text_head.load_state_dict(ckpt["text_head"])
protein_head.eval()
text_head.eval()

# load patient abundance + ESM2
abundance_df = pd.read_csv(DATA_DIR / "rank_normalized_proteomics.csv",
                            header=[0, 1], index_col=0)
with h5py.File(DATA_DIR / "esm2_embeddings.h5") as f:
    esm2_matrix      = torch.tensor(f["embeddings"][:], dtype=torch.float32)
    esm2_protein_ids = [p.decode() for p in f["protein_ids"][:]]
esm2_id_to_idx = {pid: i for i, pid in enumerate(esm2_protein_ids)}

annotations    = pd.read_csv(DATA_DIR / "protein_annotations.csv")
ensp_to_gene   = dict(zip(annotations["ENSP_ID"], annotations["Gene"]))
ensp_to_annot  = dict(zip(annotations["ENSP_ID"], annotations["Annotation"]))

# build all protein embeddings for this patient
patient_row  = abundance_df.loc[PATIENT_ID]
protein_embs = []
protein_ids  = []

for (_, protein_id), abundance in patient_row.items():
    if np.isnan(abundance):
        continue
    if protein_id not in esm2_id_to_idx:
        continue
    if protein_id not in ensp_to_annot:
        continue
    if not isinstance(ensp_to_annot[protein_id], str):
        continue
    esm2_vec = esm2_matrix[esm2_id_to_idx[protein_id]]
    protein_embs.append(torch.cat([esm2_vec, torch.tensor([abundance])]))
    protein_ids.append(protein_id)

protein_embs = torch.stack(protein_embs).to(device)
with torch.no_grad():
    protein_vecs = protein_head(protein_embs).cpu()  # (N_patient, 256)

print(f"Patient {PATIENT_ID}: {len(protein_ids)} proteins with embeddings")

# use all proteins or a random sample if N_PROTEINS is set
if N_PROTEINS is None:
    sample_ids = protein_ids
else:
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(protein_ids), size=min(N_PROTEINS, len(protein_ids)), replace=False)
    sample_ids     = [protein_ids[i] for i in sample_indices]

# encode full and masked annotations for the 100 sampled proteins
def encode_texts(texts):
    vecs = []
    BATCH = 32
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tokenizer(texts[i:i+BATCH], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = biobert(**enc)
            vecs.append(mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu())
    raw = torch.cat(vecs, dim=0)
    with torch.no_grad():
        projected = []
        for i in range(0, len(raw), 256):
            projected.append(text_head(raw[i:i+256].to(device)).cpu())
    return torch.cat(projected, dim=0)

descriptions = pd.read_csv(DATA_DIR / "descriptions.csv", index_col=0)
clinical     = descriptions.loc[PATIENT_ID, "description"]

full_texts   = [ensp_to_annot[pid] + " " + clinical for pid in sample_ids]
masked_texts = [
    (ann.split(": ", 1)[1] if ": " in ann else ann) + " " + clinical
    for ann in [ensp_to_annot[pid] for pid in sample_ids]
]

print("Encoding full annotations...")
full_vecs   = encode_texts(full_texts)    # (100, 256)
print("Encoding masked annotations...")
masked_vecs = encode_texts(masked_texts)  # (100, 256)

# compute ranks
def get_ranks(query_vecs):
    sims   = query_vecs @ protein_vecs.T  # (100, N_patient)
    ranks  = []
    for i, pid in enumerate(sample_ids):
        own_idx  = protein_ids.index(pid)
        sim_row  = sims[i]
        rank     = (sim_row > sim_row[own_idx]).sum().item() + 1
        ranks.append(rank)
    return ranks

ranks_full   = get_ranks(full_vecs)
ranks_masked = get_ranks(masked_vecs)
gene_names   = [ensp_to_gene.get(pid, pid) for pid in sample_ids]
n_total      = len(protein_ids)

# print table for top differences
df = pd.DataFrame({
    "Gene":         gene_names,
    "Rank (full)":  ranks_full,
    "Rank (masked)": ranks_masked,
    "Δ rank":       [m - f for f, m in zip(ranks_full, ranks_masked)],
})
df = df.sort_values("Rank (full)")
print(f"\n{'Gene':<12} {'Full':>8} {'Masked':>8} {'Δ (masked-full)':>16}")
print("-" * 48)
for _, row in df.iterrows():
    delta = int(row["Δ rank"])
    arrow = f"+{delta}" if delta > 0 else str(delta)
    print(f"{row['Gene']:<12} {int(row['Rank (full)']):>8} {int(row['Rank (masked)']):>8} {arrow:>16}")

GREEN  = "#0F6E56"
RED    = "#C0392B"
BLUE   = "#3C3489"
TEXT   = "#26215C"

deltas = df["Δ rank"].tolist()
n_unchanged = sum(1 for d in deltas if d == 0)
n_worse     = sum(1 for d in deltas if d > 0)
n_better    = sum(1 for d in deltas if d < 0)

CLIP = 200
deltas_clipped = [max(-CLIP, min(CLIP, d)) for d in deltas]
n_clipped = sum(1 for d in deltas if abs(d) > CLIP)

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bins = list(range(-CLIP, CLIP + 10, 10))
ax.hist(deltas_clipped, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)

for patch, left_edge in zip(ax.patches, bins):
    if left_edge < 0:
        patch.set_facecolor(GREEN)
    elif left_edge > 0:
        patch.set_facecolor(RED)
    # Δ=0 bin stays BLUE

ax.axvline(0, color="#222222", linewidth=2, zorder=4)

# annotate the three regions
ymax = ax.get_ylim()[1]
ax.annotate(f"{n_unchanged} proteins\nno change",
            xy=(0, n_unchanged), xytext=(40, ymax * 0.75),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
            fontsize=11, color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=1.2))

if n_better > 0:
    ax.text(-CLIP * 0.6, ymax * 0.6,
            f"{n_better} proteins\nimproved", color=GREEN,
            fontsize=10, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, lw=1.2))

if n_worse > 0:
    ax.text(CLIP * 0.6, ymax * 0.6,
            f"{n_worse} proteins\ngot worse", color=RED,
            fontsize=10, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, lw=1.2))

clip_note = f"  ({n_clipped} extreme outliers clipped to ±{CLIP})" if n_clipped else ""
ax.set_xlabel("Rank change when gene name is removed   (negative = improved, positive = got worse)",
              fontsize=11, color=TEXT)
ax.set_ylabel("Number of proteins", fontsize=11, color=TEXT)
ax.set_title(f"Does the model rely on gene names?   Patient {PATIENT_ID} · {N_PROTEINS} proteins{clip_note}",
             fontsize=13, fontweight="bold", color=TEXT, pad=14)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis="y", color="#E2E8F0", linewidth=1.0)
ax.tick_params(labelcolor=TEXT)

plt.tight_layout()
out = Path("data/plots/masked_rank_histogram.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out}")
plt.show()

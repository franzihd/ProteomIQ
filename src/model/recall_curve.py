"""
Plots the Recall@K curve for the LUAD test set (K = 1 … 500).
Imports compute_ranks() from recall_at_k.py and saves the figure to data/luad/recall_curve.png.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.recall_at_k import compute_ranks
import numpy as np
import matplotlib.pyplot as plt

all_ranks = compute_ranks()

# Compute recall at each rank
k_s = list(range(1, 101))
recall = []
for k in k_s:
    recall_k = np.mean([1 if rank <= k else 0 for rank in all_ranks]) * 100
    recall.append(recall_k)

pool_size = all_ranks.max()
random_baseline = [k / pool_size * 100 for k in k_s]

# plot recall curve
plt.figure(figsize=(10, 6))
plt.plot(k_s, recall, label='Model Recall', color='violet', linewidth=4)
plt.plot(k_s, random_baseline, label='Random Baseline', color='lightgray', linestyle='--', linewidth=4)
plt.xlabel('Rank (k)')
plt.ylabel('Recall (%)')
plt.title('Protein Retrieval Recall Curve')
plt.legend()
plt.savefig("data/luad/recall_curve.png", dpi=150)
plt.show()

# bar chart for R@1, R@10, R@50
ks_bar = [1, 10, 50]
values = [(all_ranks <= k).mean() * 100 for k in ks_bar]
random_values = [k / pool_size * 100 for k in ks_bar]

x = np.arange(len(ks_bar))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(x - width/2, values, width, label='ProteomIQ (LUAD)', color='rebeccapurple')
ax.bar(x + width/2, random_values, width, label='Random Baseline', color='lightgray')

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5,
            f'{val:.1f}%', ha='center', va='top', color='white', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(['R@1', 'R@10', 'R@50'])
ax.set_ylabel('Recall (%)')
ax.set_title('Recall@K — LUAD Test Set')
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig("data/luad/bar_chart.png", dpi=150)
plt.show()


"""
Plot scaling results v2: two R@K lines per metric.
  - LUAD anchor (solid): fixed 10 LUAD test patients across all checkpoints
  - All cohorts (dashed): growing test set (10 patients per trained cohort)

Usage:
    uv run python src/plotting/plot_scaling_v2.py
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "DejaVu Sans"

RESULTS_PATH = Path("data/scaling_results_v2.json")
COHORT_ORDER = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]

with open(RESULTS_PATH) as f:
    results = json.load(f)

rows = sorted(results.values(), key=lambda x: x["n_cohorts"])

x        = list(range(1, len(rows) + 1))
x_labels = [COHORT_ORDER[0].upper()] + [f"+{COHORT_ORDER[i].upper()}" for i in range(1, len(x))]

luad_r1  = [r["luad_r1"]  for r in rows]
luad_r10 = [r["luad_r10"] for r in rows]
all_r1   = [r["all_r1"]   for r in rows]
all_r10  = [r["all_r10"]  for r in rows]

GREEN  = "#0F6E56"
PURPLE = "#3C3489"
TEXT   = "#26215C"

fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# LUAD anchor — solid
ax.plot(x, luad_r10, marker="s", color=PURPLE, linewidth=2.5, markersize=9,
        label="R@10 — LUAD only", zorder=2)
ax.plot(x, luad_r1,  marker="o", color=GREEN,  linewidth=3,   markersize=10,
        label="R@1 — LUAD only", zorder=3)
for xi, yi in zip(x, luad_r1):
    ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                xytext=(0, -18), ha="center", fontsize=9, color=GREEN, fontweight="bold")

# All cohorts — dashed
ax.plot(x, all_r10, marker="s", color=PURPLE, linewidth=2.5, markersize=9,
        linestyle="--", alpha=0.7, label="R@10 — all cohorts", zorder=2)
ax.plot(x, all_r1,  marker="o", color=GREEN,  linewidth=3,   markersize=10,
        linestyle="--", alpha=0.7, label="R@1 — all cohorts", zorder=3)
for xi, yi in zip(x, all_r1):
    ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=GREEN, alpha=0.75)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_xlabel("Training cohorts", fontsize=12, labelpad=10, color=TEXT)
ax.set_ylabel("Recall@K (%)", fontsize=12, labelpad=10, color=TEXT)
ax.set_title("R@1 and R@10 across increasing training cohorts\n"
             "Solid = LUAD test patients only (fixed anchor) · Dashed = all trained cohorts",
             fontsize=13, fontweight="bold", pad=16, color=TEXT)

ax.set_xlim(0.5, len(x) + 0.5)
y_min = max(0, min(luad_r1 + all_r1) - 15)
ax.set_ylim(y_min, 105)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.tick_params(axis="both", which="both", length=0, labelcolor=TEXT)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis="y", color="#E2E8F0", linewidth=1.2)
ax.legend(fontsize=10, frameon=False, loc="lower right")

plt.tight_layout()
out = Path("data/plots/scaling_rk_v2.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()

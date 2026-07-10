"""
Plot R@1 and R@10 vs. number of cohorts from a scaling_results JSON
(as produced by src/model/run_experiments.py).
Test set = 10 patients per trained cohort (growing).

Usage:
    uv run python src/plotting/plot_scaling_v2.py               # original (with clinical text)
    uv run python src/plotting/plot_scaling_v2.py --no-clinical  # retrain without clinical text
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "DejaVu Sans"

parser = argparse.ArgumentParser()
parser.add_argument("--no-clinical", action="store_true",
                     help="plot the no-clinical-text retrain instead of the original run")
parser.add_argument("--results-path", default=None, help="override the default results JSON path")
parser.add_argument("--out", default=None, help="override the default output PNG path")
args = parser.parse_args()

if args.results_path is None:
    args.results_path = "data/scaling_results_no_clinical.json" if args.no_clinical else "data/scaling_results_v2.json"
if args.out is None:
    args.out = "data/plots/scaling_rk_no_clinical.png" if args.no_clinical else "data/plots/scaling_rk_v2.png"

RESULTS_PATH = Path(args.results_path)
COHORT_ORDER = ["luad", "brca", "ccrcc", "gbm", "hnscc", "lscc", "ov", "pdac", "ucec"]

with open(RESULTS_PATH) as f:
    results = json.load(f)

rows = sorted(results.values(), key=lambda x: x["n_cohorts"])

x        = list(range(1, len(rows) + 1))
x_labels = [COHORT_ORDER[0].upper()] + [f"+{COHORT_ORDER[i].upper()}" for i in range(1, len(x))]

r1  = [r["all_r1"]  for r in rows]
r10 = [r["all_r10"] for r in rows]

GREEN  = "#0F6E56"
PURPLE = "#3C3489"
TEXT   = "#26215C"

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# R@10 — purple, secondary
ax.plot(x, r10, marker="s", color=PURPLE, linewidth=2.5,
        markersize=9, label="R@10", zorder=2)
for xi, yi in zip(x, r10):
    ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=10, color=PURPLE)

# R@1 — green, primary
ax.plot(x, r1, marker="o", color=GREEN, linewidth=3,
        markersize=10, label="R@1", zorder=3)
for xi, yi in zip(x, r1):
    ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                xytext=(0, -18), ha="center", fontsize=10, color=GREEN, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_xlabel("Training cohorts", fontsize=12, labelpad=10, color=TEXT)
ax.set_ylabel("Recall@K (%)", fontsize=12, labelpad=10, color=TEXT)
ax.set_title("R@1 and R@10 across increasing training cohorts (evaluated on 10 patients per cohort)",
             fontsize=14, fontweight="bold", pad=16, color=TEXT)

ax.set_xlim(0.5, len(x) + 0.5)
y_min = max(0, min(r1) - 15)
ax.set_ylim(y_min, 103)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.tick_params(axis="both", which="both", length=0, labelcolor=TEXT)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis="y", color="#E2E8F0", linewidth=1.2)
ax.legend(fontsize=11, frameon=False, loc="lower right")

plt.tight_layout()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()

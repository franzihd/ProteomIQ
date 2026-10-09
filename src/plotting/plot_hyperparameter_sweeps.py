"""
Figures for the two W&B hyperparameter sweeps.

Usage:
    uv run python src/plotting/plot_hyperparameter_sweeps.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

SWEEP1_PATH = "data/evaluation/hyperparameter_sweeps/sweep1_lr_batchsize_epochs.csv"
SWEEP2_PATH = "data/evaluation/hyperparameter_sweeps/sweep2_hidden_output_dims.csv"
OUTPUT_DIR = "data/plots/hyperparameter_sweeps"

# ProteomIQ brand colors (same values as src/app.py)
BRAND_NAVY, BRAND_BLUE, BRAND_VIOLET, BRAND_BG, BRAND_MUTED = "#14213D", "#2A7BCB", "#7B5CD6", "#F5F7FA", "#5B6477"
# batch size is ordered -> one hue family, light -> dark
BATCH_COLORS = {128: "#A9CBEE", 256: BRAND_BLUE, 512: BRAND_NAVY}
EPOCH_MARKERS = {100: "o", 200: "s", 400: "^"}
INK, MUTED = BRAND_NAVY, BRAND_MUTED
WINNER = BRAND_VIOLET
# low loss = dark (better), high loss = light
LOSS_CMAP = LinearSegmentedColormap.from_list("proteomiq_loss", [BRAND_NAVY, BRAND_BLUE, BRAND_BG])


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8.5)


def text_color_on(rgba) -> str:
    """White text on dark cells, navy text on light cells (relative luminance, WCAG)."""
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if lum < 0.5 else INK


def sweep1_panel(ax, df, y, ylabel, title, winner):
    for (bs, ep), sub in df.groupby(["batch_size", "epochs"]):
        ax.scatter(sub.lr, sub[y], s=55, color=BATCH_COLORS[bs], marker=EPOCH_MARKERS[ep],
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(winner.lr, winner[y], s=190, facecolor="none", edgecolor=WINNER, linewidth=2.2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    style_axis(ax)


def plot_sweep1(s1: pd.DataFrame, output_path: Path):
    winner = s1.iloc[[s1.gap_below_random.idxmax()]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    sweep1_panel(axes[0], s1, "best_val_loss", r"$\mathcal{L}_{\mathrm{val}}$ (↓)",
                 "A  Raw validation loss", winner)
    for bs in BATCH_COLORS:
        axes[0].axhline(np.log(bs), color=BATCH_COLORS[bs], linestyle="--", linewidth=1.2)
        axes[0].annotate(rf"$\ln({bs}) = {np.log(bs):.2f}$  (random baseline)", (1.05e-5, np.log(bs)),
                         xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=MUTED)
    axes[0].set_ylim(2.0, 6.6)

    sweep1_panel(axes[1], s1, "gap_below_random", r"$\ln(N) - \mathcal{L}_{\mathrm{val}}$ (↑)",
                 "B  Corrected selection metric", winner)
    mantissa, exponent = f"{winner.lr.iloc[0]:.2e}".split("e")
    axes[1].annotate(rf"selected: $\mathrm{{lr}} = {mantissa} \times 10^{{{int(exponent)}}}$" "\nN = 512, max. 100 epochs",
                     (winner.lr.iloc[0], winner.gap_below_random.iloc[0]), xytext=(60, -95),
                     textcoords="offset points", fontsize=8, color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=f"N = {bs}")
               for bs, c in BATCH_COLORS.items()]
    handles += [plt.Line2D([], [], marker=m, linestyle="", color=MUTED, markersize=7, label=f"max. {ep} epochs")
                for ep, m in EPOCH_MARKERS.items()]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=WINNER,
                           markeredgewidth=2, markersize=11, label="selected run")]
    fig.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    s1.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")


def plot_sweep2(s2: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.9))
    hidden = sorted(s2.hidden_dim.unique())
    output = sorted(s2.output_dim.unique())
    grid = np.full((len(hidden), len(output)), np.nan)
    for _, r in s2.iterrows():
        grid[hidden.index(r.hidden_dim), output.index(r.output_dim)] = r.best_val_loss
    im = ax.imshow(grid, cmap=LOSS_CMAP, aspect="auto", origin="lower")
    for i in range(len(hidden)):
        for j in range(len(output)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "not allowed\n" r"($d_{\mathrm{out}} > d_{\mathrm{hid}}$)", ha="center", va="center", fontsize=6.5, color=MUTED)
            else:
                best = grid[i, j] == np.nanmin(grid)
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                        color=text_color_on(im.cmap(im.norm(grid[i, j]))),
                        fontweight="bold" if best else "normal")
                if best:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=WINNER, linewidth=2.5))
    ax.set_xticks(range(len(output)))
    ax.set_xticklabels(output)
    ax.set_yticks(range(len(hidden)))
    ax.set_yticklabels(hidden)
    ax.set_xlabel(r"$d_{\mathrm{output}}$ (shared embedding dimension)", color=INK)
    ax.set_ylabel(r"$d_{\mathrm{hidden}}$", color=INK)
    ax.tick_params(colors=INK, labelsize=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$\mathcal{L}_{\mathrm{val}}$ (↓)", color=INK)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    s2.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved to {output_path} (+ .csv)")


def main(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_sweep1(pd.read_csv(SWEEP1_PATH), out / "sweep1_lr_batchsize_epochs.png")
    plot_sweep2(pd.read_csv(SWEEP2_PATH), out / "sweep2_hidden_output_dims.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args.output_dir)

"""
Presentation schematic: how the training text for one protein (TP53) is
built in the three training-text designs -- M0 (function only), M3 (unmasked name + function + family + location concatenated
into one string, CLAUDE.md 11z), and multiview (one field drawn at random
per training example, 11ac). A method illustration, no results.

Example texts are TP53's real UniProt fields from
data/full_proteome/protein_annotations.csv, shortened with "..." where
noted. Concatenation format follows embed_biobert.py's
append_family_location (plain ", "-joined, no field labels).

Usage:
    uv run python src/plotting/plot_training_text_variants.py
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_PATH = "data/plots/methods/training_text_variants.png"

INK = "#333333"
MUTED = "#6b6b6b"
FIELDS = {
    "name": ("Gene name", "#d6453d", "TP53:"),
    "function": ("Function", "#2a78d6",
                 "Multifunctional transcription factor that induces cell cycle arrest, "
                 "DNA repair or apoptosis upon binding to its target DNA sequence ..."),
    "family": ("Family", "#1baf7a", "Belongs to the p53 family."),
    "location": ("Location", "#8a5cc7",
                 "Located in Cytoplasm, Nucleus, Endoplasmic reticulum, Mitochondrion matrix."),
}


def box(ax, x, y, w, h, face, edge, lw=1.2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015",
                                facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha))


def field_box(ax, x, y_top, w, key, wrap=50, label=True, fontsize=8.5):
    """Draws one text field as a tinted box with a colored left bar; returns its bottom y."""
    name, color, text = FIELDS[key]
    lines = textwrap.wrap(text, wrap)
    h = 0.031 * len(lines) + (0.058 if label else 0.024)
    y = y_top - h
    box(ax, x, y, w, h, color, color, lw=0, alpha=0.12)
    ax.add_patch(plt.Rectangle((x, y), 0.012, h, color=color, linewidth=0))
    ty = y_top - 0.012
    if label:
        ax.text(x + 0.03, ty, name.upper(), fontsize=7.5, color=color, fontweight="bold", va="top")
        ty -= 0.03
    ax.text(x + 0.03, ty, "\n".join(lines), fontsize=fontsize, color=INK, va="top", linespacing=1.35)
    return y


def protein_chip(ax, y):
    box(ax, 0.08, y, 0.84, 0.075, "#f1f1f1", "#9a9a9a")
    ax.text(0.5, y + 0.0375, "TP53 amino-acid sequence  →  ESM2 (frozen)", ha="center", va="center",
            fontsize=9, color=INK)


def header(ax, title, subtitle):
    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=13, fontweight="bold", color=INK)
    ax.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=9.5, color=MUTED)


def paired_with(ax, y):
    ax.annotate("", xy=(0.5, y - 0.045), xytext=(0.5, y), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
    ax.text(0.54, y - 0.022, "contrastive pair", fontsize=8, color=MUTED, va="center")


def footer(ax, text):
    ax.text(0.5, 0.2, text, ha="center", va="bottom", fontsize=9, color=INK, style="italic")


def main(output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0.14, 1)
        ax.axis("off")

    chip_y, arrow_y, text_top = 0.8, 0.79, 0.735

    # --- M0 ---
    ax = axes[0]
    header(ax, "M0 (baseline)", "UniProt function text only")
    protein_chip(ax, chip_y)
    paired_with(ax, arrow_y)
    field_box(ax, 0.08, text_top, 0.84, "function")
    footer(ax, "Every epoch: the same single text")

    # --- M3: one concatenated string ---
    ax = axes[1]
    header(ax, "M3 (concatenation)", "Gene name + function + family + location in one text")
    protein_chip(ax, chip_y)
    paired_with(ax, arrow_y)
    y = text_top
    for key in ("name", "function", "family", "location"):
        y = field_box(ax, 0.08, y, 0.84, key)
    box(ax, 0.065, y - 0.012, 0.87, text_top - y + 0.024, "none", INK, lw=1.4)
    ax.text(0.5, y - 0.03, "↑ joined into one string", ha="center", va="top", fontsize=8.5, color=MUTED)
    footer(ax, "Every epoch: all fields together in one text")

    # --- Multiview: one random field per example ---
    ax = axes[2]
    header(ax, "Multiview (final)", "Function, family, location as separate views")
    protein_chip(ax, chip_y)
    paired_with(ax, arrow_y)
    y = text_top
    for key in ("function", "family", "location"):
        y = field_box(ax, 0.08, y, 0.84, key) - 0.02
    ax.text(0.5, y - 0.01, "one view drawn at random each epoch", ha="center", va="top",
            fontsize=9, color=INK, fontweight="bold")
    ax.text(0.5, y - 0.055, "e.g. epoch 1 → Location,  epoch 2 → Function,  epoch 3 → Family",
            ha="center", va="top", fontsize=8.5, color=MUTED)
    footer(ax, "Every epoch: one field at a time")

    for ax in axes[1:]:
        ax.plot([-0.04, -0.04], [0.2, 0.95], color="#e0e0e0", linewidth=1, transform=ax.transAxes,
                clip_on=False)

    plt.tight_layout(w_pad=2.5)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)

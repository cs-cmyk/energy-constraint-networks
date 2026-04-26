"""
Generate architecture diagrams for the constraint network paper.

Produces 3 diagrams:
  1. Constraint network architecture (SSM + attention + energy head)
  2. Three-branch composition with gating
  3. Cross-modal transfer (text and vision side by side)

Usage:
  python generate_diagrams.py

Outputs:
  diagram_architecture.svg
  diagram_three_branch.svg
  diagram_cross_modal.svg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# ============================================================
# Color scheme
# ============================================================
COLORS = {
    "ssm": "#4A90D9",
    "attention": "#E8A838",
    "energy": "#D94A4A",
    "encoder": "#6BBF6B",
    "input": "#888888",
    "branch_s": "#4A90D9",
    "branch_f": "#9B59B6",
    "branch_l": "#E67E22",
    "gate": "#95A5A6",
    "bg": "#FAFAFA",
    "text": "#2C3E50",
    "arrow": "#555555",
}

FONT = "sans-serif"


def rounded_box(ax, x, y, w, h, label, color, fontsize=10, text_color="white",
                alpha=0.95, sublabel=None):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor="none",
                          alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x, y + (0.08 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, fontfamily=FONT, zorder=3)
    if sublabel:
        ax.text(x, y - 0.12, sublabel,
                ha="center", va="center", fontsize=fontsize - 2,
                color=text_color, fontfamily=FONT, alpha=0.85, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>", lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)


def bracket_text(ax, x, y, text, fontsize=8, color="#666666"):
    """Add descriptive text next to a component."""
    ax.text(x, y, text, ha="left", va="center", fontsize=fontsize,
            color=color, fontfamily=FONT, style="italic")


# ============================================================
# Diagram 1: Constraint Network Architecture
# ============================================================

def diagram_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    ax.set_xlim(-2, 4)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx = 1.0  # center x
    bw = 2.2  # box width
    bh = 0.5  # box height

    # Title
    ax.text(cx, 10.2, "Constraint Network", ha="center", va="center",
            fontsize=14, fontweight="bold", fontfamily=FONT, color=COLORS["text"])

    # Input
    y = 9.5
    rounded_box(ax, cx, y, bw, bh, "Input Embeddings", COLORS["input"],
                sublabel="(batch, positions, dim)")
    bracket_text(ax, cx + bw/2 + 0.2, y, "BERT windows or\nDINOv2 patches")

    # Projection
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "Linear Projection", COLORS["input"],
                sublabel="768 → 384")
    arrow(ax, cx, 9.5 - bh/2, cx, y + bh/2)

    # SSM blocks 1-2
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "SSM Block 1–2", COLORS["ssm"],
                sublabel="Causal conv + gated + decay")
    arrow(ax, cx, 8.6 - bh/2, cx, y + bh/2)
    bracket_text(ax, cx + bw/2 + 0.2, y, "Local sequential\npatterns")

    # Attention 1
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "Two-Head Attention", COLORS["attention"],
                sublabel="Causal + Bidirectional")
    arrow(ax, cx, y + 0.9 - bh/2 + 0.9, cx, y + bh/2)
    bracket_text(ax, cx + bw/2 + 0.2, y, "Head 1: forward consistency\n"
                 "Head 2: mutual compatibility")

    # SSM blocks 3-4
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "SSM Block 3–4", COLORS["ssm"],
                sublabel="Causal conv + gated + decay")
    arrow(ax, cx, y + 0.9 - bh/2 + 0.9, cx, y + bh/2)

    # Attention 2
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "Two-Head Attention", COLORS["attention"],
                sublabel="Causal + Bidirectional")
    arrow(ax, cx, y + 0.9 - bh/2 + 0.9, cx, y + bh/2)

    # SSM blocks 5-6
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "SSM Block 5–6", COLORS["ssm"],
                sublabel="Causal conv + gated + decay")
    arrow(ax, cx, y + 0.9 - bh/2 + 0.9, cx, y + bh/2)

    # Energy head
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "Energy Head", COLORS["energy"],
                sublabel="LayerNorm → MLP → per-position")
    arrow(ax, cx, y + 0.9 - bh/2 + 0.9, cx, y + bh/2)

    # Output
    y -= 0.9
    rounded_box(ax, cx, y, bw, bh, "E = mean(eᵢ) + α·max(eᵢ)", COLORS["energy"],
                fontsize=9, sublabel="Scalar energy + per-position scores")
    arrow(ax, cx, y + 0.9 - bh/2 + 0.9, cx, y + bh/2)

    # Side annotations
    # SSM bracket
    ax.annotate("", xy=(-0.4, 7.7 + bh/2), xytext=(-0.4, 3.8 - bh/2),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=1))
    ax.text(-0.6, 5.75, "6 SSM\nblocks", ha="center", va="center",
            fontsize=8, color="#999999", fontfamily=FONT, rotation=90)

    # Attention bracket
    ax.annotate("", xy=(-0.2, 6.8 + bh/2), xytext=(-0.2, 5.0 - bh/2),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=1))
    ax.text(-0.55, 5.9, "2 attn\nblocks", ha="center", va="center",
            fontsize=8, color="#999999", fontfamily=FONT, rotation=90)

    # Parameter count
    ax.text(cx, 0.8, "3.6M trainable parameters per branch",
            ha="center", va="center", fontsize=9, color="#999999",
            fontfamily=FONT, style="italic")

    plt.tight_layout()
    plt.savefig("diagram_architecture.svg", format="svg", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.savefig("diagram_architecture.pdf", format="pdf", bbox_inches="tight",
                facecolor="white", dpi=150)
    print("  Saved diagram_architecture.svg/pdf")
    plt.close()


# ============================================================
# Diagram 2: Three-Branch Composition
# ============================================================

def diagram_three_branch():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bw = 2.5
    bh = 0.55

    # Title
    ax.text(5, 6.7, "Three-Branch Composable Architecture", ha="center",
            fontsize=14, fontweight="bold", fontfamily=FONT, color=COLORS["text"])

    # Input image
    rounded_box(ax, 5, 6.0, 2.0, 0.45, "Input Image", COLORS["input"],
                fontsize=10)

    # DINOv2 encoder (shared)
    rounded_box(ax, 5, 5.2, 3.0, 0.45, "Frozen DINOv2 ViT-B/14", COLORS["encoder"],
                fontsize=10, sublabel="86M params (shared)")
    arrow(ax, 5, 6.0 - 0.45/2, 5, 5.2 + 0.45/2)

    # Three branches
    positions = {
        "structural": (1.5, "Structural\nBranch", COLORS["branch_s"],
                       "DINOv2(original)", "Face swaps,\nexpression transfers"),
        "frequency": (5.0, "Frequency\nBranch", COLORS["branch_f"],
                      "DINOv2(freq heatmap)", "GAN smoothing,\ntexture loss"),
        "local": (8.5, "Local Texture\nBranch", COLORS["branch_l"],
                  "DINOv2(original)", "Neural rendering,\nlocal inconsistency"),
    }

    for name, (x, label, color, input_desc, detects) in positions.items():
        # Input processing
        rounded_box(ax, x, 4.2, 2.3, 0.4, input_desc, "#AAAAAA",
                     fontsize=8, text_color="white")

        # Arrow from DINOv2
        if name == "frequency":
            arrow(ax, 5, 5.2 - 0.45/2, x, 4.2 + 0.4/2)
        else:
            arrow(ax, 5, 5.2 - 0.45/2, x, 4.2 + 0.4/2)

        # Constraint network
        rounded_box(ax, x, 3.3, 2.3, 0.55, label, color,
                     fontsize=10, sublabel="3.6M params")
        arrow(ax, x, 4.2 - 0.4/2, x, 3.3 + 0.55/2)

        # Energy output
        e_label = {"structural": "E_s", "frequency": "E_f", "local": "E_l"}[name]
        rounded_box(ax, x, 2.4, 1.2, 0.35, e_label, color,
                     fontsize=11, alpha=0.7)
        arrow(ax, x, 3.3 - 0.55/2, x, 2.4 + 0.35/2)

        # Detection type annotation
        ax.text(x, 1.8, detects, ha="center", va="center",
                fontsize=7, color="#666666", fontfamily=FONT, style="italic")

    # Gating
    rounded_box(ax, 5.0, 1.3, 1.8, 0.35, "Dataset Gate", COLORS["gate"],
                fontsize=9, text_color="white")
    arrow(ax, 5.0, 2.4 - 0.35/2, 5.0, 1.3 + 0.35/2, style="-|>", lw=1)

    # Composition
    rounded_box(ax, 5.0, 0.5, 4.5, 0.55, "E = E_s + gate · E_f + β · E_l",
                COLORS["energy"], fontsize=11)

    # Arrows from energies to composition
    arrow(ax, 1.5, 2.4 - 0.35/2, 3.5, 0.5 + 0.55/2, lw=1.2)
    arrow(ax, 5.0, 1.3 - 0.35/2, 5.0, 0.5 + 0.55/2, lw=1.2)
    arrow(ax, 8.5, 2.4 - 0.35/2, 6.5, 0.5 + 0.55/2, lw=1.2)

    # Training annotations
    ax.text(1.5, 2.9, "Trained on:\n5 FF++ methods", ha="center",
            fontsize=7, color=COLORS["branch_s"], fontfamily=FONT)
    ax.text(5.0, 2.9, "Trained on:\nsmoothing corruptions", ha="center",
            fontsize=7, color=COLORS["branch_f"], fontfamily=FONT)
    ax.text(8.5, 2.9, "Trained on:\nlocal corruptions + NT", ha="center",
            fontsize=7, color=COLORS["branch_l"], fontfamily=FONT)

    # Key annotation
    ax.text(5.0, -0.2, "Each branch trained independently — no gradient interference",
            ha="center", fontsize=9, color="#999999", fontfamily=FONT, style="italic")

    plt.tight_layout()
    plt.savefig("diagram_three_branch.svg", format="svg", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.savefig("diagram_three_branch.pdf", format="pdf", bbox_inches="tight",
                facecolor="white", dpi=150)
    print("  Saved diagram_three_branch.svg/pdf")
    plt.close()


# ============================================================
# Diagram 3: Cross-Modal Transfer
# ============================================================

def diagram_cross_modal():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bw = 2.2
    bh = 0.5

    # Title
    ax.text(5, 6.2, "Cross-Modal Transfer: Same Architecture, Different Encoders",
            ha="center", fontsize=13, fontweight="bold", fontfamily=FONT,
            color=COLORS["text"])

    # ---- TEXT SIDE (left) ----
    tx = 2.5

    ax.text(tx, 5.6, "TEXT DOMAIN", ha="center", fontsize=11,
            fontweight="bold", fontfamily=FONT, color=COLORS["branch_s"])

    rounded_box(ax, tx, 5.0, 2.0, 0.4, "Wikipedia paragraph", COLORS["input"],
                fontsize=9)

    rounded_box(ax, tx, 4.2, 2.0, 0.45, "Frozen BERT", COLORS["encoder"],
                fontsize=10, sublabel="110M params")
    arrow(ax, tx, 5.0 - 0.4/2, tx, 4.2 + 0.45/2)

    ax.text(tx, 3.65, "(windows, 768)", ha="center", fontsize=8,
            color="#888888", fontfamily=FONT)

    rounded_box(ax, tx, 3.1, 2.0, 0.4, "768 → 384", "#AAAAAA",
                fontsize=9)
    arrow(ax, tx, 4.2 - 0.45/2, tx, 3.1 + 0.4/2)

    # ---- VISION SIDE (right) ----
    vx = 7.5

    ax.text(vx, 5.6, "VISION DOMAIN", ha="center", fontsize=11,
            fontweight="bold", fontfamily=FONT, color=COLORS["branch_l"])

    rounded_box(ax, vx, 5.0, 2.0, 0.4, "Face image 224×224", COLORS["input"],
                fontsize=9)

    rounded_box(ax, vx, 4.2, 2.0, 0.45, "Frozen DINOv2", COLORS["encoder"],
                fontsize=10, sublabel="86M params")
    arrow(ax, vx, 5.0 - 0.4/2, vx, 4.2 + 0.45/2)

    ax.text(vx, 3.65, "(256 patches, 768)", ha="center", fontsize=8,
            color="#888888", fontfamily=FONT)

    rounded_box(ax, vx, 3.1, 2.0, 0.4, "768 → 384", "#AAAAAA",
                fontsize=9)
    arrow(ax, vx, 4.2 - 0.45/2, vx, 3.1 + 0.4/2)

    # ---- SHARED CONSTRAINT NETWORK (center, below) ----
    cx = 5.0

    # Merge arrows
    arrow(ax, tx, 3.1 - 0.4/2, cx - 0.3, 2.2 + 0.55/2, lw=1.5)
    arrow(ax, vx, 3.1 - 0.4/2, cx + 0.3, 2.2 + 0.55/2, lw=1.5)

    # Shared architecture box
    box = FancyBboxPatch((cx - 2.2, 2.2 - 0.55/2 - 0.05), 4.4, 0.65,
                          boxstyle="round,pad=0.08",
                          facecolor="#F0F0F0", edgecolor="#CCCCCC",
                          linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(box)

    rounded_box(ax, cx, 2.2, 4.0, 0.5, "Constraint Network", COLORS["ssm"],
                fontsize=11, sublabel="6 SSM + 2 Attention + Energy Head")

    ax.text(cx, 1.55, "Identical architecture — only input projection changes",
            ha="center", fontsize=9, color="#888888", fontfamily=FONT,
            style="italic")

    # Energy output
    rounded_box(ax, cx, 0.8, 2.5, 0.45, "E(x) = scalar energy", COLORS["energy"],
                fontsize=10, sublabel="+ per-position localization")
    arrow(ax, cx, 2.2 - 0.55/2, cx, 0.8 + 0.45/2)

    # Results annotations
    ax.text(tx, 0.2, "93.4% in-distribution\n87.2% unseen corruptions",
            ha="center", fontsize=8, fontweight="bold",
            color=COLORS["branch_s"], fontfamily=FONT)
    ax.text(vx, 0.2, "0.962 FF++ Deepfakes\n0.886 Celeb-DF (cross-dataset)",
            ha="center", fontsize=8, fontweight="bold",
            color=COLORS["branch_l"], fontfamily=FONT)

    plt.tight_layout()
    plt.savefig("diagram_cross_modal.svg", format="svg", bbox_inches="tight",
                facecolor="white", dpi=150)
    plt.savefig("diagram_cross_modal.pdf", format="pdf", bbox_inches="tight",
                facecolor="white", dpi=150)
    print("  Saved diagram_cross_modal.svg/pdf")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating paper diagrams...")
    diagram_architecture()
    diagram_three_branch()
    diagram_cross_modal()
    print("\nDone. Files:")
    print("  diagram_architecture.svg/pdf")
    print("  diagram_three_branch.svg/pdf")
    print("  diagram_cross_modal.svg/pdf")

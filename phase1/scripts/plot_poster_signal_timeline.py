#!/usr/bin/env python3
"""Create a compact poster figure of late-layer KC causal persistence.

The plot deliberately removes prompt-segment and transition-distribution detail.
It shows only causal recovery from restoring clean activations at the final-answer
position, by layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
CT_STATS = ROOT / "phase1/results/step3_causal_tracing/ct_stats.json"
OUT_DIR = ROOT / "phase1/results/poster"
OUT_PNG = OUT_DIR / "fig_kc_signal_emergence_timeline.png"
OUT_PDF = OUT_DIR / "fig_kc_signal_emergence_timeline.pdf"

FONT_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"


def set_fonts() -> None:
    font_manager.fontManager.addfont(FONT_REGULAR)
    font_manager.fontManager.addfont(FONT_BOLD)
    plt.rcParams.update({
        "font.family": "NanumGothic",
        "axes.unicode_minus": False,
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    set_fonts()
    causal = json.loads(CT_STATS.read_text())
    layers = np.arange(26)

    # overall_ie_matrix is stored as layer x segment; final is the last column.
    final_ie = np.array([row[-1] for row in causal["overall_ie_matrix"]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax_bottom = plt.subplots(figsize=(10.8, 3.75))
    fig.patch.set_facecolor("white")

    causal_color = "#E76F51"
    ink = "#203040"

    # Causal persistence / relevance.
    ax_bottom.plot(layers, final_ie, color=causal_color, linewidth=2.8,
                   marker="o", markersize=4.2, markerfacecolor="white",
                   markeredgewidth=1.5)
    ax_bottom.axhline(0, color="#9AA8B5", linewidth=0.9)
    ax_bottom.set_ylabel("Indirect Effect", color=ink)
    ax_bottom.set_ylim(-0.06, 0.15)

    ax_bottom.spines[["top", "right"]].set_visible(False)
    ax_bottom.spines[["left", "bottom"]].set_color("#9AA8B5")
    ax_bottom.tick_params(colors="#4F5B66", labelsize=9)
    ax_bottom.grid(axis="y", color="#E8EDF2", linewidth=0.8)
    ax_bottom.set_axisbelow(True)

    ax_bottom.set_xlim(-0.7, 25.7)
    ax_bottom.set_xticks(np.arange(0, 26, 2))
    ax_bottom.set_xlabel("Layer", color=ink, fontweight="bold")
    fig.subplots_adjust(left=0.095, right=0.985, top=0.95, bottom=0.14)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()

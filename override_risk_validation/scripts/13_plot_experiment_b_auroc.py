#!/usr/bin/env python3
"""Make the compact Experiment B specificity figure for the poster."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "override_risk_validation/results/exp_B_auroc.csv"
OUT_DIR = ROOT / "override_risk_validation/results/figures"
OUT_PNG = OUT_DIR / "fig_experiment_b_specificity_auroc.png"
OUT_PDF = OUT_DIR / "fig_experiment_b_specificity_auroc.pdf"
FONT_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

ROWS = [
    ("cond4_context_vs_cond1_context", "Harmful override vs. safe context"),
    ("cond2_closed_book_vs_cond1_closed_book", "Closed-book error vs. correct answer"),
]


def main() -> None:
    font_manager.fontManager.addfont(FONT_REGULAR)
    plt.rcParams.update({"font.family": "NanumGothic", "pdf.fonttype": 42, "ps.fonttype": 42})
    with INPUT.open(newline="") as handle:
        by_key = {row["comparison_pair"]: row for row in csv.DictReader(handle)}

    values = []
    for key, label in ROWS:
        row = by_key[key]
        values.append((
            label,
            float(row["auroc"]),
            float(row["auroc_ci_lower"]),
            float(row["auroc_ci_upper"]),
        ))

    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    fig.patch.set_facecolor("white")
    teal, gray, ink = "#1F8A8A", "#B9C4CD", "#203040"
    y_pos = list(range(len(values) - 1, -1, -1))

    ax.axvline(0.5, color="#8B98A5", linewidth=1.3, linestyle="--", zorder=0)
    for y, (label, auc, lo, hi) in zip(y_pos, values):
        color = teal if auc > 0.5 else gray
        ax.hlines(y, lo, hi, color=color, linewidth=3.0, zorder=2)
        ax.scatter(auc, y, s=75, color=color, edgecolor="white", linewidth=1.1, zorder=3)
        ax.text(0.925, y, f"{auc:.3f}\n[{lo:.3f}, {hi:.3f}]", va="center",
                ha="right", fontsize=8.6, color=ink, linespacing=1.1)

    ax.set_yticks(y_pos, [v[0] for v in values], fontsize=9.3)
    ax.set_xlim(0.35, 0.95)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("AUROC (95% bootstrap CI)", fontsize=11, color=ink)
    ax.tick_params(axis="x", labelsize=9, colors="#4F5B66")
    ax.tick_params(axis="y", length=0, colors=ink)
    ax.grid(axis="x", color="#E7EDF2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9AA8B5")
    ax.text(0.5, 1.18, "No discrimination", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.8, color="#697887")
    fig.subplots_adjust(left=0.37, right=0.98, top=0.86, bottom=0.28)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(OUT_PNG)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step1 시각화: Direction별 세 조건 평균 projection line plot

저장: phase3/results/step1/fig_projection_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT      = Path(__file__).resolve().parents[2]
RESULTS   = ROOT / "phase3/results"
STATS_PATH = RESULTS / "step1/projection_stats.json"
OUT_PATH  = RESULTS / "step1/fig_projection_comparison.png"

TARGET_LAYERS = list(range(10, 23))
DIRECTIONS    = ["A", "B", "C"]


def main() -> None:
    with open(STATS_PATH) as f:
        stats = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    condition_styles = [
        ("clean",      "gray",  "clean"),
        ("supportive", "green", "supportive"),
        ("conflict",   "red",   "conflict"),
    ]

    for i, d in enumerate(DIRECTIONS):
        ax = axes[i]
        for c, color, label in condition_styles:
            means  = [stats[d][str(l)][c]["mean"] for l in TARGET_LAYERS]
            stds   = [stats[d][str(l)][c]["std"]  for l in TARGET_LAYERS]
            means  = np.array(means)
            stds   = np.array(stds)
            ax.plot(TARGET_LAYERS, means, color=color, label=label,
                    marker="o", markersize=4, linewidth=1.8)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_title(f"Direction {d}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean projection")
        ax.set_xticks(TARGET_LAYERS)
        ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS], rotation=45, fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Projection values by Direction", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_PATH}")


if __name__ == "__main__":
    main()

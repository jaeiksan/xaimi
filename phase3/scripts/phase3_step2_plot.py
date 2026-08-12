#!/usr/bin/env python3
"""
Step2 시각화:
  그림 1: Ratio line plot (Δ_sc / Δ_cc)
  그림 2: Δ_cc와 Δ_sc 나란히 (ratio 주석 포함)

저장: phase3/results/step2/fig_ratio.png
      phase3/results/step2/fig_delta_comparison.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parents[2]
RESULTS    = ROOT / "phase3/results"

TARGET_LAYERS = list(range(10, 23))
DIRECTIONS    = ["A", "B", "C"]


def load_results():
    with open(RESULTS / "step2/ratio_results.json") as f:
        ratio_results = json.load(f)
    with open(RESULTS / "step2/delta_results.json") as f:
        delta_results = json.load(f)
    return ratio_results, delta_results


# ── 그림 1: Ratio line plot ──────────────────────────────────────────────────

def plot_ratio(ratio_results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, d in enumerate(DIRECTIONS):
        ax = axes[i]
        ratios = [
            ratio_results[d][str(l)]
            for l in TARGET_LAYERS
        ]
        valid_mask = [not (isinstance(r, float) and math.isnan(r)) for r in ratios]
        x_valid    = [l for l, m in zip(TARGET_LAYERS, valid_mask) if m]
        r_valid    = [r for r, m in zip(ratios, valid_mask) if m]

        ax.plot(x_valid, r_valid, color="steelblue", marker="o", markersize=5, linewidth=2)
        ax.axhline(0, color="green",  linewidth=1.2, linestyle="--",
                   label="conflict detection (ratio=0)")
        ax.axhline(1, color="red",    linewidth=1.2, linestyle="--",
                   label="context following (ratio=1)")
        ax.axhline(0, color="black",  linewidth=0.5, alpha=0.3)

        ax.set_title(f"Direction {d}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Ratio")
        ax.set_xticks(TARGET_LAYERS)
        ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS], rotation=45, fontsize=8)
        ax.set_ylim(-2, 3)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("$\\Delta_{sc}$ / $\\Delta_{cc}$ ratio", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = RESULTS / "step2/fig_ratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


# ── 그림 2: Δ_cc vs Δ_sc 나란히 ─────────────────────────────────────────────

def plot_delta_comparison(ratio_results: dict, delta_results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, d in enumerate(DIRECTIONS):
        ax = axes[i]
        dcc    = [delta_results[d][str(l)]["mean_delta_cc"] for l in TARGET_LAYERS]
        dsc    = [delta_results[d][str(l)]["mean_delta_sc"] for l in TARGET_LAYERS]
        ratios = [ratio_results[d][str(l)]                  for l in TARGET_LAYERS]

        ax.plot(TARGET_LAYERS, dcc, color="red",   marker="o", markersize=4,
                linewidth=2, label="Δ_cc  (conflict − clean)")
        ax.plot(TARGET_LAYERS, dsc, color="green", marker="s", markersize=4,
                linewidth=2, label="Δ_sc  (supportive − clean)")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

        # ratio 주석 (짝수 인덱스 layer만)
        for j, (l, r, sc) in enumerate(zip(TARGET_LAYERS, ratios, dsc)):
            if not (isinstance(r, float) and math.isnan(r)) and j % 2 == 0:
                ax.annotate(
                    f"r={r:.2f}",
                    xy=(l, sc),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="gray",
                )

        ax.set_title(f"Direction {d}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Δ projection")
        ax.set_xticks(TARGET_LAYERS)
        ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS], rotation=45, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Step2: Δ_cc (conflict−clean) vs Δ_sc (supportive−clean)  [r = Δ_sc / Δ_cc annotated]",
        fontsize=10,
    )
    fig.tight_layout()
    out = RESULTS / "step2/fig_delta_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


def main() -> None:
    ratio_results, delta_results = load_results()
    plot_ratio(ratio_results)
    plot_delta_comparison(ratio_results, delta_results)


if __name__ == "__main__":
    main()

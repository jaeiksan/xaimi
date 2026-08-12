#!/usr/bin/env python3
"""
Step3 시각화:
  그림 1: Effect size bar plot (direction별)
  그림 2: Flipped vs Non-flipped projection 분포 box plot (대표 layer)

저장: phase3/results/step3/fig_effect_size.png
      phase3/results/step3/fig_boxplot.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT     = Path(__file__).resolve().parents[2]
RESULTS  = ROOT / "phase3/results"

TARGET_LAYERS = list(range(10, 23))
DIRECTIONS    = ["A", "B", "C"]
REP_LAYERS    = [15, 18]


def main() -> None:
    with open(RESULTS / "step3/group_stats.json") as f:
        group_stats = json.load(f)
    with open(RESULTS / "step3/flipped_ids.json") as f:
        meta = json.load(f)

    projections    = torch.load(RESULTS / "projections_raw.pt", map_location="cpu")
    flipped_ids    = meta["flipped"]
    non_flipped_ids = meta["non_flipped"]

    # ── 그림 1: Effect size bar plot ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, d in enumerate(DIRECTIONS):
        ax = axes[i]
        effects = [group_stats[d][str(l)]["effect_size"] for l in TARGET_LAYERS]
        p_vals  = [group_stats[d][str(l)]["p_val"]       for l in TARGET_LAYERS]
        colors  = ["steelblue" if e >= 0 else "salmon"   for e in effects]
        bars = ax.bar(TARGET_LAYERS, effects, color=colors, alpha=0.75)

        # 유의미한 bar에 * 표시 (p < 0.05)
        for bar, e, p in zip(bars, effects, p_vals):
            if p < 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    e + (0.03 if e >= 0 else -0.06),
                    "*", ha="center", va="bottom", fontsize=10, color="black",
                )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"Direction {d}\n"
            f"(n_flip={meta['n_flipped']}, n_non={meta['n_non_flipped']})",
            fontsize=10,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cohen's d  (flipped − non-flipped)")
        ax.set_xticks(TARGET_LAYERS)
        ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS], rotation=45, fontsize=8)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Step3: Effect size — conflict projection (flipped vs non-flipped)\n"
        "Positive = flipped cases have higher projection  (* : p<0.05)",
        fontsize=10,
    )
    fig.tight_layout()
    out1 = RESULTS / "step3/fig_effect_size.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out1}")

    # ── 그림 2: Box plot (대표 layer별) ─────────────────────────────────────
    n_rows = len(DIRECTIONS)
    n_cols = len(REP_LAYERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.5 * n_rows))

    for i, d in enumerate(DIRECTIONS):
        for j, layer in enumerate(REP_LAYERS):
            ax    = axes[i][j]
            proj  = projections[d]["conflict"][layer].float()
            fp    = proj[flipped_ids].numpy()
            nfp   = proj[non_flipped_ids].numpy()

            bp = ax.boxplot(
                [fp, nfp],
                labels=["flipped", "non-flipped"],
                patch_artist=True,
                showfliers=False,
            )
            colors = ["#4c78a8", "#e45756"]
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            for median in bp["medians"]:
                median.set_color("black")

            es   = group_stats[d][str(layer)]["effect_size"]
            pval = group_stats[d][str(layer)]["p_val"]
            ax.set_title(
                f"Dir {d} — L{layer}\nd={es:.2f}  p={pval:.3f}",
                fontsize=9,
            )
            ax.set_ylabel("Projection (conflict)")
            ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Step3: Conflict projection distribution — flipped vs non-flipped\n"
        f"config: {meta['config']}",
        fontsize=10,
    )
    fig.tight_layout()
    out2 = RESULTS / "step3/fig_boxplot.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out2}")


if __name__ == "__main__":
    main()

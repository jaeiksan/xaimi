#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


STATS_PATH = Path("/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ct_stats.json")
OUT_DIR = Path("/home/jaeik/xaimi/phase1/results/step3_causal_tracing")
SEGMENTS = [
    "context_first_sent",
    "context_rest",
    "question",
    "options",
    "final",
]
CTYPES = ["misinformation", "temporal", "semantic"]


def draw_heatmap(matrix: np.ndarray, title: str, out_path: Path, vmin: float, vmax: float) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    im = ax.imshow(matrix.T, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Segment", fontsize=12)
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_xticklabels(range(matrix.shape[0]), fontsize=7)
    ax.set_yticks(range(len(SEGMENTS)))
    ax.set_yticklabels(SEGMENTS, fontsize=9)
    ax.set_title(title, fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Indirect effect (IE)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def draw_by_type(by_type: dict, out_path: Path, vmin: float, vmax: float) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True)

    for ax, ct in zip(axes, CTYPES):
        mat = np.array(by_type[ct])
        im = ax.imshow(mat.T, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(ct, fontsize=12)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_xticks(range(mat.shape[0]))
        ax.set_xticklabels(range(mat.shape[0]), fontsize=6)
        ax.set_yticks(range(len(SEGMENTS)))
        ax.set_yticklabels(SEGMENTS, fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Indirect effect (IE)")
    fig.suptitle("Step 3 causal tracing by conflict type", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    stats = json.loads(STATS_PATH.read_text())
    overall = np.array(stats["overall_ie_matrix"])
    by_type = stats["by_type_ie_matrix"]

    all_vals = np.concatenate(
        [overall.flatten()] + [np.array(by_type[ct]).flatten() for ct in CTYPES]
    )
    vmin = min(-0.24, float(np.percentile(all_vals, 1)))
    vmax = max(0.14, float(np.percentile(all_vals, 99)))

    draw_heatmap(
        overall,
        "IE Heatmap",
        OUT_DIR / "ie_heatmap_overall_red_positive.png",
        vmin=vmin,
        vmax=vmax,
    )
    draw_by_type(
        by_type,
        OUT_DIR / "ie_heatmap_by_type_red_positive.png",
        vmin=vmin,
        vmax=vmax,
    )


if __name__ == "__main__":
    main()

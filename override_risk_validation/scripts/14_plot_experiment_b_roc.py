#!/usr/bin/env python3
"""Create a poster-ready ROC-curve view of Experiment B specificity."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "override_risk_validation/results/exp_B_per_case.jsonl"
OUT_DIR = ROOT / "override_risk_validation/results/figures"
OUT_PNG = OUT_DIR / "fig_experiment_b_specificity_roc.png"
OUT_PDF = OUT_DIR / "fig_experiment_b_specificity_roc.pdf"
FONT_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def roc_curve(positive: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC for the rule: classify positive when projection >= threshold."""
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)])
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    scores = scores[order]
    thresholds = np.r_[True, scores[1:] != scores[:-1]]
    tp = np.cumsum(labels)[thresholds]
    fp = np.cumsum(1 - labels)[thresholds]
    return np.r_[0, fp / len(negative), 1], np.r_[0, tp / len(positive), 1]


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    delta = positive[:, None] - negative[None, :]
    return float((delta > 0).mean() + 0.5 * (delta == 0).mean())


def main() -> None:
    font_manager.fontManager.addfont(FONT_REGULAR)
    plt.rcParams.update({"font.family": "NanumGothic", "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]

    def signal(condition: str, view: str) -> np.ndarray:
        return np.array([r["projection"] for r in rows if r["condition"] == condition and r["view"] == view])

    harmful, safe = signal("cond4", "context"), signal("cond1", "context")
    error, correct = signal("cond2", "closed_book"), signal("cond1", "closed_book")

    fig, ax = plt.subplots(figsize=(5.25, 4.3))
    fig.patch.set_facecolor("white")
    ax.plot([0, 1], [0, 1], color="#9AA8B5", linewidth=1.4, linestyle="--", label="No discrimination")
    for positive, negative, color, label in [
        (harmful, safe, "#D62728", "Knowledge conflict vs. supportive context"),
        (error, correct, "#6BAED6", "No-context error vs. no-context correct"),
    ]:
        fpr, tpr = roc_curve(positive, negative)
        auc = auroc(positive, negative)
        ax.plot(fpr, tpr, color=color, linewidth=3.0, label=f"{label}\nAUROC = {auc:.3f}")

    ink = "#203040"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False Positive Rate", fontsize=11, color=ink)
    ax.set_ylabel("True Positive Rate", fontsize=11, color=ink)
    ax.tick_params(colors="#4F5B66", labelsize=9)
    ax.grid(color="#E7EDF2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9AA8B5")
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, facecolor="white",
              edgecolor="#DCE4EB", fontsize=6.8, handlelength=1.65,
              handletextpad=0.7, borderpad=0.55, labelspacing=0.55)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(OUT_PNG)


if __name__ == "__main__":
    main()

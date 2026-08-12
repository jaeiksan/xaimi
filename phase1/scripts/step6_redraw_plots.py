#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path(
    "/home/jaeik/xaimi/phase1/results/step6_intervention/intervention_results_checkpoint_summary.json"
)
OUT_DIR = Path("/home/jaeik/xaimi/phase1/results/step6_intervention")


def pct_to_float(s: str) -> float:
    return float(s.rstrip("%"))


def extract_b_points(summary: dict) -> dict[str, list[tuple[float, float, float]]]:
    search = summary["strategy_B_alpha_search"]
    cond_to_points: dict[str, list[tuple[float, float, float]]] = {"B1": [], "B2": [], "B3": []}
    for key, vals in search.items():
        cond = key.split("_")[0]
        alpha = float(key.split("_alpha_")[1])
        sup = pct_to_float(vals["kc_suppression_rate"])
        clean = pct_to_float(vals["clean_accuracy"])
        cond_to_points[cond].append((alpha, sup, clean))
    for cond in cond_to_points:
        cond_to_points[cond].sort(key=lambda x: x[0])
    return cond_to_points


def plot_main_tradeoff(summary: dict) -> None:
    base_clean = pct_to_float(summary["baseline"]["clean_accuracy"])
    strat_a = summary["strategy_A"]
    best_b = summary["strategy_B_best"]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    ax.scatter(
        [0],
        [base_clean],
        s=140,
        color="gray",
        marker="o",
        label="No intervention",
        zorder=4,
    )
    ax.scatter(
        [pct_to_float(strat_a["kc_suppression_rate"])],
        [pct_to_float(strat_a["clean_accuracy"])],
        s=180,
        color="steelblue",
        marker="D",
        label="Strategy A: L17H3 mean ablation",
        zorder=5,
    )
    ax.scatter(
        [pct_to_float(best_b["kc_suppression_rate"])],
        [pct_to_float(best_b["clean_accuracy"])],
        s=200,
        color="#8e44ad",
        marker="*",
        label=f"Best Strategy B: {best_b['condition_label']} (alpha={best_b['best_alpha']})",
        zorder=6,
    )
    ax.axhline(base_clean * 0.95, color="orange", linestyle="--", linewidth=1.2, label="95% clean-accuracy threshold")

    ax.annotate("Baseline", (0, base_clean), textcoords="offset points", xytext=(8, 6), fontsize=9, color="gray")
    ax.annotate("Strategy A", (pct_to_float(strat_a["kc_suppression_rate"]), pct_to_float(strat_a["clean_accuracy"])),
                textcoords="offset points", xytext=(8, -14), fontsize=9, color="steelblue")
    ax.annotate("Best Strategy B", (pct_to_float(best_b["kc_suppression_rate"]), pct_to_float(best_b["clean_accuracy"])),
                textcoords="offset points", xytext=(8, 6), fontsize=9, color="#8e44ad")

    ax.set_title("Phase 3: Simple intervention trade-off", fontsize=14)
    ax.set_xlabel("KC suppression rate (%)")
    ax.set_ylabel("Clean QA accuracy (%)")
    ax.set_xlim(-2, 62)
    ax.set_ylim(20, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    out = OUT_DIR / "phase3_main_tradeoff.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"saved: {out}")


def plot_strategy_b_alpha_sweep(summary: dict) -> None:
    base_clean = pct_to_float(summary["baseline"]["clean_accuracy"])
    best_b = summary["strategy_B_best"]
    cond_to_points = extract_b_points(summary)

    cond_colors = {"B1": "#e74c3c", "B2": "#e67e22", "B3": "#8e44ad"}
    cond_labels = {"B1": "L17 only", "B2": "L18 only", "B3": "L17 + L18"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax1, ax2 = axes

    for cond in ["B1", "B2", "B3"]:
        pts = cond_to_points[cond]
        alphas = [p[0] for p in pts]
        sups = [p[1] for p in pts]
        cleans = [p[2] for p in pts]
        ax1.plot(alphas, sups, marker="o", linewidth=2, color=cond_colors[cond], label=cond_labels[cond])
        ax2.plot(alphas, cleans, marker="o", linewidth=2, color=cond_colors[cond], label=cond_labels[cond])

    ax1.set_title("Strategy B: KC suppression vs alpha", fontsize=13)
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("KC suppression rate (%)")
    ax1.grid(alpha=0.3)

    ax2.set_title("Strategy B: Clean accuracy vs alpha", fontsize=13)
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Clean QA accuracy (%)")
    ax2.axhline(base_clean * 0.95, color="orange", linestyle="--", linewidth=1.2, label="95% threshold")
    ax2.grid(alpha=0.3)

    best_alpha = float(best_b["best_alpha"])
    best_clean = pct_to_float(best_b["clean_accuracy"])
    best_sup = pct_to_float(best_b["kc_suppression_rate"])
    ax1.scatter([best_alpha], [best_sup], s=180, color="#8e44ad", marker="*", edgecolor="black", linewidth=0.6, zorder=6)
    ax2.scatter([best_alpha], [best_clean], s=180, color="#8e44ad", marker="*", edgecolor="black", linewidth=0.6, zorder=6)

    ax1.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    out = OUT_DIR / "phase3_strategy_b_alpha_sweep.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"saved: {out}")


def plot_suppression_by_type(summary: dict) -> None:
    strat_a = summary["strategy_A"]
    best_b = summary["strategy_B_best"]

    types = ["misinformation", "temporal", "semantic"]
    x = np.arange(len(types))
    width = 0.36

    a_vals = [pct_to_float(strat_a["by_type"][t]["kc_suppression_rate"]) for t in types]
    b_vals = [pct_to_float(best_b["by_type"][t]["kc_suppression_rate"]) for t in types]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, a_vals, width, color="steelblue", label="Strategy A: L17H3 mean ablation")
    ax.bar(x + width / 2, b_vals, width, color="#8e44ad", label=f"Best Strategy B: {best_b['condition_label']} (alpha={best_b['best_alpha']})")

    ax.set_xticks(x)
    ax.set_xticklabels(["Misinformation", "Temporal", "Semantic"])
    ax.set_ylabel("KC suppression rate (%)")
    ax.set_title("KC suppression by conflict type", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    for xi, val in enumerate(a_vals):
        ax.text(xi - width / 2, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    for xi, val in enumerate(b_vals):
        ax.text(xi + width / 2, val + 0.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "suppression_by_type.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"saved: {out}")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    plot_main_tradeoff(summary)
    plot_strategy_b_alpha_sweep(summary)
    plot_suppression_by_type(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step3 revised — 시각화

그림 1 (fig_effect_size_revised.png): A config/B config별 Direction projection Effect size
그림 2 (fig_boxplot_revised.png): 대표 레이어(L18) flipped vs non-flipped box plot
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "phase3/results/step3_revised"
PROJ_PATH   = ROOT / "phase3/results/projections_raw.pt"

TARGET_LAYERS = list(range(10, 23))

# 그림에 표시할 config 순서와 레이블
PLOT_CONFIGS = [
    ("L17:A+L19:A",        "A",  "A solo (L17:A+L19:A)"),
    ("L18:B",              "B",  "B solo (L18:B)"),
    ("L16:B+L17:B+L18:B",  "B",  "B multi (L16:B+L17:B+L18:B)"),
]

REP_LAYER = 18


def load_data():
    with open(RESULTS_DIR / "group_stats_revised.json") as f:
        stats = json.load(f)
    with open(RESULTS_DIR / "intervention_results.json") as f:
        inter = json.load(f)
    projections = torch.load(PROJ_PATH, map_location="cpu", weights_only=False)
    return stats, inter, projections


# ── 그림 1: Effect size bar plot ─────────────────────────────────────────────

def plot_effect_size(stats: dict) -> None:
    configs_present = [c for c, d, label in PLOT_CONFIGS if c in stats]
    n = len(configs_present)
    if n == 0:
        print("[skip] no configs in group_stats_revised.json")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (cfg_label, target_dir, plot_label) in zip(axes, [(c, d, l) for c, d, l in PLOT_CONFIGS if c in stats]):
        s = stats[cfg_label]
        effects = [s["layers"][str(l)]["effect_size"] for l in TARGET_LAYERS]
        p_vals  = [s["layers"][str(l)]["p_val"]       for l in TARGET_LAYERS]
        colors  = ["steelblue" if e >= 0 else "salmon" for e in effects]

        bars = ax.bar(TARGET_LAYERS, effects, color=colors, alpha=0.8, width=0.7)

        # 유의한 bar에 * 표시
        for i, (l, e, p) in enumerate(zip(TARGET_LAYERS, effects, p_vals)):
            if p < 0.05:
                y = e + (0.05 if e >= 0 else -0.05)
                ax.text(l, y, "*", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{plot_label}\n(dir={target_dir}, flipped={s['n_flipped']}, non-flipped={s['n_non_flipped']})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cohen's d  (flipped - non-flipped)")
        ax.set_xticks(TARGET_LAYERS)
        ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS], rotation=45, fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Step3 revised: Effect size — conflict projection (flipped vs non-flipped)\n"
        "Positive = flipped cases have higher projection  (* : p<0.05)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig_effect_size_revised.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


# ── 그림 2: Box plot (대표 Layer L18) ────────────────────────────────────────

def plot_boxplot(stats: dict, inter: dict, projections: dict) -> None:
    configs_present = [(c, d, l) for c, d, l in PLOT_CONFIGS if c in stats and c in inter]
    n = len(configs_present)
    if n == 0:
        print("[skip] no configs for boxplot")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (cfg_label, target_dir, plot_label) in zip(axes, configs_present):
        case_data = inter[cfg_label]["cases"]
        kc_ids    = [i for i, c in enumerate(case_data) if c["baseline_kc"]]
        flip_ids  = [i for i in kc_ids if case_data[i]["rank_flipped"]]
        nflip_ids = [i for i in kc_ids if not case_data[i]["rank_flipped"]]

        proj = projections[target_dir]["conflict"][REP_LAYER].float()
        fp   = proj[flip_ids].numpy()
        nfp  = proj[nflip_ids].numpy()

        bp = ax.boxplot(
            [fp, nfp],
            tick_labels=["flipped", "non-flipped"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("lightcoral")

        s_layer = stats[cfg_label]["layers"].get(str(REP_LAYER), {})
        d_val   = s_layer.get("effect_size", float("nan"))
        p_val   = s_layer.get("p_val",       float("nan"))

        ax.set_title(f"{plot_label}\ndir={target_dir} @ L{REP_LAYER}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Conflict projection value")
        ax.text(0.5, 0.97, f"Cohen's d = {d_val:.2f}  p = {p_val:.4f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"Step3 revised: Conflict projection distribution @ L{REP_LAYER}\n"
        "(flipped = KC suppressed, non-flipped = KC persists)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = RESULTS_DIR / "fig_boxplot_revised.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


# ── 그림 3: A vs B effect size 직접 비교 ─────────────────────────────────────

def plot_ab_comparison(stats: dict) -> None:
    """A_solo와 B_multi(또는 B_solo) effect size를 같은 패널에 겹쳐서 비교."""
    a_key = "L17:A+L19:A"
    b_key = "L16:B+L17:B+L18:B" if "L16:B+L17:B+L18:B" in stats else "L18:B"

    if a_key not in stats or b_key not in stats:
        print("[skip] A/B comparison: missing configs")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    a_eff = [stats[a_key]["layers"][str(l)]["effect_size"] for l in TARGET_LAYERS]
    b_eff = [stats[b_key]["layers"][str(l)]["effect_size"] for l in TARGET_LAYERS]

    x = np.array(TARGET_LAYERS)
    width = 0.35

    bars_a = ax.bar(x - width/2, a_eff, width, label=f"A solo (flipped={stats[a_key]['n_flipped']})",
                    color="steelblue", alpha=0.75)
    bars_b = ax.bar(x + width/2, b_eff, width, label=f"B multi (flipped={stats[b_key]['n_flipped']})",
                    color="tomato", alpha=0.75)

    # p<0.05 표시
    for bars, key in [(bars_a, a_key), (bars_b, b_key)]:
        for bar, l in zip(bars, TARGET_LAYERS):
            p = stats[key]["layers"][str(l)]["p_val"]
            if p < 0.05:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + (0.03 if h >= 0 else -0.06),
                        "*", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d  (flipped - non-flipped)")
    ax.set_title("Direction-specific intervention: A solo vs B multi\n"
                 "Conflict projection effect size by layer (* : p<0.05)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS], rotation=45, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = RESULTS_DIR / "fig_ab_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


def main() -> None:
    stats, inter, projections = load_data()
    plot_effect_size(stats)
    plot_boxplot(stats, inter, projections)
    plot_ab_comparison(stats)


if __name__ == "__main__":
    main()

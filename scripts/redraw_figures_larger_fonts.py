#!/usr/bin/env python3
"""
4개 그림을 범례·숫자 크기를 키워서 다시 그리는 스크립트.

대상:
  phase2/results/exp_grid/figure3_top4_comparison.png
  phase2/results/exp_grid/figure5_direction_comparison.png
  phase1/results/step3_causal_tracing/ie_heatmap_overall_red_positive.png
  phase1/results/step4_path_patching/component_effects_by_layer.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# Korean font setup
_korean_fonts = [f.name for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ["Nanum", "Malgun", "Noto Sans CJK"])]
if _korean_fonts:
    plt.rcParams["font.family"] = _korean_fonts[0]
plt.rcParams["axes.unicode_minus"] = False  # ASCII hyphen for minus signs

# ── 공통 경로 ─────────────────────────────────────────────────────────────────
ROOT = Path("/home/jaeik/xaimi")
GRID_DIR  = ROOT / "phase2/results/exp_grid"
CT_DIR    = ROOT / "phase1/results/step3_causal_tracing"
PP_DIR    = ROOT / "phase1/results/step4_path_patching"

SEGMENTS = [
    "context_first_sent",
    "context_rest",
    "question",
    "options",
    "final",
]

DIR_MAP = {
    "A_svd_conf_clean":   "A (conflict - clean SVD)",
    "B_svd_conf_support": "B (conflict - supportive SVD)",
    "C_cpca":             "C (cPCA)",
    "D_regression":       "D (Ridge regression)",
}
DIR_NAMES_ORDERED = ["A_svd_conf_clean", "B_svd_conf_support", "C_cpca", "D_regression"]
DIR_COLORS = {"A": "#e74c3c", "B": "#2a9d8f", "C": "#9b59b6", "D": "#f4a261"}
# dname → color (DIR_MAP이 변경돼도 색상은 첫 글자 A/B/C/D 로 결정)
DIR_COLOR_BY_DNAME = {
    "A_svd_conf_clean":   "#e74c3c",
    "B_svd_conf_support": "#2a9d8f",
    "C_cpca":             "#9b59b6",
    "D_regression":       "#f4a261",
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: KC Suppression Top 4 조합 비교
# ─────────────────────────────────────────────────────────────────────────────
def draw_figure3() -> None:
    data = json.loads((GRID_DIR / "all_results_ranked.json").read_text())
    phase1 = [r for r in data if r["phase"] == 1]
    phase3 = [r for r in data if r["phase"] == 3]

    # baseline: L18:B (best single-layer config used as reference)
    baseline = next(r for r in phase1 if r["config"] == "L18:B")
    top4     = phase3[:4]

    configs    = [baseline["config"]] + [r["config"] for r in top4]
    kc_values  = [baseline["kc_suppression"]] + [r["kc_suppression"] for r in top4]
    clean_vals = [baseline["clean_accuracy"]]  + [r["clean_accuracy"]  for r in top4]
    is_top4    = [False] + [True] * 4

    x     = np.arange(len(configs))
    colors = ["#4da6d9" if not t else "#e07070" for t in is_top4]

    fig, ax = plt.subplots(figsize=(13, 7))

    bars = ax.bar(x, kc_values, color=colors, zorder=3, width=0.55)

    # percentage labels on bars
    for xi, val, top in zip(x, kc_values, is_top4):
        color = "#c0392b" if top else "#1a6fa8"
        ax.text(xi, val + 0.010, f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=21, fontweight="bold", color=color)

    # clean accuracy line
    ax2 = ax.twinx()
    ax2.plot(x, clean_vals, color="#2a9d8f", marker="o", markersize=10,
             linewidth=2.5, linestyle="--", zorder=5, label="Clean Accuracy")
    for xi, val in zip(x, clean_vals):
        ax2.text(xi, val + 0.006, f"{val*100:.1f}%",
                 ha="center", va="bottom", fontsize=21, color="#2a9d8f", fontweight="bold")
    ax2.axhline(0.80, color="gray", lw=1.2, ls=":", alpha=0.6)
    ax2.text(x[-1] + 0.45, 0.802, "80%", color="gray", fontsize=16, va="bottom")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Clean Accuracy", fontsize=16)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.tick_params(axis="y", labelsize=19)

    # divider between baseline and top4
    ax.axvline(x=0.5, color="gray", lw=1.5, ls="--", alpha=0.6)

    # legend patches
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#4da6d9", label="KC Suppression (baseline)"),
        Patch(facecolor="#e07070", label="KC Suppression (top 4)"),
        Line2D([0], [0], color="#2a9d8f", marker="o", lw=2.5, ls="--",
               markersize=9, label="Clean Accuracy"),
    ]
    ax.legend(handles=legend_elements, fontsize=13, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=15, fontweight="bold", rotation=12, ha="right")
    ax.set_ylabel("(%)", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(axis="y", labelsize=19)
    ax.set_ylim(0, 1.0)
    ax.set_title("KC Suppression Top 4 조합 비교", fontsize=18, fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    out = GRID_DIR / "figure3_top4_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: 단일 방향 제거 — KC Suppression & Clean Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def draw_figure5() -> None:
    data = json.loads((GRID_DIR / "phase1_results.json").read_text())
    layers = sorted({r["layers"][0] for r in data})
    dir_names = DIR_NAMES_ORDERED  # A, B, C, D

    # Build lookup: (layer, dir_key) → metrics
    lookup: dict[tuple, dict] = {}
    for r in data:
        key = (r["layers"][0], r["directions"][0])
        lookup[key] = r

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
    fig.suptitle("단일 방향 제거: KC Suppression & Clean Accuracy",
                 fontsize=20, fontweight="bold", y=0.98)

    n_dirs   = len(dir_names)
    bar_w    = 0.18
    x        = np.arange(len(layers))
    offsets  = np.linspace(-(n_dirs - 1) / 2, (n_dirs - 1) / 2, n_dirs) * bar_w

    # ── Top: KC Suppression bar chart ─────────────────────────────────────
    ax_top.set_title("KC Suppression", fontsize=16, pad=8)
    for di, dname in enumerate(dir_names):
        label = DIR_MAP[dname]
        color = DIR_COLOR_BY_DNAME[dname]
        vals  = [lookup.get((l, dname), {}).get("kc_suppression", 0) for l in layers]
        rects = ax_top.bar(x + offsets[di], vals, bar_w, label=label, color=color, alpha=0.85)
        for rect, val in zip(rects, vals):
            if val > 0.001:
                ax_top.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.001,
                    f"{val*100:.1f}%",
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=color,
                )

    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"L{l}" for l in layers], fontsize=14)
    ax_top.set_ylabel("KC Suppression (%)", fontsize=16, fontweight="bold")
    ax_top.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_top.tick_params(axis="y", labelsize=18)
    for label in ax_top.get_yticklabels():
        label.set_fontweight("bold")
    ax_top.set_ylim(top=ax_top.get_ylim()[1] * 1.20)  # 상단 여백 확보
    ax_top.legend(title="Direction", fontsize=13, title_fontsize=13, loc="upper left")
    ax_top.grid(axis="y", alpha=0.25)

    # ── Bottom: Clean Accuracy line chart ─────────────────────────────────
    ax_bot.set_title("Clean Accuracy", fontsize=16, pad=8)
    for dname in dir_names:
        label = DIR_MAP[dname]
        color = DIR_COLOR_BY_DNAME[dname]
        vals  = [lookup.get((l, dname), {}).get("clean_accuracy", float("nan")) for l in layers]
        ax_bot.plot(x, vals, marker="o", markersize=8, linewidth=2.5,
                    color=color, label=label)
        for xi, val in zip(x, vals):
            if not np.isnan(val):
                ax_bot.text(xi, val + 0.004, f"{val*100:.1f}%",
                            ha="center", va="bottom", fontsize=13,
                            color=color, fontweight="bold")

    ax_bot.set_ylim(ax_bot.get_ylim()[0], min(ax_bot.get_ylim()[1] * 1.08, 1.02))  # 상단 여백
    ax_bot.axhline(0.80, color="gray", lw=1.5, ls="--", alpha=0.7)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"L{l}" for l in layers], fontsize=14)
    ax_bot.set_xlabel("Patch Layer", fontsize=15)
    ax_bot.set_ylabel("Clean Accuracy (%)", fontsize=16, fontweight="bold")
    ax_bot.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_bot.tick_params(axis="y", labelsize=18)
    for label in ax_bot.get_yticklabels():
        label.set_fontweight("bold")
    ax_bot.legend(title="Direction", fontsize=13, title_fontsize=13, loc="lower right")
    ax_bot.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = GRID_DIR / "figure5_direction_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# IE Heatmap (step3)
# ─────────────────────────────────────────────────────────────────────────────
def draw_ie_heatmap() -> None:
    stats = json.loads((CT_DIR / "ct_stats.json").read_text())
    overall = np.array(stats["overall_ie_matrix"])

    all_vals = overall.flatten()
    vmin = min(-0.24, float(np.percentile(all_vals, 1)))
    vmax = max(0.14, float(np.percentile(all_vals, 99)))

    cmap = plt.get_cmap("RdYlGn_r")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(overall.T, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xlabel("Layer", fontsize=15)
    ax.set_ylabel("Segment", fontsize=15)
    ax.set_xticks(range(overall.shape[0]))
    ax.set_xticklabels(range(overall.shape[0]), fontsize=12)
    ax.set_yticks(range(len(SEGMENTS)))
    ax.set_yticklabels(SEGMENTS, fontsize=13)
    ax.set_title("IE Heatmap", fontsize=18, fontweight="bold", pad=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Indirect effect (IE)", fontsize=14)
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout()
    out = CT_DIR / "ie_heatmap_overall_red_positive.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Component effects by layer (step4)
# ─────────────────────────────────────────────────────────────────────────────
def draw_component_effects() -> None:
    stats = json.loads((PP_DIR / "pp_stats.json").read_text())
    layers = stats["layers"]
    overall = stats["overall"]

    ie_a = np.array(overall["IE_attn_mean"])
    ie_m = np.array(overall["IE_mlp_mean"])
    ie_f = np.array(overall["IE_full_mean"])

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, ie_a, marker="o",  linewidth=2.5, markersize=8,
            color="steelblue", label="Attention output restored")
    ax.plot(layers, ie_m, marker="^",  linewidth=2.5, markersize=8,
            color="tomato",    label="MLP output restored")
    ax.plot(layers, ie_f, marker="s",  linewidth=2.8, markersize=8,
            color="dimgray",   label="Full residual restored")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=13)
    ax.set_xlabel("Layer", fontsize=15)
    ax.set_ylabel("Indirect Effect", fontsize=18, fontweight="bold")
    ax.set_title("Step 4: Attention vs MLP vs full residual restoration",
                 fontsize=17, fontweight="bold", pad=12)
    ax.legend(fontsize=14)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = PP_DIR / "component_effects_by_layer.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== figure3_top4_comparison.png ===")
    draw_figure3()

    print("=== figure5_direction_comparison.png ===")
    draw_figure5()

    print("=== ie_heatmap_overall_red_positive.png ===")
    draw_ie_heatmap()

    print("=== component_effects_by_layer.png ===")
    draw_component_effects()

    print("\n완료!")

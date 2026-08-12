#!/usr/bin/env python3
"""
Step 4 Analysis — Path Patching 결과 분석 및 시각화

pp_stats.json을 읽어:
  1. 레이어별 설명률  (IE_attn + IE_mlp) / IE_full × 100
  2. dominant component 정리 (attn / mlp / unexplained)
  3. type별 설명률 비교 (misinformation / temporal / semantic)
  4. ie_explanation_ratio.png
  5. ie_stacked.png

실행:
  python phase1/scripts/step4_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATS_PATH = Path("/home/jaeik/xaimi/phase1/results/step4_path_patching/pp_stats.json")
OUT_DIR    = Path("/home/jaeik/xaimi/phase1/results/step4_path_patching")

UNEXPLAINED_THRESHOLD = 50.0  # 설명률 < 50% → unexplained


# ── helpers ──────────────────────────────────────────────────────────────────

def explanation_ratio(ie_attn: list[float], ie_mlp: list[float], ie_full: list[float]) -> list[float]:
    ratios = []
    for a, m, f in zip(ie_attn, ie_mlp, ie_full):
        if abs(f) < 1e-9:
            ratios.append(0.0)
        else:
            ratios.append((a + m) / f * 100.0)
    return ratios


def dominant_label(ie_attn: float, ie_mlp: float, ratio: float) -> str:
    if ratio < UNEXPLAINED_THRESHOLD:
        return "unexplained"
    return "attn" if abs(ie_attn) >= abs(ie_mlp) else "mlp"


# ── 분석 1: 레이어별 설명률 ────────────────────────────────────────────────────

def analysis1(layers: list[int], data: dict) -> list[float]:
    ie_a = data["IE_attn_mean"]
    ie_m = data["IE_mlp_mean"]
    ie_f = data["IE_full_mean"]
    ratios = explanation_ratio(ie_a, ie_m, ie_f)

    print("=" * 65)
    print("분석 1: 레이어별 설명률  (IE_attn + IE_mlp) / IE_full × 100")
    print("=" * 65)
    for i, layer in enumerate(layers):
        combo = ie_a[i] + ie_m[i]
        full  = ie_f[i]
        r     = ratios[i]
        print(f"  layer {layer:2d}: attn+mlp={combo:+.4f}  full={full:.4f}  설명률={r:6.1f}%")
    print()
    return ratios


# ── 분석 2: dominant component 정리 ───────────────────────────────────────────

def analysis2(layers: list[int], data: dict, ratios: list[float]) -> None:
    ie_a = data["IE_attn_mean"]
    ie_m = data["IE_mlp_mean"]

    attn_layers, mlp_layers, unexplained_layers = [], [], []
    for i, layer in enumerate(layers):
        label = dominant_label(ie_a[i], ie_m[i], ratios[i])
        if label == "attn":
            attn_layers.append(layer)
        elif label == "mlp":
            mlp_layers.append(layer)
        else:
            unexplained_layers.append(layer)

    print("=" * 65)
    print("분석 2: 레이어별 dominant component (기준: 설명률 ≥ 50%)")
    print("=" * 65)
    print(f"  attn-dominant  : {attn_layers}")
    print(f"  mlp-dominant   : {mlp_layers}")
    print(f"  unexplained    : {unexplained_layers}  (설명률 < {UNEXPLAINED_THRESHOLD:.0f}%)")
    print()


# ── 분석 3: type별 비교 ──────────────────────────────────────────────────────

def analysis3(layers: list[int], by_type: dict) -> None:
    print("=" * 65)
    print("분석 3: type별 설명률 비교")
    print("=" * 65)

    type_ratios: dict[str, list[float]] = {}
    for t_name, t_data in by_type.items():
        ratios = explanation_ratio(
            t_data["IE_attn_mean"],
            t_data["IE_mlp_mean"],
            t_data["IE_full_mean"],
        )
        type_ratios[t_name] = ratios

    # header
    header = f"  {'layer':>5} | " + " | ".join(f"{t:>14}" for t in type_ratios)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, layer in enumerate(layers):
        row = f"  {layer:5d} | "
        for t_name in type_ratios:
            r = type_ratios[t_name][i]
            row += f"{r:12.1f}%  "
        print(row)

    print()
    print("  평균 설명률 (전체 레이어 평균):")
    for t_name, ratios in type_ratios.items():
        mean_r = float(np.mean(ratios))
        print(f"    {t_name:16s}: {mean_r:.1f}%")
    print()


# ── 시각화 1: ie_explanation_ratio.png ───────────────────────────────────────

def plot_explanation_ratio(layers: list[int], ratios: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, ratios, marker="o", color="steelblue", linewidth=2, markersize=7, label="Explanation ratio")
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1.2, label="50% threshold")

    ax.fill_between(layers, ratios, 50,
                    where=[r >= 50 for r in ratios],
                    alpha=0.15, color="steelblue", label="Ratio >= 50%")
    ax.fill_between(layers, ratios, 50,
                    where=[r < 50 for r in ratios],
                    alpha=0.15, color="tomato", label="Ratio < 50%")

    for i, (layer, r) in enumerate(zip(layers, ratios)):
        ax.annotate(f"{r:.0f}%", (layer, r),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="steelblue")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Explanation Ratio (%)", fontsize=12)
    ax.set_title("How much of the full late-layer effect is explained by attention + MLP?", fontsize=13)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = OUT_DIR / "ie_explanation_ratio.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── 시각화 2: ie_stacked.png ─────────────────────────────────────────────────

def plot_stacked(layers: list[int], data: dict) -> None:
    ie_a = np.array(data["IE_attn_mean"])
    ie_m = np.array(data["IE_mlp_mean"])
    ie_f = np.array(data["IE_full_mean"])

    x = np.arange(len(layers))
    width = 0.55

    fig, ax = plt.subplots(figsize=(12, 5))

    # Split positive and negative values for stacked bars.
    ie_a_pos = np.where(ie_a > 0, ie_a, 0)
    ie_a_neg = np.where(ie_a < 0, ie_a, 0)
    ie_m_pos = np.where(ie_m > 0, ie_m, 0)
    ie_m_neg = np.where(ie_m < 0, ie_m, 0)

    # Positive stack
    ax.bar(x, ie_a_pos, width, color="steelblue", label="Attention output restored (+)")
    ax.bar(x, ie_m_pos, width, bottom=ie_a_pos, color="tomato", label="MLP output restored (+)")

    # Negative stack
    ax.bar(x, ie_a_neg, width, color="steelblue", alpha=0.4, label="Attention output restored (-)")
    ax.bar(x, ie_m_neg, width, bottom=ie_a_neg, color="tomato", alpha=0.4, label="MLP output restored (-)")

    # IE_full line
    ax.plot(x, ie_f, color="dimgray", marker="s", linewidth=2,
            markersize=6, label="Full residual restored", zorder=5)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Indirect Effect", fontsize=12)
    ax.set_title("Late-layer causal effect by restored component", fontsize=13)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = OUT_DIR / "ie_stacked.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_component_lines(layers: list[int], data: dict) -> None:
    ie_a = np.array(data["IE_attn_mean"])
    ie_m = np.array(data["IE_mlp_mean"])
    ie_f = np.array(data["IE_full_mean"])

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    ax.plot(
        layers,
        ie_a,
        marker="o",
        linewidth=2.2,
        color="steelblue",
        label="Attention output restored",
    )
    ax.plot(
        layers,
        ie_m,
        marker="^",
        linewidth=2.2,
        color="tomato",
        label="MLP output restored",
    )
    ax.plot(
        layers,
        ie_f,
        marker="s",
        linewidth=2.6,
        color="dimgray",
        label="Full residual restored",
    )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Indirect Effect on correct-answer recovery", fontsize=12)
    ax.set_title("Step 4: Attention vs MLP vs full residual restoration", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = OUT_DIR / "component_effects_by_layer.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with STATS_PATH.open() as f:
        stats = json.load(f)

    layers: list[int] = stats["layers"]
    overall = stats["overall"]
    by_type = stats["by_type"]

    print(f"\npp_stats.json 로드 완료 — n={stats['n']}, 레이어 {layers[0]}~{layers[-1]}\n")

    ratios = analysis1(layers, overall)
    analysis2(layers, overall, ratios)
    analysis3(layers, by_type)

    print("=" * 65)
    print("Save plots")
    print("=" * 65)
    plot_explanation_ratio(layers, ratios)
    plot_stacked(layers, overall)
    plot_component_lines(layers, overall)
    print()


if __name__ == "__main__":
    main()

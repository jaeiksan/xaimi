#!/usr/bin/env python3
"""
Exp1b Supplementary: Cohen's d plot for candidates A and B

A_dir = SVD(conflict − clean)     → "context direction"
B_dir = SVD(conflict − support)   → "conflict-vs-supportive direction"

Paired Cohen's d = mean(diff) / std(diff)

4 combinations × 14 layers:
  A_dir:  conflict vs clean     (A designed for this)
  A_dir:  conflict vs support   (cross-check)
  B_dir:  conflict vs clean     (cross-check)
  B_dir:  conflict vs support   (B designed for this)

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u scripts/exp1b_cohens_d.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u scripts/exp1b_cohens_d.py --n_cases 200
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# 한글 폰트 설정
_font_path = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
_font_prop = fm.FontProperties(fname=_font_path)
matplotlib.rcParams["font.family"] = _font_prop.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
from transformer_lens import HookedTransformer

# ── 상수 ──────────────────────────────────────────────────────────────────────
LAYERS     = list(range(6, 20))
MODEL_NAME = "google/gemma-2-2b"
KC_PATH    = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
OUT_DIR    = Path("phase2/results/exp1b_candidates")

CAND_NAME = {"A": "A_svd_conf_clean", "B": "B_svd_conf_support"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_cases", type=int, default=500)
    return p.parse_args()


def build_conflict_prompt(ctx, q, opts):
    lines = ["Context:", ctx, "", f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_neutral_prompt(q, opts):
    lines = [f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_supportive_prompt(ctx, q, opts, conflict_answer, correct_answer):
    sup_ctx = ctx.replace(conflict_answer, correct_answer)
    lines = ["Context:", sup_ctx, "", f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(a − b) / std(a − b)."""
    diff = a - b
    s = diff.std()
    return float(diff.mean() / s) if s > 1e-10 else 0.0


def main():
    args = parse_args()
    n_cases = args.n_cases

    kc_cases = json.load(open(KC_PATH))[:n_cases]
    cands    = torch.load(CANDS_PATH, map_location="cpu")

    # 방향 벡터 로드 (float32, CPU)
    dirs: dict = {}
    for k, name in CAND_NAME.items():
        dirs[k] = {}
        for l in LAYERS:
            for entry in cands[l]:
                if entry["name"] == name:
                    dirs[k][l] = entry["dir"].float()
                    break

    # 모델 로드
    print("모델 로드 ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16
    )
    model.eval()
    device = next(model.parameters()).device

    # 누적 배열: proj[cand][cond][layer] = list of scalar projections
    projs = {
        cand: {cond: {l: [] for l in LAYERS} for cond in ("conf", "clean", "sup")}
        for cand in ("A", "B")
    }

    print(f"[cohens_d] {n_cases}개 케이스 residual 계산 ...", flush=True)
    t0 = time.time()

    for i, case in enumerate(kc_cases):
        ctx = case["conflict_context"]
        q   = case["question"]
        opt = case["options"]
        ca  = case["conflict_answer"]
        coa = case["correct_answer"]

        prompts = {
            "conf":  build_conflict_prompt(ctx, q, opt),
            "clean": build_neutral_prompt(q, opt),
            "sup":   build_supportive_prompt(ctx, q, opt, ca, coa),
        }

        for cond, prompt in prompts.items():
            tokens = model.to_tokens(prompt, prepend_bos=True)
            seq_len = tokens.shape[1]
            pos_idx = seq_len - 1        # final token

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: "resid_post" in name,
                    return_type=None,
                )

            for l in LAYERS:
                h = cache[f"blocks.{l}.hook_resid_post"][0, pos_idx, :].float().cpu()
                for cand in ("A", "B"):
                    proj = (h @ dirs[cand][l]).item()
                    projs[cand][cond][l].append(proj)

            del cache

        if (i + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  {i+1}/{n_cases} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[cohens_d] 계산 완료 ({time.time()-t0:.0f}s)", flush=True)

    # Cohen's d 계산
    # d[cand][comparison][layer]
    # comparisons: conf_vs_clean, conf_vs_sup
    d_vals = {cand: {} for cand in ("A", "B")}
    for cand in ("A", "B"):
        for comp, (g1, g2) in [("conf_vs_clean", ("conf", "clean")),
                                ("conf_vs_sup",   ("conf", "sup"))]:
            d_vals[cand][comp] = []
            for l in LAYERS:
                a = np.array(projs[cand][g1][l])
                b = np.array(projs[cand][g2][l])
                d_vals[cand][comp].append(cohens_d_paired(a, b))

    # 결과 출력
    print("\n=== Cohen's d (paired) ===")
    print(f"{'Layer':<6} {'A: conf-clean':>14} {'A: conf-sup':>12} {'B: conf-clean':>14} {'B: conf-sup':>12}")
    for i, l in enumerate(LAYERS):
        print(f"  L{l:<4} {d_vals['A']['conf_vs_clean'][i]:>14.3f} "
              f"{d_vals['A']['conf_vs_sup'][i]:>12.3f} "
              f"{d_vals['B']['conf_vs_clean'][i]:>14.3f} "
              f"{d_vals['B']['conf_vs_sup'][i]:>12.3f}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)

    layers_arr = np.array(LAYERS)

    COMP_LABELS = {
        "conf_vs_clean": "Conflict vs Clean",
        "conf_vs_sup":   "Conflict vs Supportive",
    }
    CAND_COLORS = {"A": "#e74c3c", "B": "#3498db"}
    CAND_LABELS_FULL = {
        "A": "A: SVD(conflict − clean)",
        "B": "B: SVD(conflict − support)",
    }

    # ── Subplot 1: A direction, 두 비교 ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])  # 상단 전체 폭
    for comp, ls, marker in [("conf_vs_clean", "-",  "o"),
                              ("conf_vs_sup",   "--", "s")]:
        ax1.plot(layers_arr, d_vals["A"][comp],
                 color=CAND_COLORS["A"], linestyle=ls, marker=marker,
                 linewidth=2.2, markersize=7,
                 label=f"A → {COMP_LABELS[comp]}")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.axvline(14, color="#e67e22", linewidth=1.2, linestyle="--", alpha=0.7,
                label="L14 (transition)")
    ax1.set_title("Candidate A (context direction) — Cohen's d by comparison pair",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cohen's d (paired)", fontsize=11)
    ax1.set_xticks(LAYERS)
    ax1.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.25)

    # ── Subplot 2: B direction, 두 비교 ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])  # 하단 전체 폭
    for comp, ls, marker in [("conf_vs_clean", "-",  "o"),
                              ("conf_vs_sup",   "--", "s")]:
        ax2.plot(layers_arr, d_vals["B"][comp],
                 color=CAND_COLORS["B"], linestyle=ls, marker=marker,
                 linewidth=2.2, markersize=7,
                 label=f"B → {COMP_LABELS[comp]}")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax2.axvline(14, color="#e67e22", linewidth=1.2, linestyle="--", alpha=0.7,
                label="L14 (transition)")
    ax2.set_title("Candidate B (conflict−support direction) — Cohen's d by comparison pair",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Cohen's d (paired)", fontsize=11)
    ax2.set_xticks(LAYERS)
    ax2.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.25)

    plt.suptitle(
        "Exp1b: Cohen's d — A vs B Candidate Direction Separation\n"
        "실선: 해당 방향이 설계된 비교쌍 / 점선: 교차 비교",
        fontsize=13, fontweight="bold", y=1.01,
    )

    path = OUT_DIR / "cohens_d_A_B.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)

    # ── 보조 그래프: A vs B 직접 비교 (각자의 주 비교쌍) ────────────────────
    fig2, ax = plt.subplots(figsize=(11, 5))

    ax.plot(layers_arr, d_vals["A"]["conf_vs_clean"],
            color=CAND_COLORS["A"], marker="o", linewidth=2.5, markersize=8,
            label="A: conflict vs clean  (context signal)")
    ax.plot(layers_arr, d_vals["B"]["conf_vs_sup"],
            color=CAND_COLORS["B"], marker="s", linewidth=2.5, markersize=8,
            linestyle="--", label="B: conflict vs supportive  (KC-divergence signal)")

    # 갈라짐 구간 하이라이트
    d_A = np.array(d_vals["A"]["conf_vs_clean"])
    d_B = np.array(d_vals["B"]["conf_vs_sup"])
    for i, l in enumerate(LAYERS):
        if d_A[i] > d_B[i]:
            ax.axvspan(l - 0.45, l + 0.45, alpha=0.08, color="#e74c3c", zorder=0)
        else:
            ax.axvspan(l - 0.45, l + 0.45, alpha=0.08, color="#3498db", zorder=0)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(14, color="#e67e22", linewidth=1.5, linestyle="--", alpha=0.8,
               label="L14 transition")

    # A-B diff 수치 표기
    for i, l in enumerate(LAYERS):
        diff = d_A[i] - d_B[i]
        y_top = max(d_A[i], d_B[i]) + 0.05
        if abs(diff) > 0.1:
            ax.annotate(f"{diff:+.2f}", (l, y_top),
                        ha="center", va="bottom", fontsize=7,
                        color="#c0392b" if diff > 0 else "#2980b9")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cohen's d (paired)", fontsize=12)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.title(
        "A vs B: Signal Separation Strength\n"
        "A=빨강 우세 구간 vs B=파랑 우세 구간 (배경색)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path2 = OUT_DIR / "cohens_d_A_vs_B.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path2}", flush=True)

    print(f"\n[cohens_d] 완료. 결과: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exp1b Candidate C (cPCA) 분석 — 그래프 및 요약

Figure 1: cPCA eigenvalue + C↔A/B/D overlap 레이어별 변화
Figure 2: Cohen's d for C direction (conflict vs clean / conflict vs support)
Figure 3: C eigenvalue vs IE_C (Exp2) 비교 — representational vs causal

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u scripts/exp1b_candidate_C_analysis.py
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from transformer_lens import HookedTransformer

_fp = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
matplotlib.rcParams["font.family"] = _fp.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 상수 ──────────────────────────────────────────────────────────────────────
LAYERS     = list(range(6, 20))
MODEL_NAME = "google/gemma-2-2b"
KC_PATH    = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
EXP1B_SUM  = Path("phase2/results/exp1b_candidates/summary.json")
EXP2_SUM   = Path("phase2/results/exp2_causal_patching/IE_summary.json")
OUT_DIR    = Path("phase2/results/exp1b_candidates")
N_CASES    = 500


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


def build_supportive_prompt(ctx, q, opts, ca, coa):
    sup_ctx = ctx.replace(ca, coa)
    lines = ["Context:", sup_ctx, "", f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    s = diff.std()
    return float(diff.mean() / s) if s > 1e-10 else 0.0


# ── Figure 1: eigenvalue + overlap ────────────────────────────────────────────
def plot_figure1(exp1b: dict, out_dir: Path) -> None:
    layers_arr = np.array(LAYERS)
    eig     = [exp1b[str(l)]["C_top_eigenvalue"]  for l in LAYERS]
    ov_A    = [exp1b[str(l)]["C_overlap_with_A"]  for l in LAYERS]
    ov_B    = [exp1b[str(l)]["C_overlap_with_B"]  for l in LAYERS]
    ov_D    = [exp1b[str(l)]["D_overlap_with_C"]  for l in LAYERS]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                    gridspec_kw={"hspace": 0.35})

    # ── 상단: eigenvalue ──────────────────────────────────────────────────────
    ax1.fill_between(layers_arr, eig, alpha=0.15, color="#2ecc71")
    ax1.plot(layers_arr, eig, color="#27ae60", marker="o", linewidth=2.5,
             markersize=8, label="cPCA top eigenvalue")
    ax1.axvline(14, color="#e67e22", linewidth=1.4, linestyle="--", alpha=0.8,
                label="L14 transition")

    # L17-19 하이라이트
    for l in [17, 18, 19]:
        ax1.axvspan(l - 0.45, l + 0.45, alpha=0.12, color="#27ae60", zorder=0)

    # 수치 표기 (L14+)
    for i, l in enumerate(LAYERS):
        if l >= 14:
            ax1.annotate(f"{eig[i]:.0f}", (l, eig[i] + 30),
                         ha="center", va="bottom", fontsize=8, color="#27ae60")

    ax1.set_ylabel("cPCA Top Eigenvalue\n(conflict variance excess over support)", fontsize=10)
    ax1.set_title("Figure 1-A: cPCA (Candidate C) Eigenvalue — Layer별 KC 특화 분산",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.set_yscale("log")
    ax1.set_ylabel("cPCA Top Eigenvalue (log scale)", fontsize=10)

    # ── 하단: overlap ─────────────────────────────────────────────────────────
    ax2.plot(layers_arr, ov_A, color="#e74c3c", marker="o", linewidth=2.2,
             markersize=7, label="|cos(C, A)| — C↔context dir")
    ax2.plot(layers_arr, ov_B, color="#3498db", marker="s", linewidth=2.2,
             markersize=7, linestyle="--", label="|cos(C, B)| — C↔conflict-support dir")
    ax2.plot(layers_arr, ov_D, color="#9b59b6", marker="^", linewidth=1.8,
             markersize=7, linestyle=":", label="|cos(C, D)| — C↔regression dir")

    ax2.axhline(0.3, color="gray", linewidth=0.8, linestyle="--", alpha=0.6,
                label="0.3 threshold")
    ax2.axvline(14, color="#e67e22", linewidth=1.4, linestyle="--", alpha=0.8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("|cos| overlap", fontsize=10)
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_title("Figure 1-B: C 방향의 다른 후보들과의 overlap\n(낮을수록 해당 후보와 독립적인 방향)",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(LAYERS)
    ax2.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.25)

    plt.suptitle("Exp1b Candidate C (cPCA, α=0.5): Eigenvalue & Direction Overlap",
                 fontsize=13, fontweight="bold")
    path = out_dir / "candidate_C_figure1_eigenvalue_overlap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


# ── Figure 2: Cohen's d ────────────────────────────────────────────────────────
def compute_cohens_d_C(model, kc_cases, c_dirs) -> dict:
    """C 방향으로 conflict/clean/support 투영 → Cohen's d 계산."""
    projs = {cond: {l: [] for l in LAYERS} for cond in ("conf", "clean", "sup")}

    print(f"[cohens_d_C] {len(kc_cases)}개 케이스 ...", flush=True)
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
            tokens  = model.to_tokens(prompt, prepend_bos=True)
            seq_len = tokens.shape[1]
            pos_idx = seq_len - 1

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: "resid_post" in name,
                    return_type=None,
                )

            for l in LAYERS:
                h    = cache[f"blocks.{l}.hook_resid_post"][0, pos_idx, :].float()
                proj = (h @ c_dirs[l].to(h.device)).item()
                projs[cond][l].append(proj)

            del cache

        if (i + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  {i+1}/{len(kc_cases)} ({time.time()-t0:.0f}s)", flush=True)

    d_conf_clean = []
    d_conf_sup   = []
    for l in LAYERS:
        a_cc = np.array(projs["conf"][l])
        b_cc = np.array(projs["clean"][l])
        b_sup = np.array(projs["sup"][l])
        d_conf_clean.append(cohens_d_paired(a_cc, b_cc))
        d_conf_sup.append(cohens_d_paired(a_cc, b_sup))

    return {"conf_vs_clean": d_conf_clean, "conf_vs_sup": d_conf_sup}


def plot_figure2(d_vals: dict, out_dir: Path) -> None:
    layers_arr = np.array(LAYERS)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(layers_arr, d_vals["conf_vs_clean"],
            color="#e74c3c", marker="o", linewidth=2.4, markersize=8,
            label="C방향: conflict vs clean  (context signal 포착 여부)")
    ax.plot(layers_arr, d_vals["conf_vs_sup"],
            color="#27ae60", marker="s", linewidth=2.4, markersize=8,
            linestyle="--", label="C방향: conflict vs supportive  (KC-specific signal 포착 여부)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(14, color="#e67e22", linewidth=1.4, linestyle="--", alpha=0.8,
               label="L14 transition")

    # 수치 표기
    for i, l in enumerate(LAYERS):
        for vals, color, dy in [(d_vals["conf_vs_clean"], "#c0392b", 0.05),
                                (d_vals["conf_vs_sup"],   "#1a8a50", -0.15)]:
            if abs(vals[i]) > 0.05:
                ax.annotate(f"{vals[i]:.2f}", (l, vals[i] + dy),
                            ha="center", va="bottom", fontsize=7.5, color=color)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Cohen's d (paired)", fontsize=11)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.title("Figure 2: Candidate C — Cohen's d by Comparison Pair\n"
              "C가 context 신호(빨강)와 KC-specific 신호(초록) 중 어느 것을 더 포착하는가",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "candidate_C_figure2_cohens_d.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


# ── Figure 3: eigenvalue vs IE_C ──────────────────────────────────────────────
def plot_figure3(exp1b: dict, exp2: dict, out_dir: Path) -> None:
    layers_arr = np.array(LAYERS)
    eig    = np.array([exp1b[str(l)]["C_top_eigenvalue"] for l in LAYERS])
    ie_c   = np.array([exp2["C"][str(l)]["final"]["mean"] for l in LAYERS])
    pos_r  = np.array([exp2["C"][str(l)]["final"]["positive_ratio"] for l in LAYERS])

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                              gridspec_kw={"hspace": 0.4})

    # ── 상단: eigenvalue vs IE_C (dual y-axis) ────────────────────────────────
    ax1 = axes[0]
    ax1r = ax1.twinx()

    ax1.fill_between(layers_arr, np.log10(eig), alpha=0.12, color="#27ae60")
    ax1.plot(layers_arr, np.log10(eig), color="#27ae60", marker="o",
             linewidth=2.2, markersize=7, label="log₁₀(C eigenvalue)  [좌축]")

    colors_ie = ["#e74c3c" if v > 0 else "#3498db" for v in ie_c]
    ax1r.bar(layers_arr, ie_c, color=colors_ie, alpha=0.6, zorder=2,
             label="IE_C (Exp2)  [우축]")
    ax1r.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax1.axvline(14, color="#e67e22", linewidth=1.4, linestyle="--", alpha=0.8)

    # 음수 IE_C 구간 강조
    for i, l in enumerate(LAYERS):
        if ie_c[i] < -0.1:
            ax1.axvspan(l - 0.45, l + 0.45, alpha=0.08, color="#3498db", zorder=0)

    ax1.set_ylabel("log₁₀(cPCA eigenvalue)", color="#27ae60", fontsize=10)
    ax1r.set_ylabel("IE_C mean  (↑ = KC 감소)", color="#e74c3c", fontsize=10)
    ax1.tick_params(axis="y", colors="#27ae60")
    ax1r.tick_params(axis="y", colors="#e74c3c")
    ax1.set_title("Figure 3-A: cPCA Eigenvalue vs Causal Effect (IE_C)\n"
                  "파랑 배경: IE_C < −0.1 (방향 제거 시 KC 오히려 증가)",
                  fontsize=11, fontweight="bold")

    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbs1 + lbs2, fontsize=9, loc="upper left")
    ax1.grid(alpha=0.2)

    # ── 하단: IE_C + pos_ratio ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2r = ax2.twinx()

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.errorbar(layers_arr, ie_c,
                 yerr=[exp2["C"][str(l)]["final"]["std"] for l in LAYERS],
                 color="#27ae60", marker="o", linewidth=2.2, capsize=3,
                 markersize=7, label="IE_C mean ± std  [좌축]")

    ax2r.bar(layers_arr, pos_r, alpha=0.3, color="#9b59b6",
             label="positive ratio  [우축]")
    ax2r.axhline(0.5, color="#9b59b6", linewidth=0.8, linestyle=":", alpha=0.7)
    ax2r.set_ylim(0, 1.2)

    ax2.axvline(14, color="#e67e22", linewidth=1.4, linestyle="--", alpha=0.8,
                label="L14 transition")

    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("IE_C mean", color="#27ae60", fontsize=10)
    ax2r.set_ylabel("Positive ratio", color="#9b59b6", fontsize=10)
    ax2.tick_params(axis="y", colors="#27ae60")
    ax2r.tick_params(axis="y", colors="#9b59b6")
    ax2.set_xticks(LAYERS)
    ax2.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
    ax2.set_title("Figure 3-B: IE_C detail — 효과 크기 + positive ratio",
                  fontsize=11, fontweight="bold")

    lines1, lbs1 = ax2.get_legend_handles_labels()
    lines2, lbs2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbs1 + lbs2, fontsize=9)
    ax2.grid(alpha=0.2)

    plt.suptitle("Exp1b Candidate C (cPCA): Representational Strength vs Causal Effect",
                 fontsize=13, fontweight="bold")
    path = out_dir / "candidate_C_figure3_eig_vs_IE.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    exp1b = json.load(open(EXP1B_SUM))
    exp2  = json.load(open(EXP2_SUM))

    # Figure 1 (모델 불필요)
    print("[C analysis] Figure 1 ...", flush=True)
    plot_figure1(exp1b, OUT_DIR)

    # Figure 3 (모델 불필요)
    print("[C analysis] Figure 3 ...", flush=True)
    plot_figure3(exp1b, exp2, OUT_DIR)

    # Figure 2: Cohen's d (모델 필요)
    print("[C analysis] 모델 로드 (Figure 2용) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16
    )
    model.eval()

    cands = torch.load(CANDS_PATH, map_location="cpu")
    device = next(model.parameters()).device
    c_dirs = {}
    for l in LAYERS:
        for entry in cands[l]:
            if entry["name"] == "C_cpca":
                c_dirs[l] = entry["dir"].float().to(device)
                break

    kc_cases = json.load(open(KC_PATH))[:N_CASES]
    d_vals = compute_cohens_d_C(model, kc_cases, c_dirs)
    plot_figure2(d_vals, OUT_DIR)

    # 수치 요약 출력
    print("\n=== Candidate C 수치 요약 ===")
    print(f"{'Layer':<6} {'C_eig':>10} {'C↔A':>7} {'C↔B':>7} {'d(cc)':>8} {'d(cs)':>8} {'IE_C':>8} {'pos_r':>7}")
    for i, l in enumerate(LAYERS):
        r    = exp1b[str(l)]
        ie_r = exp2["C"][str(l)]["final"]
        print(f"  L{l:<4} {r['C_top_eigenvalue']:>10.1f} "
              f"{r['C_overlap_with_A']:>7.3f} {r['C_overlap_with_B']:>7.3f} "
              f"{d_vals['conf_vs_clean'][i]:>8.3f} {d_vals['conf_vs_sup'][i]:>8.3f} "
              f"{ie_r['mean']:>+8.4f} {ie_r['positive_ratio']:>7.3f}")

    print(f"\n[C analysis] 완료. 결과: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

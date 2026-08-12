#!/usr/bin/env python3
"""
실험 1: KC Direction Trajectory

KC 신호가 residual space에서 어느 방향으로, 얼마나 KC에 특화되어 존재하는가를 SVD로 분석.

세 가지 condition:
  Condition 1 (conflict):   기존 conflict_prompt 그대로 사용
  Condition 2 (clean):      기존 clean_prompt 그대로 사용
  Condition 3 (supportive): correct answer를 지지하는 context를 추가한 prompt
    → kc_cases에 supportive_prompt 필드가 없으므로,
      clean_prompt 앞에 "The answer is {correct_answer}."를 context로 prepend하는 방식 사용.

출력:
  phase2/results/exp1_trajectory/
    trajectory_stats.pt       — layer별 kc_dir, 수치 (exp2/exp3에서 사용)
    trajectory_plot.png       — Figure 1: kc_strength, sup_strength, overlap, continuity
    direction_overlap_plot.png — Figure 2: kc_sup_overlap by layer

실행:
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1_trajectory.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1_trajectory.py --test 20
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from exp_utils import (
    LAYERS, N_CASES, SEED,
    load_model, load_cases,
    build_clean_prompt, build_conflict_prompt, build_supportive_prompt,
    collect_residuals,
    get_pos_indices,
    save_pt, save_json,
)

OUT_DIR = Path("/home/jaeik/xaimi/phase2/results/exp1_trajectory")
CKPT_PATH = OUT_DIR / "residuals_checkpoint.pt"
CHECKPOINT_EVERY = 50


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실험 1: KC Direction Trajectory")
    p.add_argument("--test",    type=int, default=None, help="N개만 처리 (테스트용)")
    p.add_argument("--n_cases", type=int, default=None, help="사용할 케이스 수 (기본: N_CASES=200)")
    return p.parse_args()


# ── Residual 수집 (3 conditions) ──────────────────────────────────────────────
def collect_all_residuals(
    model,
    cases: List[Dict],
    layers: List[int],
) -> Dict[str, torch.Tensor]:
    """
    각 case에 대해 3개 condition에서 final token position의 residual을 수집.

    Returns:
        {
          "conflict": (N, len(layers), d_model),
          "clean":    (N, len(layers), d_model),
          "support":  (N, len(layers), d_model),
        }
    """
    N = len(cases)
    L = len(layers)

    # 체크포인트 로드
    if CKPT_PATH.exists():
        print(f"[exp1] 체크포인트 로드: {CKPT_PATH}", flush=True)
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        h_conflict = ckpt["conflict"]
        h_clean    = ckpt["clean"]
        h_support  = ckpt["support"]
        start_i    = ckpt["n_done"]
        print(f"[exp1] {start_i}개 이미 완료, 이어서 진행", flush=True)
    else:
        d_model = model.cfg.d_model
        h_conflict = torch.zeros(N, L, d_model)
        h_clean    = torch.zeros(N, L, d_model)
        h_support  = torch.zeros(N, L, d_model)
        start_i    = 0

    t0 = time.time()
    for i in range(start_i, N):
        case = cases[i]
        q    = case["question"]
        opts = case["options"]
        ctx  = case["conflict_context"]
        cor  = case["correct_option"]
        cor_ans = case["correct_answer"]

        conflict_p  = build_conflict_prompt(ctx, q, opts)
        clean_p     = build_clean_prompt(q, opts)
        support_p   = build_supportive_prompt(q, opts, cor_ans, ctx, case.get("conflict_answer"))

        final_pos, _ = get_pos_indices(model, conflict_p)

        res_cf, _ = collect_residuals(model, conflict_p, layers, final_pos)
        h_conflict[i] = res_cf

        clean_fp, _ = get_pos_indices(model, clean_p)
        res_cl, _ = collect_residuals(model, clean_p, layers, clean_fp)
        h_clean[i] = res_cl

        sup_fp, _ = get_pos_indices(model, support_p)
        res_sp, _ = collect_residuals(model, support_p, layers, sup_fp)
        h_support[i] = res_sp

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i - start_i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_pt(CKPT_PATH, {
                "conflict": h_conflict,
                "clean":    h_clean,
                "support":  h_support,
                "n_done":   i + 1,
            })

    return {"conflict": h_conflict, "clean": h_clean, "support": h_support}


# ── SVD 분석 ─────────────────────────────────────────────────────────────────
def analyze_trajectory(
    residuals: Dict[str, torch.Tensor],
    layers: List[int],
) -> Dict:
    """
    Layer별 SVD로 KC direction, 강도, overlap, continuity를 계산.

    residuals: {"conflict": (N, L, d_model), "clean": ..., "support": ...}

    Returns:
        trajectory_stats[layer] = {
            "kc_strength":        float  — 1위 singular value (KC 신호 강도)
            "sup_strength":       float  — 1위 singular value (supportive 신호 강도)
            "kc_sup_overlap":     float  — |cos(kc_dir, sup_dir)| (낮을수록 KC-specific)
            "direction_continuity": float or None — |cos(kc_dir_l, kc_dir_{l-1})|
            "kc_dir":             Tensor (d_model,)  — KC 주방향 (unit norm)
            "clean_mean_proj":    float  — clean residual의 kc_dir 방향 평균 projection
        }
    """
    h_conflict = residuals["conflict"]  # (N, L, d_model)
    h_clean    = residuals["clean"]
    h_support  = residuals["support"]

    stats = {}
    prev_kc_dir = None

    for li, layer in enumerate(layers):
        # D_kc = conflict - clean  (N, d_model)
        D_kc  = (h_conflict[:, li, :] - h_clean[:, li, :]).float()
        D_sup = (h_support[:, li, :]  - h_clean[:, li, :]).float()

        # SVD
        _, S_kc,  Vt_kc  = torch.linalg.svd(D_kc,  full_matrices=False)
        _, S_sup, Vt_sup = torch.linalg.svd(D_sup, full_matrices=False)

        kc_dir  = F.normalize(Vt_kc[0],  dim=-1)   # (d_model,)
        sup_dir = F.normalize(Vt_sup[0], dim=-1)

        # KC ↔ supportive direction overlap
        kc_sup_overlap = float(torch.abs(kc_dir @ sup_dir).item())

        # Layer 간 KC direction continuity
        if prev_kc_dir is not None:
            continuity = float(torch.abs(kc_dir @ prev_kc_dir).item())
        else:
            continuity = None

        # clean residual의 kc_dir 방향 평균 projection (exp2/3에서 사용)
        clean_projs = (h_clean[:, li, :].float() @ kc_dir).tolist()
        clean_mean_proj = float(np.mean(clean_projs))

        stats[layer] = {
            "kc_strength":          float(S_kc[0].item()),
            "sup_strength":         float(S_sup[0].item()),
            "kc_sup_overlap":       kc_sup_overlap,
            "direction_continuity": continuity,
            "kc_dir":               kc_dir,           # Tensor (d_model,)
            "clean_mean_proj":      clean_mean_proj,
        }
        prev_kc_dir = kc_dir

        cont_str = f"{continuity:.3f}" if continuity is not None else "N/A"
        print(
            f"  layer {layer:2d}: kc_str={S_kc[0]:.2f}  sup_str={S_sup[0]:.2f}"
            f"  overlap={kc_sup_overlap:.3f}  cont={cont_str}",
            flush=True,
        )

    return stats


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_trajectory(stats: Dict, layers: List[int], out_dir: Path) -> None:
    """Figure 1: kc_strength, sup_strength, kc_sup_overlap, direction_continuity."""
    x = layers

    kc_str   = [stats[l]["kc_strength"]          for l in x]
    sup_str  = [stats[l]["sup_strength"]          for l in x]
    overlap  = [stats[l]["kc_sup_overlap"]         for l in x]
    cont     = [stats[l]["direction_continuity"] or 0.0 for l in x]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # (1) KC vs Supportive strength
    axes[0].plot(x, kc_str,  color="#e74c3c", linewidth=2, label="KC strength (top singular value)")
    axes[0].plot(x, sup_str, color="#3498db", linewidth=2, label="Supportive strength (top singular value)", linestyle="--")
    axes[0].set_ylabel("1st Singular Value")
    axes[0].legend(fontsize=9)
    axes[0].set_title("KC vs Supportive Signal Strength by Layer", fontsize=12)
    axes[0].grid(alpha=0.3)

    # (2) KC-Supportive direction overlap
    axes[1].plot(x, overlap, color="#2ecc71", linewidth=2, marker="o", markersize=5)
    axes[1].axhline(0.3, color="gray", linestyle=":", linewidth=1, label="0.3 threshold")
    axes[1].set_ylabel("|cos(kc_dir, sup_dir)|")
    axes[1].set_title("KC-Supportive Direction Overlap (lower = more KC-specific)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    # (3) Direction continuity (adjacent layers)
    axes[2].plot(x, cont, color="#9b59b6", linewidth=2, marker="s", markersize=5)
    axes[2].axhline(0.7, color="gray", linestyle=":", linewidth=1, label="0.7 threshold")
    axes[2].set_ylabel("|cos(kc_dir_l, kc_dir_{l-1})|")
    axes[2].set_xlabel("Layer")
    axes[2].set_title("KC Direction Continuity (lower = larger direction shift)", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(x)

    plt.tight_layout()
    path = out_dir / "trajectory_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_direction_overlap(stats: Dict, layers: List[int], out_dir: Path) -> None:
    """Figure 2: kc_sup_overlap by layer."""
    x = layers
    overlap = [stats[l]["kc_sup_overlap"] for l in x]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(x, overlap, color=[
        "#e74c3c" if v < 0.3 else ("#e67e22" if v < 0.5 else "#95a5a6")
        for v in overlap
    ], edgecolor="white")
    ax.axhline(0.3, color="red",  linestyle="--", linewidth=1.5, label="0.3 (KC-specific)")
    ax.axhline(0.5, color="gray", linestyle=":",  linewidth=1.0, label="0.5 threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|cos(kc_dir, sup_dir)|")
    ax.set_title(
        "KC Direction vs Supportive Direction Overlap\n"
        "(lower = layer direction is more KC-specific)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "direction_overlap_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)
    n      = args.test if args.test else (args.n_cases if args.n_cases else N_CASES)
    cases  = load_cases(n)

    print(f"\n[exp1] {len(cases)}개 케이스, layers={LAYERS}", flush=True)

    # 1. Residual 수집
    print("\n[exp1] Step 1: 3 conditions residual 수집 ...", flush=True)
    t0 = time.time()
    residuals = collect_all_residuals(model, cases, LAYERS)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    # 2. SVD 분석
    print("\n[exp1] Step 2: Layer별 SVD 분석 ...", flush=True)
    stats = analyze_trajectory(residuals, LAYERS)

    # 3. 저장
    save_pt(OUT_DIR / "trajectory_stats.pt", stats)

    # JSON 호환 summary (tensor 제외)
    stats_json = {}
    for layer, s in stats.items():
        stats_json[str(layer)] = {
            k: v for k, v in s.items() if not isinstance(v, torch.Tensor)
        }
    save_json(OUT_DIR / "trajectory_stats_summary.json", stats_json)

    # 4. 시각화
    print("\n[exp1] Step 3: 시각화 ...", flush=True)
    plot_trajectory(stats, LAYERS, OUT_DIR)
    plot_direction_overlap(stats, LAYERS, OUT_DIR)

    # 5. 체크포인트 정리
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print(f"[exp1] 체크포인트 삭제: {CKPT_PATH}", flush=True)

    print(f"\n[exp1] 완료. 결과: {OUT_DIR}", flush=True)

    # 핵심 결과 요약
    print("\n=== 실험 1 핵심 결과 ===")
    low_overlap = [(l, stats[l]["kc_sup_overlap"]) for l in LAYERS
                   if stats[l]["kc_sup_overlap"] < 0.3]
    print(f"KC-specific layer (overlap < 0.3): {[l for l, _ in low_overlap]}")
    kc_strong = max(LAYERS, key=lambda l: stats[l]["kc_strength"])
    print(f"KC 신호가 가장 강한 layer: {kc_strong}")
    breaks = [(l, stats[l]["direction_continuity"]) for l in LAYERS
              if stats[l]["direction_continuity"] is not None
              and stats[l]["direction_continuity"] < 0.7]
    print(f"Direction continuity < 0.7 (방향 꺾임): {[l for l, _ in breaks]}")


if __name__ == "__main__":
    main()

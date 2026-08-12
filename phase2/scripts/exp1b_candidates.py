#!/usr/bin/env python3
"""
Exp1b: KC Direction Candidate Extraction

네 가지 방법으로 방향 후보를 추출해 Exp2 causal patching에 넘긴다.

방향 후보:
  A: SVD(conflict - clean)         — Exp1 결과 재사용, context-following 기준선
  B: SVD(conflict - supportive)    — context 효과 제거 후 KC 방향
  C: cPCA                          — conflict vs supportive 분산 구조 차이
  D: Regression (KC severity)      — KC 강도와 직접 연결된 방향

출력:
  phase2/results/exp1b_candidates/
    all_candidates.pt         — 네 후보 + 수렴 분석
    candidates_for_exp2.pt    — Exp2 입력 형식 (unit-norm 확인)
    summary.json
    figure1_candidate_strength.png
    figure2_convergence.png
    figure3_diff_from_exp1.png

실행:
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1b_candidates.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1b_candidates.py --test 20
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1b_candidates.py --n_cases 1000
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    load_trajectory_stats,
    save_pt, save_json,
)

RESULTS_DIR = Path("/home/jaeik/xaimi/phase2/results/exp1b_candidates")
EXP1_DIR    = Path("/home/jaeik/xaimi/phase2/results/exp1_trajectory")
CKPT_PATH   = RESULTS_DIR / "residuals_checkpoint.pt"
CHECKPOINT_EVERY = 50
ALPHA_VALUES     = [0.5, 1.0, 2.0]
PCA_COMPONENTS   = 100
RIDGE_ALPHA      = 1.0


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1b: KC Direction Candidate Extraction")
    p.add_argument("--test",    type=int, default=None, help="N개만 처리 (테스트용)")
    p.add_argument("--n_cases", type=int, default=None, help="케이스 수 (기본: N_CASES=200)")
    return p.parse_args()


# ── PyTorch PCA + Ridge (sklearn 미설치 환경) ─────────────────────────────────
def torch_pca(
    X: torch.Tensor, n_components: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: (N, d) float32
    Returns: X_proj (N, k), components (k, d)
    """
    X_c = X - X.mean(0)
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components]       # (k, d)
    X_proj     = X_c @ components.T     # (N, k)
    return X_proj, components


def torch_ridge(
    X_train: torch.Tensor, y_train: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Ridge regression closed form: w = (XᵀX + αI)⁻¹ Xᵀy"""
    k = X_train.shape[1]
    A = X_train.T @ X_train + alpha * torch.eye(k, dtype=X_train.dtype)
    b = X_train.T @ y_train
    return torch.linalg.solve(A, b)


def pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = (a_c.norm() * b_c.norm()).item()
    return 0.0 if denom < 1e-8 else float((a_c @ b_c).item() / denom)


# ── Residual 수집 ─────────────────────────────────────────────────────────────
def load_or_compute_residuals(
    model,
    cases: List[Dict],
    layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    conflict / clean / supportive 각 final token residual을 수집.

    Returns:
        h_conflict: {layer: (N, d_model)}
        h_clean:    {layer: (N, d_model)}
        h_support:  {layer: (N, d_model)}
    """
    N       = len(cases)
    d_model = model.cfg.d_model

    if CKPT_PATH.exists():
        print(f"[exp1b] 체크포인트 로드: {CKPT_PATH}", flush=True)
        ckpt       = torch.load(CKPT_PATH, map_location="cpu")
        h_conflict = ckpt["h_conflict"]
        h_clean    = ckpt["h_clean"]
        h_support  = ckpt["h_support"]
        start_i    = ckpt["n_done"]
        print(f"[exp1b] {start_i}개 완료, 이어서 진행", flush=True)
    else:
        h_conflict = {l: torch.zeros(N, d_model) for l in layers}
        h_clean    = {l: torch.zeros(N, d_model) for l in layers}
        h_support  = {l: torch.zeros(N, d_model) for l in layers}
        start_i    = 0

    t0 = time.time()
    for i in range(start_i, N):
        case    = cases[i]
        q       = case["question"]
        opts    = case["options"]
        ctx     = case["conflict_context"]
        cor_ans = case["correct_answer"]
        con_ans = case.get("conflict_answer")

        conflict_p = build_conflict_prompt(ctx, q, opts)
        clean_p    = build_clean_prompt(q, opts)
        support_p  = build_supportive_prompt(q, opts, cor_ans, ctx, con_ans)

        conf_pos, _ = get_pos_indices(model, conflict_p)
        cln_pos,  _ = get_pos_indices(model, clean_p)
        sup_pos,  _ = get_pos_indices(model, support_p)

        res_cf, _ = collect_residuals(model, conflict_p, layers, conf_pos)
        res_cl, _ = collect_residuals(model, clean_p,    layers, cln_pos)
        res_sp, _ = collect_residuals(model, support_p,  layers, sup_pos)

        for li, l in enumerate(layers):
            h_conflict[l][i] = res_cf[li]
            h_clean[l][i]    = res_cl[li]
            h_support[l][i]  = res_sp[li]

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i - start_i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_pt(CKPT_PATH, {
                "h_conflict": h_conflict,
                "h_clean":    h_clean,
                "h_support":  h_support,
                "n_done":     i + 1,
            })

    return h_conflict, h_clean, h_support


# ── 후보 A: SVD(conflict - clean) — Exp1 kc_dir 재사용 ───────────────────────
def extract_candidate_A(
    traj_stats: Dict,
    h_conflict: Dict[int, torch.Tensor],
    h_clean:    Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict:
    candidate_A = {}
    for l in layers:
        if "kc_dir" in traj_stats.get(l, {}):
            kc_dir = traj_stats[l]["kc_dir"]
            expl   = None
            s1     = None
        else:
            D = (h_conflict[l] - h_clean[l]).float()
            _, S, Vt = torch.linalg.svd(D, full_matrices=False)
            kc_dir = F.normalize(Vt[0], dim=0)
            s1     = S[0].item()
            expl   = (S[0]**2 / (S**2).sum()).item()

        candidate_A[l] = {
            "dir":              F.normalize(kc_dir.float(), dim=0),
            "method":           "svd_conflict_clean",
            "s1":               s1,
            "explained_ratio":  expl,
        }
    return candidate_A


# ── 후보 B: SVD(conflict - supportive) ───────────────────────────────────────
def extract_candidate_B(
    h_conflict:  Dict[int, torch.Tensor],
    h_support:   Dict[int, torch.Tensor],
    candidate_A: Dict,
    layers: List[int],
) -> Dict:
    candidate_B = {}
    for l in layers:
        D = (h_conflict[l] - h_support[l]).float()
        _, S, Vt = torch.linalg.svd(D, full_matrices=False)
        d = F.normalize(Vt[0], dim=0)

        candidate_B[l] = {
            "dir":             d,
            "method":          "svd_conflict_supportive",
            "s1":              S[0].item(),
            "explained_ratio": (S[0]**2 / (S**2).sum()).item(),
            "overlap_with_A":  float(torch.abs(
                torch.cosine_similarity(d, candidate_A[l]["dir"], dim=0)
            ).item()),
        }
        print(
            f"  [B] L{l:2d}: s1={S[0]:.1f}  explained={candidate_B[l]['explained_ratio']:.3f}"
            f"  overlap_A={candidate_B[l]['overlap_with_A']:.3f}",
            flush=True,
        )
    return candidate_B


# ── 후보 C: cPCA ──────────────────────────────────────────────────────────────
def extract_candidate_C(
    h_conflict:  Dict[int, torch.Tensor],
    h_support:   Dict[int, torch.Tensor],
    candidate_A: Dict,
    candidate_B: Dict,
    layers: List[int],
    device: str,
) -> Dict:
    candidate_C = {}
    for l in layers:
        X_conf = (h_conflict[l] - h_conflict[l].mean(0)).float().to(device)
        X_sup  = (h_support[l]  - h_support[l].mean(0)).float().to(device)

        C_conflict = (X_conf.T @ X_conf) / len(X_conf)
        C_support  = (X_sup.T  @ X_sup)  / len(X_sup)

        best_dir   = None
        best_eig   = -float("inf")
        best_alpha = None

        for alpha in ALPHA_VALUES:
            C_contrast  = C_conflict - alpha * C_support
            eigenvalues, eigenvectors = torch.linalg.eigh(C_contrast)
            top_dir = F.normalize(eigenvectors[:, -1].float().cpu(), dim=0)
            top_eig = eigenvalues[-1].item()
            if top_eig > best_eig:
                best_eig   = top_eig
                best_dir   = top_dir
                best_alpha = alpha

        del X_conf, X_sup, C_conflict, C_support
        torch.cuda.empty_cache() if device == "cuda" else None

        candidate_C[l] = {
            "dir":             best_dir,
            "method":          "cpca",
            "best_alpha":      best_alpha,
            "top_eigenvalue":  best_eig,
            "overlap_with_A":  float(torch.abs(
                torch.cosine_similarity(best_dir, candidate_A[l]["dir"], dim=0)
            ).item()),
            "overlap_with_B":  float(torch.abs(
                torch.cosine_similarity(best_dir, candidate_B[l]["dir"], dim=0)
            ).item()),
        }
        print(
            f"  [C] L{l:2d}: best_alpha={best_alpha}"
            f"  eig={best_eig:.2f}"
            f"  overlap_A={candidate_C[l]['overlap_with_A']:.3f}"
            f"  overlap_B={candidate_C[l]['overlap_with_B']:.3f}",
            flush=True,
        )
    return candidate_C


# ── 후보 D: Regression Probing (KC severity) ──────────────────────────────────
def extract_candidate_D(
    h_conflict:  Dict[int, torch.Tensor],
    y_kc:        torch.Tensor,
    candidate_A: Dict,
    candidate_B: Dict,
    candidate_C: Dict,
    layers: List[int],
) -> Dict:
    N = y_kc.shape[0]
    rng = torch.Generator()
    rng.manual_seed(SEED)
    perm      = torch.randperm(N, generator=rng)
    split     = int(N * 0.7)
    idx_train = perm[:split].tolist()
    idx_test  = perm[split:].tolist()

    candidate_D = {}
    for l in layers:
        X = h_conflict[l].float()
        k = min(PCA_COMPONENTS, len(idx_train) - 1)

        X_proj, components = torch_pca(X, k)     # (N, k), (k, d)

        X_train = X_proj[idx_train]
        X_test  = X_proj[idx_test]
        y_train = y_kc[idx_train].float()
        y_test  = y_kc[idx_test].float()

        w       = torch_ridge(X_train, y_train, RIDGE_ALPHA)   # (k,)
        y_pred  = X_test @ w
        test_r  = pearson_r(y_pred, y_test)

        # PCA space → d_model
        reg_dir = F.normalize(components.T @ w, dim=0)         # (d_model,)

        candidate_D[l] = {
            "dir":            reg_dir,
            "method":         "regression_kc_severity",
            "test_r":         test_r,
            "overlap_with_A": float(torch.abs(
                torch.cosine_similarity(reg_dir, candidate_A[l]["dir"], dim=0)
            ).item()),
            "overlap_with_B": float(torch.abs(
                torch.cosine_similarity(reg_dir, candidate_B[l]["dir"], dim=0)
            ).item()),
            "overlap_with_C": float(torch.abs(
                torch.cosine_similarity(reg_dir, candidate_C[l]["dir"], dim=0)
            ).item()),
        }
        print(
            f"  [D] L{l:2d}: test_r={test_r:.4f}"
            f"  overlap_A={candidate_D[l]['overlap_with_A']:.3f}"
            f"  overlap_B={candidate_D[l]['overlap_with_B']:.3f}",
            flush=True,
        )
    return candidate_D


# ── 수렴 분석 ─────────────────────────────────────────────────────────────────
def run_convergence(
    candidate_A: Dict,
    candidate_B: Dict,
    candidate_C: Dict,
    candidate_D: Dict,
    layers: List[int],
) -> Dict:
    convergence = {}
    for l in layers:
        dirs = {
            "A": candidate_A[l]["dir"],
            "B": candidate_B[l]["dir"],
            "C": candidate_C[l]["dir"],
            "D": candidate_D[l]["dir"],
        }
        pairs = {}
        names = list(dirs.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                pairs[f"{n1}_vs_{n2}"] = float(torch.abs(
                    torch.cosine_similarity(dirs[n1], dirs[n2], dim=0)
                ).item())
        pairs["mean_convergence"] = float(np.mean(list(pairs.values())))
        convergence[l] = pairs
    return convergence


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_figure1_candidate_strength(
    candidate_A: Dict,
    candidate_B: Dict,
    candidate_C: Dict,
    candidate_D: Dict,
    layers: List[int],
    out_dir: Path,
) -> None:
    """Figure 1: 후보별 방향 강도 by layer"""
    expl_A = [candidate_A[l].get("explained_ratio") or 0.0 for l in layers]
    expl_B = [candidate_B[l]["explained_ratio"] for l in layers]
    eig_C  = [candidate_C[l]["top_eigenvalue"]  for l in layers]
    r_D    = [candidate_D[l]["test_r"]           for l in layers]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(layers, expl_A, "b-o", markersize=5, label="A: SVD(conf−clean)")
    axes[0].plot(layers, expl_B, "r-o", markersize=5, label="B: SVD(conf−sup)")
    axes[0].axhline(0.15, color="gray", linestyle="--", linewidth=1, label="0.15")
    axes[0].set_ylabel("Explained Ratio (σ₁²/Σσ²)")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Candidates A & B: SVD Explained Ratio by Layer", fontsize=11)
    axes[0].grid(alpha=0.3)

    color_C = ["#2ecc71" if v > 0 else "#e74c3c" for v in eig_C]
    axes[1].bar(layers, eig_C, color=color_C, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Top Eigenvalue")
    axes[1].set_title("Candidate C: cPCA Top Eigenvalue (양수 = conflict 분산 우세)", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].plot(layers, r_D, "m-o", markersize=5, linewidth=2)
    axes[2].axhline(0.2, color="gray", linestyle="--", linewidth=1, label="r=0.2")
    axes[2].axhline(0,   color="black", linewidth=0.8)
    axes[2].set_ylabel("Pearson r (held-out)")
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Candidate D: Regression Test r (KC severity 예측력)", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].set_xticks(layers)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure1_candidate_strength.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure2_convergence(
    convergence: Dict, layers: List[int], out_dir: Path
) -> None:
    """Figure 2: 후보 간 수렴도 by layer"""
    pairs  = ["A_vs_B", "A_vs_C", "A_vs_D", "B_vs_C", "B_vs_D", "C_vs_D"]
    colors = {"A_vs_B": "#e74c3c", "A_vs_C": "#e67e22", "A_vs_D": "#f1c40f",
              "B_vs_C": "#2ecc71", "B_vs_D": "#3498db", "C_vs_D": "#9b59b6"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                              gridspec_kw={"height_ratios": [2, 1]})

    for pair in pairs:
        vals = [convergence[l].get(pair, np.nan) for l in layers]
        axes[0].plot(layers, vals, "o-", color=colors[pair], markersize=4,
                     linewidth=1.5, label=pair.replace("_vs_", " ↔ "))
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="0.5 threshold")
    axes[0].set_ylabel("|cos(dir_i, dir_j)|")
    axes[0].set_title("후보 간 Direction 수렴도", fontsize=11)
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    mean_conv = [convergence[l]["mean_convergence"] for l in layers]
    axes[1].plot(layers, mean_conv, "ko-", markersize=5, linewidth=2,
                 label="mean convergence")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Mean Convergence")
    axes[1].set_xlabel("Layer")
    axes[1].set_xticks(layers)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure2_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure3_diff_from_exp1(
    candidate_B: Dict,
    candidate_C: Dict,
    candidate_D: Dict,
    layers: List[int],
    out_dir: Path,
) -> None:
    """Figure 3: Exp1 kc_dir(=후보 A)와의 차이 by layer"""
    diff_B = [1.0 - candidate_B[l]["overlap_with_A"] for l in layers]
    diff_C = [1.0 - candidate_C[l]["overlap_with_A"] for l in layers]
    diff_D = [1.0 - candidate_D[l]["overlap_with_A"] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(layers, diff_B, "r-o", markersize=5, linewidth=2,
            label="B: SVD(conf−sup)")
    ax.plot(layers, diff_C, "g-o", markersize=5, linewidth=2,
            label="C: cPCA")
    ax.plot(layers, diff_D, "b-o", markersize=5, linewidth=2,
            label="D: Regression")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
               label="0.5 (≥50% 다른 방향)")
    ax.axhline(0.9, color="red",  linestyle=":",  linewidth=1,
               label="0.9 (≥90% 다른 방향, KC-specific 후보)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("1 − |cos(dir, A)|")
    ax.set_title("Exp1 kc_dir(후보 A)와의 방향 차이\n(높을수록 context direction과 다른 방향 → KC-specific 가능성)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure3_diff_from_exp1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n     = args.test if args.test else (args.n_cases if args.n_cases else N_CASES)
    cases = load_cases(n)
    N     = len(cases)

    # Exp1 trajectory stats (후보 A에서 kc_dir 재사용)
    traj_stats = load_trajectory_stats(EXP1_DIR)

    # kc_severity
    y_kc = torch.tensor(
        [cases[i]["kc_severity"] for i in range(N)], dtype=torch.float32
    )
    print(f"\n  y_kc: mean={y_kc.mean():.3f}  std={y_kc.std():.3f}", flush=True)

    # ── Step 1: Residual 수집 ─────────────────────────────────────────────────
    print("\n[exp1b] Step 1: Residual 수집 ...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)
    t0 = time.time()
    h_conflict, h_clean, h_support = load_or_compute_residuals(model, cases, LAYERS)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Step 2: 후보 A ────────────────────────────────────────────────────────
    print("\n[exp1b] Step 2: 후보 A (SVD conflict−clean, Exp1 재사용) ...", flush=True)
    candidate_A = extract_candidate_A(traj_stats, h_conflict, h_clean, LAYERS)

    # ── Step 3: 후보 B ────────────────────────────────────────────────────────
    print("\n[exp1b] Step 3: 후보 B (SVD conflict−supportive) ...", flush=True)
    candidate_B = extract_candidate_B(h_conflict, h_support, candidate_A, LAYERS)

    # ── Step 4: 후보 C (cPCA) — GPU 사용 ─────────────────────────────────────
    print("\n[exp1b] Step 4: 후보 C (cPCA) ...", flush=True)
    cpca_device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_C = extract_candidate_C(
        h_conflict, h_support, candidate_A, candidate_B, LAYERS, cpca_device
    )

    # ── Step 5: 후보 D ────────────────────────────────────────────────────────
    print("\n[exp1b] Step 5: 후보 D (Regression KC severity) ...", flush=True)
    candidate_D = extract_candidate_D(
        h_conflict, y_kc, candidate_A, candidate_B, candidate_C, LAYERS
    )

    # ── Step 6: 수렴 분석 ─────────────────────────────────────────────────────
    print("\n[exp1b] Step 6: 수렴 분석 ...", flush=True)
    convergence = run_convergence(
        candidate_A, candidate_B, candidate_C, candidate_D, LAYERS
    )

    # ── Step 7: 저장 ──────────────────────────────────────────────────────────
    print("\n[exp1b] Step 7: 저장 ...", flush=True)

    # 전체 저장
    save_pt(
        RESULTS_DIR / "all_candidates.pt",
        {
            "candidate_A": candidate_A,
            "candidate_B": candidate_B,
            "candidate_C": candidate_C,
            "candidate_D": candidate_D,
            "convergence": convergence,
        },
    )

    # Exp2 입력 형식 — unit norm 확인 후 저장
    candidates_for_exp2: Dict[int, List[Dict]] = {}
    for l in LAYERS:
        candidates_for_exp2[l] = [
            {"name": "A_svd_conf_clean",   "dir": candidate_A[l]["dir"]},
            {"name": "B_svd_conf_support", "dir": candidate_B[l]["dir"]},
            {"name": "C_cpca",             "dir": candidate_C[l]["dir"]},
            {"name": "D_regression",       "dir": candidate_D[l]["dir"]},
        ]

    for l in LAYERS:
        for cand in candidates_for_exp2[l]:
            norm = cand["dir"].norm().item()
            assert abs(norm - 1.0) < 1e-4, \
                f"L{l} {cand['name']} norm={norm:.6f} (not unit)"
    print("  [check] 모든 후보 unit norm 확인 완료", flush=True)

    save_pt(RESULTS_DIR / "candidates_for_exp2.pt", candidates_for_exp2)

    # 수치 요약 JSON
    def _s(v):
        return None if v is None else round(float(v), 6)

    summary = {
        str(l): {
            "A_explained_ratio":  _s(candidate_A[l].get("explained_ratio")),
            "B_explained_ratio":  _s(candidate_B[l]["explained_ratio"]),
            "B_overlap_with_A":   _s(candidate_B[l]["overlap_with_A"]),
            "C_best_alpha":       candidate_C[l]["best_alpha"],
            "C_top_eigenvalue":   _s(candidate_C[l]["top_eigenvalue"]),
            "C_overlap_with_A":   _s(candidate_C[l]["overlap_with_A"]),
            "C_overlap_with_B":   _s(candidate_C[l]["overlap_with_B"]),
            "D_test_r":           _s(candidate_D[l]["test_r"]),
            "D_overlap_with_A":   _s(candidate_D[l]["overlap_with_A"]),
            "D_overlap_with_B":   _s(candidate_D[l]["overlap_with_B"]),
            "D_overlap_with_C":   _s(candidate_D[l]["overlap_with_C"]),
            "mean_convergence":   _s(convergence[l]["mean_convergence"]),
            "A_vs_B":             _s(convergence[l]["A_vs_B"]),
            "A_vs_C":             _s(convergence[l]["A_vs_C"]),
            "A_vs_D":             _s(convergence[l]["A_vs_D"]),
            "B_vs_C":             _s(convergence[l]["B_vs_C"]),
            "B_vs_D":             _s(convergence[l]["B_vs_D"]),
            "C_vs_D":             _s(convergence[l]["C_vs_D"]),
        }
        for l in LAYERS
    }
    save_json(RESULTS_DIR / "summary.json", summary)

    # ── Step 8: 시각화 ────────────────────────────────────────────────────────
    print("\n[exp1b] Step 8: 시각화 ...", flush=True)
    plot_figure1_candidate_strength(
        candidate_A, candidate_B, candidate_C, candidate_D, LAYERS, RESULTS_DIR
    )
    plot_figure2_convergence(convergence, LAYERS, RESULTS_DIR)
    plot_figure3_diff_from_exp1(candidate_B, candidate_C, candidate_D, LAYERS, RESULTS_DIR)

    # ── Checkpoint 정리 ───────────────────────────────────────────────────────
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print(f"[exp1b] 체크포인트 삭제", flush=True)

    # ── 핵심 결과 출력 ────────────────────────────────────────────────────────
    print(f"\n[exp1b] 완료. 결과: {RESULTS_DIR}", flush=True)
    print("\n=== Exp1b 핵심 결과 ===")
    print(f"{'Layer':>5}  {'B_expl':>6}  {'B↔A':>5}  {'C_eig':>8}  {'C↔A':>5}  {'D_r':>6}  {'D↔A':>5}  {'mean_conv':>9}")
    for l in LAYERS:
        s = summary[str(l)]
        print(
            f"  L{l:2d}  "
            f"{s['B_explained_ratio'] or 0:6.3f}  "
            f"{s['B_overlap_with_A'] or 0:5.3f}  "
            f"{s['C_top_eigenvalue'] or 0:8.1f}  "
            f"{s['C_overlap_with_A'] or 0:5.3f}  "
            f"{s['D_test_r'] or 0:6.3f}  "
            f"{s['D_overlap_with_A'] or 0:5.3f}  "
            f"{s['mean_convergence'] or 0:9.4f}"
        )

    print("\n=== candidates_for_exp2.pt 저장 완료 ===")
    print(f"  레이어: {LAYERS}")
    print(f"  후보: A_svd_conf_clean, B_svd_conf_support, C_cpca, D_regression")


if __name__ == "__main__":
    main()

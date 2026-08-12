#!/usr/bin/env python3
"""
실험 1b: KC-specific Direction Extraction

Exp1 결과: kc_dir ≈ context_dir (overlap ≥ 0.88 전 레이어)
→ conflict - supportive 기반으로 KC-specific direction을 네 가지 방법으로 추출·수렴 분석.

방법:
  A: Mean Difference (conflict - supportive)
  B: SVD on (conflict - supportive)
  C: cPCA (Contrastive PCA)
  D: Regression Probing with KC severity (kc_severity, excess)

출력:
  phase2/results/exp1b_direction_extraction/
    method_A_mean_diff.pt / method_B_svd_conf_sup.pt
    method_C_cpca.pt      / method_D_regression.pt
    stability_stats.json  / type_alignment_stats.json
    convergence_stats.json / direction_to_use.pt
    exp1b_summary.json    / exp1b_report.md
    figure1~4.png

실행:
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1b_direction_extraction.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1b_direction_extraction.py --test 20
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp1b_direction_extraction.py --n_cases 1000
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from exp_utils import (
    LAYERS, N_CASES, SEED,
    load_model, load_cases,
    build_conflict_prompt, build_supportive_prompt,
    collect_residuals,
    get_pos_indices,
    get_letter_lp, LETTER_VARIANTS,
    load_trajectory_stats,
    save_pt, save_json,
)

RESULTS_DIR  = Path("/home/jaeik/xaimi/phase2/results/exp1b_direction_extraction")
EXP1_DIR     = Path("/home/jaeik/xaimi/phase2/results/exp1_trajectory")
CKPT_PATH    = RESULTS_DIR / "residuals_checkpoint.pt"
CHECKPOINT_EVERY = 50
ALPHA_VALUES = [0.5, 1.0, 2.0]
PCA_COMPONENTS  = 100   # regression probing 차원 축소
RIDGE_ALPHA     = 1.0


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실험 1b: KC-specific Direction Extraction")
    p.add_argument("--test",    type=int, default=None, help="N개만 처리 (테스트용)")
    p.add_argument("--n_cases", type=int, default=None, help="케이스 수 (기본: N_CASES=200)")
    return p.parse_args()


# ── PyTorch PCA + Ridge (sklearn 없이) ───────────────────────────────────────
def torch_pca(X: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    X: (N, d) float32
    Returns: X_proj (N, k), components (k, d), mean (d,)
    """
    mean = X.mean(0)
    X_c = X - mean
    _, _, Vt = torch.linalg.svd(X_c, full_matrices=False)
    components = Vt[:n_components]         # (k, d)
    X_proj = X_c @ components.T            # (N, k)
    return X_proj, components, mean


def torch_ridge(X_train: torch.Tensor, y_train: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
    """
    Ridge regression closed form: w = (X^T X + alpha I)^{-1} X^T y
    Returns: w (k,)
    """
    k = X_train.shape[1]
    A = X_train.T @ X_train + alpha * torch.eye(k, dtype=X_train.dtype)
    b = X_train.T @ y_train
    return torch.linalg.solve(A, b)


def pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = (a_c.norm() * b_c.norm()).item()
    if denom < 1e-8:
        return 0.0
    return float((a_c @ b_c).item() / denom)


# ── Step 1: Residual 수집 ─────────────────────────────────────────────────────
def collect_all_residuals(
    model,
    cases: List[Dict],
    layers: List[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
    """
    conflict, supportive 각각의 final token residual을 수집.
    supportive run에서 logit_diff(replace - correct)도 함께 계산.

    Returns:
        h_conflict: {layer: (N, d_model)}
        h_support:  {layer: (N, d_model)}
        y_sup_ld:   (N,) — supportive run logit diff (replace - correct)
    """
    N = len(cases)
    d_model = model.cfg.d_model

    if CKPT_PATH.exists():
        print(f"[exp1b] 체크포인트 로드: {CKPT_PATH}", flush=True)
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        h_conflict = ckpt["h_conflict"]
        h_support  = ckpt["h_support"]
        y_sup_ld   = ckpt["y_sup_ld"]
        start_i    = ckpt["n_done"]
        print(f"[exp1b] {start_i}개 완료, 이어서 진행", flush=True)
    else:
        h_conflict = {l: torch.zeros(N, d_model) for l in layers}
        h_support  = {l: torch.zeros(N, d_model) for l in layers}
        y_sup_ld   = torch.zeros(N)
        start_i    = 0

    t0 = time.time()
    for i in range(start_i, N):
        case    = cases[i]
        q       = case["question"]
        opts    = case["options"]
        ctx     = case["conflict_context"]
        cor     = case["correct_option"]
        rep     = case["replace_option"]
        cor_ans = case["correct_answer"]
        con_ans = case.get("conflict_answer")

        conflict_p = build_conflict_prompt(ctx, q, opts)
        support_p  = build_supportive_prompt(q, opts, cor_ans, ctx, con_ans)

        conf_pos, _ = get_pos_indices(model, conflict_p)
        sup_pos,  _ = get_pos_indices(model, support_p)

        # conflict residuals (N, L, d_model) 형식으로 collect_residuals 호출
        res_cf, _ = collect_residuals(model, conflict_p, layers, conf_pos)
        for li, l in enumerate(layers):
            h_conflict[l][i] = res_cf[li]

        # supportive residuals + logit diff
        res_sp, logits_sp = collect_residuals(model, support_p, layers, sup_pos)
        for li, l in enumerate(layers):
            h_support[l][i] = res_sp[li]

        lp_sp = torch.log_softmax(logits_sp.float(), dim=-1)
        y_sup_ld[i] = get_letter_lp(lp_sp, rep) - get_letter_lp(lp_sp, cor)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i - start_i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_pt(CKPT_PATH, {
                "h_conflict": h_conflict,
                "h_support":  h_support,
                "y_sup_ld":   y_sup_ld,
                "n_done":     i + 1,
            })

    return h_conflict, h_support, y_sup_ld


# ── Method A: Mean Difference ─────────────────────────────────────────────────
def run_method_A(
    h_conflict: Dict[int, torch.Tensor],
    h_support:  Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict:
    stats = {}
    for l in layers:
        diff = (h_conflict[l] - h_support[l]).float()   # (N, d_model)
        mean_dir = diff.mean(0)
        mean_dir_norm = F.normalize(mean_dir, dim=0)

        diff_norm = F.normalize(diff, dim=1)
        alignment = diff_norm @ mean_dir_norm            # (N,)

        stats[l] = {
            "mean_dir":               mean_dir_norm,
            "mean_dir_magnitude":     mean_dir.norm().item(),
            "mean_alignment":         alignment.mean().item(),
            "alignment_std":          alignment.std().item(),
            "positive_alignment_ratio": (alignment > 0).float().mean().item(),
        }
        print(
            f"  [A] L{l:2d}: magnitude={stats[l]['mean_dir_magnitude']:.2f}"
            f"  mean_align={stats[l]['mean_alignment']:.3f}"
            f"  pos_ratio={stats[l]['positive_alignment_ratio']:.3f}",
            flush=True,
        )
    return stats


# ── Method B: SVD on (conflict - supportive) ─────────────────────────────────
def run_method_B(
    h_conflict: Dict[int, torch.Tensor],
    h_support:  Dict[int, torch.Tensor],
    layers: List[int],
    kc_dir_exp1: Dict[int, torch.Tensor],
    mean_diff_stats: Dict,
) -> Dict:
    stats = {}
    prev_dir = None

    for l in layers:
        D_diff = (h_conflict[l] - h_support[l]).float()   # (N, d_model)
        _, S, Vt = torch.linalg.svd(D_diff, full_matrices=False)

        conf_sup_dir = F.normalize(Vt[0], dim=0)
        s2 = S ** 2
        explained_ratio = (s2[0] / s2.sum()).item()

        overlap_exp1 = float(torch.abs(
            torch.cosine_similarity(conf_sup_dir, kc_dir_exp1[l].float(), dim=0)
        ).item())
        overlap_A = float(torch.abs(
            torch.cosine_similarity(conf_sup_dir, mean_diff_stats[l]["mean_dir"], dim=0)
        ).item())

        continuity = None
        if prev_dir is not None:
            continuity = float(torch.abs(
                torch.cosine_similarity(conf_sup_dir, prev_dir, dim=0)
            ).item())

        # Cohen's d on this direction
        proj_conf = (h_conflict[l].float() @ conf_sup_dir).numpy()
        proj_sup  = (h_support[l].float()  @ conf_sup_dir).numpy()
        pooled_std = np.sqrt((proj_conf.std()**2 + proj_sup.std()**2) / 2) + 1e-8
        cohens_d  = float((proj_conf.mean() - proj_sup.mean()) / pooled_std)

        stats[l] = {
            "s1":                     S[0].item(),
            "top3_singular":          S[:3].tolist(),
            "explained_ratio":        explained_ratio,
            "overlap_with_exp1_dir":  overlap_exp1,
            "overlap_with_mean_diff": overlap_A,
            "direction_continuity":   continuity,
            "conf_sup_dir":           conf_sup_dir,
            "cohens_d":               cohens_d,
            "proj_conf_mean":         float(proj_conf.mean()),
            "proj_sup_mean":          float(proj_sup.mean()),
        }
        cont_str = f"{continuity:.3f}" if continuity is not None else "N/A"
        print(
            f"  [B] L{l:2d}: s1={S[0]:.1f}  explained={explained_ratio:.3f}"
            f"  overlap_exp1={overlap_exp1:.3f}  cohens_d={cohens_d:.3f}"
            f"  cont={cont_str}",
            flush=True,
        )
        prev_dir = conf_sup_dir

    return stats


# ── Method C: cPCA ────────────────────────────────────────────────────────────
def run_method_C(
    h_conflict: Dict[int, torch.Tensor],
    h_support:  Dict[int, torch.Tensor],
    layers: List[int],
    mean_diff_stats: Dict,
    svd_stats: Dict,
    device: str,
) -> Dict:
    stats = {}

    for l in layers:
        X_conf = h_conflict[l].float()
        X_sup  = h_support[l].float()

        X_conf = X_conf - X_conf.mean(0)
        X_sup  = X_sup  - X_sup.mean(0)

        # covariance matrices on device
        Xc = X_conf.to(device)
        Xs = X_sup.to(device)
        C_conflict = (Xc.T @ Xc) / len(Xc)
        C_support  = (Xs.T @ Xs) / len(Xs)

        stats[l] = {}

        for alpha in ALPHA_VALUES:
            C_contrast = C_conflict - alpha * C_support
            eigenvalues, eigenvectors = torch.linalg.eigh(C_contrast)
            # eigh: 오름차순 → 마지막 = 최대 고유값
            cpca_dir = F.normalize(eigenvectors[:, -1].float().cpu(), dim=0)
            cpca_eig = eigenvalues[-1].item()

            overlap_A = float(torch.abs(torch.cosine_similarity(
                cpca_dir, mean_diff_stats[l]["mean_dir"], dim=0)).item())
            overlap_B = float(torch.abs(torch.cosine_similarity(
                cpca_dir, svd_stats[l]["conf_sup_dir"], dim=0)).item())

            stats[l][f"alpha_{alpha}"] = {
                "cpca_dir":       cpca_dir,
                "top_eigenvalue": cpca_eig,
                "overlap_with_A": overlap_A,
                "overlap_with_B": overlap_B,
            }

        # best alpha: overlap_A + overlap_B 최대
        best_alpha = max(
            ALPHA_VALUES,
            key=lambda a: (
                stats[l][f"alpha_{a}"]["overlap_with_A"] +
                stats[l][f"alpha_{a}"]["overlap_with_B"]
            ),
        )
        stats[l]["best_alpha"] = best_alpha
        stats[l]["best_dir"]   = stats[l][f"alpha_{best_alpha}"]["cpca_dir"]

        best = stats[l][f"alpha_{best_alpha}"]
        print(
            f"  [C] L{l:2d}: best_alpha={best_alpha}"
            f"  eig={best['top_eigenvalue']:.2f}"
            f"  overlap_A={best['overlap_with_A']:.3f}"
            f"  overlap_B={best['overlap_with_B']:.3f}",
            flush=True,
        )

        del Xc, Xs, C_conflict, C_support
        if device == "cuda":
            torch.cuda.empty_cache()

    return stats


# ── Method D: Regression Probing ─────────────────────────────────────────────
def run_method_D(
    h_conflict: Dict[int, torch.Tensor],
    layers: List[int],
    y_kc_severity: torch.Tensor,
    y_excess: torch.Tensor,
    mean_diff_stats: Dict,
    svd_stats: Dict,
    cpca_stats: Dict,
) -> Dict:
    N = y_kc_severity.shape[0]
    rng = torch.Generator()
    rng.manual_seed(SEED)
    perm = torch.randperm(N, generator=rng)
    split = int(N * 0.7)
    idx_train = perm[:split].tolist()
    idx_test  = perm[split:].tolist()

    stats = {}

    for l in layers:
        X = h_conflict[l].float()   # (N, d_model)

        # PCA 차원 축소
        k = min(PCA_COMPONENTS, len(idx_train) - 1)
        X_proj, components, X_mean = torch_pca(X, k)   # (N, k), (k, d_model)

        X_train = X_proj[idx_train]   # (n_train, k)
        X_test  = X_proj[idx_test]    # (n_test,  k)

        stats[l] = {}

        for target_name, y in [("kc_severity", y_kc_severity), ("excess", y_excess)]:
            y_train = y[idx_train].float()
            y_test  = y[idx_test].float()

            w = torch_ridge(X_train, y_train, RIDGE_ALPHA)   # (k,)

            y_pred = X_test @ w
            test_r = pearson_r(y_pred, y_test)

            # PCA space → d_model: reg_dir = components.T @ w
            reg_dir = F.normalize(components.T @ w, dim=0)   # (d_model,)

            overlap_A = float(torch.abs(torch.cosine_similarity(
                reg_dir, mean_diff_stats[l]["mean_dir"], dim=0)).item())
            overlap_B = float(torch.abs(torch.cosine_similarity(
                reg_dir, svd_stats[l]["conf_sup_dir"], dim=0)).item())
            overlap_C = float(torch.abs(torch.cosine_similarity(
                reg_dir, cpca_stats[l]["best_dir"], dim=0)).item())

            stats[l][target_name] = {
                "test_r":         test_r,
                "reg_dir":        reg_dir,
                "overlap_with_A": overlap_A,
                "overlap_with_B": overlap_B,
                "overlap_with_C": overlap_C,
            }

        print(
            f"  [D] L{l:2d}: r_kc={stats[l]['kc_severity']['test_r']:.3f}"
            f"  r_excess={stats[l]['excess']['test_r']:.3f}",
            flush=True,
        )

    return stats


# ── Stability Diagnosis ───────────────────────────────────────────────────────
def run_stability(
    h_conflict: Dict[int, torch.Tensor],
    h_support:  Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict:
    stats = {}
    for l in layers:
        D_diff = (h_conflict[l] - h_support[l]).float()
        D_norm = F.normalize(D_diff, dim=1)

        N = len(D_diff)
        cos_mat = D_norm @ D_norm.T
        mask = ~torch.eye(N, dtype=torch.bool)
        mean_pairwise = float(cos_mat[mask].mean().item())

        _, S, _ = torch.linalg.svd(D_diff, full_matrices=False)
        s2 = S ** 2
        s2_sum = s2.sum()
        top1_energy   = float((s2[0] / s2_sum).item())
        top5_energy   = float((s2[:5].sum() / s2_sum).item())
        p = s2 / (s2_sum + 1e-10)
        effective_rank = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()).item())

        stats[l] = {
            "mean_pairwise_cos": mean_pairwise,
            "top1_energy":       top1_energy,
            "top5_energy":       top5_energy,
            "effective_rank":    effective_rank,
        }
    return stats


# ── Type Alignment ────────────────────────────────────────────────────────────
def run_type_alignment(
    h_conflict: Dict[int, torch.Tensor],
    h_support:  Dict[int, torch.Tensor],
    layers: List[int],
    cases: List[Dict],
) -> Dict:
    ctypes = ["misinformation", "temporal", "semantic"]
    type_alignment = {}

    for l in layers:
        type_dirs: Dict[str, torch.Tensor] = {}
        for ctype in ctypes:
            idx = [i for i, c in enumerate(cases) if c["conflict_type"] == ctype]
            if len(idx) < 10:
                continue
            D = (h_conflict[l][idx] - h_support[l][idx]).float()
            d = F.normalize(D.mean(0), dim=0)
            type_dirs[ctype] = d

        type_alignment[l] = {}
        types = list(type_dirs.keys())
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                key = f"{types[i]}_vs_{types[j]}"
                type_alignment[l][key] = float(torch.abs(
                    torch.cosine_similarity(type_dirs[types[i]], type_dirs[types[j]], dim=0)
                ).item())

    return type_alignment


# ── Convergence Analysis ──────────────────────────────────────────────────────
def run_convergence(
    layers: List[int],
    mean_diff_stats: Dict,
    svd_stats: Dict,
    cpca_stats: Dict,
    reg_stats: Dict,
) -> Dict:
    convergence = {}
    for l in layers:
        dirs = {
            "A_mean_diff":   mean_diff_stats[l]["mean_dir"],
            "B_svd":         svd_stats[l]["conf_sup_dir"],
            "C_cpca":        cpca_stats[l]["best_dir"],
            "D_reg_kc":      reg_stats[l]["kc_severity"]["reg_dir"],
            "D_reg_excess":  reg_stats[l]["excess"]["reg_dir"],
        }
        convergence[l] = {}
        names = list(dirs.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                convergence[l][f"{n1}_vs_{n2}"] = float(torch.abs(
                    torch.cosine_similarity(dirs[n1], dirs[n2], dim=0)
                ).item())
        pair_vals = [v for k, v in convergence[l].items()]
        convergence[l]["mean_convergence"] = float(np.mean(pair_vals))
    return convergence


# ── Conclusion Determination ──────────────────────────────────────────────────
def determine_conclusion(
    svd_stats: Dict,
    stability_stats: Dict,
    reg_stats: Dict,
    convergence: Dict,
    layers: List[int],
) -> Tuple[str, Optional[List[int]], str]:

    case1_layers = [
        l for l in layers
        if (svd_stats[l]["explained_ratio"] > 0.15
            and svd_stats[l]["overlap_with_exp1_dir"] < 0.85
            and stability_stats[l]["top1_energy"] > 0.15
            and convergence[l]["mean_convergence"] > 0.5)
    ]
    case2_layers = [
        l for l in layers
        if reg_stats[l]["excess"]["test_r"] > 0.2
    ]

    if len(case1_layers) >= 3:
        conclusion = "case1_structured_direction"
        best_layers = sorted(
            case1_layers,
            key=lambda l: svd_stats[l]["explained_ratio"],
            reverse=True,
        )[:3]
        best_dir_key = "B_svd"
    elif len(case2_layers) >= 3:
        conclusion = "case2_weak_direction"
        best_layers = sorted(
            case2_layers,
            key=lambda l: reg_stats[l]["excess"]["test_r"],
            reverse=True,
        )[:3]
        best_dir_key = "D_reg_excess"
    else:
        conclusion = "case3_magnitude_excess"
        best_layers = None
        best_dir_key = "A_mean_diff"

    return conclusion, best_layers, best_dir_key


# ── direction_to_use.pt 생성 ──────────────────────────────────────────────────
def build_direction_to_use(
    conclusion: str,
    best_layers: Optional[List[int]],
    best_dir_key: str,
    layers: List[int],
    mean_diff_stats: Dict,
    svd_stats: Dict,
    cpca_stats: Dict,
    reg_stats: Dict,
    h_support: Dict[int, torch.Tensor],
) -> Dict:
    dir_map = {
        "A_mean_diff":   lambda l: mean_diff_stats[l]["mean_dir"],
        "B_svd":         lambda l: svd_stats[l]["conf_sup_dir"],
        "C_cpca":        lambda l: cpca_stats[l]["best_dir"],
        "D_reg_kc":      lambda l: reg_stats[l]["kc_severity"]["reg_dir"],
        "D_reg_excess":  lambda l: reg_stats[l]["excess"]["reg_dir"],
    }
    get_dir = dir_map[best_dir_key]

    directions = {}
    for l in layers:
        patch_dir = get_dir(l)
        if conclusion in ("case1_structured_direction", "case2_weak_direction"):
            patch_target = 0.0
        else:
            # Case 3: supportive 평균 projection 수준으로 낮추기
            sup_projs = (h_support[l].float() @ patch_dir).numpy()
            patch_target = float(sup_projs.mean())

        directions[l] = {
            "patch_dir":         patch_dir,
            "patch_target_proj": patch_target,
        }

    return {
        "conclusion":   conclusion,
        "best_dir_key": best_dir_key,
        "best_layers":  best_layers,
        "directions":   directions,
        "patching_formula": {
            "case1": "h_patched = h_conf - (proj_conf - 0) * patch_dir",
            "case2": "h_patched = h_conf - (proj_conf - 0) * patch_dir",
            "case3": "h_patched = h_conf - (proj_conf - proj_sup_mean) * patch_dir",
        },
    }


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_figure1_convergence(convergence: Dict, layers: List[int], out_dir: Path) -> None:
    """Figure 1: 방법 간 direction 수렴 heatmap (layer × pair)"""
    pairs = [
        "A_mean_diff_vs_B_svd", "A_mean_diff_vs_C_cpca",
        "A_mean_diff_vs_D_reg_kc", "A_mean_diff_vs_D_reg_excess",
        "B_svd_vs_C_cpca", "B_svd_vs_D_reg_kc", "B_svd_vs_D_reg_excess",
        "C_cpca_vs_D_reg_kc", "C_cpca_vs_D_reg_excess",
        "D_reg_kc_vs_D_reg_excess",
    ]
    labels = [p.replace("_vs_", " ↔ ").replace("_", " ") for p in pairs]
    mat = np.array([[convergence[l].get(p, np.nan) for l in layers] for p in pairs])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [3, 1]})

    im = axes[0].imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(layers)))
    axes[0].set_xticklabels(layers, fontsize=9)
    axes[0].set_yticks(range(len(pairs)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title("방법 간 Direction 수렴 행렬\n(녹색=높음=같은 방향을 가리킴)", fontsize=11)
    plt.colorbar(im, ax=axes[0], label="|cos(dir_i, dir_j)|", fraction=0.03)

    mean_conv = [convergence[l]["mean_convergence"] for l in layers]
    axes[1].plot(layers, mean_conv, "ko-", markersize=5, linewidth=2)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 threshold")
    axes[1].set_ylabel("Mean\nConvergence")
    axes[1].set_xlabel("Layer")
    axes[1].set_xticks(layers)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure1_convergence_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure2_kc_strength(
    svd_stats: Dict,
    reg_stats: Dict,
    layers: List[int],
    kc_str_exp1: List[float],
    out_dir: Path,
) -> None:
    """Figure 2: KC-specific strength 지표 (layer별)"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # subplot 1: SVD s1 비교
    s1_exp1b = [svd_stats[l]["s1"] for l in layers]
    axes[0].plot(layers, kc_str_exp1, "b-o", markersize=5, label="Exp1 σ₁ (conf−clean)")
    axes[0].plot(layers, s1_exp1b,   "r-o", markersize=5, label="Exp1b σ₁ (conf−sup)")
    axes[0].set_ylabel("1st Singular Value")
    axes[0].legend(fontsize=9)
    axes[0].set_title("SVD σ₁: Exp1 (conf−clean) vs Exp1b (conf−sup)", fontsize=11)
    axes[0].grid(alpha=0.3)

    # subplot 2: explained_ratio vs overlap_with_exp1
    explained  = [svd_stats[l]["explained_ratio"] for l in layers]
    overlap_e1 = [svd_stats[l]["overlap_with_exp1_dir"] for l in layers]
    cohens_d   = [svd_stats[l]["cohens_d"] for l in layers]

    ax2 = axes[1].twinx()
    axes[1].bar(layers, explained, color="#e74c3c", alpha=0.6, label="explained_ratio", width=0.5)
    axes[1].axhline(0.15, color="#e74c3c", linestyle="--", linewidth=1)
    ax2.plot(layers, overlap_e1, "g--o", markersize=5, label="|cos(B_dir, exp1_dir)|")
    ax2.plot(layers, cohens_d,   "m:s",  markersize=5, label="Cohen's d")
    ax2.axhline(0.85, color="gray", linestyle=":", linewidth=1)
    axes[1].set_ylabel("explained_ratio", color="#e74c3c")
    ax2.set_ylabel("|cos| / Cohen's d", color="g")
    ax2.set_ylim(0, 1)
    lines1, lab1 = axes[1].get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    axes[1].set_title("Method B: Explained Ratio, Overlap with Exp1 Dir, Cohen's d", fontsize=11)
    axes[1].grid(alpha=0.3)

    # subplot 3: regression test_r
    r_kc = [reg_stats[l]["kc_severity"]["test_r"] for l in layers]
    r_ex = [reg_stats[l]["excess"]["test_r"] for l in layers]
    axes[2].plot(layers, r_kc, "b-o", markersize=5, label="r (kc_severity)")
    axes[2].plot(layers, r_ex, "r-o", markersize=5, label="r (excess)")
    axes[2].axhline(0.2, color="gray", linestyle="--", linewidth=1, label="0.2 threshold")
    axes[2].axhline(0,   color="black", linewidth=0.8)
    axes[2].set_ylabel("Pearson r (held-out)")
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Method D: Regression Test Pearson r", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].set_xticks(layers)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure2_kc_specific_strength.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure3_stability(stability_stats: Dict, layers: List[int], out_dir: Path) -> None:
    """Figure 3: Stability by layer"""
    top1  = [stability_stats[l]["top1_energy"]   for l in layers]
    top5  = [stability_stats[l]["top5_energy"]   for l in layers]
    eff_r = [stability_stats[l]["effective_rank"] for l in layers]
    mpc   = [stability_stats[l]["mean_pairwise_cos"] for l in layers]

    x  = np.array(layers)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(x - 0.25, top1, width=0.45, color="#3498db", alpha=0.8, label="top1_energy")
    axes[0].bar(x + 0.25, top5, width=0.45, color="#85c1e9", alpha=0.8, label="top5_energy")
    axes[0].axhline(0.15, color="red", linestyle="--", linewidth=1, label="0.15 threshold")
    axes[0].set_ylabel("Energy Fraction")
    axes[0].legend(fontsize=9)
    axes[0].set_title("SVD Energy Concentration — (conflict−supportive)", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(layers, eff_r, "m-o", markersize=5, linewidth=2)
    axes[1].set_ylabel("Effective Rank")
    axes[1].set_title("Effective Rank (낮을수록 단일 방향에 집중)", fontsize=11)
    axes[1].grid(alpha=0.3)

    axes[2].plot(layers, mpc, "g-o", markersize=5, linewidth=2)
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_ylabel("Mean Pairwise cos")
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Mean Pairwise Cosine of Case-wise Diffs\n(양수=케이스들이 같은 방향, ≈0=랜덤)", fontsize=11)
    axes[2].set_xticks(layers)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure3_stability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure4_type_alignment(
    type_alignment: Dict, layers: List[int], out_dir: Path
) -> None:
    """Figure 4: Conflict type별 KC direction 정렬도"""
    pairs = [
        "misinformation_vs_temporal",
        "misinformation_vs_semantic",
        "temporal_vs_semantic",
    ]
    colors = {"misinformation_vs_temporal": "#e74c3c",
              "misinformation_vs_semantic":  "#3498db",
              "temporal_vs_semantic":         "#2ecc71"}

    fig, ax = plt.subplots(figsize=(12, 4))
    for pair in pairs:
        vals = [type_alignment[l].get(pair, np.nan) for l in layers]
        ax.plot(layers, vals, "o-", color=colors[pair], markersize=5,
                linewidth=2, label=pair.replace("_vs_", " vs "))

    ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, label="0.8 (high)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|cos(type_dir_i, type_dir_j)|")
    ax.set_title("Conflict Type별 KC Direction 정렬도\n(높을수록 type 무관하게 단일 방향)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "figure4_type_alignment.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


# ── Report MD 생성 ───────────────────────────────────────────────────────────
def generate_report(
    conclusion: str,
    best_layers: Optional[List[int]],
    best_dir_key: str,
    svd_stats: Dict,
    reg_stats: Dict,
    convergence: Dict,
    layers: List[int],
) -> str:
    lines = [
        "# Exp1b: KC-specific Direction Extraction — 결과 보고서",
        "",
        f"## 결론: `{conclusion}`",
        "",
    ]

    desc = {
        "case1_structured_direction": (
            "**KC-specific direction이 구조적으로 존재한다.**\n\n"
            f"- Exp2/3에서 Method B (SVD conf−sup) direction 사용\n"
            f"- best_layers: {best_layers}\n"
            "- `h_patched = h_conf − (proj_conf − 0) × patch_dir`"
        ),
        "case2_weak_direction": (
            "**KC severity와 상관된 방향이 있지만 약하다.**\n\n"
            f"- Exp2/3에서 Method D (regression excess) direction 사용\n"
            f"- best_layers: {best_layers}\n"
            "- `h_patched = h_conf − (proj_conf − 0) × patch_dir`"
        ),
        "case3_magnitude_excess": (
            "**독립적인 KC-specific direction을 찾지 못했다.**\n\n"
            "→ KC는 context-following circuit의 **excess magnitude**로 나타남\n"
            "→ Exp2/3: conflict → supportive level patching\n"
            "- `h_patched = h_conf − (proj_conf − proj_sup_mean) × patch_dir`\n"
            "- Contribution: 'KC selective intervention이 어려운 이유' 설명"
        ),
    }
    lines.append(desc.get(conclusion, ""))
    lines += [
        "",
        "## 레이어별 핵심 수치",
        "",
        "| Layer | explained_ratio | overlap_exp1 | cohens_d | r_excess | mean_convergence |",
        "|-------|----------------|-------------|----------|----------|-----------------|",
    ]
    for l in layers:
        er  = svd_stats[l]["explained_ratio"]
        oe  = svd_stats[l]["overlap_with_exp1_dir"]
        cd  = svd_stats[l]["cohens_d"]
        rr  = reg_stats[l]["excess"]["test_r"]
        mc  = convergence[l]["mean_convergence"]
        lines.append(f"| {l} | {er:.4f} | {oe:.4f} | {cd:.3f} | {rr:.3f} | {mc:.4f} |")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n = args.test if args.test else (args.n_cases if args.n_cases else N_CASES)
    cases = load_cases(n)
    N = len(cases)

    # Exp1 trajectory stats 로드
    traj_stats   = load_trajectory_stats(EXP1_DIR)
    kc_dir_exp1  = {l: traj_stats[l]["kc_dir"] for l in LAYERS}
    kc_str_exp1  = [traj_stats[l]["kc_strength"] for l in LAYERS]

    y_kc_severity = torch.tensor(
        [cases[i]["kc_severity"] for i in range(N)], dtype=torch.float32
    )

    # ── Step 1: Residual 수집 ─────────────────────────────────────────────────
    print("\n[exp1b] Step 1: Residual 수집 ...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)
    t0 = time.time()
    h_conflict, h_support, y_sup_ld = collect_all_residuals(model, cases, LAYERS)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    y_excess = y_kc_severity - y_sup_ld

    del model
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else None

    print(f"\n  y_kc_severity: mean={y_kc_severity.mean():.3f}  std={y_kc_severity.std():.3f}")
    print(f"  y_sup_ld:      mean={y_sup_ld.mean():.3f}       std={y_sup_ld.std():.3f}")
    print(f"  y_excess:      mean={y_excess.mean():.3f}       std={y_excess.std():.3f}")

    # ── Step 2: Method A ──────────────────────────────────────────────────────
    print("\n[exp1b] Step 2: Method A (Mean Difference) ...", flush=True)
    mean_diff_stats = run_method_A(h_conflict, h_support, LAYERS)
    save_pt(RESULTS_DIR / "method_A_mean_diff.pt", mean_diff_stats)

    # ── Step 3: Method B ──────────────────────────────────────────────────────
    print("\n[exp1b] Step 3: Method B (SVD conf−sup) ...", flush=True)
    svd_stats = run_method_B(h_conflict, h_support, LAYERS, kc_dir_exp1, mean_diff_stats)
    save_pt(RESULTS_DIR / "method_B_svd_conf_sup.pt", svd_stats)

    # ── Step 4: Method C (cPCA) — GPU 사용 ───────────────────────────────────
    print("\n[exp1b] Step 4: Method C (cPCA) ...", flush=True)
    cpca_device = "cuda" if torch.cuda.is_available() else "cpu"
    cpca_stats  = run_method_C(h_conflict, h_support, LAYERS,
                               mean_diff_stats, svd_stats, cpca_device)
    save_pt(RESULTS_DIR / "method_C_cpca.pt", cpca_stats)

    # ── Step 5: Method D ──────────────────────────────────────────────────────
    print("\n[exp1b] Step 5: Method D (Regression Probing) ...", flush=True)
    reg_stats = run_method_D(h_conflict, LAYERS, y_kc_severity, y_excess,
                              mean_diff_stats, svd_stats, cpca_stats)
    save_pt(RESULTS_DIR / "method_D_regression.pt", reg_stats)

    # ── Step 6: Stability ─────────────────────────────────────────────────────
    print("\n[exp1b] Step 6: Stability diagnosis ...", flush=True)
    stability_stats = run_stability(h_conflict, h_support, LAYERS)
    save_json(RESULTS_DIR / "stability_stats.json",
              {str(l): v for l, v in stability_stats.items()})

    # ── Step 7: Type Alignment ────────────────────────────────────────────────
    print("\n[exp1b] Step 7: Type alignment ...", flush=True)
    type_alignment = run_type_alignment(h_conflict, h_support, LAYERS, cases)
    save_json(RESULTS_DIR / "type_alignment_stats.json",
              {str(l): v for l, v in type_alignment.items()})

    # ── Step 8: Convergence ───────────────────────────────────────────────────
    print("\n[exp1b] Step 8: Convergence analysis ...", flush=True)
    convergence = run_convergence(LAYERS, mean_diff_stats, svd_stats, cpca_stats, reg_stats)
    save_json(RESULTS_DIR / "convergence_stats.json",
              {str(l): v for l, v in convergence.items()})

    # ── Step 9: Conclusion ────────────────────────────────────────────────────
    print("\n[exp1b] Step 9: Conclusion determination ...", flush=True)
    conclusion, best_layers, best_dir_key = determine_conclusion(
        svd_stats, stability_stats, reg_stats, convergence, LAYERS
    )
    print(f"  결론:        {conclusion}", flush=True)
    print(f"  best_dir:    {best_dir_key}", flush=True)
    print(f"  best_layers: {best_layers}", flush=True)

    # ── Step 10: direction_to_use.pt ─────────────────────────────────────────
    direction_to_use = build_direction_to_use(
        conclusion, best_layers, best_dir_key, LAYERS,
        mean_diff_stats, svd_stats, cpca_stats, reg_stats, h_support,
    )
    save_pt(RESULTS_DIR / "direction_to_use.pt", direction_to_use)

    # ── Step 11: 시각화 ───────────────────────────────────────────────────────
    print("\n[exp1b] Step 10: 시각화 ...", flush=True)
    plot_figure1_convergence(convergence, LAYERS, RESULTS_DIR)
    plot_figure2_kc_strength(svd_stats, reg_stats, LAYERS, kc_str_exp1, RESULTS_DIR)
    plot_figure3_stability(stability_stats, LAYERS, RESULTS_DIR)
    plot_figure4_type_alignment(type_alignment, LAYERS, RESULTS_DIR)

    # ── Step 12: Summary JSON ─────────────────────────────────────────────────
    def _scalar(v):
        if isinstance(v, (int, float, bool)):
            return v
        if isinstance(v, list):
            return v
        return None

    summary = {
        "conclusion":    conclusion,
        "best_dir_key":  best_dir_key,
        "best_layers":   best_layers,
        "n_cases":       N,
        "layers":        LAYERS,
        "method_A": {
            str(l): {k: _scalar(v) for k, v in s.items() if _scalar(v) is not None}
            for l, s in mean_diff_stats.items()
        },
        "method_B": {
            str(l): {k: _scalar(v) for k, v in s.items() if _scalar(v) is not None}
            for l, s in svd_stats.items()
        },
        "method_C": {
            str(l): {
                "best_alpha": s["best_alpha"],
                **{
                    f"alpha_{a}": {
                        k: _scalar(v)
                        for k, v in s[f"alpha_{a}"].items()
                        if _scalar(v) is not None
                    }
                    for a in ALPHA_VALUES
                },
            }
            for l, s in cpca_stats.items()
        },
        "method_D": {
            str(l): {
                tgt: {k: _scalar(v) for k, v in s[tgt].items() if _scalar(v) is not None}
                for tgt in ["kc_severity", "excess"]
            }
            for l, s in reg_stats.items()
        },
        "convergence": {str(l): v for l, v in convergence.items()},
        "stability":   {str(l): v for l, v in stability_stats.items()},
    }
    save_json(RESULTS_DIR / "exp1b_summary.json", summary)

    # ── Step 13: Report MD ───────────────────────────────────────────────────
    report = generate_report(
        conclusion, best_layers, best_dir_key,
        svd_stats, reg_stats, convergence, LAYERS,
    )
    (RESULTS_DIR / "exp1b_report.md").write_text(report, encoding="utf-8")
    print(f"[save] {RESULTS_DIR / 'exp1b_report.md'}", flush=True)

    # ── Checkpoint 정리 ───────────────────────────────────────────────────────
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print(f"[exp1b] 체크포인트 삭제", flush=True)

    # ── 핵심 결과 출력 ────────────────────────────────────────────────────────
    print(f"\n[exp1b] 완료. 결과: {RESULTS_DIR}", flush=True)
    print("\n=== 실험 1b 핵심 결과 ===")
    print(f"결론:         {conclusion}")
    print(f"Exp2/3 direction: {best_dir_key}")
    print(f"우선 patch layer: {best_layers}")
    print(f"\n{'Layer':>5}  {'explained':>9}  {'ovlp_e1':>7}  {'cohen_d':>7}  {'r_excess':>8}  {'mean_conv':>9}")
    for l in LAYERS:
        print(
            f"  L{l:2d}  "
            f"{svd_stats[l]['explained_ratio']:9.4f}  "
            f"{svd_stats[l]['overlap_with_exp1_dir']:7.4f}  "
            f"{svd_stats[l]['cohens_d']:7.3f}  "
            f"{reg_stats[l]['excess']['test_r']:8.3f}  "
            f"{convergence[l]['mean_convergence']:9.4f}"
        )


if __name__ == "__main__":
    main()

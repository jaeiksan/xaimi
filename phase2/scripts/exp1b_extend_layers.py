#!/usr/bin/env python3
"""
exp1b 방향 벡터를 L20~L22로 확장한다.

기존 method_A/B/C .pt 파일에 L20, L21, L22 레이어를 추가로 추출해 append한다.
Method A: Mean Difference (conflict - supportive)
Method B: SVD on (conflict - supportive)
Method C: cPCA (best alpha 선택)

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u \
    phase2/scripts/exp1b_extend_layers.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from exp_utils import (
    build_conflict_prompt,
    build_supportive_prompt,
    collect_residuals,
    get_pos_indices,
    load_cases,
    load_model,
)

EXP1B_DIR   = ROOT / "phase2/results/exp1b_direction_extraction"
NEW_LAYERS  = [20, 21, 22]
ALPHA_VALUES = [0.5, 1.0, 2.0]
CHECKPOINT_EVERY = 100
DEVICE = "cuda"


# ── Residual 수집 ─────────────────────────────────────────────────────────────

def collect_residuals_for_layers(
    model,
    cases: List[Dict],
    layers: List[int],
) -> tuple:
    N = len(cases)
    d_model = model.cfg.d_model
    h_conflict = {l: torch.zeros(N, d_model) for l in layers}
    h_support  = {l: torch.zeros(N, d_model) for l in layers}
    t0 = time.time()

    for i, case in enumerate(cases):
        q       = case["question"]
        opts    = case["options"]
        ctx     = case["conflict_context"]
        cor_ans = case["correct_answer"]
        con_ans = case.get("conflict_answer")

        conflict_p = build_conflict_prompt(ctx, q, opts)
        support_p  = build_supportive_prompt(q, opts, cor_ans, ctx, con_ans)

        conf_pos, _ = get_pos_indices(model, conflict_p)
        sup_pos,  _ = get_pos_indices(model, support_p)

        res_cf, _ = collect_residuals(model, conflict_p, layers, conf_pos)
        for li, l in enumerate(layers):
            h_conflict[l][i] = res_cf[li].cpu()

        res_sp, _ = collect_residuals(model, support_p, layers, sup_pos)
        for li, l in enumerate(layers):
            h_support[l][i] = res_sp[li].cpu()

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [{i+1}/{N}] elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"[residuals] done  elapsed={time.time()-t0:.0f}s")
    return h_conflict, h_support


# ── Method A: Mean Difference ─────────────────────────────────────────────────

def run_method_A(h_conflict, h_support, layers):
    stats = {}
    for l in layers:
        diff     = (h_conflict[l] - h_support[l]).float()
        mean_dir = F.normalize(diff.mean(0), dim=0)
        stats[l] = {"mean_dir": mean_dir}
        print(f"  [A] L{l}")
    return stats


# ── Method B: SVD ─────────────────────────────────────────────────────────────

def run_method_B(h_conflict, h_support, layers):
    stats = {}
    for l in layers:
        D_diff = (h_conflict[l] - h_support[l]).float()
        _, S, Vt = torch.linalg.svd(D_diff, full_matrices=False)
        conf_sup_dir = F.normalize(Vt[0], dim=0)

        s2 = S ** 2
        explained = (s2[0] / s2.sum()).item()

        proj_conf = (h_conflict[l].float() @ conf_sup_dir).numpy()
        proj_sup  = (h_support[l].float()  @ conf_sup_dir).numpy()
        pooled_std = np.sqrt((proj_conf.std()**2 + proj_sup.std()**2) / 2) + 1e-8
        cohens_d = float((proj_conf.mean() - proj_sup.mean()) / pooled_std)

        stats[l] = {
            "conf_sup_dir":    conf_sup_dir,
            "explained_ratio": explained,
            "cohens_d":        cohens_d,
            "proj_conf_mean":  float(proj_conf.mean()),
            "proj_sup_mean":   float(proj_sup.mean()),
        }
        print(f"  [B] L{l}: explained={explained:.3f}  cohens_d={cohens_d:.3f}")
    return stats


# ── Method C: cPCA ────────────────────────────────────────────────────────────

def run_method_C(h_conflict, h_support, layers, mean_diff_stats, svd_stats):
    stats = {}
    for l in layers:
        X_conf = (h_conflict[l].float() - h_conflict[l].float().mean(0)).to(DEVICE)
        X_sup  = (h_support[l].float()  - h_support[l].float().mean(0)).to(DEVICE)
        C_conflict = (X_conf.T @ X_conf) / len(X_conf)
        C_support  = (X_sup.T  @ X_sup)  / len(X_sup)

        stats[l] = {}
        for alpha in ALPHA_VALUES:
            C_contrast = C_conflict - alpha * C_support
            eigenvalues, eigenvectors = torch.linalg.eigh(C_contrast)
            cpca_dir = F.normalize(eigenvectors[:, -1].float().cpu(), dim=0)

            ov_A = float(torch.abs(torch.cosine_similarity(
                cpca_dir, mean_diff_stats[l]["mean_dir"], dim=0)).item())
            ov_B = float(torch.abs(torch.cosine_similarity(
                cpca_dir, svd_stats[l]["conf_sup_dir"], dim=0)).item())

            stats[l][f"alpha_{alpha}"] = {
                "cpca_dir": cpca_dir,
                "overlap_with_A": ov_A,
                "overlap_with_B": ov_B,
            }

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
        print(f"  [C] L{l}: best_alpha={best_alpha}  ov_A={best['overlap_with_A']:.3f}  ov_B={best['overlap_with_B']:.3f}")
    return stats


# ── 기존 파일에 append ────────────────────────────────────────────────────────

def append_to_file(path: Path, new_entries: Dict) -> None:
    data = torch.load(path, map_location="cpu")
    data.update(new_entries)
    torch.save(data, path)
    print(f"[save] {path}  (layers now: {sorted(data.keys())})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    model = load_model(device=DEVICE)
    cases = load_cases(n=1000)

    print(f"\n[collect] L{NEW_LAYERS} residuals for {len(cases)} cases ...")
    h_conflict, h_support = collect_residuals_for_layers(model, cases, NEW_LAYERS)

    print("\n[method A]")
    stats_A = run_method_A(h_conflict, h_support, NEW_LAYERS)

    print("\n[method B]")
    stats_B = run_method_B(h_conflict, h_support, NEW_LAYERS)

    print("\n[method C]")
    stats_C = run_method_C(h_conflict, h_support, NEW_LAYERS, stats_A, stats_B)

    print("\n[append to existing files]")
    append_to_file(EXP1B_DIR / "method_A_mean_diff.pt",    {l: stats_A[l] for l in NEW_LAYERS})
    append_to_file(EXP1B_DIR / "method_B_svd_conf_sup.pt", {l: stats_B[l] for l in NEW_LAYERS})
    append_to_file(EXP1B_DIR / "method_C_cpca.pt",         {l: stats_C[l] for l in NEW_LAYERS})

    print(f"\n[done] total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

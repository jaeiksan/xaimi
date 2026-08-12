#!/usr/bin/env python3
"""
실험 3: Mid-to-L17H3 Mediation Test

핵심 질문: layer 6-16의 KC direction을 patch하면, L17H3의 conflict-supporting behavior가 바뀌는가?

각 case에서 측정하는 세 가지 값:
  1. delta_logit_diff:  logit_diff_before − logit_diff_after
     (양수 = patch 후 KC 경향 감소, good)

  2. delta_ev_attn:    ev_attn_before − ev_attn_after
     L17H3의 evidence token span에 대한 attention sum
     (양수 = patch 후 L17H3가 evidence를 덜 보게 됨, good)

  3. delta_kc_proj:    kc_proj_before − kc_proj_after
     L17H3의 z output (hook_z)을 kc_dir_head (d_head 공간)에 projection
     (양수 = patch 후 L17H3 output에서 KC 성분 감소, good)

kc_dir 차원 처리 (방법 A):
  kc_dir (d_model) → W_O[17][3] @ kc_dir → kc_dir_head (d_head)
  step5_head_ablation.py 참고: hook_z shape = [batch, pos, n_heads, d_head]

evidence span 위치:
  conflict_prompt = "Context:\n{context}\n\n..."
  근사: "Context:\n" tokenization으로 시작 위치를 찾고, context tokenization으로 끝 위치 계산

출력:
  phase2/results/exp3_mediation/
    mediation_results.pt        — layer별 측정값
    mediation_plot.png          — Figure 4
    mediation_summary.json      — 요약 통계

실행:
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp3_mediation.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp3_mediation.py --test 20
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
    MED_LAYER, MED_HEAD,
    load_model, load_cases,
    build_clean_prompt, build_conflict_prompt,
    collect_residuals_two_pos,
    get_pos_indices, get_evidence_span,
    get_logit_diff, get_correct_prob,
    patch_kc_direction, make_patch_hook,
    load_trajectory_stats,
    save_pt, save_json,
)

EXP1_DIR = Path("/home/jaeik/xaimi/phase2/results/exp1_trajectory")
EXP2_DIR = Path("/home/jaeik/xaimi/results/exp2_direction_patching")
OUT_DIR  = Path("/home/jaeik/xaimi/phase2/results/exp3_mediation")
CKPT_PATH = OUT_DIR / "mediation_checkpoint.pt"
CHECKPOINT_EVERY = 50

# 기본 patch target layers (exp2 top IE + 균등 간격)
DEFAULT_PATCH_LAYERS = [6, 8, 10, 12, 14, 16]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실험 3: Mid-to-L17H3 Mediation Test")
    p.add_argument("--test",    type=int, default=None, help="N개만 처리 (테스트용)")
    p.add_argument("--n_cases", type=int, default=None, help="사용할 케이스 수 (기본: N_CASES=200)")
    return p.parse_args()


# ── patch target layer 결정 ────────────────────────────────────────────────────
def get_patch_layers(exp2_dir: Path) -> List[int]:
    """
    exp2의 top-IE layers (final position 기준 상위 4개)와
    기본 간격 layers를 합쳐 patch target을 결정.
    """
    patch_layers = list(DEFAULT_PATCH_LAYERS)

    comparison_path = exp2_dir / "comparison_with_step3.json"
    if comparison_path.exists():
        with comparison_path.open() as f:
            comp = json.load(f)
        top_ie = comp.get("top_ie_layers_final", [])
        for l in top_ie:
            if l not in patch_layers and 6 <= l <= 16:
                patch_layers.append(l)
        patch_layers = sorted(set(patch_layers))
        print(f"[exp3] exp2 top-IE layers 추가: {top_ie}", flush=True)

    print(f"[exp3] patch target layers: {patch_layers}", flush=True)
    return patch_layers


# ── L17H3 kc_dir 투영 (d_model → d_head) ─────────────────────────────────────
def get_kc_dir_head(model, kc_dir_dmodel: torch.Tensor) -> torch.Tensor:
    """
    방법 A: kc_dir (d_model) → W_O[17][3] @ kc_dir → kc_dir_head (d_head)
    W_O[layer, head]: shape (d_head, d_model)
    kc_dir_head = W_O[17, 3] @ kc_dir  → (d_head,)
    """
    W_O = model.W_O[MED_LAYER, MED_HEAD].float().cpu()  # (d_head, d_model)
    kc_dir_head = W_O @ kc_dir_dmodel.float()            # (d_head,)
    return F.normalize(kc_dir_head, dim=-1)


# ── "before" 측정 (conflict run, no patch) ────────────────────────────────────
def _ev_entropy(pattern_slice: torch.Tensor) -> float:
    """Entropy of L17H3 attention distribution over evidence span tokens."""
    if len(pattern_slice) == 0:
        return 0.0
    p = pattern_slice / (pattern_slice.sum() + 1e-10)
    return float(-(p * torch.log(p + 1e-10)).sum().item())


def measure_before(
    model,
    conflict_ids: torch.Tensor,
    final_pos: int,
    ev_start: int,
    ev_end: int,
    cor: str,
    rep: str,
    kc_dir_head: torch.Tensor,
) -> Tuple[float, float, float, float]:
    """
    conflict run 그대로 실행, 네 가지 값 측정.

    Returns:
        logit_diff_before:   logP(replace) - logP(correct)
        ev_attn_before:      L17H3의 evidence attention sum
        kc_proj_before:      L17H3 z output의 kc_dir_head projection
        ev_entropy_before:   L17H3의 evidence span attention entropy
    """
    store = {}

    def capture_z(val: torch.Tensor, hook) -> torch.Tensor:
        # val: [1, seq, n_heads, d_head]
        store["z"] = val[0, final_pos, MED_HEAD, :].detach().float().cpu()
        return val

    def capture_pattern(val: torch.Tensor, hook) -> torch.Tensor:
        # val: [1, n_heads, seq, seq]
        store["pattern"] = val[0, MED_HEAD, final_pos, :].detach().float().cpu()
        return val

    with torch.no_grad():
        logits = model.run_with_hooks(
            conflict_ids,
            fwd_hooks=[
                (f"blocks.{MED_LAYER}.attn.hook_z",       capture_z),
                (f"blocks.{MED_LAYER}.attn.hook_pattern",  capture_pattern),
            ],
        )

    # 1. logit_diff
    ld_before = get_logit_diff(logits, rep, cor)

    # 2. ev_attn + entropy: attention to evidence span (clamp to valid range)
    seq_len = conflict_ids.shape[1]
    ev_s = min(ev_start, seq_len - 1)
    ev_e = min(ev_end, seq_len)
    ev_slice = store["pattern"][ev_s:ev_e] if ev_s < ev_e else torch.zeros(0)
    ev_attn    = float(ev_slice.sum().item())
    ev_entropy = _ev_entropy(ev_slice)

    # 3. kc_proj
    z = store["z"]  # (d_head,)
    kc_proj = float((z @ kc_dir_head).item())

    return ld_before, ev_attn, kc_proj, ev_entropy


# ── "after" 측정 (patch layer l, KC direction correction) ─────────────────────
def measure_after(
    model,
    conflict_ids: torch.Tensor,
    patch_layer: int,
    h_patched: torch.Tensor,
    final_pos: int,
    ev_start: int,
    ev_end: int,
    cor: str,
    rep: str,
    kc_dir_head: torch.Tensor,
    device: str,
) -> Tuple[float, float, float, float]:
    """
    layer patch_layer에서 KC direction 보정 후 실행, 네 가지 값 측정.
    """
    store = {}

    patch_hook = make_patch_hook(h_patched, final_pos, device)

    def capture_z(val: torch.Tensor, hook) -> torch.Tensor:
        store["z"] = val[0, final_pos, MED_HEAD, :].detach().float().cpu()
        return val

    def capture_pattern(val: torch.Tensor, hook) -> torch.Tensor:
        store["pattern"] = val[0, MED_HEAD, final_pos, :].detach().float().cpu()
        return val

    with torch.no_grad():
        logits = model.run_with_hooks(
            conflict_ids,
            fwd_hooks=[
                (f"blocks.{patch_layer}.hook_resid_post",  patch_hook),
                (f"blocks.{MED_LAYER}.attn.hook_z",        capture_z),
                (f"blocks.{MED_LAYER}.attn.hook_pattern",  capture_pattern),
            ],
        )

    ld_after = get_logit_diff(logits, rep, cor)

    seq_len = conflict_ids.shape[1]
    ev_s = min(ev_start, seq_len - 1)
    ev_e = min(ev_end, seq_len)
    ev_slice   = store["pattern"][ev_s:ev_e] if ev_s < ev_e else torch.zeros(0)
    ev_attn    = float(ev_slice.sum().item())
    ev_entropy = _ev_entropy(ev_slice)

    z = store["z"]
    kc_proj = float((z @ kc_dir_head).item())

    return ld_after, ev_attn, kc_proj, ev_entropy


# ── 메인 측정 루프 ─────────────────────────────────────────────────────────────
def run_mediation(
    model,
    cases: List[Dict],
    patch_layers: List[int],
    kc_dirs: Dict[int, torch.Tensor],
    clean_mean_projs: Dict[int, float],
    kc_dir_17: torch.Tensor,
) -> Dict:
    """
    mediation_results[patch_layer] = {
        "delta_logit_diff":     [floats],
        "delta_ev_attn":        [floats],
        "delta_kc_proj":        [floats],
    }
    """
    N = len(cases)
    device = str(next(model.parameters()).device)

    # 체크포인트 로드
    if CKPT_PATH.exists():
        print(f"[exp3] 체크포인트 로드: {CKPT_PATH}", flush=True)
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        med_lists = ckpt["med_lists"]
        start_i   = ckpt["n_done"]
        print(f"[exp3] {start_i}개 이미 완료, 이어서 진행", flush=True)
    else:
        med_lists = {
            pl: {"delta_logit_diff": [], "delta_ev_attn": [], "delta_kc_proj": [], "delta_ev_entropy": []}
            for pl in patch_layers
        }
        start_i = 0

    # kc_dir_head: L17H3 output을 투영할 d_head 방향
    # layer 17의 kc_dir (d_model)을 W_O[17][3]으로 d_head 공간으로 변환
    kc_dir_head = get_kc_dir_head(model, kc_dir_17)

    t0 = time.time()

    for i in range(start_i, N):
        case = cases[i]
        q    = case["question"]
        opts = case["options"]
        ctx  = case["conflict_context"]
        cor  = case["correct_option"]
        rep  = case["replace_option"]

        conflict_p = build_conflict_prompt(ctx, q, opts)
        final_pos, _ = get_pos_indices(model, conflict_p)
        ev_start, ev_end = get_evidence_span(model, ctx)

        conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)

        # "before" 측정 (conflict run 그대로)
        ld_before, ev_attn_before, kc_proj_before, ev_entropy_before = measure_before(
            model, conflict_ids, final_pos, ev_start, ev_end,
            cor, rep, kc_dir_head,
        )

        # "after" 측정: 각 patch_layer에서 KC direction 보정
        # conflict residuals (final position) 수집 — patch에 필요
        res_final, _, _ = collect_residuals_two_pos(
            model, conflict_p, patch_layers, final_pos, max(0, final_pos - 1)
        )

        for li, pl in enumerate(patch_layers):
            if pl >= MED_LAYER:
                # patch가 L17H3보다 늦으면 L17H3 측정에 영향 없음 (skip)
                continue

            kc_dir = kc_dirs.get(pl)
            if kc_dir is None:
                continue

            cmp = clean_mean_projs.get(pl, 0.0)
            h_patched = patch_kc_direction(res_final[li], kc_dir, cmp)

            ld_after, ev_attn_after, kc_proj_after, ev_entropy_after = measure_after(
                model, conflict_ids, pl, h_patched,
                final_pos, ev_start, ev_end,
                cor, rep, kc_dir_head, device,
            )

            med_lists[pl]["delta_logit_diff"].append(ld_before - ld_after)
            med_lists[pl]["delta_ev_attn"].append(ev_attn_before - ev_attn_after)
            med_lists[pl]["delta_kc_proj"].append(kc_proj_before - kc_proj_after)
            med_lists[pl]["delta_ev_entropy"].append(ev_entropy_before - ev_entropy_after)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i - start_i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_pt(CKPT_PATH, {"med_lists": med_lists, "n_done": i + 1})

        gc.collect()
        torch.cuda.empty_cache()

    return med_lists


# ── 집계 ─────────────────────────────────────────────────────────────────────
def aggregate(med_lists: Dict, patch_layers: List[int]) -> Dict:
    results = {}
    for pl in patch_layers:
        n = len(med_lists[pl]["delta_logit_diff"])
        if n == 0:
            continue
        results[pl] = {}
        for metric in ["delta_logit_diff", "delta_ev_attn", "delta_kc_proj", "delta_ev_entropy"]:
            vals = med_lists[pl][metric]
            results[pl][metric]              = float(np.mean(vals)) if vals else 0.0
            results[pl][f"{metric}_std"]     = float(np.std(vals))  if vals else 0.0
        results[pl]["n_cases"] = n
    return results


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_mediation(results: Dict, patch_layers: List[int], out_dir: Path) -> None:
    """Figure 4: 네 가지 delta 값을 한 그래프에 표시."""
    valid_layers = [l for l in patch_layers if l in results and results[l]["n_cases"] > 0]
    if not valid_layers:
        print("[exp3] 유효한 결과 없음, 시각화 스킵", flush=True)
        return

    metrics = {
        "delta_logit_diff":  ("#e74c3c", "Δ logit_diff\n(↑ = KC 감소)"),
        "delta_ev_attn":     ("#3498db", "Δ ev_attn L17H3\n(↑ = evidence attention 감소)"),
        "delta_kc_proj":     ("#2ecc71", "Δ kc_proj L17H3\n(↑ = KC projection 감소)"),
        "delta_ev_entropy":  ("#9b59b6", "Δ ev_entropy L17H3\n(↑ = evidence attention 더 분산됨)"),
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 13), sharex=True)

    for ax, (metric, (color, ylabel)) in zip(axes, metrics.items()):
        vals = [results[l][metric] for l in valid_layers]
        stds = [results[l][f"{metric}_std"] for l in valid_layers]

        ax.bar(valid_layers, vals, color=color, alpha=0.7, edgecolor="white", width=0.7)
        ax.errorbar(
            valid_layers, vals, yerr=stds,
            fmt="none", color="black", capsize=4, linewidth=1.5,
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Patch Layer")
    axes[-1].set_xticks(valid_layers)
    fig.suptitle(
        f"Mediation Analysis: Mid-Layer KC Direction → L{MED_LAYER}H{MED_HEAD}\n"
        "(네 값이 동시에 높은 layer = KC mediation 핵심 layer)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = out_dir / "mediation_plot.png"
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

    # exp1 결과 로드
    traj_stats = load_trajectory_stats(EXP1_DIR)
    kc_dirs = {layer: traj_stats[layer]["kc_dir"] for layer in LAYERS}

    # patch target layers 결정 (exp2 결과 + 기본값)
    patch_layers = get_patch_layers(EXP2_DIR)
    # L17H3보다 이후 layer는 의미 없으므로 필터
    patch_layers = [l for l in patch_layers if l < MED_LAYER]

    # clean_mean_proj 로드 (exp1에서 계산된 값)
    clean_mean_projs = {
        layer: traj_stats[layer].get("clean_mean_proj", 0.0)
        for layer in patch_layers
        if layer in traj_stats
    }

    # L17H3 projection을 위한 kc_dir[17] 사용
    # layer 17이 LAYERS 범위(6-19)에 있으면 사용, 없으면 가장 가까운 layer 사용
    if MED_LAYER in traj_stats:
        kc_dir_17 = traj_stats[MED_LAYER]["kc_dir"]
    else:
        closest = min(traj_stats.keys(), key=lambda l: abs(l - MED_LAYER))
        kc_dir_17 = traj_stats[closest]["kc_dir"]
        print(f"[exp3] layer 17 kc_dir 없음, layer {closest} 사용", flush=True)

    print(f"\n[exp3] {len(cases)}개 케이스, patch_layers={patch_layers}", flush=True)
    print(f"[exp3] 측정 대상: L{MED_LAYER}H{MED_HEAD} (evidence attention, KC projection)", flush=True)

    # 메인 측정
    t0 = time.time()
    med_lists = run_mediation(
        model, cases, patch_layers, kc_dirs, clean_mean_projs, kc_dir_17,
    )
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    # 집계
    results = aggregate(med_lists, patch_layers)

    # 저장
    save_pt(OUT_DIR / "mediation_results.pt", {"results": results, "med_lists": med_lists})

    summary = {
        str(l): {
            k: v for k, v in results[l].items()
        }
        for l in patch_layers if l in results
    }
    save_json(OUT_DIR / "mediation_summary.json", summary)

    # 체크포인트 정리
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print(f"[exp3] 체크포인트 삭제: {CKPT_PATH}", flush=True)

    # 시각화
    print("\n[exp3] 시각화 ...", flush=True)
    plot_mediation(results, patch_layers, OUT_DIR)

    # 핵심 결과 요약
    print("\n=== 실험 3 핵심 결과 ===")
    if results:
        best = max(results, key=lambda l: results[l]["delta_logit_diff"])
        print(f"delta_logit_diff 최대 layer: {best}  ({results[best]['delta_logit_diff']:.4f})")
        best_ev = max(results, key=lambda l: results[l]["delta_ev_attn"])
        print(f"delta_ev_attn 최대 layer:    {best_ev}  ({results[best_ev]['delta_ev_attn']:.4f})")
        best_kc = max(results, key=lambda l: results[l]["delta_kc_proj"])
        print(f"delta_kc_proj 최대 layer:    {best_kc}  ({results[best_kc]['delta_kc_proj']:.4f})")
        best_ent = max(results, key=lambda l: results[l]["delta_ev_entropy"])
        print(f"delta_ev_entropy 최대 layer: {best_ent}  ({results[best_ent]['delta_ev_entropy']:.4f})")

    print(f"\n[exp3] 완료. 결과: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

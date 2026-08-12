#!/usr/bin/env python3
"""
실험 2: Direction-Specific Causal Patching

실험 1에서 찾은 KC direction 성분만 clean 평균값으로 복원하고,
나머지 방향은 그대로 두어 해당 direction의 causal contribution을 측정한다.

핵심 질문: kc_sup_overlap이 낮았던 layer의 KC direction이 실제로 final answer에 causal한가?

IE_dir[layer][pos] = mean(P(correct | patched) - P(correct | conflict))
  양수: KC direction 제거 후 정답 확률 증가 → causal
  음수: 오히려 정답 확률 감소 → KC driving 방향이 아님

두 position:
  final:      마지막 token (answer prediction 위치)
  pre_answer: "\nAnswer:" 직전 token

출력:
  results/exp2_direction_patching/
    IE_dir.pt                  — layer × position별 IE 수치
    IE_dir_heatmap.png         — Figure 3
    comparison_with_step3.json — step3 결과와 비교

실행:
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u scripts/exp2_direction_patching.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u scripts/exp2_direction_patching.py --test 20
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

from exp_utils import (
    LAYERS, N_CASES, SEED,
    load_model, load_cases,
    build_clean_prompt, build_conflict_prompt,
    collect_residuals_two_pos,
    get_pos_indices,
    get_correct_prob,
    patch_kc_direction, make_patch_hook,
    load_trajectory_stats,
    save_pt, save_json,
)

EXP1_DIR = Path("/home/jaeik/xaimi/phase2/results/exp1_trajectory")
STEP3_STATS = Path("/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ct_stats.json")
OUT_DIR  = Path("/home/jaeik/xaimi/results/exp2_direction_patching")
CKPT_PATH = OUT_DIR / "ie_checkpoint.pt"
CHECKPOINT_EVERY = 50
POSITIONS = ("final", "pre_answer")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실험 2: Direction-Specific Causal Patching")
    p.add_argument("--test",    type=int, default=None, help="N개만 처리 (테스트용)")
    p.add_argument("--n_cases", type=int, default=None, help="사용할 케이스 수 (기본: N_CASES=200)")
    return p.parse_args()


# ── clean_mean_proj 계산 ─────────────────────────────────────────────────────
def compute_clean_mean_projs(
    model,
    cases: List[Dict],
    layers: List[int],
    kc_dirs: Dict[int, torch.Tensor],
) -> Dict[int, Dict[str, float]]:
    """
    각 layer, 각 position에서 clean run residual의 kc_dir 방향 평균 projection 계산.

    Returns:
        clean_mean_projs[layer]["final"]      = float
        clean_mean_projs[layer]["pre_answer"] = float
    """
    N = len(cases)
    sums: Dict[int, Dict[str, float]] = {
        l: {"final": 0.0, "pre_answer": 0.0} for l in layers
    }

    print("[exp2] clean_mean_proj 계산 중 ...", flush=True)
    t0 = time.time()

    for i, case in enumerate(cases):
        q    = case["question"]
        opts = case["options"]
        clean_p = build_clean_prompt(q, opts)
        final_pos, pre_ans_pos = get_pos_indices(model, clean_p)

        res_final, res_pre, _ = collect_residuals_two_pos(
            model, clean_p, layers, final_pos, pre_ans_pos
        )

        for li, layer in enumerate(layers):
            d = kc_dirs[layer]
            sums[layer]["final"]      += (res_final[li] @ d).item()
            sums[layer]["pre_answer"] += (res_pre[li]   @ d).item()

        if (i + 1) % 50 == 0:
            print(f"  clean_mean [{i+1}/{N}]  {time.time()-t0:.0f}s", flush=True)

    return {
        l: {pos: sums[l][pos] / N for pos in POSITIONS}
        for l in layers
    }


# ── IE 계산 (패칭) ────────────────────────────────────────────────────────────
def compute_ie_dir(
    model,
    cases: List[Dict],
    layers: List[int],
    kc_dirs: Dict[int, torch.Tensor],
    clean_mean_projs: Dict[int, Dict[str, float]],
) -> Tuple[Dict, Dict]:
    """
    각 (layer, position) 조합에서 KC direction 패칭 후 IE를 계산.

    IE_dir[layer][pos] = mean over cases of
        P(correct | patched) - P(correct | conflict)

    Returns:
        ie_means: {layer: {"final": float, "pre_answer": float}}
        ie_lists: {layer: {"final": [floats], "pre_answer": [floats]}}
    """
    N = len(cases)

    # 체크포인트 로드
    if CKPT_PATH.exists():
        print(f"[exp2] 체크포인트 로드: {CKPT_PATH}", flush=True)
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        ie_lists = ckpt["ie_lists"]
        start_i  = ckpt["n_done"]
        print(f"[exp2] {start_i}개 이미 완료, 이어서 진행", flush=True)
    else:
        ie_lists = {
            l: {"final": [], "pre_answer": []} for l in layers
        }
        start_i = 0

    device = str(next(model.parameters()).device)
    t0 = time.time()

    for i in range(start_i, N):
        case = cases[i]
        q    = case["question"]
        opts = case["options"]
        ctx  = case["conflict_context"]
        cor  = case["correct_option"]

        conflict_p = build_conflict_prompt(ctx, q, opts)
        final_pos, pre_ans_pos = get_pos_indices(model, conflict_p)

        # conflict run: residuals + baseline P(correct)
        res_final, res_pre, logits_base = collect_residuals_two_pos(
            model, conflict_p, layers, final_pos, pre_ans_pos
        )
        p_cor_base = get_correct_prob(logits_base.unsqueeze(0).unsqueeze(0), cor)

        conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)

        for li, layer in enumerate(layers):
            kc_dir = kc_dirs[layer]  # (d_model,) CPU

            for pos_name, h_res, pos_idx in [
                ("final",      res_final[li], final_pos),
                ("pre_answer", res_pre[li],   pre_ans_pos),
            ]:
                cmp = clean_mean_projs[layer][pos_name]

                h_patched = patch_kc_direction(h_res, kc_dir, cmp)

                hook = make_patch_hook(h_patched, pos_idx, device)
                with torch.no_grad():
                    logits_p = model.run_with_hooks(
                        conflict_ids,
                        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)],
                    )

                p_cor_patch = get_correct_prob(logits_p, cor)
                ie = p_cor_patch - p_cor_base
                ie_lists[layer][pos_name].append(ie)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i - start_i + 1) * (N - i - 1)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_pt(CKPT_PATH, {"ie_lists": ie_lists, "n_done": i + 1})

        gc.collect()
        torch.cuda.empty_cache()

    ie_means = {
        l: {pos: float(np.mean(ie_lists[l][pos])) if ie_lists[l][pos] else 0.0
            for pos in POSITIONS}
        for l in layers
    }
    return ie_means, ie_lists


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_ie_heatmap(ie_means: Dict, layers: List[int], out_dir: Path) -> None:
    """Figure 3: IE_dir heatmap (layer × position)."""
    mat = np.array([
        [ie_means[l]["final"], ie_means[l]["pre_answer"]]
        for l in layers
    ])  # (n_layers, 2)

    vabs = max(0.02, float(np.percentile(np.abs(mat), 95)))
    vmax =  vabs
    vmin = -vabs

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(
        mat.T, aspect="auto", cmap="RdYlGn",
        vmin=vmin, vmax=vmax,
    )
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=9)
    ax.set_yticks(range(len(POSITIONS)))
    ax.set_yticklabels(POSITIONS, fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_title(
        "IE_dir: P(correct|patched) − P(correct|conflict)\n"
        "(빨간색=양수=KC direction 제거 후 정답 확률 증가, causal)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="IE")
    plt.tight_layout()
    path = out_dir / "IE_dir_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_ie_with_step3(ie_means: Dict, layers: List[int], step3_stats: Dict, out_dir: Path) -> None:
    """IE_dir (final position)과 step3 IE를 같은 그래프에 표시."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # exp2 IE (final position)
    ie_final = [ie_means[l]["final"] for l in layers]
    ax.plot(layers, ie_final, color="#e74c3c", linewidth=2, marker="o",
            markersize=5, label="Exp2: IE_dir (final, KC direction only)")

    # step3 최대 IE per layer (context_first_sent 기준)
    if step3_stats:
        overall = step3_stats.get("overall_ie_matrix", [])
        if overall:
            n_step3_layers = len(overall)
            # context_first_sent = column 0
            ie_step3 = [overall[l][0] if l < n_step3_layers else 0.0 for l in layers]
            ax.plot(layers, ie_step3, color="#3498db", linewidth=2, marker="s",
                    markersize=5, linestyle="--", label="Step3: IE (context_first_sent, full restore)")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("IE")
    ax.set_title("Direction-Specific IE vs Step3 Full Restore IE")
    ax.legend(fontsize=9)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "IE_dir_vs_step3.png"
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

    # 실험 1 결과 로드
    traj_stats = load_trajectory_stats(EXP1_DIR)
    kc_dirs = {layer: traj_stats[layer]["kc_dir"] for layer in LAYERS}

    print(f"\n[exp2] {len(cases)}개 케이스, layers={LAYERS}", flush=True)

    # 1. clean_mean_proj 계산
    print("\n[exp2] Step 1: clean_mean_proj 계산 ...", flush=True)
    clean_mean_projs = compute_clean_mean_projs(model, cases, LAYERS, kc_dirs)

    # 2. IE 계산 (패칭)
    print("\n[exp2] Step 2: KC direction 패칭 IE 계산 ...", flush=True)
    t0 = time.time()
    ie_means, ie_lists = compute_ie_dir(model, cases, LAYERS, kc_dirs, clean_mean_projs)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    # 3. 저장
    ie_payload = {
        "ie_means": ie_means,
        "ie_stds": {
            l: {pos: float(np.std(ie_lists[l][pos])) if ie_lists[l][pos] else 0.0
                for pos in POSITIONS}
            for l in LAYERS
        },
        "n_cases": len(cases),
        "layers": LAYERS,
    }
    save_pt(OUT_DIR / "IE_dir.pt", ie_payload)

    # 4. step3와 비교
    step3_stats = {}
    if STEP3_STATS.exists():
        with STEP3_STATS.open() as f:
            step3_stats = json.load(f)

    # top-IE layers 찾기 (exp3에서 사용)
    top_ie_layers = sorted(LAYERS, key=lambda l: ie_means[l]["final"], reverse=True)[:4]
    comparison = {
        "exp2_ie_means": {str(l): ie_means[l] for l in LAYERS},
        "top_ie_layers_final": top_ie_layers,
        "step3_available": STEP3_STATS.exists(),
    }
    if step3_stats:
        overall = step3_stats.get("overall_ie_matrix", [])
        if overall:
            step3_max_layer = int(np.argmax([row[0] for row in overall]))
            comparison["step3_max_ie_layer"] = step3_max_layer
    save_json(OUT_DIR / "comparison_with_step3.json", comparison)

    # 5. 시각화
    print("\n[exp2] Step 3: 시각화 ...", flush=True)
    plot_ie_heatmap(ie_means, LAYERS, OUT_DIR)
    plot_ie_with_step3(ie_means, LAYERS, step3_stats, OUT_DIR)

    # 6. 체크포인트 정리
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print(f"[exp2] 체크포인트 삭제: {CKPT_PATH}", flush=True)

    # 핵심 결과 요약
    print("\n=== 실험 2 핵심 결과 ===")
    print(f"IE가 가장 높은 layer (final pos): {top_ie_layers}")
    for l in top_ie_layers:
        print(f"  layer {l}: final={ie_means[l]['final']:.4f}  pre_answer={ie_means[l]['pre_answer']:.4f}")
    print(f"\n[exp2] 완료. 결과: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

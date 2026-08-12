#!/usr/bin/env python3
"""
Phase 3: 메커니즘 기반 KC 억제 개입 실험  (v2 — 세분화 alpha + 레이어 분리 + 정규화)

전략 A : Layer 17 Head 3 Mean Ablation  (hook_z, step5 mean_values.pt 재사용)
전략 B1: Direction Vector Steering, L17만
전략 B2: Direction Vector Steering, L18만
전략 B3: Direction Vector Steering, L17+L18 동시
  alpha : [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]  (정규화 벡터)
기준선 : No intervention

Phase 1 — conflict/neutral 양쪽 → baseline + 방향벡터 계산 + 정규화
Phase 2 — 전략 A×2 + 전략 B×3조건×8alpha×2 = 50 pass/case

실행:
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step6_intervention.py
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step6_intervention.py --test 10
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from transformer_lens import HookedTransformer

# ── 상수 ──────────────────────────────────────────────────────────────────────
KC_PATH    = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
MEAN_PATH  = Path("/home/jaeik/xaimi/phase1/results/step5_head_ablation/mean_values.pt")
OUT_DIR    = Path("/home/jaeik/xaimi/phase1/results/step6_intervention")
MODEL_NAME = "google/gemma-2-2b"

TARGET_LAYER = 17
TARGET_HEAD  = 3

# 전략 B 조건
B_CONFIGS: Dict[str, List[int]] = {
    "B1": [17],
    "B2": [18],
    "B3": [17, 18],
}
B_LABELS: Dict[str, str] = {
    "B1": "L17only",
    "B2": "L18only",
    "B3": "L17+18 ",
}
STEER_LAYERS = [17, 18]          # 방향벡터 계산 대상 레이어

ALPHAS     = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
ALPHA_STRS = [str(a) for a in ALPHAS]

CTYPES           = ("misinformation", "temporal", "semantic")
CHECKPOINT_EVERY = 50
SEED             = 42

LETTER_VARIANTS: Dict[str, List[int]] = {}
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=int, default=None, metavar="N",
                   help="N개만 처리 (테스트용)")
    return p.parse_args()


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model() -> HookedTransformer:
    print("모델 로드 (bfloat16) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16,
    )
    model.eval()
    return model


# ── 토크나이저 ────────────────────────────────────────────────────────────────
def setup_letter_variants(tokenizer) -> None:
    global LETTER_VARIANTS
    for letter in "ABCD":
        ids: List[int] = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.append(toks[-1] if len(toks) > 1 else toks[0])
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
    print(f"[setup] LETTER_VARIANTS={LETTER_VARIANTS}", flush=True)


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────
def build_conflict_prompt(ctx: str, q: str, opts: Dict[str, str]) -> str:
    lines = ["Context:", ctx, "", f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_neutral_prompt(q: str, opts: Dict[str, str]) -> str:
    lines = [f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── 확률 / 예측 ───────────────────────────────────────────────────────────────
def get_letter_prob(probs: torch.Tensor, letter: str) -> float:
    return max(probs[tid].item() for tid in LETTER_VARIANTS[letter])


def get_top_prediction(probs: torch.Tensor) -> str:
    best, best_p = "?", -1.0
    for l in "ABCD":
        p = get_letter_prob(probs, l)
        if p > best_p:
            best_p, best = p, l
    return best


def run_fwd(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    fwd_hooks: Optional[List] = None,
) -> torch.Tensor:
    hooks = fwd_hooks or []
    with torch.no_grad():
        logits = (
            model.run_with_hooks(input_ids.unsqueeze(0), fwd_hooks=hooks)
            if hooks
            else model(input_ids.unsqueeze(0))
        )
    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    del logits
    return probs


# ── Phase 1: baseline + 방향벡터 계산 + 정규화 ───────────────────────────────
def phase1(
    model: HookedTransformer,
    kc_cases: list,
    n_cases: int,
) -> Tuple[list, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Returns:
      baselines        : [{baseline_diff, clean_top, cor, rep, ctype}]
      directions       : {17: tensor[d_model], 18: tensor[d_model]}  (CPU, float32, 원본)
      directions_norm  : {17: tensor[d_model], 18: tensor[d_model]}  (CPU, float32, 정규화)
    """
    d_model = model.cfg.d_model
    sum_n = {l: torch.zeros(d_model, dtype=torch.float32) for l in STEER_LAYERS}
    sum_c = {l: torch.zeros(d_model, dtype=torch.float32) for l in STEER_LAYERS}
    baselines: list = []

    print(f"\n[Phase 1] baseline + 방향벡터 계산 ({n_cases}개) ...", flush=True)
    t0 = time.time()

    for i, case in enumerate(kc_cases[:n_cases]):
        ctx  = case["conflict_context"]
        q    = case["question"]
        opts = case["options"]
        cor  = case["correct_option"]
        rep  = case["replace_option"]
        ct   = case["conflict_type"]

        neutral_p    = build_neutral_prompt(q, opts)
        conflict_p   = build_conflict_prompt(ctx, q, opts)
        neutral_ids  = model.to_tokens(neutral_p,  prepend_bos=True)[0]
        conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]

        fp_n = int(neutral_ids.shape[0]) - 1
        fp_c = int(conflict_ids.shape[0]) - 1

        # neutral pass — resid 캡처 + clean 예측
        store_n: Dict[int, torch.Tensor] = {}

        def make_cap_n(l: int, fp: int):
            def _fn(val: torch.Tensor, hook) -> torch.Tensor:
                store_n[l] = val[0, fp, :].detach().float().cpu()
                return val
            return _fn

        probs_n   = run_fwd(model, neutral_ids,
                            [(f"blocks.{l}.hook_resid_post", make_cap_n(l, fp_n))
                             for l in STEER_LAYERS])
        clean_top = get_top_prediction(probs_n)

        # conflict pass — resid 캡처 + baseline_diff
        store_c: Dict[int, torch.Tensor] = {}

        def make_cap_c(l: int, fp: int):
            def _fn(val: torch.Tensor, hook) -> torch.Tensor:
                store_c[l] = val[0, fp, :].detach().float().cpu()
                return val
            return _fn

        probs_c       = run_fwd(model, conflict_ids,
                                [(f"blocks.{l}.hook_resid_post", make_cap_c(l, fp_c))
                                 for l in STEER_LAYERS])
        baseline_diff = get_letter_prob(probs_c, rep) - get_letter_prob(probs_c, cor)

        for l in STEER_LAYERS:
            sum_n[l] += store_n[l]
            sum_c[l] += store_c[l]

        baselines.append({
            "baseline_diff": baseline_diff,
            "clean_top":     clean_top,
            "cor":           cor,
            "rep":           rep,
            "ctype":         ct,
        })

        if (i + 1) % 100 == 0:
            print(f"  phase1 {i+1}/{n_cases}  ({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()

    n = len(baselines)
    directions: Dict[int, torch.Tensor] = {
        l: (sum_n[l] - sum_c[l]) / n for l in STEER_LAYERS
    }
    directions_norm: Dict[int, torch.Tensor] = {}
    for l in STEER_LAYERS:
        norm = directions[l].norm().item()
        directions_norm[l] = directions[l] / (norm + 1e-10)
        print(f"  방향벡터 L{l}: norm={norm:.2f}  → 정규화 후 norm="
              f"{directions_norm[l].norm().item():.4f}", flush=True)

    return baselines, directions, directions_norm


# ── Phase 2: 전략 평가 ────────────────────────────────────────────────────────
def phase2(
    model: HookedTransformer,
    kc_cases: list,
    baselines: list,
    mean_vals: Dict[int, torch.Tensor],
    directions_norm: Dict[int, torch.Tensor],
    n_cases: int,
) -> list:
    """
    전략 A  : hook_z L17H3 ablation
    전략 B1/B2/B3 : hook_resid_post steering (정규화 벡터, 8 alphas)

    per-case result keys:
      cor, ctype, baseline_diff, clean_top,
      ablated_diff_A, clean_top_A,
      ablated_diff_B[cond][a_str], clean_top_B[cond][a_str]
        cond ∈ {B1, B2, B3}
    """
    n_B_passes = len(B_CONFIGS) * len(ALPHAS) * 2   # conflict + clean
    print(f"\n[Phase 2] 전략 평가 ({n_cases}개, "
          f"전략A×2 + 전략B×{n_B_passes} = {2 + n_B_passes} pass/case) ...",
          flush=True)

    dev   = next(model.parameters()).device
    dtype = torch.bfloat16

    mean_h17 = mean_vals[TARGET_LAYER][TARGET_HEAD].to(dev, dtype=dtype)
    dir_gpu: Dict[int, torch.Tensor] = {
        l: directions_norm[l].to(dev, dtype=dtype) for l in STEER_LAYERS
    }

    results: list = []
    t0 = time.time()

    for i, (case, base) in enumerate(zip(kc_cases[:n_cases], baselines)):
        ctx  = case["conflict_context"]
        q    = case["question"]
        opts = case["options"]
        cor  = base["cor"]
        rep  = base["rep"]

        neutral_p    = build_neutral_prompt(q, opts)
        conflict_p   = build_conflict_prompt(ctx, q, opts)
        neutral_ids  = model.to_tokens(neutral_p,  prepend_bos=True)[0]
        conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]

        fp_n = int(neutral_ids.shape[0]) - 1
        fp_c = int(conflict_ids.shape[0]) - 1

        # ── 전략 A ────────────────────────────────────────────────────────────
        def make_ablate_z(fp: int, hi: int, mv: torch.Tensor):
            def _fn(val: torch.Tensor, hook) -> torch.Tensor:
                val[0, fp, hi, :] = mv
                return val
            return _fn

        probs_Ac = run_fwd(model, conflict_ids,
                           [(f"blocks.{TARGET_LAYER}.attn.hook_z",
                             make_ablate_z(fp_c, TARGET_HEAD, mean_h17))])
        ablated_diff_A = get_letter_prob(probs_Ac, rep) - get_letter_prob(probs_Ac, cor)

        probs_An    = run_fwd(model, neutral_ids,
                              [(f"blocks.{TARGET_LAYER}.attn.hook_z",
                                make_ablate_z(fp_n, TARGET_HEAD, mean_h17))])
        clean_top_A = get_top_prediction(probs_An)

        # ── 전략 B (정규화 벡터) ──────────────────────────────────────────────
        ablated_diff_B: Dict[str, Dict[str, float]] = {c: {} for c in B_CONFIGS}
        clean_top_B:   Dict[str, Dict[str, str]]  = {c: {} for c in B_CONFIGS}

        for cond, layers in B_CONFIGS.items():
            for a_str, alpha in zip(ALPHA_STRS, ALPHAS):
                def make_steer(fp: int, l_list: List[int], a: float):
                    def _fn_factory(l: int):
                        d = dir_gpu[l]
                        def _fn(val: torch.Tensor, hook) -> torch.Tensor:
                            val[0, fp, :] = val[0, fp, :] + a * d
                            return val
                        return _fn
                    return [(f"blocks.{l}.hook_resid_post", _fn_factory(l))
                            for l in l_list]

                probs_Bc = run_fwd(model, conflict_ids, make_steer(fp_c, layers, alpha))
                ablated_diff_B[cond][a_str] = (
                    get_letter_prob(probs_Bc, rep) - get_letter_prob(probs_Bc, cor)
                )

                probs_Bn = run_fwd(model, neutral_ids, make_steer(fp_n, layers, alpha))
                clean_top_B[cond][a_str] = get_top_prediction(probs_Bn)

        results.append({
            "cor":            cor,
            "ctype":          base["ctype"],
            "baseline_diff":  base["baseline_diff"],
            "clean_top":      base["clean_top"],
            "ablated_diff_A": ablated_diff_A,
            "clean_top_A":    clean_top_A,
            "ablated_diff_B": ablated_diff_B,
            "clean_top_B":    clean_top_B,
        })

        if (i + 1) % 50 == 0:
            print(f"  phase2 {i+1}/{n_cases}  ({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            ckpt = OUT_DIR / "phase2_checkpoint.json"
            with ckpt.open("w") as f:
                json.dump({"processed": i + 1, "partial": results}, f)
            gc.collect()
            torch.cuda.empty_cache()

    del mean_h17, dir_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ── 집계 ─────────────────────────────────────────────────────────────────────
def _sup_rate(mean_base: float, mean_abl: float) -> float:
    if abs(mean_base) < 1e-9:
        return 0.0
    return (mean_base - mean_abl) / mean_base


def _tradeoff_score(sup_pct: float, ca_pct: float) -> float:
    """KC억제율 - (100 - 일반성능유지율). 양수일수록 좋음."""
    return sup_pct - (100.0 - ca_pct)


def _type_stats(results: list, key_diff: str, key_top: str = "") -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ct in CTYPES:
        sub = [r for r in results if r["ctype"] == ct]
        if not sub:
            continue
        m_base = float(np.mean([r["baseline_diff"] for r in sub]))
        m_abl  = float(np.mean([r[key_diff] for r in sub]))
        sup    = _sup_rate(m_base, m_abl)
        ca     = float(np.mean([r[key_top] == r["cor"] for r in sub])) if key_top else 0.0
        out[ct] = {
            "kc_suppression_rate": f"{sup*100:.1f}%",
            "clean_accuracy":      f"{ca*100:.1f}%",
            "_sup": sup, "_ca": ca,
        }
    return out


def _type_stats_B(results: list, cond: str, a_str: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ct in CTYPES:
        sub = [r for r in results if r["ctype"] == ct]
        if not sub:
            continue
        m_base = float(np.mean([r["baseline_diff"] for r in sub]))
        m_abl  = float(np.mean([r["ablated_diff_B"][cond][a_str] for r in sub]))
        sup    = _sup_rate(m_base, m_abl)
        ca     = float(np.mean([r["clean_top_B"][cond][a_str] == r["cor"] for r in sub]))
        out[ct] = {
            "kc_suppression_rate": f"{sup*100:.1f}%",
            "clean_accuracy":      f"{ca*100:.1f}%",
            "_sup": sup, "_ca": ca,
        }
    return out


def aggregate(results: list) -> dict:
    n          = len(results)
    mean_base  = float(np.mean([r["baseline_diff"] for r in results]))
    base_clean = float(np.mean([r["clean_top"] == r["cor"] for r in results]))

    # ── 전략 A ────────────────────────────────────────────────────────────────
    mean_A  = float(np.mean([r["ablated_diff_A"] for r in results]))
    sup_A   = _sup_rate(mean_base, mean_A)
    ca_A    = float(np.mean([r["clean_top_A"] == r["cor"] for r in results]))
    score_A = _tradeoff_score(sup_A * 100, ca_A * 100)
    type_A  = _type_stats(results, "ablated_diff_A", "clean_top_A")

    # ── 전략 B ────────────────────────────────────────────────────────────────
    # b_stats[cond][a_str] = {sup, ca, score}
    b_stats: Dict[str, Dict[str, dict]] = {c: {} for c in B_CONFIGS}
    for cond in B_CONFIGS:
        for a_str in ALPHA_STRS:
            m_abl = float(np.mean([r["ablated_diff_B"][cond][a_str] for r in results]))
            sup   = _sup_rate(mean_base, m_abl)
            ca    = float(np.mean(
                [r["clean_top_B"][cond][a_str] == r["cor"] for r in results]))
            score = _tradeoff_score(sup * 100, ca * 100)
            b_stats[cond][a_str] = {"_sup": sup, "_ca": ca, "_score": score}

    # best (cond, alpha): tradeoff_score 최대, 단 clean_acc >= 0.95 * base_clean 우선
    threshold = 0.95 * base_clean
    all_combos = [
        (cond, a_str, b_stats[cond][a_str])
        for cond in B_CONFIGS
        for a_str in ALPHA_STRS
    ]
    eligible = [(c, a, v) for c, a, v in all_combos if v["_ca"] >= threshold]
    candidates = eligible if eligible else all_combos
    best_cond, best_a_str, best_v = max(candidates, key=lambda x: x[2]["_score"])
    best_alpha = float(best_a_str)

    sup_B_best   = best_v["_sup"]
    ca_B_best    = best_v["_ca"]
    score_B_best = best_v["_score"]
    type_B_best  = _type_stats_B(results, best_cond, best_a_str)

    def clean_type(d: Dict[str, dict]) -> Dict[str, dict]:
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in d.items()}

    # alpha_search: per-cond summary
    alpha_search: Dict[str, dict] = {}
    for cond in B_CONFIGS:
        for a_str in ALPHA_STRS:
            key = f"{B_LABELS[cond].strip()}|α={a_str}"
            v   = b_stats[cond][a_str]
            alpha_search[key] = {
                "cond":           cond,
                "alpha":          float(a_str),
                "kc_suppression": f"{v['_sup']*100:.1f}%",
                "clean_acc":      f"{v['_ca']*100:.1f}%",
                "tradeoff_score": round(v["_score"], 2),
            }

    return {
        "n_cases": n,
        "baseline": {
            "kc_suppression_rate": "0.0%",
            "clean_accuracy":      f"{base_clean*100:.1f}%",
        },
        "strategy_A": {
            "kc_suppression_rate": f"{sup_A*100:.1f}%",
            "clean_accuracy":      f"{ca_A*100:.1f}%",
            "tradeoff_score":      round(score_A, 2),
            "by_type":             clean_type(type_A),
        },
        "strategy_B_best": {
            "condition":           best_cond,
            "condition_label":     B_LABELS[best_cond].strip(),
            "best_alpha":          best_alpha,
            "kc_suppression_rate": f"{sup_B_best*100:.1f}%",
            "clean_accuracy":      f"{ca_B_best*100:.1f}%",
            "tradeoff_score":      round(score_B_best, 2),
            "by_type":             clean_type(type_B_best),
        },
        "strategy_B_alpha_search": alpha_search,
        "_internal": {
            "mean_baseline_diff": mean_base,
            "base_clean_acc":     base_clean,
            "sup_A": sup_A, "ca_A": ca_A, "score_A": score_A,
            "b_stats": b_stats,
            "best_cond": best_cond, "best_a_str": best_a_str,
            "sup_B_best": sup_B_best, "ca_B_best": ca_B_best,
        },
    }


# ── 콘솔 출력 ────────────────────────────────────────────────────────────────
def print_summary(summary: dict) -> None:
    bar = "━" * 52
    best = summary["strategy_B_best"]

    print(f"\n{bar}", flush=True)
    print("Phase 3 결과", flush=True)
    print(bar, flush=True)
    print(f"{'':16s}  KC 억제율    일반 성능   score", flush=True)
    print(f"No interv:      "
          f"{summary['baseline']['kc_suppression_rate']:>8s}   "
          f"{summary['baseline']['clean_accuracy']}", flush=True)
    sa = summary["strategy_A"]
    print(f"전략 A:         "
          f"{sa['kc_suppression_rate']:>8s}   "
          f"{sa['clean_accuracy']:>8s}   {sa['tradeoff_score']:+.1f}", flush=True)
    print(f"전략 B 최적:    "
          f"{best['kc_suppression_rate']:>8s}   "
          f"{best['clean_accuracy']:>8s}   {best['tradeoff_score']:+.1f}  "
          f"({best['condition_label']}, α={best['best_alpha']})", flush=True)
    print(bar, flush=True)

    # 전략 B 세부
    print(f"\n{bar}", flush=True)
    print("전략 B 세부 결과 (정규화 벡터)", flush=True)
    print(bar, flush=True)
    print(f"  {'alpha':>5s}  {'조건':>8s}  KC억제율   일반성능   score", flush=True)

    internal   = summary["_internal"]
    b_stats    = internal["b_stats"]
    base_clean = internal["base_clean_acc"] * 100

    for a_str in ALPHA_STRS:
        for cond in B_CONFIGS:
            v     = b_stats[cond][a_str]
            sup_p = v["_sup"] * 100
            ca_p  = v["_ca"]  * 100
            sc    = v["_score"]
            label = B_LABELS[cond]
            marker = " ★" if (cond == internal["best_cond"]
                               and a_str == internal["best_a_str"]) else "  "
            print(f"  α={a_str:<5s} {label}: "
                  f"KC={sup_p:5.1f}%  clean={ca_p:5.1f}%  score={sc:+6.1f}{marker}",
                  flush=True)
        print("", flush=True)

    print(f"최적 조합: α={best['best_alpha']} {best['condition_label']}", flush=True)
    print(f"  KC억제율={best['kc_suppression_rate']}  "
          f"일반성능={best['clean_accuracy']}  "
          f"score={best['tradeoff_score']:+.1f}", flush=True)
    print(bar, flush=True)

    # type별
    for label, strat in [("전략 A", sa), (f"전략 B 최적 ({best['condition_label']})", best)]:
        print(f"\n[{label}  type별]", flush=True)
        for ct, info in strat["by_type"].items():
            print(f"  {ct:18s}: KC억제={info['kc_suppression_rate']:>7s}  "
                  f"clean={info['clean_accuracy']}", flush=True)


# ── 시각화 ───────────────────────────────────────────────────────────────────
def plot_tradeoff(summary: dict) -> None:
    internal   = summary["_internal"]
    base_ca    = internal["base_clean_acc"] * 100
    b_stats    = internal["b_stats"]
    best_cond  = internal["best_cond"]
    best_a_str = internal["best_a_str"]

    fig, ax = plt.subplots(figsize=(10, 7))

    # 기준선
    ax.scatter(0, base_ca, s=140, color="gray", zorder=6, label="No intervention")
    ax.annotate("No interv.", (0, base_ca), textcoords="offset points",
                xytext=(8, -14), fontsize=8, color="gray")

    # 전략 A
    sup_A = internal["sup_A"] * 100
    ca_A  = internal["ca_A"]  * 100
    ax.scatter(sup_A, ca_A, s=160, color="steelblue", marker="D",
               zorder=6, label="전략 A (L17H3 Ablation)")
    ax.annotate("Strategy A", (sup_A, ca_A), textcoords="offset points",
                xytext=(6, 5), fontsize=8, color="steelblue")

    # 전략 B 조건별
    cond_colors  = {"B1": "#e74c3c", "B2": "#e67e22", "B3": "#8e44ad"}
    cond_markers = {"B1": "o", "B2": "s", "B3": "^"}

    for cond in B_CONFIGS:
        sups = [b_stats[cond][a]["_sup"] * 100 for a in ALPHA_STRS]
        cas  = [b_stats[cond][a]["_ca"]  * 100 for a in ALPHA_STRS]
        color  = cond_colors[cond]
        marker = cond_markers[cond]
        ax.plot(sups, cas, color=color, linewidth=1, alpha=0.4, zorder=3)
        for j, (s, c, a) in enumerate(zip(sups, cas, ALPHA_STRS)):
            is_best = (cond == best_cond and a == best_a_str)
            ax.scatter(s, c, s=160 if is_best else 50,
                       color=color, marker="*" if is_best else marker,
                       zorder=5 if is_best else 4, alpha=1.0 if is_best else 0.7)
            if is_best or j in (0, len(ALPHA_STRS) - 1):
                ax.annotate(f"α={a}", (s, c), textcoords="offset points",
                            xytext=(5, 3), fontsize=7, color=color)
        ax.scatter([], [], s=60, color=color, marker=marker,
                   label=f"전략 {cond} ({B_LABELS[cond].strip()})")

    ax.axhline(base_ca * 0.95, color="orange", linestyle="--", linewidth=1,
               alpha=0.7, label="clean acc 95% 기준선")
    ax.set_xlabel("KC 억제율 (%)", fontsize=12)
    ax.set_ylabel("일반 QA 성능 유지율 (%)", fontsize=12)
    ax.set_title("KC 억제 vs 부작용 트레이드오프 (Phase 3 v2)\n"
                 "정규화 방향벡터, alpha=[0.01~0.3]", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "tradeoff_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  저장: {out}", flush=True)


def plot_suppression_by_type(summary: dict) -> None:
    best       = summary["strategy_B_best"]
    best_cond  = best["condition"]
    best_a_str = str(best["best_alpha"])
    internal   = summary["_internal"]
    b_stats    = internal["b_stats"]

    types = list(CTYPES)
    x     = np.arange(len(types))
    width = 0.25

    def get_type_sup(strat_type_dict: dict) -> List[float]:
        return [
            float(strat_type_dict.get(ct, {}).get(
                "kc_suppression_rate", "0%").rstrip("%"))
            for ct in types
        ]

    sup_A = get_type_sup(summary["strategy_A"]["by_type"])
    sup_B = get_type_sup(best["by_type"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, sup_A, width, color="steelblue", alpha=0.85,
           label="전략 A (L17H3 Ablation)")
    ax.bar(x + width / 2, sup_B, width, color="#8e44ad", alpha=0.85,
           label=f"전략 B 최적 ({best['condition_label']}, α={best['best_alpha']})")

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("KC 억제율 (%)", fontsize=12)
    ax.set_title("Type별 KC 억제율 비교 (Phase 3)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for xi, (sa, sb) in enumerate(zip(sup_A, sup_B)):
        ax.text(xi - width / 2, sa + 0.3, f"{sa:.1f}%", ha="center",
                va="bottom", fontsize=8, color="steelblue")
        ax.text(xi + width / 2, sb + 0.3, f"{sb:.1f}%", ha="center",
                va="bottom", fontsize=8, color="#8e44ad")
    fig.tight_layout()

    out = OUT_DIR / "suppression_by_type.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  저장: {out}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with KC_PATH.open() as f:
        kc_cases = json.load(f)

    n_cases = args.test if args.test is not None else len(kc_cases)
    print(f"KC 케이스: {len(kc_cases)}개 (사용: {n_cases}개)", flush=True)

    model = load_model()
    setup_letter_variants(model.tokenizer)
    print(f"cfg: n_heads={model.cfg.n_heads}, d_model={model.cfg.d_model}, "
          f"d_head={model.cfg.d_head}", flush=True)

    if not MEAN_PATH.exists():
        raise FileNotFoundError(
            f"mean_values.pt 없음: {MEAN_PATH}\n"
            "step5_head_ablation.py를 먼저 실행하세요.")
    mean_vals: Dict[int, torch.Tensor] = torch.load(MEAN_PATH, map_location="cpu")
    print(f"mean_values.pt 로드 완료", flush=True)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    baselines, directions, directions_norm = phase1(model, kc_cases, n_cases)

    dir_path = OUT_DIR / "direction_vectors.pt"
    torch.save(
        {f"{l}_raw":  directions[l]      for l in STEER_LAYERS} |
        {f"{l}_norm": directions_norm[l] for l in STEER_LAYERS},
        dir_path,
    )
    print(f"방향벡터 저장: {dir_path}", flush=True)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    results = phase2(model, kc_cases, baselines, mean_vals, directions_norm, n_cases)

    # ── 집계 및 출력 ──────────────────────────────────────────────────────────
    summary = aggregate(results)
    print_summary(summary)

    # ── JSON 저장 (_internal 제외) ────────────────────────────────────────────
    save_obj = {k: v for k, v in summary.items() if k != "_internal"}
    out_json = OUT_DIR / "intervention_results.json"
    with out_json.open("w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"\n결과 저장: {out_json}", flush=True)

    ckpt = OUT_DIR / "phase2_checkpoint.json"
    if ckpt.exists():
        ckpt.unlink()

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print("\n시각화 저장 ...", flush=True)
    plot_tradeoff(summary)
    plot_suppression_by_type(summary)

    print("\n완료!", flush=True)


if __name__ == "__main__":
    main()

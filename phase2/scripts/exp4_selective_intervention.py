#!/usr/bin/env python3
"""
Exp4: Selective Intervention — KC 완화율 + Clean Accuracy

파트 1: Alpha scaling (핵심)  — A × L16, alpha ∈ [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
파트 2: Layer 비교 (보조)    — alpha=1.0 고정, layers ∈ [14, 15, 16, 17, 18, 19]
파트 3: Type별 분해          — alpha=1.0, L16 고정 (파트 1 결과 재사용)

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp4_selective_intervention.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp4_selective_intervention.py --test 20
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp4_selective_intervention.py --skip_layers
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

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
MODEL_NAME     = "google/gemma-2-2b"
KC_PATH        = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH     = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
OUT_DIR        = Path("phase2/results/exp4")

ALPHAS         = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
LAYERS_SWEEP   = [14, 15, 16, 17, 18, 19]
MAIN_LAYER     = 16
CTYPES         = ("misinformation", "temporal", "semantic")
CHECKPOINT_EVERY = 100

# 파일럿 Phase 3 참조값 (L17H3 ablation)
PILOT_PHASE3 = {"kc_suppression": 0.571, "clean_accuracy": 0.272}

# Exp2 pos_ratio 참조 (Figure 2 보조 표시용)
EXP2_POS_RATIO = {14: 0.981, 15: 0.972, 16: 0.981, 17: 0.772, 18: 0.722, 19: 0.709}

LETTER_VARIANTS: Dict[str, List[int]] = {}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",       type=int, default=None, metavar="N",
                   help="빠른 테스트: N개 케이스만 사용")
    p.add_argument("--n_cases",    type=int, default=None)
    p.add_argument("--skip_layers", action="store_true",
                   help="Layer sweep 생략 (alpha sweep만 실행)")
    return p.parse_args()


# ── 모델 ──────────────────────────────────────────────────────────────────────
def load_model() -> HookedTransformer:
    print("모델 로드 (bfloat16) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16
    )
    model.eval()
    return model


def setup_letter_variants(tokenizer) -> None:
    global LETTER_VARIANTS
    for letter in "ABCD":
        ids: List[int] = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.append(toks[-1] if len(toks) > 1 else toks[0])
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
    print(f"[setup] LETTER_VARIANTS={LETTER_VARIANTS}", flush=True)


# ── 프롬프트 빌더 ──────────────────────────────────────────────────────────────
def build_conflict_prompt(ctx: str, q: str, opts: Dict) -> str:
    lines = ["Context:", ctx, "", f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_clean_prompt(q: str, opts: Dict) -> str:
    lines = [f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_supportive_prompt(ctx: str, q: str, opts: Dict,
                             conflict_answer: str, correct_answer: str) -> str:
    sup_ctx = ctx.replace(conflict_answer, correct_answer)
    lines = ["Context:", sup_ctx, "", f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── 로그 확률 ──────────────────────────────────────────────────────────────────
def _logprobs_from_logits(logits: torch.Tensor, cor: str, rep: str):
    lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    lp_cor = max(lp[tid].item() for tid in LETTER_VARIANTS[cor])
    lp_rep = max(lp[tid].item() for tid in LETTER_VARIANTS[rep])
    return lp_cor, lp_rep


def run_logprobs(model: HookedTransformer, tokens: torch.Tensor,
                 cor: str, rep: str):
    with torch.no_grad():
        logits = model(tokens)
    return _logprobs_from_logits(logits, cor, rep)


def run_logprobs_hooked(model: HookedTransformer, tokens: torch.Tensor,
                        cor: str, rep: str, hook_name: str, hook_fn):
    with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        logits = model(tokens)
    return _logprobs_from_logits(logits, cor, rep)


# ── Scale hook ────────────────────────────────────────────────────────────────
def make_scale_hook(direction: torch.Tensor, alpha: float):
    """모든 position에서 direction 성분을 alpha 비율만큼 제거."""
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        v = value.float()
        d = direction.to(value.device).float()
        proj = v @ d                                # [batch, seq_len]
        v    = v - alpha * proj.unsqueeze(-1) * d  # [batch, seq_len, d_model]
        return v.to(value.dtype)
    return hook_fn


# ── 후보 방향 로드 ────────────────────────────────────────────────────────────
def load_direction(cands: dict, layer: int) -> torch.Tensor:
    for entry in cands[layer]:
        if entry["name"] == "A_svd_conf_clean":
            return entry["dir"].float()
    raise KeyError(f"A_svd_conf_clean not found at layer {layer}")


# ── 베이스라인 전처리 ─────────────────────────────────────────────────────────
def prepare_baselines(model: HookedTransformer, kc_cases: list) -> List[Dict]:
    """
    세 가지 프롬프트 토크나이즈 + 베이스라인(무패치) 로그 확률 계산.
    반환값: 케이스별 dict 리스트
    """
    n = len(kc_cases)
    print(f"[baseline] {n}개 케이스 전처리 ...", flush=True)
    t0 = time.time()
    data = []

    for i, case in enumerate(kc_cases):
        ctx = case["conflict_context"]
        q   = case["question"]
        opt = case["options"]
        cor = case["correct_option"]
        rep = case["replace_option"]
        ca  = case["conflict_answer"]
        coa = case["correct_answer"]

        tok_conf  = model.to_tokens(build_conflict_prompt(ctx, q, opt),   prepend_bos=True)
        tok_clean = model.to_tokens(build_clean_prompt(q, opt),            prepend_bos=True)
        tok_sup   = model.to_tokens(build_supportive_prompt(ctx, q, opt, ca, coa), prepend_bos=True)

        lp_cor_conf,  lp_rep_conf  = run_logprobs(model, tok_conf,  cor, rep)
        lp_cor_clean, lp_rep_clean = run_logprobs(model, tok_clean, cor, rep)
        lp_cor_sup,   lp_rep_sup   = run_logprobs(model, tok_sup,   cor, rep)

        data.append({
            "tok_conf":       tok_conf,
            "tok_clean":      tok_clean,
            "tok_sup":        tok_sup,
            "cor":            cor,
            "rep":            rep,
            "conflict_type":  case["conflict_type"],
            "case_id":        case["id"],
            "lp_cor_conf":    lp_cor_conf,
            "lp_rep_conf":    lp_rep_conf,
            "lp_cor_clean":   lp_cor_clean,
            "lp_rep_clean":   lp_rep_clean,
            "lp_cor_sup":     lp_cor_sup,
            "lp_rep_sup":     lp_rep_sup,
            "has_kc":         lp_rep_conf > lp_cor_conf,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [baseline] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    kc_frac   = sum(1 for d in data if d["has_kc"]) / n
    clean_acc = sum(1 for d in data if d["lp_cor_clean"] > d["lp_rep_clean"]) / n
    sup_acc   = sum(1 for d in data if d["lp_cor_sup"]   > d["lp_rep_sup"])   / n
    print(f"[baseline] 완료: kc_fraction={kc_frac:.3f}  "
          f"clean_acc={clean_acc:.3f}  sup_acc={sup_acc:.3f}  ({time.time()-t0:.0f}s)",
          flush=True)
    return data


# ── 조건 평가 ─────────────────────────────────────────────────────────────────
def evaluate_condition(
    model: HookedTransformer,
    case_data: List[Dict],
    hook_fn,
    hook_name: str,
    tag: str = "",
) -> Dict:
    """
    세 가지 지표 측정:
      kc_suppression    = flipped / N  (KC가 있던 케이스 중 correct로 뒤집힌 비율 / N)
      clean_accuracy    = correct / N  (clean prompt에서 correct > replace 비율)
      supportive_accuracy = correct / N (supportive prompt에서 correct > replace 비율)
    """
    n         = len(case_data)
    kc_flip   = 0
    clean_ok  = 0
    sup_ok    = 0
    per_case  = []

    for i, d in enumerate(case_data):
        cor = d["cor"]
        rep = d["rep"]

        # ── KC suppression ──────────────────────────────────────────────────
        kc_flipped = False
        if d["has_kc"]:
            lp_c, lp_r = run_logprobs_hooked(model, d["tok_conf"], cor, rep, hook_name, hook_fn)
            kc_flipped = lp_c > lp_r
            if kc_flipped:
                kc_flip += 1

        # ── Clean accuracy ──────────────────────────────────────────────────
        lp_c_cl, lp_r_cl = run_logprobs_hooked(model, d["tok_clean"], cor, rep, hook_name, hook_fn)
        clean_correct = lp_c_cl > lp_r_cl
        if clean_correct:
            clean_ok += 1

        # ── Supportive accuracy ─────────────────────────────────────────────
        lp_c_sp, lp_r_sp = run_logprobs_hooked(model, d["tok_sup"], cor, rep, hook_name, hook_fn)
        sup_correct = lp_c_sp > lp_r_sp
        if sup_correct:
            sup_ok += 1

        per_case.append({
            "case_id":       d["case_id"],
            "conflict_type": d["conflict_type"],
            "has_kc":        d["has_kc"],
            "kc_flipped":    kc_flipped,
            "clean_correct": clean_correct,
            "sup_correct":   sup_correct,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()
            kc_so_far = sum(1 for dd in case_data[:i+1] if dd["has_kc"])
            print(f"  [{tag}] {i+1}/{n}  kc_flip={kc_flip}/{kc_so_far}  "
                  f"clean={clean_ok/(i+1):.3f}  sup={sup_ok/(i+1):.3f}", flush=True)

    kc_total = sum(1 for d in case_data if d["has_kc"])
    return {
        "kc_suppression":     kc_flip / n,
        "clean_accuracy":     clean_ok / n,
        "supportive_accuracy": sup_ok / n,
        "kc_total":           kc_total,
        "kc_flipped":         kc_flip,
        "per_case":           per_case,
    }


# ── Type별 집계 ────────────────────────────────────────────────────────────────
def aggregate_by_type(per_case: List[Dict]) -> Dict:
    n_total = len(per_case)
    result = {}
    for ct in CTYPES:
        sub = [r for r in per_case if r["conflict_type"] == ct]
        if not sub:
            continue
        n_sub     = len(sub)
        kc_flip   = sum(1 for r in sub if r["has_kc"] and r["kc_flipped"])
        clean_ok  = sum(1 for r in sub if r["clean_correct"])
        sup_ok    = sum(1 for r in sub if r["sup_correct"])
        result[ct] = {
            "n":                   n_sub,
            "kc_suppression":      kc_flip / n_sub,
            "clean_accuracy":      clean_ok / n_sub,
            "supportive_accuracy": sup_ok / n_sub,
        }
    return result


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_alpha_curve(alpha_rows: List[Dict], baseline: Dict, out_dir: Path) -> None:
    """Figure 1: Alpha 값에 따른 세 가지 지표 변화."""
    alphas = [r["alpha"] for r in alpha_rows]
    kc_sup = [r["kc_suppression"]     for r in alpha_rows]
    clean  = [r["clean_accuracy"]     for r in alpha_rows]
    sup    = [r["supportive_accuracy"] for r in alpha_rows]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(alphas, kc_sup, "o-", color="#e74c3c", lw=2.5, ms=8, label="KC Suppression")
    ax.plot(alphas, clean,  "s-", color="#3498db", lw=2.5, ms=8, label="Clean Accuracy")
    ax.plot(alphas, sup,    "^-", color="#2ecc71", lw=2.5, ms=8, label="Supportive Accuracy")

    # 수치 표기 (alpha > 0)
    for a, kc, cl in zip(alphas, kc_sup, clean):
        if a > 0:
            ax.annotate(f"{kc:.3f}", (a, kc), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7.5, color="#c0392b")
            ax.annotate(f"{cl:.3f}", (a, cl), xytext=(0, -13),
                        textcoords="offset points", ha="center", fontsize=7.5, color="#2980b9")

    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Alpha (direction 제거 강도)", fontsize=12)
    ax.set_ylabel("비율", fontsize=12)
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas], fontsize=10)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("KC 완화율 및 일반 QA 성능 방어", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure1_alpha_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def plot_layer_comparison(layer_rows: List[Dict], out_dir: Path) -> None:
    """Figure 2: Layer별 세 가지 지표 (alpha=1.0 고정)."""
    layers = [r["layer"] for r in layer_rows]
    kc_sup = [r["kc_suppression"]     for r in layer_rows]
    clean  = [r["clean_accuracy"]     for r in layer_rows]
    sup    = [r["supportive_accuracy"] for r in layer_rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers, kc_sup, "o-", color="#e74c3c", lw=2.5, ms=9, label="KC Suppression")
    ax.plot(layers, clean,  "s-", color="#3498db", lw=2.5, ms=9, label="Clean Accuracy")
    ax.plot(layers, sup,    "^-", color="#2ecc71", lw=2.5, ms=9, label="Supportive Accuracy")

    ax.axvline(MAIN_LAYER, color="#e67e22", lw=1.5, ls="--", alpha=0.8,
               label=f"L{MAIN_LAYER} (Exp2 권장)")

    # Exp2 pos_ratio 보조 막대 (우측 y축)
    ax2 = ax.twinx()
    ax2.bar(layers, [EXP2_POS_RATIO.get(l, 0) for l in layers],
            alpha=0.13, color="#9b59b6", width=0.65, label="Exp2 pos_ratio")
    ax2.set_ylabel("Exp2 pos_ratio (참조)", color="#9b59b6", fontsize=10)
    ax2.set_ylim(0, 1.6)
    ax2.tick_params(axis="y", colors="#9b59b6")

    ax.set_xlim(layers[0] - 0.8, layers[-1] + 0.8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Patch Layer (A 방향, alpha=1.0)", fontsize=12)
    ax.set_ylabel("비율", fontsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=11)
    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    ax.set_title(
        "Figure 2: Layer Comparison — alpha=1.0 고정\n"
        "막대(보라): Exp2 pos_ratio 참조  |  주황 점선: L16 (Exp2 권장 레이어)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "figure2_layer_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def plot_type_breakdown(type_results: Dict, out_dir: Path) -> None:
    """Figure 3: 충돌 유형별 지표 (alpha=1.0, L16)."""
    types = [ct for ct in CTYPES if ct in type_results]
    if not types:
        return

    x       = np.arange(len(types))
    width   = 0.25
    metrics = [
        ("kc_suppression",     "#e74c3c", "KC Suppression"),
        ("clean_accuracy",     "#3498db", "Clean Accuracy"),
        ("supportive_accuracy","#2ecc71", "Supportive Accuracy"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, (key, color, label) in enumerate(metrics):
        vals = [type_results[ct][key] for ct in types]
        bars = ax.bar(x + (j - 1) * width, vals, width,
                      color=color, alpha=0.82, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{ct}\n(n={type_results[ct]['n']})" for ct in types], fontsize=10
    )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("비율", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        f"Figure 3: 충돌 유형별 분해 — A × L{MAIN_LAYER}, alpha=1.0\n"
        "KC suppression이 충돌 유형에 따라 다른지 확인",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "figure3_type_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def plot_tradeoff_scatter(
    alpha_rows: List[Dict],
    layer_rows: List[Dict],
    out_dir: Path,
) -> None:
    """Figure 4: KC suppression vs Clean accuracy tradeoff space."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Alpha curve (L16)
    kc_a = [r["kc_suppression"] for r in alpha_rows]
    cl_a = [r["clean_accuracy"] for r in alpha_rows]
    ax.plot(kc_a, cl_a, "-", color="#aaa", lw=1.2, alpha=0.7, zorder=1)

    cmap = plt.cm.RdYlGn
    for r in alpha_rows:
        a = r["alpha"]
        ax.scatter(r["kc_suppression"], r["clean_accuracy"],
                   s=130, c=[cmap(a)], zorder=3, edgecolors="black", lw=0.8)
        if a > 0:
            ax.annotate(f"α={a:.1f}",
                        (r["kc_suppression"], r["clean_accuracy"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
        else:
            ax.annotate("baseline",
                        (r["kc_suppression"], r["clean_accuracy"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)

    # Layer sweep (alpha=1.0)
    blues = plt.cm.Blues(np.linspace(0.45, 0.9, len(layer_rows)))
    for r, c in zip(layer_rows, blues):
        ax.scatter(r["kc_suppression"], r["clean_accuracy"],
                   s=150, c=[c], marker="D", zorder=4, edgecolors="navy", lw=0.8)
        ax.annotate(f"L{r['layer']}",
                    (r["kc_suppression"], r["clean_accuracy"]),
                    xytext=(5, -13), textcoords="offset points", fontsize=8, color="navy")

    # Phase 3 참조점
    ax.scatter(PILOT_PHASE3["kc_suppression"], PILOT_PHASE3["clean_accuracy"],
               s=220, c="red", marker="X", zorder=5)
    ax.annotate("Phase 3\n(L17H3)",
                (PILOT_PHASE3["kc_suppression"], PILOT_PHASE3["clean_accuracy"]),
                xytext=(8, 8), textcoords="offset points", fontsize=8.5, color="red")

    # 목표 영역
    ax.fill_between([0.3, 1.0], [0.8, 0.8], [1.05, 1.05],
                    alpha=0.06, color="green", zorder=0, label="목표: KC>30% & clean>80%")

    ax.axhline(0.8, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.3, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("KC Suppression", fontsize=12)
    ax.set_ylabel("Clean Accuracy", fontsize=12)
    ax.grid(alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.0),
               markeredgecolor="black", ms=9, label="원: alpha scaling (L16, 빨강→녹색)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="steelblue",
               markeredgecolor="navy", ms=9, label="다이아몬드: layer sweep (alpha=1.0)"),
        Line2D([0], [0], marker="X", color="red", ms=11, label="X: Phase 3 (L17H3 ablation)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left")
    ax.set_title(
        "Figure 4: KC Suppression vs Clean Accuracy Tradeoff\n"
        "목표 영역 (초록): KC sup > 30% AND clean > 80%",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "figure4_tradeoff_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


# ── 결과 출력 ──────────────────────────────────────────────────────────────────
def print_summary(alpha_rows, layer_rows, type_results, baseline):
    print(f"\n{'='*62}")
    print("  Exp4 결과 요약")
    print(f"{'='*62}")
    print(f"  Baseline : KC_frac={baseline['kc_fraction']:.3f}  "
          f"clean={baseline['clean_accuracy']:.3f}  sup={baseline['supportive_accuracy']:.3f}")
    print(f"  Phase 3  : KC_sup={PILOT_PHASE3['kc_suppression']:.3f}  "
          f"clean={PILOT_PHASE3['clean_accuracy']:.3f}")
    print()
    print(f"  [Alpha sweep — A × L{MAIN_LAYER}]")
    print(f"  {'alpha':>6} {'KC_sup':>9} {'clean':>9} {'sup_acc':>9}")
    for r in alpha_rows:
        flag = " ← Phase3 비교" if abs(r["alpha"] - 1.0) < 1e-6 else ""
        print(f"  {r['alpha']:>6.1f} {r['kc_suppression']:>9.3f} "
              f"{r['clean_accuracy']:>9.3f} {r['supportive_accuracy']:>9.3f}{flag}")

    if layer_rows:
        print()
        print(f"  [Layer sweep — alpha=1.0]")
        print(f"  {'layer':>7} {'KC_sup':>9} {'clean':>9} {'sup_acc':>9}")
        for r in layer_rows:
            flag = " ← main" if r["layer"] == MAIN_LAYER else ""
            print(f"  L{r['layer']:<6} {r['kc_suppression']:>9.3f} "
                  f"{r['clean_accuracy']:>9.3f} {r['supportive_accuracy']:>9.3f}{flag}")

    print()
    print(f"  [Type breakdown — A × L{MAIN_LAYER}, alpha=1.0]")
    print(f"  {'type':>15} {'n':>5} {'KC_sup':>9} {'clean':>9} {'sup_acc':>9}")
    for ct, info in type_results.items():
        print(f"  {ct:>15} {info['n']:>5} {info['kc_suppression']:>9.3f} "
              f"{info['clean_accuracy']:>9.3f} {info['supportive_accuracy']:>9.3f}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kc_cases = json.load(open(KC_PATH))
    n_cases  = args.test or args.n_cases or len(kc_cases)
    kc_cases = kc_cases[:n_cases]
    print(f"[exp4] N_CASES={n_cases}  skip_layers={args.skip_layers}", flush=True)

    cands = torch.load(CANDS_PATH, map_location="cpu")
    model = load_model()
    setup_letter_variants(model.tokenizer)
    device = next(model.parameters()).device

    t0 = time.time()

    # ── Phase 0: 베이스라인 전처리 ───────────────────────────────────────────
    case_data = prepare_baselines(model, kc_cases)
    n = len(case_data)
    kc_total_base = sum(1 for d in case_data if d["has_kc"])
    clean_acc_base = sum(1 for d in case_data if d["lp_cor_clean"] > d["lp_rep_clean"]) / n
    sup_acc_base   = sum(1 for d in case_data if d["lp_cor_sup"]   > d["lp_rep_sup"])   / n

    baseline = {
        "kc_suppression":      0.0,
        "kc_fraction":         kc_total_base / n,
        "clean_accuracy":      clean_acc_base,
        "supportive_accuracy": sup_acc_base,
    }

    # ── Phase 1: Alpha sweep at MAIN_LAYER ──────────────────────────────────
    alpha_rows = [
        {"alpha": 0.0,
         "kc_suppression": 0.0,
         "clean_accuracy": clean_acc_base,
         "supportive_accuracy": sup_acc_base,
         "kc_total": kc_total_base,
         "kc_flipped": 0,
         "per_case": None}
    ]

    for alpha in ALPHAS[1:]:
        dir_A     = load_direction(cands, MAIN_LAYER).to(device)
        hook_fn   = make_scale_hook(dir_A, alpha)
        hook_name = f"blocks.{MAIN_LAYER}.hook_resid_post"

        print(f"\n[alpha={alpha:.1f}, L{MAIN_LAYER}] 평가 중 ...", flush=True)
        result = evaluate_condition(model, case_data, hook_fn, hook_name,
                                    tag=f"a={alpha:.1f},L{MAIN_LAYER}")
        alpha_rows.append({"alpha": alpha, **result})
        print(f"  → KC_sup={result['kc_suppression']:.3f}  "
              f"clean={result['clean_accuracy']:.3f}  "
              f"sup={result['supportive_accuracy']:.3f}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    # ── Phase 2: Layer sweep (alpha=1.0) ────────────────────────────────────
    layer_rows: List[Dict] = []

    if not args.skip_layers:
        for layer in LAYERS_SWEEP:
            dir_A     = load_direction(cands, layer).to(device)
            hook_fn   = make_scale_hook(dir_A, alpha=1.0)
            hook_name = f"blocks.{layer}.hook_resid_post"

            print(f"\n[L{layer}, alpha=1.0] 평가 중 ...", flush=True)
            result = evaluate_condition(model, case_data, hook_fn, hook_name,
                                        tag=f"L{layer},a=1.0")
            layer_rows.append({"layer": layer, **result})
            print(f"  → KC_sup={result['kc_suppression']:.3f}  "
                  f"clean={result['clean_accuracy']:.3f}  "
                  f"sup={result['supportive_accuracy']:.3f}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # alpha=1.0 at MAIN_LAYER 결과 재사용
        main_a1 = next(r for r in alpha_rows if r["alpha"] == 1.0)
        layer_rows = [{"layer": MAIN_LAYER,
                       "kc_suppression":     main_a1["kc_suppression"],
                       "clean_accuracy":     main_a1["clean_accuracy"],
                       "supportive_accuracy":main_a1["supportive_accuracy"],
                       "kc_total":           main_a1["kc_total"],
                       "kc_flipped":         main_a1["kc_flipped"]}]

    # ── Phase 3: Type별 분해 (alpha=1.0 at MAIN_LAYER per_case 재사용) ──────
    main_a1_row  = next(r for r in alpha_rows if r["alpha"] == 1.0)
    type_results = aggregate_by_type(main_a1_row["per_case"])

    # ── 저장 ─────────────────────────────────────────────────────────────────
    def _strip(row):
        return {k: v for k, v in row.items() if k != "per_case"}

    with open(OUT_DIR / "alpha_results.json", "w") as f:
        json.dump({
            "baseline":     baseline,
            "alpha_results": [_strip(r) for r in alpha_rows],
            "pilot_phase3":  PILOT_PHASE3,
        }, f, indent=2)

    with open(OUT_DIR / "layer_results.json", "w") as f:
        json.dump([_strip(r) for r in layer_rows], f, indent=2)

    with open(OUT_DIR / "type_results.json", "w") as f:
        json.dump(type_results, f, indent=2)

    print("\n[save] alpha_results.json, layer_results.json, type_results.json")

    print_summary(alpha_rows, layer_rows, type_results, baseline)

    # ── 시각화 ───────────────────────────────────────────────────────────────
    print("\n[exp4] 시각화 ...", flush=True)
    plot_alpha_curve(alpha_rows, baseline, OUT_DIR)
    if layer_rows:
        plot_layer_comparison(layer_rows, OUT_DIR)
    plot_type_breakdown(type_results, OUT_DIR)
    print(f"\n[exp4] 완료 ({time.time()-t0:.0f}s). 결과: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

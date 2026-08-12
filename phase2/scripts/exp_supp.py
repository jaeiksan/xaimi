#!/usr/bin/env python3
"""
Exp Supp: 보충 실험 3종

Exp1: Logit Tracking (A × L16)
  — delta_cor_conf, delta_rep_conf, delta_cor_sup, delta_rep_sup 측정
  — IE_kc와 delta_rep_conf의 관계, supportive vs conflict 비교

Exp2: KC Severity Analysis
  — kc_severity 구간별 IE + rank_flip 비율
  — r(severity, IE) 상관계수

Exp3: Multi-layer Patch Comparison
  — L16_only / L19_only / L16+L19 / L16+L17+L19 구성 비교

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp_supp.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp_supp.py --test 20
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp_supp.py --exp 1
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp_supp.py --exp 2 3
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
import matplotlib.font_manager as fm
import numpy as np
import torch
from scipy import stats
from transformer_lens import HookedTransformer

_fp = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
matplotlib.rcParams["font.family"] = _fp.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/gemma-2-2b"
KC_PATH      = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH   = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
OUT_DIR      = Path("phase2/results/exp_supp")

PATCH_LAYER  = 16   # A 방향 기본 레이어 (Exp1, Exp2)
CHECKPOINT_EVERY = 100

# Exp2 severity bins
SEVERITY_BINS: List[Tuple[float, float, str]] = [
    (0.0,   1.0,  "low"),
    (1.0,   2.0,  "medium"),
    (2.0,   3.5,  "high"),
    (3.5, 999.0,  "severe"),
]

# Exp3 multi-layer configs: {config_name: [(layer, alpha), ...]}
MULTI_CONFIGS: Dict[str, List[Tuple[int, float]]] = {
    "L16_only":    [(16, 1.0)],
    "L19_only":    [(19, 1.0)],
    "L16_L19":     [(16, 1.0), (19, 1.0)],
    "L16_L17_L19": [(16, 1.0), (17, 1.0), (19, 1.0)],
}

LETTER_VARIANTS: Dict[str, List[int]] = {}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",    type=int, default=None, metavar="N",
                   help="빠른 테스트: N개 케이스만 사용")
    p.add_argument("--n_cases", type=int, default=None)
    p.add_argument("--exp",     type=int, nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3], metavar="N",
                   help="실행할 실험 번호 (기본: 1 2 3)")
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
                         cor: str, rep: str,
                         fwd_hooks: List[Tuple]):
    with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)
    return _logprobs_from_logits(logits, cor, rep)


# ── Scale hook ────────────────────────────────────────────────────────────────
def make_scale_hook(direction: torch.Tensor, alpha: float):
    """모든 position에서 direction 성분을 alpha 비율만큼 제거."""
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        v = value.float()
        d = direction.to(value.device).float()
        proj = v @ d
        v    = v - alpha * proj.unsqueeze(-1) * d
        return v.to(value.dtype)
    return hook_fn


# ── 후보 방향 로드 ────────────────────────────────────────────────────────────
def load_direction(cands: dict, layer: int,
                   name: str = "A_svd_conf_clean") -> torch.Tensor:
    for entry in cands[layer]:
        if entry["name"] == name:
            return entry["dir"].float()
    raise KeyError(f"{name} not found at layer {layer}")


# ── 베이스라인 전처리 ─────────────────────────────────────────────────────────
def prepare_baselines(model: HookedTransformer, kc_cases: list) -> List[Dict]:
    """세 가지 프롬프트 토크나이즈 + 베이스라인 로그 확률 계산."""
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

        tok_conf  = model.to_tokens(build_conflict_prompt(ctx, q, opt),
                                    prepend_bos=True)
        tok_clean = model.to_tokens(build_clean_prompt(q, opt),
                                    prepend_bos=True)
        tok_sup   = model.to_tokens(
            build_supportive_prompt(ctx, q, opt, ca, coa), prepend_bos=True
        )

        lp_cor_conf,  lp_rep_conf  = run_logprobs(model, tok_conf,  cor, rep)
        lp_cor_clean, lp_rep_clean = run_logprobs(model, tok_clean, cor, rep)
        lp_cor_sup,   lp_rep_sup   = run_logprobs(model, tok_sup,   cor, rep)

        data.append({
            "tok_conf":      tok_conf,
            "tok_clean":     tok_clean,
            "tok_sup":       tok_sup,
            "cor":           cor,
            "rep":           rep,
            "conflict_type": case["conflict_type"],
            "case_id":       case["id"],
            "kc_severity":   case["kc_severity"],
            "lp_cor_conf":   lp_cor_conf,
            "lp_rep_conf":   lp_rep_conf,
            "lp_cor_clean":  lp_cor_clean,
            "lp_rep_clean":  lp_rep_clean,
            "lp_cor_sup":    lp_cor_sup,
            "lp_rep_sup":    lp_rep_sup,
            "has_kc":        lp_rep_conf > lp_cor_conf,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [baseline] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    kc_frac   = sum(1 for d in data if d["has_kc"]) / n
    clean_acc = sum(1 for d in data if d["lp_cor_clean"] > d["lp_rep_clean"]) / n
    sup_acc   = sum(1 for d in data if d["lp_cor_sup"]   > d["lp_rep_sup"])   / n
    print(f"[baseline] 완료: kc_frac={kc_frac:.3f}  "
          f"clean_acc={clean_acc:.3f}  sup_acc={sup_acc:.3f}  ({time.time()-t0:.0f}s)",
          flush=True)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Logit Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp1(
    model: HookedTransformer,
    case_data: List[Dict],
    dir_A: torch.Tensor,
    alpha: float = 1.0,
) -> List[Dict]:
    """
    A × L{PATCH_LAYER} 패치 후 4가지 delta 측정:
      delta_cor_conf = lp_cor_conf_patched - lp_cor_conf_baseline  (IE_kc)
      delta_rep_conf = lp_rep_conf_patched - lp_rep_conf_baseline
      delta_cor_sup  = lp_cor_sup_patched  - lp_cor_sup_baseline   (IE_sup)
      delta_rep_sup  = lp_rep_sup_patched  - lp_rep_sup_baseline
    """
    hook_name = f"blocks.{PATCH_LAYER}.hook_resid_post"
    hook_fn   = make_scale_hook(dir_A, alpha)
    fwd_hooks = [(hook_name, hook_fn)]

    n = len(case_data)
    print(f"\n[Exp1] Logit Tracking (A × L{PATCH_LAYER}, alpha={alpha}) — {n}개 케이스",
          flush=True)
    t0 = time.time()
    results = []

    for i, d in enumerate(case_data):
        cor = d["cor"]
        rep = d["rep"]

        lp_c_conf_p, lp_r_conf_p = run_logprobs_hooked(
            model, d["tok_conf"], cor, rep, fwd_hooks)
        lp_c_sup_p,  lp_r_sup_p  = run_logprobs_hooked(
            model, d["tok_sup"],  cor, rep, fwd_hooks)

        results.append({
            "case_id":        d["case_id"],
            "conflict_type":  d["conflict_type"],
            "kc_severity":    d["kc_severity"],
            "has_kc":         d["has_kc"],
            "delta_cor_conf": lp_c_conf_p - d["lp_cor_conf"],
            "delta_rep_conf": lp_r_conf_p - d["lp_rep_conf"],
            "delta_cor_sup":  lp_c_sup_p  - d["lp_cor_sup"],
            "delta_rep_sup":  lp_r_sup_p  - d["lp_rep_sup"],
            "kc_flipped":     d["has_kc"] and (lp_c_conf_p > lp_r_conf_p),
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  [Exp1] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    return results


def summarize_exp1(results: List[Dict]) -> Dict:
    keys = ["delta_cor_conf", "delta_rep_conf", "delta_cor_sup", "delta_rep_sup"]

    def _stats(vals):
        arr = np.array(vals)
        return {
            "mean":      float(arr.mean()),
            "std":       float(arr.std()),
            "median":    float(np.median(arr)),
            "pos_ratio": float((arr > 0).mean()),
        }

    summary = {k: _stats([r[k] for r in results]) for k in keys}

    # IE_kc (delta_cor_conf) vs delta_rep_conf 상관
    ie_kc   = np.array([r["delta_cor_conf"] for r in results])
    d_rep   = np.array([r["delta_rep_conf"] for r in results])
    r_val, p_val = stats.pearsonr(ie_kc, d_rep)
    summary["r_IE_vs_delta_rep"] = float(r_val)
    summary["p_IE_vs_delta_rep"] = float(p_val)

    # 전체 dissociation (IE_kc > IE_sup)
    n_disso = sum(1 for r in results if r["delta_cor_conf"] > r["delta_cor_sup"])
    summary["dissociation_ratio"] = n_disso / len(results)

    # kc_flipped
    summary["kc_flipped"]     = sum(1 for r in results if r["kc_flipped"])
    summary["kc_suppression"] = summary["kc_flipped"] / len(results)

    # 충돌 유형별
    by_type = {}
    for ct in ("misinformation", "temporal", "semantic"):
        sub = [r for r in results if r["conflict_type"] == ct]
        if not sub:
            continue
        by_type[ct] = {
            "n":              len(sub),
            "delta_cor_conf": float(np.mean([r["delta_cor_conf"] for r in sub])),
            "delta_cor_sup":  float(np.mean([r["delta_cor_sup"]  for r in sub])),
            "delta_rep_conf": float(np.mean([r["delta_rep_conf"] for r in sub])),
        }
    summary["by_type"] = by_type

    return summary


def plot_exp1(results: List[Dict], summary: Dict, out_dir: Path) -> None:
    # Figure 1: 4-delta bar chart
    keys   = ["delta_cor_conf", "delta_rep_conf", "delta_cor_sup", "delta_rep_sup"]
    labels = ["Δcor_conflict\n(IE_kc)", "Δrep_conflict", "Δcor_supp\n(IE_sup)", "Δrep_supp"]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    means  = [summary[k]["mean"] for k in keys]
    stds   = [summary[k]["std"]  for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, color=colors, alpha=0.82,
                  yerr=stds, capsize=6, error_kw={"linewidth": 1.5})
    ax.axhline(0, color="black", lw=0.8, ls="--")
    for bar, m in zip(bars, means):
        yt = bar.get_height() + (0.03 if m >= 0 else -0.07)
        ax.text(bar.get_x() + bar.get_width() / 2, yt, f"{m:+.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Delta log-prob (평균 ± std)", fontsize=12)
    ax.set_title(
        f"Exp Supp-1 Figure 1: Logit Tracking — A × L{PATCH_LAYER}\n"
        f"dissociation={summary['dissociation_ratio']:.3f}  "
        f"KC_sup={summary['kc_suppression']:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp1_figure1_delta_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: scatter IE_kc vs delta_rep_conf
    ie_vals  = [r["delta_cor_conf"] for r in results]
    rep_vals = [r["delta_rep_conf"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ie_vals, rep_vals, s=10, alpha=0.35, color="#555", linewidths=0)
    # regression line
    slope, intercept, rv, pv, _ = stats.linregress(ie_vals, rep_vals)
    x_line = np.linspace(min(ie_vals), max(ie_vals), 200)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=2,
            label=f"r={rv:.3f}  p={pv:.3f}")
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("IE_kc (Δcor_conflict)", fontsize=12)
    ax.set_ylabel("Δrep_conflict", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(
        "Exp Supp-1 Figure 2: IE_kc vs Δrep_conflict\n"
        "correct 상승이 replace 하락을 동반하는지",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / "exp1_figure2_scatter_IE_vs_rep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: KC vs Supportive dissociation (IE_kc vs IE_sup scatter)
    ie_sup = [r["delta_cor_sup"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ie_vals, ie_sup, s=10, alpha=0.35, color="#555", linewidths=0)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("IE_kc (Δcor_conflict)", fontsize=12)
    ax.set_ylabel("IE_sup (Δcor_supp)", fontsize=12)
    ax.set_title(
        f"Exp Supp-1 Figure 3: Double Dissociation Scatter\n"
        f"dissociation={summary['dissociation_ratio']:.3f} "
        f"(좌상 영역 = IE_kc>0 & IE_sup<0)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / "exp1_figure3_dissociation_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Exp1] 그림 저장 완료 → {out_dir}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: KC Severity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp2(
    model: HookedTransformer,
    case_data: List[Dict],
    dir_A: torch.Tensor,
    alpha: float = 1.0,
) -> List[Dict]:
    """
    케이스별 IE_kc + rank_flip 측정. (severity는 case_data에서 바로 사용)
    """
    hook_name = f"blocks.{PATCH_LAYER}.hook_resid_post"
    hook_fn   = make_scale_hook(dir_A, alpha)
    fwd_hooks = [(hook_name, hook_fn)]

    n = len(case_data)
    print(f"\n[Exp2] KC Severity Analysis (A × L{PATCH_LAYER}, alpha={alpha}) — {n}개 케이스",
          flush=True)
    t0 = time.time()
    results = []

    for i, d in enumerate(case_data):
        cor = d["cor"]
        rep = d["rep"]

        lp_c_p, lp_r_p = run_logprobs_hooked(
            model, d["tok_conf"], cor, rep, fwd_hooks)

        ie_kc      = lp_c_p - d["lp_cor_conf"]
        kc_flipped = d["has_kc"] and (lp_c_p > lp_r_p)

        results.append({
            "case_id":       d["case_id"],
            "conflict_type": d["conflict_type"],
            "kc_severity":   d["kc_severity"],
            "has_kc":        d["has_kc"],
            "ie_kc":         ie_kc,
            "kc_flipped":    kc_flipped,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  [Exp2] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    return results


def summarize_exp2(results: List[Dict]) -> Dict:
    sev_arr = np.array([r["kc_severity"] for r in results])
    ie_arr  = np.array([r["ie_kc"]       for r in results])

    # 전체 상관
    r_val, p_val = stats.pearsonr(sev_arr, ie_arr)
    r_sp, p_sp   = stats.spearmanr(sev_arr, ie_arr)

    summary = {
        "n":           len(results),
        "r_pearson":   float(r_val),
        "p_pearson":   float(p_val),
        "r_spearman":  float(r_sp),
        "p_spearman":  float(p_sp),
        "ie_kc_mean":  float(ie_arr.mean()),
        "ie_kc_std":   float(ie_arr.std()),
    }

    # 구간별 집계
    bins_summary = {}
    for lo, hi, label in SEVERITY_BINS:
        sub = [r for r in results if lo <= r["kc_severity"] < hi]
        if not sub:
            bins_summary[label] = {"n": 0}
            continue
        ie_sub = np.array([r["ie_kc"] for r in sub])
        sev_sub = np.array([r["kc_severity"] for r in sub])
        kc_flip = sum(1 for r in sub if r["kc_flipped"])
        n_kc    = sum(1 for r in sub if r["has_kc"])
        bins_summary[label] = {
            "n":              len(sub),
            "sev_range":      [lo, hi if hi < 999 else None],
            "sev_mean":       float(sev_sub.mean()),
            "ie_kc_mean":     float(ie_sub.mean()),
            "ie_kc_std":      float(ie_sub.std()),
            "pos_ratio":      float((ie_sub > 0).mean()),
            "kc_flipped":     kc_flip,
            "kc_total":       n_kc,
            "flip_rate":      kc_flip / len(sub),
        }
    summary["bins"] = bins_summary

    return summary


def plot_exp2(results: List[Dict], summary: Dict, out_dir: Path) -> None:
    # Figure 1: severity bin별 mean_IE + flip_rate (dual axis)
    bin_labels = [label for _, _, label in SEVERITY_BINS
                  if summary["bins"].get(label, {}).get("n", 0) > 0]
    ie_means = [summary["bins"][lb]["ie_kc_mean"]  for lb in bin_labels]
    ie_stds  = [summary["bins"][lb]["ie_kc_std"]   for lb in bin_labels]
    flip_rates = [summary["bins"][lb]["flip_rate"] for lb in bin_labels]
    ns       = [summary["bins"][lb]["n"]           for lb in bin_labels]

    x = np.arange(len(bin_labels))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x - 0.18, ie_means, 0.34, color="#e74c3c", alpha=0.78,
                   yerr=ie_stds, capsize=5, label="IE_kc mean")
    ax2.bar(x + 0.18, flip_rates, 0.34, color="#3498db", alpha=0.68, label="Flip rate")

    ax1.axhline(0, color="k", lw=0.7, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{lb}\n(n={ns[i]})" for i, lb in enumerate(bin_labels)], fontsize=10
    )
    ax1.set_ylabel("IE_kc (Δlog-prob, 빨강)", fontsize=11, color="#c0392b")
    ax2.set_ylabel("Rank-flip rate (파랑)", fontsize=11, color="#2980b9")
    ax1.tick_params(axis="y", colors="#c0392b")
    ax2.tick_params(axis="y", colors="#2980b9")

    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbs1 + lbs2, fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_title(
        f"Exp Supp-2 Figure 1: Severity Bin별 IE_kc + Flip Rate\n"
        f"r_pearson={summary['r_pearson']:.3f} (p={summary['p_pearson']:.3f})  "
        f"r_spearman={summary['r_spearman']:.3f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "exp2_figure1_severity_bins.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: scatter kc_severity vs IE_kc with regression
    sev_vals = [r["kc_severity"] for r in results]
    ie_vals  = [r["ie_kc"]       for r in results]

    fig, ax = plt.subplots(figsize=(7, 6))

    # color by flip status
    colors_pt = ["#e74c3c" if r["kc_flipped"] else "#aaa" for r in results]
    ax.scatter(sev_vals, ie_vals, s=12, c=colors_pt, alpha=0.5, linewidths=0)

    slope, intercept, rv, pv, _ = stats.linregress(sev_vals, ie_vals)
    x_line = np.linspace(min(sev_vals), max(sev_vals), 200)
    ax.plot(x_line, slope * x_line + intercept, "k-", lw=2,
            label=f"r={rv:.3f}  p={pv:.3f}")
    ax.axhline(0, color="k", lw=0.7, ls="--")

    # bin boundaries
    for lo, hi, label in SEVERITY_BINS:
        if 0 < lo < 9:
            ax.axvline(lo, color="gray", lw=0.8, ls=":", alpha=0.6)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="k", lw=2, label=f"회귀선 r={rv:.3f}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               ms=8, label="rank flip (빨강)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaa",
               ms=8, label="no flip (회색)"),
    ], fontsize=9)
    ax.set_xlabel("KC Severity (초기 logit gap)", fontsize=12)
    ax.set_ylabel("IE_kc (Δcor_conflict)", fontsize=12)
    ax.set_title(
        "Exp Supp-2 Figure 2: Severity vs IE_kc Scatter\n"
        "심각도가 높을수록 IE도 크지만 rank flip은 어려운가?",
        fontsize=11, fontweight="bold",
    )
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / "exp2_figure2_scatter_severity_IE.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Exp2] 그림 저장 완료 → {out_dir}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Multi-layer Patch Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp3(
    model: HookedTransformer,
    case_data: List[Dict],
    cands: dict,
) -> Dict[str, Dict]:
    """
    4가지 multi-layer config에 대해 kc_suppression, clean_accuracy, sup_accuracy 측정.
    """
    results: Dict[str, Dict] = {}
    n = len(case_data)

    for config_name, layer_alpha_list in MULTI_CONFIGS.items():
        # 이 config의 hooks 준비
        fwd_hooks = []
        for layer, alpha in layer_alpha_list:
            dir_v    = load_direction(cands, layer)
            hook_nm  = f"blocks.{layer}.hook_resid_post"
            hook_fn  = make_scale_hook(dir_v, alpha)
            fwd_hooks.append((hook_nm, hook_fn))

        layer_str = "+".join(f"L{l}(α{a})" for l, a in layer_alpha_list)
        print(f"\n[Exp3] Config={config_name}  [{layer_str}]  — {n}개 케이스",
              flush=True)
        t0 = time.time()

        kc_flip  = 0
        clean_ok = 0
        sup_ok   = 0
        per_case = []

        for i, d in enumerate(case_data):
            cor = d["cor"]
            rep = d["rep"]

            kc_flipped = False
            if d["has_kc"]:
                lp_c, lp_r = run_logprobs_hooked(
                    model, d["tok_conf"], cor, rep, fwd_hooks)
                kc_flipped = lp_c > lp_r
                if kc_flipped:
                    kc_flip += 1

            lp_c_cl, lp_r_cl = run_logprobs_hooked(
                model, d["tok_clean"], cor, rep, fwd_hooks)
            clean_correct = lp_c_cl > lp_r_cl
            if clean_correct:
                clean_ok += 1

            lp_c_sp, lp_r_sp = run_logprobs_hooked(
                model, d["tok_sup"], cor, rep, fwd_hooks)
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
                print(f"  [{config_name}] {i+1}/{n}  kc_flip={kc_flip}/{kc_so_far}  "
                      f"clean={clean_ok/(i+1):.3f}  sup={sup_ok/(i+1):.3f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        kc_total = sum(1 for d in case_data if d["has_kc"])
        results[config_name] = {
            "kc_suppression":      kc_flip / n,
            "clean_accuracy":      clean_ok / n,
            "supportive_accuracy": sup_ok / n,
            "kc_total":            kc_total,
            "kc_flipped":          kc_flip,
            "layers":              [l for l, _ in layer_alpha_list],
            "alphas":              [a for _, a in layer_alpha_list],
        }
        print(f"  [{config_name}] 완료: "
              f"kc_sup={kc_flip/n:.4f}  "
              f"clean={clean_ok/n:.3f}  "
              f"sup={sup_ok/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    return results


def plot_exp3(config_results: Dict[str, Dict], out_dir: Path) -> None:
    # Figure 1: grouped bar chart (4 configs × 3 metrics)
    configs = list(MULTI_CONFIGS.keys())
    metrics = [
        ("kc_suppression",     "#e74c3c", "KC Suppression"),
        ("clean_accuracy",     "#3498db", "Clean Accuracy"),
        ("supportive_accuracy","#2ecc71", "Supportive Accuracy"),
    ]

    x     = np.arange(len(configs))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (key, color, label) in enumerate(metrics):
        vals = [config_results[c][key] for c in configs]
        bars = ax.bar(x + (j - 1) * width, vals, width,
                      color=color, alpha=0.82, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in configs], fontsize=10
    )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("비율", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        "Exp Supp-3 Figure 1: Multi-layer Patch Comparison\n"
        "KC Suppression vs Clean/Supportive Accuracy",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "exp3_figure1_multiconfig_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: tradeoff scatter (KC_sup vs clean_acc)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_cfg = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for c, color in zip(configs, colors_cfg):
        r = config_results[c]
        ax.scatter(r["kc_suppression"], r["clean_accuracy"],
                   s=180, c=color, zorder=4, edgecolors="black", lw=0.8)
        ax.annotate(c.replace("_", "\n"),
                    (r["kc_suppression"], r["clean_accuracy"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=color)

    ax.fill_between([0.3, 1.0], [0.8, 0.8], [1.05, 1.05],
                    alpha=0.07, color="green", zorder=0, label="목표: KC>30% & clean>80%")
    ax.axhline(0.8, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.3, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(-0.02, max(r["kc_suppression"] for r in config_results.values()) + 0.08)
    ax.set_ylim(0.6, 1.05)
    ax.set_xlabel("KC Suppression (rank-flip rate)", fontsize=12)
    ax.set_ylabel("Clean Accuracy", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_title(
        "Exp Supp-3 Figure 2: Multi-layer KC-Clean Tradeoff\n"
        "레이어 추가가 tradeoff를 개선하는가?",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "exp3_figure2_tradeoff_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Exp3] 그림 저장 완료 → {out_dir}", flush=True)


# ── JSON 직렬화 helper ─────────────────────────────────────────────────────────
def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    return obj


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    n_cases = args.test or args.n_cases
    exps    = set(args.exp)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    print(f"[data] KC 케이스 로드: {KC_PATH}", flush=True)
    kc_cases: list = json.loads(KC_PATH.read_text())
    if n_cases:
        kc_cases = kc_cases[:n_cases]
    print(f"[data] {len(kc_cases)}개 케이스 사용", flush=True)

    # 모델 로드
    model = load_model()
    setup_letter_variants(model.tokenizer)

    # 후보 방향 로드
    print(f"[data] 후보 방향 로드: {CANDS_PATH}", flush=True)
    cands = torch.load(CANDS_PATH, map_location="cpu")

    # 베이스라인 (공통)
    case_data = prepare_baselines(model, kc_cases)

    # A 방향 (Exp1, 2에서 사용)
    dir_A_main = load_direction(cands, PATCH_LAYER)

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    if 1 in exps:
        exp1_results = run_exp1(model, case_data, dir_A_main)
        exp1_summary = summarize_exp1(exp1_results)

        print("\n[Exp1] 결과 요약:", flush=True)
        for k in ["delta_cor_conf", "delta_rep_conf", "delta_cor_sup", "delta_rep_sup"]:
            s = exp1_summary[k]
            print(f"  {k}: mean={s['mean']:+.4f}  std={s['std']:.4f}  "
                  f"pos={s['pos_ratio']:.3f}")
        print(f"  dissociation_ratio: {exp1_summary['dissociation_ratio']:.4f}")
        print(f"  r_IE_vs_delta_rep:  {exp1_summary['r_IE_vs_delta_rep']:.4f}"
              f"  p={exp1_summary['p_IE_vs_delta_rep']:.4f}")
        print(f"  kc_suppression:     {exp1_summary['kc_suppression']:.4f}")

        plot_exp1(exp1_results, exp1_summary, OUT_DIR)

        exp1_save = {
            "summary":    _to_serializable(exp1_summary),
            "per_case":   _to_serializable(
                [{k: v for k, v in r.items() if k not in ("tok_conf",)} for r in exp1_results]
            ),
        }
        (OUT_DIR / "exp1_results.json").write_text(
            json.dumps(exp1_save, indent=2, ensure_ascii=False)
        )
        print(f"[Exp1] 저장: {OUT_DIR}/exp1_results.json", flush=True)

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    if 2 in exps:
        exp2_results = run_exp2(model, case_data, dir_A_main)
        exp2_summary = summarize_exp2(exp2_results)

        print("\n[Exp2] 결과 요약:", flush=True)
        print(f"  r_pearson:  {exp2_summary['r_pearson']:.4f}  p={exp2_summary['p_pearson']:.4f}")
        print(f"  r_spearman: {exp2_summary['r_spearman']:.4f}  p={exp2_summary['p_spearman']:.4f}")
        for lb, bs in exp2_summary["bins"].items():
            if bs.get("n", 0) == 0:
                continue
            print(f"  [{lb:8s}] n={bs['n']:4d}  ie_mean={bs['ie_kc_mean']:+.4f}  "
                  f"flip_rate={bs['flip_rate']:.4f}")

        plot_exp2(exp2_results, exp2_summary, OUT_DIR)

        exp2_save = {
            "summary":  _to_serializable(exp2_summary),
            "per_case": _to_serializable(
                [{k: v for k, v in r.items()} for r in exp2_results]
            ),
        }
        (OUT_DIR / "exp2_results.json").write_text(
            json.dumps(exp2_save, indent=2, ensure_ascii=False)
        )
        print(f"[Exp2] 저장: {OUT_DIR}/exp2_results.json", flush=True)

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    if 3 in exps:
        exp3_results = run_exp3(model, case_data, cands)

        print("\n[Exp3] 결과 요약:", flush=True)
        for cfg, r in exp3_results.items():
            print(f"  {cfg:15s}: kc_sup={r['kc_suppression']:.4f}  "
                  f"clean={r['clean_accuracy']:.3f}  sup={r['supportive_accuracy']:.3f}")

        plot_exp3(exp3_results, OUT_DIR)

        exp3_save = {k: _to_serializable(v) for k, v in exp3_results.items()}
        (OUT_DIR / "exp3_results.json").write_text(
            json.dumps(exp3_save, indent=2, ensure_ascii=False)
        )
        print(f"[Exp3] 저장: {OUT_DIR}/exp3_results.json", flush=True)

    print("\n[완료] 모든 실험 완료.", flush=True)


if __name__ == "__main__":
    main()

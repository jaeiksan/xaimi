#!/usr/bin/env python3
"""
Exp3: KC-specific 판별 + L17H3 Mediation

파트 1: Double Dissociation (A × L16)
  conflict prompt vs supportive prompt에서 IE 비교
  IE_kc >> IE_sup → A 방향이 KC-specific

파트 2: L17H3 Mediation (A × L16)
  A 제거 → L17H3 evidence attention 감소 → logit diff 감소 체인 검증

파트 3 (--run_D): D × L15 동일 분석

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp3_dissociation_mediation.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp3_dissociation_mediation.py --test 20
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp3_dissociation_mediation.py --run_D
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
from scipy.stats import pearsonr
from transformer_lens import HookedTransformer

_fp = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
matplotlib.rcParams["font.family"] = _fp.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 상수 ──────────────────────────────────────────────────────────────────────
PATCH_LAYER_A  = 16
PATCH_LAYER_D  = 15
MED_LAYER      = 17
MED_HEAD       = 3

MODEL_NAME     = "google/gemma-2-2b"
KC_PATH        = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH     = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
RESULTS_DIR    = Path("phase2/results/exp3")
CHECKPOINT_EVERY = 100
SEED = 42

CTYPES = ("misinformation", "temporal", "semantic")
LETTER_VARIANTS: Dict[str, List[int]] = {}

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=int, default=None, metavar="N")
    p.add_argument("--n_cases", type=int, default=None)
    p.add_argument("--run_D", action="store_true", help="파트 3: D × L15 추가 분석")
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


def build_supportive_prompt(ctx: str, q: str, opts: Dict,
                             conflict_answer: str, correct_answer: str) -> str:
    sup_ctx = ctx.replace(conflict_answer, correct_answer)
    lines = ["Context:", sup_ctx, "", f"Question: {q}"]
    for l in "ABCD":
        lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── 로그 확률 ──────────────────────────────────────────────────────────────────
def get_letter_logprob(logits: torch.Tensor, letter: str) -> float:
    lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    return max(lp[tid].item() for tid in LETTER_VARIANTS[letter])


def get_logit_diff(logits: torch.Tensor, replace: str, correct: str) -> float:
    """logP(replace) - logP(correct)"""
    lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    lp_rep = max(lp[tid].item() for tid in LETTER_VARIANTS[replace])
    lp_cor = max(lp[tid].item() for tid in LETTER_VARIANTS[correct])
    return lp_rep - lp_cor


# ── Evidence token 탐색 ────────────────────────────────────────────────────────
def find_evidence_indices(
    tokenizer,
    conflict_prompt: str,
    conflict_answer: str,
) -> List[int]:
    """conflict_prompt에서 conflict_answer 텍스트에 해당하는 토큰 인덱스 반환."""
    if not conflict_answer or len(conflict_answer) < 2:
        return [1]

    enc = tokenizer(
        conflict_prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = enc["offset_mapping"]

    indices: List[int] = []
    search_from = 0
    while True:
        pos = conflict_prompt.find(conflict_answer, search_from)
        if pos == -1:
            break
        end_pos = pos + len(conflict_answer)
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s < end_pos and e > pos and i not in indices:
                indices.append(i)
        search_from = pos + 1

    return sorted(indices) if indices else [1]


# ── Patch hook ────────────────────────────────────────────────────────────────
def make_patch_hook(direction: torch.Tensor):
    """모든 position에서 direction 성분을 제거하는 hook."""
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        v = value.float()
        d = direction.to(value.device).float()
        proj = v @ d                               # [batch, seq_len]
        v = v - proj.unsqueeze(-1) * d            # [batch, seq_len, d_model]
        return v.to(value.dtype)
    return hook_fn


# ── 후보 방향 로드 ────────────────────────────────────────────────────────────
def load_direction(cands: dict, layer: int, name_key: str) -> torch.Tensor:
    name_map = {
        "A": "A_svd_conf_clean",
        "D": "D_regression",
    }
    target = name_map[name_key]
    for entry in cands[layer]:
        if entry["name"] == target:
            return entry["dir"].float()
    raise KeyError(f"{target} not found at layer {layer}")


# ── 메인 실험 루프 ────────────────────────────────────────────────────────────
def run_experiment(
    model: HookedTransformer,
    kc_cases: list,
    dir_A: torch.Tensor,   # L16, float32 cpu
    dir_D: Optional[torch.Tensor],  # L15, float32 cpu (파트 3용)
    run_D: bool,
) -> Tuple[list, list, list, list]:
    """
    Returns:
        results_diss_A : 파트 1 (double dissociation, A × L16)
        results_med_A  : 파트 2 (mediation, A × L16)
        results_diss_D : 파트 3a (double dissociation, D × L15)
        results_med_D  : 파트 3b (mediation, D × L15)
    """
    device = next(model.parameters()).device
    dir_A_dev = dir_A.to(device)
    dir_D_dev = dir_D.to(device) if (dir_D is not None and run_D) else None

    hook_A = make_patch_hook(dir_A_dev)
    hook_D = make_patch_hook(dir_D_dev) if dir_D_dev is not None else None

    results_diss_A: list = []
    results_med_A:  list = []
    results_diss_D: list = []
    results_med_D:  list = []

    n = len(kc_cases)
    print(f"\n[exp3] {n}개 케이스 처리 ...", flush=True)
    t0 = time.time()

    for i, case in enumerate(kc_cases):
        ctx = case["conflict_context"]
        q   = case["question"]
        opt = case["options"]
        cor = case["correct_option"]
        rep = case["replace_option"]
        ca  = case["conflict_answer"]
        coa = case["correct_answer"]
        cid = case["id"]
        ct  = case["conflict_type"]

        conflict_prompt   = build_conflict_prompt(ctx, q, opt)
        supportive_prompt = build_supportive_prompt(ctx, q, opt, ca, coa)

        tok_conf = model.to_tokens(conflict_prompt,   prepend_bos=True)
        tok_sup  = model.to_tokens(supportive_prompt, prepend_bos=True)
        seq_conf = tok_conf.shape[1]

        # evidence 토큰 인덱스 (mediation용)
        ev_idx = find_evidence_indices(model.tokenizer, conflict_prompt, ca)
        # seq_len 범위 안으로 clip
        ev_idx = [x for x in ev_idx if x < seq_conf]
        if not ev_idx:
            ev_idx = [1]
        final_pos = seq_conf - 1

        # ── A × L16 ────────────────────────────────────────────────────────
        hook_name_A = f"blocks.{PATCH_LAYER_A}.hook_resid_post"
        med_filter  = lambda name: name == f"blocks.{MED_LAYER}.attn.hook_pattern"

        # 1) Baseline conflict (cache L17 attention)
        with torch.no_grad():
            logits_base, cache_base = model.run_with_cache(
                tok_conf, names_filter=med_filter
            )
        logp_conf_base  = get_letter_logprob(logits_base, cor)
        logit_diff_base = get_logit_diff(logits_base, rep, cor)
        attn_base = cache_base[f"blocks.{MED_LAYER}.attn.hook_pattern"][
            0, MED_HEAD, final_pos, ev_idx
        ].float().sum().item()
        del logits_base, cache_base

        # 2) Patched conflict (cache L17 attention)
        with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name_A, hook_A)]):
            logits_patch, cache_patch = model.run_with_cache(
                tok_conf,
                names_filter=med_filter,
            )
        logp_conf_patch  = get_letter_logprob(logits_patch, cor)
        logit_diff_patch = get_logit_diff(logits_patch, rep, cor)
        attn_patch = cache_patch[f"blocks.{MED_LAYER}.attn.hook_pattern"][
            0, MED_HEAD, final_pos, ev_idx
        ].float().sum().item()
        del logits_patch, cache_patch

        IE_kc = logp_conf_patch - logp_conf_base

        results_diss_A.append({
            "case_id": cid,
            "conflict_type": ct,
            "IE_kc": float(IE_kc),
        })
        results_med_A.append({
            "case_id":          cid,
            "conflict_type":    ct,
            "attn_before":      float(attn_base),
            "attn_after":       float(attn_patch),
            "delta_attn":       float(attn_patch - attn_base),
            "logit_diff_before": float(logit_diff_base),
            "logit_diff_after":  float(logit_diff_patch),
            "delta_logit_diff":  float(logit_diff_patch - logit_diff_base),
            "IE":               float(IE_kc),
        })

        # 3) Baseline supportive
        with torch.no_grad():
            logits_sup_base = model(tok_sup)
        logp_sup_base = get_letter_logprob(logits_sup_base, cor)
        del logits_sup_base

        # 4) Patched supportive
        hook_name_A_sup = f"blocks.{PATCH_LAYER_A}.hook_resid_post"
        with torch.no_grad():
            logits_sup_patch = model.run_with_hooks(
                tok_sup, fwd_hooks=[(hook_name_A_sup, hook_A)]
            )
        logp_sup_patch = get_letter_logprob(logits_sup_patch, cor)
        del logits_sup_patch

        IE_sup = logp_sup_patch - logp_sup_base
        results_diss_A[-1]["IE_sup"]             = float(IE_sup)
        results_diss_A[-1]["IE_kc_minus_IE_sup"] = float(IE_kc - IE_sup)

        # ── D × L15 (파트 3) ───────────────────────────────────────────────
        if run_D and hook_D is not None:
            hook_name_D = f"blocks.{PATCH_LAYER_D}.hook_resid_post"

            # baseline은 A와 동일 (재사용 불가 — logits_base 이미 삭제)
            # D용 baseline: 다시 conflict forward
            with torch.no_grad():
                logits_D_base, cache_D_base = model.run_with_cache(
                    tok_conf, names_filter=med_filter
                )
            logp_D_base       = get_letter_logprob(logits_D_base, cor)
            logit_diff_D_base = get_logit_diff(logits_D_base, rep, cor)
            attn_D_base       = cache_D_base[f"blocks.{MED_LAYER}.attn.hook_pattern"][
                0, MED_HEAD, final_pos, ev_idx
            ].float().sum().item()
            del logits_D_base, cache_D_base

            # Patched conflict with D dir at L15
            with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name_D, hook_D)]):
                logits_D_patch, cache_D_patch = model.run_with_cache(
                    tok_conf,
                    names_filter=med_filter,
                )
            logp_D_patch       = get_letter_logprob(logits_D_patch, cor)
            logit_diff_D_patch = get_logit_diff(logits_D_patch, rep, cor)
            attn_D_patch       = cache_D_patch[f"blocks.{MED_LAYER}.attn.hook_pattern"][
                0, MED_HEAD, final_pos, ev_idx
            ].float().sum().item()
            del logits_D_patch, cache_D_patch

            IE_D_kc = logp_D_patch - logp_D_base

            # Supportive with D
            with torch.no_grad():
                logits_D_sup_base = model(tok_sup)
            logp_D_sup_base = get_letter_logprob(logits_D_sup_base, cor)
            del logits_D_sup_base

            with torch.no_grad():
                logits_D_sup_patch = model.run_with_hooks(
                    tok_sup, fwd_hooks=[(hook_name_D, hook_D)]
                )
            logp_D_sup_patch = get_letter_logprob(logits_D_sup_patch, cor)
            del logits_D_sup_patch

            IE_D_sup = logp_D_sup_patch - logp_D_base

            results_diss_D.append({
                "case_id":          cid,
                "conflict_type":    ct,
                "IE_kc":            float(IE_D_kc),
                "IE_sup":           float(IE_D_sup),
                "IE_kc_minus_IE_sup": float(IE_D_kc - IE_D_sup),
            })
            results_med_D.append({
                "case_id":           cid,
                "conflict_type":     ct,
                "attn_before":       float(attn_D_base),
                "attn_after":        float(attn_D_patch),
                "delta_attn":        float(attn_D_patch - attn_D_base),
                "logit_diff_before": float(logit_diff_D_base),
                "logit_diff_after":  float(logit_diff_D_patch),
                "delta_logit_diff":  float(logit_diff_D_patch - logit_diff_D_base),
                "IE":                float(IE_D_kc),
            })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.save(results_diss_A, RESULTS_DIR / "dissociation_A_checkpoint.pt")
            torch.save(results_med_A,  RESULTS_DIR / "mediation_A_checkpoint.pt")
            if run_D:
                torch.save(results_diss_D, RESULTS_DIR / "dissociation_D_checkpoint.pt")
                torch.save(results_med_D,  RESULTS_DIR / "mediation_D_checkpoint.pt")
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{n}] ({elapsed:.0f}s, {rate:.2f} case/s, "
                  f"ETA {(n-i-1)/rate:.0f}s)", flush=True)

    return results_diss_A, results_med_A, results_diss_D, results_med_D


# ── 통계 집계 ─────────────────────────────────────────────────────────────────
def aggregate_dissociation(results: list) -> dict:
    ie_kc  = np.array([r["IE_kc"]  for r in results])
    ie_sup = np.array([r["IE_sup"] for r in results])
    ratio  = float(ie_kc.mean() / ie_sup.mean()) if abs(ie_sup.mean()) > 1e-6 else float("nan")

    stats = {
        "IE_kc_mean":        float(ie_kc.mean()),
        "IE_kc_std":         float(ie_kc.std()),
        "IE_kc_median":      float(np.median(ie_kc)),
        "IE_sup_mean":       float(ie_sup.mean()),
        "IE_sup_std":        float(ie_sup.std()),
        "IE_sup_median":     float(np.median(ie_sup)),
        "ratio_kc_over_sup": ratio,
        "dissociation_ratio": float(np.mean(ie_kc > ie_sup)),
        "IE_kc_positive_ratio": float(np.mean(ie_kc > 0)),
        "IE_sup_positive_ratio": float(np.mean(ie_sup > 0)),
        "by_type": {},
    }
    for ct in CTYPES:
        sub = [r for r in results if r["conflict_type"] == ct]
        if not sub:
            continue
        kc_s  = np.array([r["IE_kc"]  for r in sub])
        sup_s = np.array([r["IE_sup"] for r in sub])
        stats["by_type"][ct] = {
            "n":       len(sub),
            "IE_kc":   float(kc_s.mean()),
            "IE_sup":  float(sup_s.mean()),
            "ratio":   float(kc_s.mean() / sup_s.mean()) if abs(sup_s.mean()) > 1e-6 else float("nan"),
        }
    return stats


def aggregate_mediation(results: list) -> dict:
    da  = np.array([r["delta_attn"]       for r in results])
    dl  = np.array([r["delta_logit_diff"] for r in results])
    ie  = np.array([r["IE"]               for r in results])

    r_med, p_med = pearsonr(da, dl)

    chain = np.array([
        r["delta_attn"] < 0 and r["delta_logit_diff"] < 0
        for r in results
    ])

    return {
        "delta_attn_mean":       float(da.mean()),
        "delta_attn_std":        float(da.std()),
        "attn_decrease_ratio":   float(np.mean(da < 0)),
        "delta_logit_diff_mean": float(dl.mean()),
        "delta_logit_diff_std":  float(dl.std()),
        "logit_decrease_ratio":  float(np.mean(dl < 0)),
        "chain_ratio":           float(chain.mean()),
        "r_mediation":           float(r_med),
        "p_mediation":           float(p_med),
        "IE_mean":               float(ie.mean()),
        "IE_std":                float(ie.std()),
    }


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_dissociation(
    stats_A: dict,
    stats_D: Optional[dict],
    out_dir: Path,
    run_D: bool,
) -> None:
    """Figure 1: IE_kc vs IE_sup 비교 (A × L16, 선택적으로 D × L15 병행)."""
    n_groups = 2 if run_D else 1
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6), sharey=False)
    if n_groups == 1:
        axes = [axes]

    def _plot_one(ax, stats, title):
        labels = ["IE_kc\n(conflict)", "IE_sup\n(supportive)"]
        means  = [stats["IE_kc_mean"],  stats["IE_sup_mean"]]
        stds   = [stats["IE_kc_std"],   stats["IE_sup_std"]]
        colors = ["#e74c3c", "#3498db"]

        bars = ax.bar(labels, means, color=colors, alpha=0.8, width=0.45)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + (0.02 if m >= 0 else -0.05),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("IE mean", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    _plot_one(axes[0], stats_A, "")
    if run_D and stats_D:
        _plot_one(axes[1], stats_D, "D × L15  (비교용)")

    plt.suptitle("Exp3. A방향의 KC 영향", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure1_double_dissociation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def plot_mediation(
    med_A: list,
    stats_A: dict,
    med_D: Optional[list],
    stats_D_med: Optional[dict],
    out_dir: Path,
    run_D: bool,
) -> None:
    """Figure 2: delta_attn vs delta_logit_diff scatter + 분포."""
    n_cols = 2 if run_D else 1
    fig, axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 11))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    def _plot_med(col, med_results, stats_med, label):
        da = [r["delta_attn"]       for r in med_results]
        dl = [r["delta_logit_diff"] for r in med_results]
        r_val = stats_med["r_mediation"]
        chain  = stats_med["chain_ratio"]

        # 상단: scatter
        ax_sc = axes[0, col]
        ct_colors = {
            "misinformation": "#e74c3c",
            "temporal":       "#3498db",
            "semantic":       "#2ecc71",
        }
        for ct in CTYPES:
            sub = [(r["delta_attn"], r["delta_logit_diff"])
                   for r in med_results if r["conflict_type"] == ct]
            if not sub:
                continue
            xs, ys = zip(*sub)
            ax_sc.scatter(xs, ys, c=ct_colors[ct], alpha=0.3, s=15, label=ct)

        ax_sc.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax_sc.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        # 3사분면 하이라이트 (chain 성립 구간)
        xlim = ax_sc.get_xlim(); ylim = ax_sc.get_ylim()
        ax_sc.fill_betweenx(
            [min(yl for yl in [min(dl), 0] if True), 0],
            min(xl for xl in [min(da), 0] if True), 0,
            alpha=0.08, color="#e74c3c",
        )
        # trend line
        z = np.polyfit(da, dl, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(da), max(da), 100)
        ax_sc.plot(x_line, p(x_line), color="black", linewidth=1.5, linestyle="--",
                   label=f"r={r_val:.3f}")
        ax_sc.set_xlabel("Δ attn (L17H3 evidence attention 변화)", fontsize=9)
        ax_sc.set_ylabel("Δ logit_diff (logP(rep)−logP(cor) 변화)", fontsize=9)
        ax_sc.set_title(f"{label}: Mediation Scatter\n"
                        f"r={r_val:.3f}  chain_ratio={chain:.1%}",
                        fontsize=10, fontweight="bold")
        ax_sc.legend(fontsize=8)
        ax_sc.grid(alpha=0.2)

        # 하단: delta_attn 분포
        ax_dist = axes[1, col]
        bins = 40
        ax_dist.hist(da, bins=bins, color="#3498db", alpha=0.6, label="Δ attn")
        ax_dist.axvline(np.mean(da), color="#2980b9", linewidth=2,
                        linestyle="--", label=f"mean={np.mean(da):.4f}")
        ax_dist.axvline(0, color="gray", linewidth=0.8)
        ax_dist2 = ax_dist.twinx()
        ax_dist2.hist(dl, bins=bins, color="#e74c3c", alpha=0.4, label="Δ logit_diff")
        ax_dist2.axvline(np.mean(dl), color="#c0392b", linewidth=2,
                         linestyle=":", label=f"mean={np.mean(dl):.4f}")
        ax_dist.set_xlabel("변화량", fontsize=9)
        ax_dist.set_ylabel("count (Δ attn)", color="#3498db", fontsize=9)
        ax_dist2.set_ylabel("count (Δ logit_diff)", color="#e74c3c", fontsize=9)
        ax_dist.set_title(
            f"Δ attn mean={np.mean(da):.4f} (decrease ratio={stats_med['attn_decrease_ratio']:.1%})\n"
            f"Δ logit mean={np.mean(dl):.4f} (decrease ratio={stats_med['logit_decrease_ratio']:.1%})",
            fontsize=9,
        )
        lines1, lbs1 = ax_dist.get_legend_handles_labels()
        lines2, lbs2 = ax_dist2.get_legend_handles_labels()
        ax_dist.legend(lines1 + lines2, lbs1 + lbs2, fontsize=8)

    _plot_med(0, med_A, stats_A, "A × L16")
    if run_D and med_D and stats_D_med:
        _plot_med(1, med_D, stats_D_med, "D × L15")

    plt.suptitle("Figure 2: L17H3 Mediation — A 제거 → evidence attention 감소 → KC 감소\n"
                 "3사분면(빨강): chain 성립 구간 (Δattn<0 + Δlogit<0)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure2_mediation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def plot_summary(
    diss_A: dict,
    med_A:  dict,
    diss_D: Optional[dict],
    med_D:  Optional[dict],
    out_dir: Path,
    run_D: bool,
) -> None:
    """Figure 3: 핵심 지표 요약."""
    metrics_A = [
        ("IE_kc\nmean",    diss_A["IE_kc_mean"],   "#e74c3c"),
        ("IE_sup\nmean",   diss_A["IE_sup_mean"],   "#3498db"),
        ("chain\nratio",   med_A["chain_ratio"],    "#2ecc71"),
        ("r_med\n|r|",     abs(med_A["r_mediation"]), "#9b59b6"),
    ]
    labels = [m[0] for m in metrics_A]
    vals_A = [m[1] for m in metrics_A]
    colors = [m[2] for m in metrics_A]

    x = np.arange(len(labels))
    width = 0.35 if run_D else 0.6

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_A = ax.bar(x - (width / 2 if run_D else 0), vals_A,
                    width, label="A × L16", color=colors, alpha=0.8)

    if run_D and diss_D and med_D:
        vals_D = [
            diss_D["IE_kc_mean"],
            diss_D["IE_sup_mean"],
            med_D["chain_ratio"],
            abs(med_D["r_mediation"]),
        ]
        bars_D = ax.bar(x + width / 2, vals_D,
                        width, label="D × L15", color=colors, alpha=0.4,
                        hatch="//")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("값", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # 수치 표기
    for bar in bars_A:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.title("Figure 3: Exp3 핵심 지표 요약\n"
              "IE_kc/IE_sup ratio & chain_ratio & |r_mediation|",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure3_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


# ── 결과 출력 ──────────────────────────────────────────────────────────────────
def print_results(diss: dict, med: dict, label: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  [Double Dissociation]")
    print(f"    IE_kc  = {diss['IE_kc_mean']:+.4f} ± {diss['IE_kc_std']:.4f}  (pos_ratio={diss['IE_kc_positive_ratio']:.3f})")
    print(f"    IE_sup = {diss['IE_sup_mean']:+.4f} ± {diss['IE_sup_std']:.4f}  (pos_ratio={diss['IE_sup_positive_ratio']:.3f})")
    print(f"    ratio  = {diss['ratio_kc_over_sup']:.3f}")
    print(f"    KC>sup = {diss['dissociation_ratio']:.1%}")
    print(f"  [Type breakdown]")
    for ct, info in diss["by_type"].items():
        print(f"    {ct:<15}: KC={info['IE_kc']:+.4f}  sup={info['IE_sup']:+.4f}  ratio={info['ratio']:.2f}")
    print(f"  [Mediation]")
    print(f"    Δ attn      = {med['delta_attn_mean']:+.5f} ± {med['delta_attn_std']:.5f}  "
          f"(decrease {med['attn_decrease_ratio']:.1%})")
    print(f"    Δ logit_diff= {med['delta_logit_diff_mean']:+.4f} ± {med['delta_logit_diff_std']:.4f}  "
          f"(decrease {med['logit_decrease_ratio']:.1%})")
    print(f"    chain_ratio = {med['chain_ratio']:.3f}")
    print(f"    r_mediation = {med['r_mediation']:+.4f}  (p={med['p_mediation']:.2e})")
    print(f"    IE_mean     = {med['IE_mean']:+.4f}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    kc_cases: list = json.load(open(KC_PATH))
    n_cases = args.test or args.n_cases or len(kc_cases)
    kc_cases = kc_cases[:n_cases]
    print(f"[exp3] N_CASES={n_cases}  run_D={args.run_D}", flush=True)

    # 후보 방향 로드
    cands = torch.load(CANDS_PATH, map_location="cpu")
    dir_A = load_direction(cands, PATCH_LAYER_A, "A")  # L16
    dir_D = load_direction(cands, PATCH_LAYER_D, "D") if args.run_D else None
    print(f"[exp3] A dir norm={dir_A.norm():.5f}  (L{PATCH_LAYER_A})", flush=True)
    if dir_D is not None:
        print(f"[exp3] D dir norm={dir_D.norm():.5f}  (L{PATCH_LAYER_D})", flush=True)

    # 모델 로드
    model = load_model()
    setup_letter_variants(model.tokenizer)

    # 실험 실행
    t0 = time.time()
    res_diss_A, res_med_A, res_diss_D, res_med_D = run_experiment(
        model, kc_cases, dir_A, dir_D, args.run_D
    )
    print(f"\n[exp3] 실험 완료 ({time.time()-t0:.0f}s)", flush=True)

    # 저장
    torch.save(res_diss_A, RESULTS_DIR / "dissociation_A_results.pt")
    torch.save(res_med_A,  RESULTS_DIR / "mediation_A_results.pt")
    print("[save] dissociation_A_results.pt, mediation_A_results.pt")

    if args.run_D:
        torch.save(res_diss_D, RESULTS_DIR / "dissociation_D_results.pt")
        torch.save(res_med_D,  RESULTS_DIR / "mediation_D_results.pt")
        print("[save] dissociation_D_results.pt, mediation_D_results.pt")

    # 집계
    stats_diss_A = aggregate_dissociation(res_diss_A)
    stats_med_A  = aggregate_mediation(res_med_A)
    stats_diss_D = aggregate_dissociation(res_diss_D) if args.run_D and res_diss_D else None
    stats_med_D  = aggregate_mediation(res_med_D)     if args.run_D and res_med_D  else None

    with open(RESULTS_DIR / "dissociation_stats.json", "w") as f:
        json.dump({"A": stats_diss_A,
                   **({"D": stats_diss_D} if stats_diss_D else {})}, f, indent=2)
    with open(RESULTS_DIR / "mediation_stats.json", "w") as f:
        json.dump({"A": stats_med_A,
                   **({"D": stats_med_D} if stats_med_D else {})}, f, indent=2)
    print("[save] dissociation_stats.json, mediation_stats.json")

    # 결과 출력
    print_results(stats_diss_A, stats_med_A, f"A × L{PATCH_LAYER_A} (context direction)")
    if args.run_D and stats_diss_D and stats_med_D:
        print_results(stats_diss_D, stats_med_D, f"D × L{PATCH_LAYER_D} (KC-specific direction)")

    # 시각화
    print("\n[exp3] 시각화 ...", flush=True)
    plot_dissociation(stats_diss_A, stats_diss_D, RESULTS_DIR, args.run_D)
    plot_mediation(res_med_A, stats_med_A, res_med_D, stats_med_D, RESULTS_DIR, args.run_D)
    plot_summary(stats_diss_A, stats_med_A, stats_diss_D, stats_med_D, RESULTS_DIR, args.run_D)

    print(f"\n[exp3] 완료. 결과: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()

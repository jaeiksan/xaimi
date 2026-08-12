#!/usr/bin/env python3
"""
Exp Grid: 방향 × 레이어 전체 grid search

Phase 1: 단일 레이어 × 4방향      — 4 layers × 4 dirs = 16 configs
Phase 2: 다중 레이어 서브셋 × 동일 방향  — 11 subsets × 4 dirs = 44 configs
Phase 3: 다중 레이어 서브셋 × 레이어별 최적 방향  — 11 configs (Phase 1 기반)
총 71 configs

레이어 범위: L16–L19 (Exp2/4/Supp에서 유의미했던 구간)
방향: A=SVD(conf−clean), B=SVD(conf−sup), C=cPCA, D=regression

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_grid.py
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_grid.py --test 20
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_grid.py --phase 1
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
MODEL_NAME  = "google/gemma-2-2b"
KC_PATH     = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH  = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
OUT_DIR     = Path("phase2/results/exp_grid")

GRID_LAYERS = [16, 17, 18, 19]
DIR_NAMES   = ["A_svd_conf_clean", "B_svd_conf_support", "C_cpca", "D_regression"]
DIR_LABELS  = {"A_svd_conf_clean": "A", "B_svd_conf_support": "B",
               "C_cpca": "C", "D_regression": "D"}
ALPHA       = 1.0
CHECKPOINT_EVERY = 100

LETTER_VARIANTS: Dict[str, List[int]] = {}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",    type=int, default=None, metavar="N",
                   help="빠른 테스트: N개 케이스만 사용")
    p.add_argument("--n_cases", type=int, default=None)
    p.add_argument("--phase",   type=int, nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3], metavar="N",
                   help="실행할 phase 번호 (기본: 1 2 3)")
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


def run_logprobs(model: HookedTransformer, tokens: torch.Tensor, cor: str, rep: str):
    with torch.no_grad():
        logits = model(tokens)
    return _logprobs_from_logits(logits, cor, rep)


def run_logprobs_hooked(model, tokens, cor, rep, fwd_hooks):
    with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)
    return _logprobs_from_logits(logits, cor, rep)


# ── Scale hook ────────────────────────────────────────────────────────────────
def make_scale_hook(direction: torch.Tensor, alpha: float):
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        v = value.float()
        d = direction.to(value.device).float()
        proj = v @ d
        v    = v - alpha * proj.unsqueeze(-1) * d
        return v.to(value.dtype)
    return hook_fn


# ── 베이스라인 전처리 ─────────────────────────────────────────────────────────
def prepare_baselines(model: HookedTransformer, kc_cases: list) -> List[Dict]:
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

        tok_conf  = model.to_tokens(build_conflict_prompt(ctx, q, opt),  prepend_bos=True)
        tok_clean = model.to_tokens(build_clean_prompt(q, opt),           prepend_bos=True)
        tok_sup   = model.to_tokens(
            build_supportive_prompt(ctx, q, opt, ca, coa), prepend_bos=True)

        lp_cor_conf,  lp_rep_conf  = run_logprobs(model, tok_conf,  cor, rep)
        lp_cor_clean, lp_rep_clean = run_logprobs(model, tok_clean, cor, rep)
        lp_cor_sup,   lp_rep_sup   = run_logprobs(model, tok_sup,   cor, rep)

        data.append({
            "tok_conf":     tok_conf,
            "tok_clean":    tok_clean,
            "tok_sup":      tok_sup,
            "cor":          cor,
            "rep":          rep,
            "case_id":      case["id"],
            "lp_cor_conf":  lp_cor_conf,
            "lp_rep_conf":  lp_rep_conf,
            "lp_cor_clean": lp_cor_clean,
            "lp_rep_clean": lp_rep_clean,
            "lp_cor_sup":   lp_cor_sup,
            "lp_rep_sup":   lp_rep_sup,
            "has_kc":       lp_rep_conf > lp_cor_conf,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [baseline] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    kc_frac   = sum(1 for d in data if d["has_kc"]) / n
    clean_acc = sum(1 for d in data if d["lp_cor_clean"] > d["lp_rep_clean"]) / n
    print(f"[baseline] 완료: kc_frac={kc_frac:.3f}  clean_acc={clean_acc:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return data


# ── 단일 config 평가 ──────────────────────────────────────────────────────────
def evaluate_config(
    model: HookedTransformer,
    case_data: List[Dict],
    fwd_hooks: List[Tuple],
    config_label: str,
) -> Dict:
    n        = len(case_data)
    kc_flip  = 0
    clean_ok = 0
    sup_ok   = 0

    for i, d in enumerate(case_data):
        cor = d["cor"]
        rep = d["rep"]

        if d["has_kc"]:
            lp_c, lp_r = run_logprobs_hooked(model, d["tok_conf"], cor, rep, fwd_hooks)
            if lp_c > lp_r:
                kc_flip += 1

        lp_c_cl, lp_r_cl = run_logprobs_hooked(model, d["tok_clean"], cor, rep, fwd_hooks)
        if lp_c_cl > lp_r_cl:
            clean_ok += 1

        lp_c_sp, lp_r_sp = run_logprobs_hooked(model, d["tok_sup"], cor, rep, fwd_hooks)
        if lp_c_sp > lp_r_sp:
            sup_ok += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()
            kc_so_far = sum(1 for dd in case_data[:i+1] if dd["has_kc"])
            print(f"  [{config_label}] {i+1}/{n}  "
                  f"kc={kc_flip}/{kc_so_far}  clean={clean_ok/(i+1):.3f}  "
                  f"sup={sup_ok/(i+1):.3f}", flush=True)

    return {
        "kc_suppression":      kc_flip / n,
        "clean_accuracy":      clean_ok / n,
        "supportive_accuracy": sup_ok / n,
        "kc_flipped":          kc_flip,
    }


# ── Config 생성 ────────────────────────────────────────────────────────────────
def make_fwd_hooks(layer_dir_pairs: List[Tuple[int, str]], cands: dict) -> List[Tuple]:
    """[(layer, dir_name), ...] → fwd_hooks 리스트"""
    hooks = []
    for layer, dir_name in layer_dir_pairs:
        direction = next(e["dir"].float() for e in cands[layer] if e["name"] == dir_name)
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn   = make_scale_hook(direction, ALPHA)
        hooks.append((hook_name, hook_fn))
    return hooks


def config_label(layer_dir_pairs: List[Tuple[int, str]]) -> str:
    parts = [f"L{l}:{DIR_LABELS[d]}" for l, d in sorted(layer_dir_pairs)]
    return "+".join(parts)


# ── Phase 1: 단일 레이어 × 모든 방향 ──────────────────────────────────────────
def run_phase1(model, case_data, cands) -> List[Dict]:
    configs = [(l, d) for l in GRID_LAYERS for d in DIR_NAMES]
    total   = len(configs)
    print(f"\n[Phase 1] 단일 레이어 grid — {total} configs", flush=True)
    results = []
    t0 = time.time()

    for idx, (layer, dir_name) in enumerate(configs):
        label   = config_label([(layer, dir_name)])
        fwd_hooks = make_fwd_hooks([(layer, dir_name)], cands)
        print(f"\n  [{idx+1}/{total}] {label}", flush=True)
        r = evaluate_config(model, case_data, fwd_hooks, label)
        results.append({
            "config":           label,
            "layers":           [layer],
            "directions":       [dir_name],
            "layer_dir_pairs":  [(layer, dir_name)],
            "n_layers":         1,
            "phase":            1,
            **r,
        })
        print(f"  → kc_sup={r['kc_suppression']:.4f}  "
              f"clean={r['clean_accuracy']:.3f}  sup={r['supportive_accuracy']:.3f}  "
              f"(총 경과: {time.time()-t0:.0f}s)", flush=True)

    return results


# ── Phase 2: 다중 레이어 서브셋 × 동일 방향 ───────────────────────────────────
def run_phase2(model, case_data, cands) -> List[Dict]:
    # 2개 이상 레이어의 모든 non-empty 서브셋
    multi_subsets = []
    for r in range(2, len(GRID_LAYERS) + 1):
        for combo in combinations(GRID_LAYERS, r):
            multi_subsets.append(list(combo))

    configs = [(subset, d) for subset in multi_subsets for d in DIR_NAMES]
    total   = len(configs)
    print(f"\n[Phase 2] 다중 레이어 × 동일 방향 — {total} configs", flush=True)
    results = []
    t0 = time.time()

    for idx, (layers, dir_name) in enumerate(configs):
        layer_dir_pairs = [(l, dir_name) for l in layers]
        label           = config_label(layer_dir_pairs)
        fwd_hooks       = make_fwd_hooks(layer_dir_pairs, cands)
        print(f"\n  [{idx+1}/{total}] {label}", flush=True)
        r = evaluate_config(model, case_data, fwd_hooks, label)
        results.append({
            "config":           label,
            "layers":           layers,
            "directions":       [dir_name] * len(layers),
            "layer_dir_pairs":  layer_dir_pairs,
            "n_layers":         len(layers),
            "phase":            2,
            **r,
        })
        print(f"  → kc_sup={r['kc_suppression']:.4f}  "
              f"clean={r['clean_accuracy']:.3f}  sup={r['supportive_accuracy']:.3f}  "
              f"(총 경과: {time.time()-t0:.0f}s)", flush=True)

    return results


# ── Phase 3: 다중 레이어 서브셋 × 레이어별 최적 방향 ─────────────────────────
def run_phase3(model, case_data, cands, phase1_results: List[Dict]) -> List[Dict]:
    """
    Phase 1 결과에서 레이어별 best direction (KC_sup 기준) 결정.
    그 방향을 조합하여 다중 레이어 서브셋에 적용.
    """
    # 레이어별 best direction 선택 (kc_suppression 우선, clean_acc>0.8 조건)
    best_dir: Dict[int, str] = {}
    for layer in GRID_LAYERS:
        layer_rows = [r for r in phase1_results if r["layers"] == [layer]]
        # clean_acc > 0.8 만족하는 것 중 kc_sup 최고
        valid = [r for r in layer_rows if r["clean_accuracy"] > 0.8]
        if not valid:
            valid = layer_rows  # 조건 완화
        best = max(valid, key=lambda r: r["kc_suppression"])
        best_dir[layer] = best["directions"][0]

    print(f"\n[Phase 3] 레이어별 best direction: "
          + "  ".join(f"L{l}:{DIR_LABELS[d]}" for l, d in sorted(best_dir.items())),
          flush=True)

    multi_subsets = []
    for r in range(2, len(GRID_LAYERS) + 1):
        for combo in combinations(GRID_LAYERS, r):
            multi_subsets.append(list(combo))

    total   = len(multi_subsets)
    print(f"[Phase 3] 다중 레이어 × 레이어별 최적 방향 — {total} configs", flush=True)
    results = []
    t0 = time.time()

    for idx, layers in enumerate(multi_subsets):
        layer_dir_pairs = [(l, best_dir[l]) for l in layers]
        label           = config_label(layer_dir_pairs)
        fwd_hooks       = make_fwd_hooks(layer_dir_pairs, cands)
        print(f"\n  [{idx+1}/{total}] {label}", flush=True)
        r = evaluate_config(model, case_data, fwd_hooks, label)
        results.append({
            "config":           label,
            "layers":           layers,
            "directions":       [best_dir[l] for l in layers],
            "layer_dir_pairs":  layer_dir_pairs,
            "n_layers":         len(layers),
            "phase":            3,
            **r,
        })
        print(f"  → kc_sup={r['kc_suppression']:.4f}  "
              f"clean={r['clean_accuracy']:.3f}  sup={r['supportive_accuracy']:.3f}  "
              f"(총 경과: {time.time()-t0:.0f}s)", flush=True)

    return results


# ── 결과 집계 및 랭킹 ─────────────────────────────────────────────────────────
def rank_results(all_results: List[Dict]) -> List[Dict]:
    """
    복합 점수: KC_sup 우선 (clean_acc > 0.8 조건 내에서)
    score = kc_suppression  if clean_accuracy >= 0.8
            kc_suppression - 10 * (0.8 - clean_accuracy)  otherwise
    """
    for r in all_results:
        penalty = max(0.0, 0.8 - r["clean_accuracy"])
        r["score"] = r["kc_suppression"] - 10 * penalty
    return sorted(all_results, key=lambda r: r["score"], reverse=True)


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_heatmap_phase1(phase1: List[Dict], out_dir: Path) -> None:
    """Figure 1: L16–L19 × A/B/C/D heatmap (KC_sup + clean_acc)."""
    dirs   = DIR_NAMES
    layers = GRID_LAYERS
    dir_lb = [DIR_LABELS[d] for d in dirs]

    kc_mat    = np.zeros((len(layers), len(dirs)))
    clean_mat = np.zeros((len(layers), len(dirs)))

    for r in phase1:
        l = r["layers"][0]
        d = r["directions"][0]
        li = layers.index(l)
        di = dirs.index(d)
        kc_mat[li, di]    = r["kc_suppression"]
        clean_mat[li, di] = r["clean_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title, cmap in zip(
        axes,
        [kc_mat, clean_mat],
        ["KC Suppression", "Clean Accuracy"],
        ["YlOrRd", "YlGn"],
    ):
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=mat.max())
        ax.set_xticks(range(len(dirs)))
        ax.set_xticklabels(dir_lb, fontsize=12)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=12)
        ax.set_xlabel("Direction", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        for i in range(len(layers)):
            for j in range(len(dirs)):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if mat[i,j] > mat.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle("Phase 1: Single Layer × Direction Grid", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "figure1_phase1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] figure1_phase1_heatmap.png", flush=True)


def plot_tradeoff_all(all_results: List[Dict], out_dir: Path) -> None:
    """Figure 2: KC_sup vs clean_acc scatter (모든 configs, phase별 색상)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71"}
    labels_ph = {1: "Phase 1 (single)", 2: "Phase 2 (multi, same dir)", 3: "Phase 3 (multi, best dir)"}
    plotted = set()

    for r in all_results:
        ph    = r["phase"]
        color = colors[ph]
        label = labels_ph[ph] if ph not in plotted else ""
        ax.scatter(r["kc_suppression"], r["clean_accuracy"],
                   s=60, c=color, alpha=0.7, edgecolors="none",
                   label=label, zorder=3)
        if label:
            plotted.add(ph)

    # 상위 10개 라벨
    ranked = rank_results(list(all_results))
    for r in ranked[:10]:
        ax.annotate(r["config"],
                    (r["kc_suppression"], r["clean_accuracy"]),
                    fontsize=6, xytext=(4, 3), textcoords="offset points",
                    color=colors[r["phase"]])

    ax.fill_between([0.3, 1.0], [0.8, 0.8], [1.05, 1.05],
                    alpha=0.07, color="green", zorder=0, label="목표 영역 (KC>30% & clean>80%)")
    ax.axhline(0.8, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.3, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("KC Suppression", fontsize=12)
    ax.set_ylabel("Clean Accuracy", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.2)
    ax.set_title("전체 Grid: KC-Clean Tradeoff Space\n(상위 10개 라벨 표시)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "figure2_tradeoff_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] figure2_tradeoff_all.png", flush=True)


def plot_top_configs(ranked: List[Dict], out_dir: Path, top_n: int = 20) -> None:
    """Figure 3: 상위 N개 config의 3가지 지표 가로 막대."""
    top = ranked[:top_n]
    labels = [r["config"] for r in top]
    kc_sup = [r["kc_suppression"]     for r in top]
    clean  = [r["clean_accuracy"]     for r in top]
    sup    = [r["supportive_accuracy"] for r in top]

    y  = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4)))

    ax.barh(y - 0.25, kc_sup, 0.22, color="#e74c3c", alpha=0.82, label="KC Suppression")
    ax.barh(y,        clean,  0.22, color="#3498db", alpha=0.82, label="Clean Accuracy")
    ax.barh(y + 0.25, sup,    0.22, color="#2ecc71", alpha=0.82, label="Supportive Accuracy")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.8, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("비율", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"상위 {top_n}개 Config (score 기준 = KC_sup - 10×clean_penalty)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_dir / "figure3_top_configs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] figure3_top_configs.png", flush=True)


def plot_direction_summary(all_results: List[Dict], out_dir: Path) -> None:
    """Figure 4: 방향별 KC_sup × clean_acc 분포 (violin 또는 box)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dir_order = [DIR_LABELS[d] for d in DIR_NAMES]

    for ax, metric, title in zip(
        axes,
        ["kc_suppression", "clean_accuracy"],
        ["KC Suppression", "Clean Accuracy"],
    ):
        data_by_dir = []
        for dir_name in DIR_NAMES:
            lb = DIR_LABELS[dir_name]
            vals = [r[metric] for r in all_results
                    if lb in [DIR_LABELS[d] for d in r["directions"]]]
            data_by_dir.append(vals)

        bp = ax.boxplot(data_by_dir, tick_labels=dir_order, patch_artist=True,
                        medianprops={"color": "black", "lw": 2})
        colors_dir = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
        for patch, color in zip(bp["boxes"], colors_dir):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"방향별 {title} 분포", fontsize=11, fontweight="bold")
        ax.set_xlabel("Direction", fontsize=11)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_dir / "figure4_direction_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] figure4_direction_summary.png", flush=True)


# ── JSON 직렬화 helper ─────────────────────────────────────────────────────────
def _ser(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):           return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [_ser(x) for x in obj]
    return obj


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    n_cases = args.test or args.n_cases
    phases  = set(args.phase)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    kc_cases: list = json.loads(KC_PATH.read_text())
    if n_cases:
        kc_cases = kc_cases[:n_cases]
    print(f"[data] {len(kc_cases)}개 케이스 사용", flush=True)

    # 모델 로드
    model = load_model()
    setup_letter_variants(model.tokenizer)

    # 후보 방향 로드
    cands = torch.load(CANDS_PATH, map_location="cpu")

    # 베이스라인
    case_data = prepare_baselines(model, kc_cases)

    all_results: List[Dict] = []
    phase1_results: List[Dict] = []

    # Phase 1
    if 1 in phases:
        phase1_results = run_phase1(model, case_data, cands)
        all_results.extend(phase1_results)
        (OUT_DIR / "phase1_results.json").write_text(
            json.dumps(_ser(phase1_results), indent=2, ensure_ascii=False))
        print(f"\n[Phase 1] 저장 완료", flush=True)
    else:
        # Phase 2/3만 실행 시 Phase 1 결과 로드 시도
        p1_path = OUT_DIR / "phase1_results.json"
        if p1_path.exists():
            phase1_results = json.loads(p1_path.read_text())
            all_results.extend(phase1_results)
            print(f"[Phase 1] 기존 결과 로드 ({len(phase1_results)} configs)", flush=True)

    # Phase 2
    if 2 in phases:
        phase2_results = run_phase2(model, case_data, cands)
        all_results.extend(phase2_results)
        (OUT_DIR / "phase2_results.json").write_text(
            json.dumps(_ser(phase2_results), indent=2, ensure_ascii=False))
        print(f"\n[Phase 2] 저장 완료", flush=True)
    else:
        p2_path = OUT_DIR / "phase2_results.json"
        if p2_path.exists():
            p2 = json.loads(p2_path.read_text())
            all_results.extend(p2)

    # Phase 3
    if 3 in phases:
        if not phase1_results:
            print("[Phase 3] Phase 1 결과가 없어 건너뜀", flush=True)
        else:
            phase3_results = run_phase3(model, case_data, cands, phase1_results)
            all_results.extend(phase3_results)
            (OUT_DIR / "phase3_results.json").write_text(
                json.dumps(_ser(phase3_results), indent=2, ensure_ascii=False))
            print(f"\n[Phase 3] 저장 완료", flush=True)

    if not all_results:
        print("[완료] 결과 없음 — 실행된 phase가 없거나 결과 파일이 없습니다.", flush=True)
        return

    # 랭킹 및 전체 저장
    ranked = rank_results(all_results)
    (OUT_DIR / "all_results_ranked.json").write_text(
        json.dumps(_ser(ranked), indent=2, ensure_ascii=False))

    # 상위 20개 출력
    print(f"\n{'='*70}", flush=True)
    print(f"[결과] 전체 {len(ranked)}개 config 랭킹 (상위 20개):", flush=True)
    print(f"{'순위':>4} {'Config':<35} {'KC_sup':>8} {'clean':>7} {'sup':>7} {'score':>8}")
    print("-" * 70)
    for i, r in enumerate(ranked[:20], 1):
        flag = " ★" if r["kc_suppression"] > 0.3 and r["clean_accuracy"] > 0.8 else ""
        print(f"{i:>4} {r['config']:<35} {r['kc_suppression']:>8.4f} "
              f"{r['clean_accuracy']:>7.3f} {r['supportive_accuracy']:>7.3f} "
              f"{r['score']:>8.4f}{flag}")

    # 시각화
    if phase1_results:
        plot_heatmap_phase1(phase1_results, OUT_DIR)
    plot_tradeoff_all(all_results, OUT_DIR)
    plot_top_configs(ranked, OUT_DIR, top_n=min(20, len(ranked)))
    plot_direction_summary(all_results, OUT_DIR)

    print(f"\n[완료] 결과 저장: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

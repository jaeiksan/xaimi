#!/usr/bin/env python3
"""
Exp IE-Dissociation: Layer별 IE + KC-specificity 증명

Exp 1: 방향별 layer sweep — mean IE_kc (꺾은선)
Exp 2: KC-specificity — IE_kc vs IE_sup double dissociation (4방향)

각 (direction, layer) 조합에서 패치 1회로 IE_kc + IE_sup 동시 측정.
총 14 layers × 4 dirs × 2 prompts × N cases

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_ie_dissociation.py
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_ie_dissociation.py --test 30
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_ie_dissociation.py --layers 14 19
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
import matplotlib.font_manager as fm
import numpy as np
import torch
from scipy import stats
from transformer_lens import HookedTransformer

_fp = fm.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf")
matplotlib.rcParams["font.family"] = _fp.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b"
KC_PATH    = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
OUT_DIR    = Path("phase2/results/exp_ie_dissociation")

ALL_LAYERS = list(range(6, 20))   # L6–L19
DIR_NAMES  = ["A_svd_conf_clean", "B_svd_conf_support", "C_cpca", "D_regression"]
DIR_LABELS = {"A_svd_conf_clean": "A", "B_svd_conf_support": "B",
              "C_cpca": "C", "D_regression": "D"}
DIR_COLORS = {"A_svd_conf_clean": "#e67e22", "B_svd_conf_support": "#e74c3c",
              "C_cpca": "#9b59b6",            "D_regression": "#3498db"}
ALPHA      = 1.0
CHECKPOINT_EVERY = 200

LETTER_VARIANTS: Dict[str, List[int]] = {}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",    type=int, default=None)
    p.add_argument("--n_cases", type=int, default=None)
    p.add_argument("--layers",  type=int, nargs=2, default=None,
                   metavar=("FROM", "TO"), help="레이어 범위 (기본: 6 19)")
    return p.parse_args()


# ── 모델 ──────────────────────────────────────────────────────────────────────
def load_model() -> HookedTransformer:
    print("모델 로드 (bfloat16) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16)
    model.eval()
    return model


def setup_letter_variants(tokenizer):
    global LETTER_VARIANTS
    for letter in "ABCD":
        ids = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.append(toks[-1] if len(toks) > 1 else toks[0])
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
    print(f"[setup] LETTER_VARIANTS={LETTER_VARIANTS}", flush=True)


# ── 프롬프트 빌더 ──────────────────────────────────────────────────────────────
def build_conflict_prompt(ctx, q, opts):
    lines = ["Context:", ctx, "", f"Question: {q}"]
    for l in "ABCD": lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_supportive_prompt(ctx, q, opts, ca, coa):
    sup_ctx = ctx.replace(ca, coa)
    lines = ["Context:", sup_ctx, "", f"Question: {q}"]
    for l in "ABCD": lines.append(f"{l}. {opts[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── 로그 확률 ──────────────────────────────────────────────────────────────────
def _lp_correct(logits: torch.Tensor, cor: str) -> float:
    lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    return max(lp[tid].item() for tid in LETTER_VARIANTS[cor])


def run_lp(model, tokens, cor):
    with torch.no_grad():
        logits = model(tokens)
    return _lp_correct(logits, cor)


def run_lp_hooked(model, tokens, cor, hook_name, hook_fn):
    with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        logits = model(tokens)
    return _lp_correct(logits, cor)


# ── Scale hook ────────────────────────────────────────────────────────────────
def make_scale_hook(direction: torch.Tensor, alpha: float):
    def hook_fn(value, hook):
        v = value.float()
        d = direction.to(value.device).float()
        v = v - alpha * (v @ d).unsqueeze(-1) * d
        return v.to(value.dtype)
    return hook_fn


# ── 베이스라인 전처리 ─────────────────────────────────────────────────────────
def prepare_baselines(model, kc_cases) -> List[Dict]:
    n = len(kc_cases)
    print(f"[baseline] {n}개 케이스 전처리 ...", flush=True)
    t0 = time.time()
    data = []

    for i, case in enumerate(kc_cases):
        ctx = case["conflict_context"]
        q   = case["question"]
        opt = case["options"]
        cor = case["correct_option"]
        ca  = case["conflict_answer"]
        coa = case["correct_answer"]

        tok_conf = model.to_tokens(build_conflict_prompt(ctx, q, opt), prepend_bos=True)
        tok_sup  = model.to_tokens(build_supportive_prompt(ctx, q, opt, ca, coa), prepend_bos=True)

        lp_cor_conf = run_lp(model, tok_conf, cor)
        lp_cor_sup  = run_lp(model, tok_sup,  cor)

        data.append({
            "tok_conf":    tok_conf,
            "tok_sup":     tok_sup,
            "cor":         cor,
            "case_id":     case["id"],
            "lp_cor_conf": lp_cor_conf,
            "lp_cor_sup":  lp_cor_sup,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [baseline] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[baseline] 완료 ({time.time()-t0:.0f}s)", flush=True)
    return data


# ── (direction, layer) 단위 IE 측정 ─────────────────────────────────────────
def run_one_combo(
    model, case_data, direction: torch.Tensor, layer: int, label: str
) -> Dict:
    """
    IE_kc = lp_cor_conf_patched - lp_cor_conf_baseline
    IE_sup = lp_cor_sup_patched  - lp_cor_sup_baseline
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn   = make_scale_hook(direction, ALPHA)
    n         = len(case_data)
    ie_kc_list, ie_sup_list = [], []

    for d in case_data:
        lp_c_conf_p = run_lp_hooked(model, d["tok_conf"], d["cor"], hook_name, hook_fn)
        lp_c_sup_p  = run_lp_hooked(model, d["tok_sup"],  d["cor"], hook_name, hook_fn)
        ie_kc_list.append(lp_c_conf_p - d["lp_cor_conf"])
        ie_sup_list.append(lp_c_sup_p  - d["lp_cor_sup"])

    ie_kc = np.array(ie_kc_list)
    ie_sup = np.array(ie_sup_list)

    disso = float((ie_kc > ie_sup).mean())
    r_val, p_val = stats.pearsonr(ie_kc, ie_sup)

    return {
        "label":            label,
        "layer":            layer,
        "direction":        label.split(":")[1] if ":" in label else label,
        "ie_kc_mean":       float(ie_kc.mean()),
        "ie_kc_std":        float(ie_kc.std()),
        "ie_kc_pos":        float((ie_kc > 0).mean()),
        "ie_sup_mean":      float(ie_sup.mean()),
        "ie_sup_std":       float(ie_sup.std()),
        "ie_sup_pos":       float((ie_sup > 0).mean()),
        "dissociation":     disso,
        "r_kc_sup":         float(r_val),
        "p_kc_sup":         float(p_val),
        "ie_kc_per_case":   ie_kc.tolist(),
        "ie_sup_per_case":  ie_sup.tolist(),
    }


# ── 전체 sweep ────────────────────────────────────────────────────────────────
def run_sweep(model, case_data, cands, layers) -> List[Dict]:
    total   = len(layers) * len(DIR_NAMES)
    results = []
    done    = 0
    t0      = time.time()

    print(f"\n[sweep] {len(layers)} layers × {len(DIR_NAMES)} dirs = {total} combos", flush=True)

    for layer in layers:
        for dir_name in DIR_NAMES:
            dir_label = DIR_LABELS[dir_name]
            label     = f"L{layer}:{dir_label}"
            direction = next(e["dir"].float() for e in cands[layer] if e["name"] == dir_name)
            done += 1
            print(f"  [{done}/{total}] {label} ...", end=" ", flush=True)
            t1 = time.time()

            r = run_one_combo(model, case_data, direction, layer, label)
            results.append(r)

            gc.collect()
            torch.cuda.empty_cache()
            print(f"IE_kc={r['ie_kc_mean']:+.3f}  IE_sup={r['ie_sup_mean']:+.3f}  "
                  f"disso={r['dissociation']:.3f}  ({time.time()-t1:.0f}s, 총 {time.time()-t0:.0f}s)",
                  flush=True)

    return results


# ── Figure 1: Layer별 IE 꺾은선 ───────────────────────────────────────────────
def plot_ie_by_layer(results: List[Dict], layers: List[int], out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5))

    for dir_name in DIR_NAMES:
        dlabel = DIR_LABELS[dir_name]
        color  = DIR_COLORS[dir_name]
        rows   = [r for r in results if r["direction"] == dlabel]
        rows   = sorted(rows, key=lambda r: r["layer"])
        xs     = [r["layer"] for r in rows]
        ys     = [r["ie_kc_mean"] for r in rows]
        stds   = [r["ie_kc_std"]  for r in rows]

        full_labels = {
            "A": "A (conflict - clean SVD)",
            "B": "B (conflict - supportive SVD)",
            "C": "C (cPCA)",
            "D": "D (Ridge regression)",
        }
        ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=7, label=full_labels.get(dlabel, dlabel), zorder=3)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_xlabel("Patch Layer", fontsize=12)
    ax.set_ylabel("IE_KC", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, title="Direction", title_fontsize=10, loc="upper left")
    ax.grid(alpha=0.2)
    ax.set_title("Figure 1: Layer별 IE_kc — 방향 후보 A/B/C/D 비교", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure1_ie_by_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path.name}", flush=True)


# ── Figure 2: KC-specificity (IE_kc vs IE_sup 라인 비교) ──────────────────────
def plot_kc_specificity_lines(results: List[Dict], layers: List[int], out_dir: Path):
    """4방향 × 2패널 (IE_kc / IE_sup) 꺾은선"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, metric, title in zip(
        axes,
        ["ie_kc_mean", "ie_sup_mean"],
        ["IE_kc (conflict prompt)", "IE_sup (supportive prompt)"],
    ):
        for dir_name in DIR_NAMES:
            dlabel = DIR_LABELS[dir_name]
            color  = DIR_COLORS[dir_name]
            rows   = sorted([r for r in results if r["direction"] == dlabel],
                            key=lambda r: r["layer"])
            xs = [r["layer"] for r in rows]
            ys = [r[metric]  for r in rows]
            ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=7, label=dlabel, zorder=3)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
        ax.set_xlabel("Patch Layer", fontsize=11)
        ax.set_ylabel("mean Δ log-prob", fontsize=11)
        ax.legend(fontsize=9, title="Direction")
        ax.grid(alpha=0.2)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("Figure 2: KC-Specificity 증명 — IE_kc >> IE_sup", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure2_kc_specificity_lines.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path.name}", flush=True)


# ── Figure 3: KC-specificity scatter (4방향 × best layer) ─────────────────────
def plot_kc_specificity_scatter(results: List[Dict], out_dir: Path):
    """각 방향의 best layer에서 case별 IE_kc vs IE_sup scatter"""
    # 방향별 best layer (IE_kc_mean 최고)
    best_per_dir = {}
    for dir_name in DIR_NAMES:
        dlabel = DIR_LABELS[dir_name]
        rows   = [r for r in results if r["direction"] == dlabel]
        best   = max(rows, key=lambda r: r["ie_kc_mean"])
        best_per_dir[dlabel] = best

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, dir_name in zip(axes, DIR_NAMES):
        dlabel = DIR_LABELS[dir_name]
        color  = DIR_COLORS[dir_name]
        r      = best_per_dir[dlabel]
        ie_kc  = np.array(r["ie_kc_per_case"])
        ie_sup = np.array(r["ie_sup_per_case"])

        ax.scatter(ie_kc, ie_sup, s=8, alpha=0.35, color=color, linewidths=0)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.axvline(0, color="k", lw=0.7, ls="--")

        # 대각선 (IE_kc == IE_sup)
        lim = max(abs(ie_kc).max(), abs(ie_sup).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, ls=":", alpha=0.6)

        disso = r["dissociation"]
        ie_kc_m = r["ie_kc_mean"]
        ie_sup_m = r["ie_sup_mean"]

        ax.set_title(
            f"{dlabel}  (L{r['layer']})\n"
            f"IE_kc={ie_kc_m:+.3f}  IE_sup={ie_sup_m:+.3f}\n"
            f"dissociation={disso:.3f}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("IE_kc", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("IE_sup", fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.15)

        # 사분면 주석
        n_kc_pos_sup_neg = int(((ie_kc > 0) & (ie_sup < 0)).sum())
        ax.text(lim * 0.55, -lim * 0.85,
                f"KC↓ + sup↑\nn={n_kc_pos_sup_neg}",
                fontsize=7, color="#555", ha="center")

    fig.suptitle("Figure 3: KC-Specificity Scatter — 방향별 best layer에서 case별 IE_kc vs IE_sup",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure3_kc_specificity_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path.name}", flush=True)


# ── Figure 4: dissociation ratio by layer ────────────────────────────────────
def plot_dissociation_by_layer(results: List[Dict], layers: List[int], out_dir: Path):
    """각 방향의 dissociation ratio (IE_kc > IE_sup 비율) 꺾은선"""
    fig, ax = plt.subplots(figsize=(11, 5))

    for dir_name in DIR_NAMES:
        dlabel = DIR_LABELS[dir_name]
        color  = DIR_COLORS[dir_name]
        rows   = sorted([r for r in results if r["direction"] == dlabel],
                        key=lambda r: r["layer"])
        xs = [r["layer"]      for r in rows]
        ys = [r["dissociation"] for r in rows]
        ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=7, label=dlabel, zorder=3)

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5, label="chance (0.5)")
    ax.axhline(0.99, color="green", lw=0.8, ls=":", alpha=0.4)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_xlabel("Patch Layer", fontsize=12)
    ax.set_ylabel("Dissociation ratio (IE_kc > IE_sup)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, title="Direction", title_fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_title("Figure 4: KC-Specificity — Dissociation Ratio by Layer\n"
                 "높을수록 해당 방향이 KC-specific함", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure4_dissociation_by_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path.name}", flush=True)


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
    layers  = list(range(args.layers[0], args.layers[1] + 1)) if args.layers else ALL_LAYERS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kc_cases = json.loads(KC_PATH.read_text())
    if n_cases:
        kc_cases = kc_cases[:n_cases]
    print(f"[data] {len(kc_cases)}개 케이스 / layers={layers}", flush=True)

    model = load_model()
    setup_letter_variants(model.tokenizer)
    cands = torch.load(CANDS_PATH, map_location="cpu")

    case_data = prepare_baselines(model, kc_cases)
    results   = run_sweep(model, case_data, cands, layers)

    # 결과 저장
    save = [{k: v for k, v in r.items() if k not in ("ie_kc_per_case", "ie_sup_per_case")}
            for r in results]
    (OUT_DIR / "ie_stats.json").write_text(json.dumps(_ser(save), indent=2, ensure_ascii=False))

    # per-case는 별도 저장
    (OUT_DIR / "ie_per_case.json").write_text(
        json.dumps(_ser([{
            "label": r["label"], "layer": r["layer"], "direction": r["direction"],
            "ie_kc": r["ie_kc_per_case"], "ie_sup": r["ie_sup_per_case"]
        } for r in results]), indent=2, ensure_ascii=False)
    )

    # 요약 출력
    print(f"\n{'='*65}", flush=True)
    print(f"{'Config':<12} {'IE_kc':>8} {'IE_sup':>8} {'pos_kc':>7} {'disso':>7}", flush=True)
    print("-" * 65)
    for r in sorted(results, key=lambda r: (r["direction"], r["layer"])):
        print(f"  {r['label']:<10} {r['ie_kc_mean']:>+8.4f} {r['ie_sup_mean']:>+8.4f} "
              f"{r['ie_kc_pos']:>7.3f} {r['dissociation']:>7.3f}")

    # 그림
    plot_ie_by_layer(results, layers, OUT_DIR)
    plot_kc_specificity_lines(results, layers, OUT_DIR)
    plot_kc_specificity_scatter(results, OUT_DIR)
    plot_dissociation_by_layer(results, layers, OUT_DIR)

    print(f"\n[완료] → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

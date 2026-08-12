#!/usr/bin/env python3
"""
Exp B-Logit-Severity: Direction B × L13-L19

Exp1: Logit Tracking
  - 각 레이어에서 B 제거 후 delta_cor_conf / delta_rep_conf / delta_cor_sup / delta_rep_sup
  - A × L16 결과와 비교

Exp2: Severity별 flip 분석
  - 각 레이어에서 severity 구간별 flip_rate
  - B가 어느 severity 구간까지 gap을 넘기는지 확인

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_logit_severity.py
  CUDA_VISIBLE_DEVICES=0 /home/jaeik/miniconda3/envs/apa/bin/python -u phase2/scripts/exp_logit_severity.py --test 30
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
OUT_DIR    = Path("phase2/results/exp_logit_severity")

SWEEP_LAYERS = list(range(13, 20))   # L13–L19
DIR_NAME     = "B_svd_conf_support"
ALPHA        = 1.0
CHECKPOINT_EVERY = 200

SEVERITY_BINS: List[Tuple[float, float, str]] = [
    (0.0,   1.0,  "low"),
    (1.0,   2.0,  "medium"),
    (2.0,   3.5,  "high"),
    (3.5, 999.0,  "severe"),
]

LETTER_VARIANTS: Dict[str, List[int]] = {}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",    type=int, default=None, metavar="N")
    p.add_argument("--n_cases", type=int, default=None)
    return p.parse_args()


# ── 모델 ──────────────────────────────────────────────────────────────────────
def load_model() -> HookedTransformer:
    print("모델 로드 (bfloat16) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16)
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
def _logprobs(logits: torch.Tensor, cor: str, rep: str):
    lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    lp_cor = max(lp[tid].item() for tid in LETTER_VARIANTS[cor])
    lp_rep = max(lp[tid].item() for tid in LETTER_VARIANTS[rep])
    return lp_cor, lp_rep


def run_logprobs(model, tokens, cor, rep):
    with torch.no_grad():
        logits = model(tokens)
    return _logprobs(logits, cor, rep)


def run_logprobs_hooked(model, tokens, cor, rep, fwd_hooks):
    with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)
    return _logprobs(logits, cor, rep)


# ── hook ──────────────────────────────────────────────────────────────────────
def make_scale_hook(direction: torch.Tensor, alpha: float):
    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        v = value.float()
        d = direction.to(value.device).float()
        proj = v @ d
        v = v - alpha * proj.unsqueeze(-1) * d
        return v.to(value.dtype)
    return hook_fn


# ── 방향 로드 ─────────────────────────────────────────────────────────────────
def load_direction(cands: dict, layer: int, name: str = DIR_NAME) -> torch.Tensor:
    for entry in cands[layer]:
        if entry["name"] == name:
            return entry["dir"].float()
    raise KeyError(f"{name} not found at layer {layer}")


# ── 베이스라인 전처리 ─────────────────────────────────────────────────────────
def prepare_baselines(model, kc_cases: list) -> List[Dict]:
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

        tok_conf = model.to_tokens(build_conflict_prompt(ctx, q, opt), prepend_bos=True)
        tok_sup  = model.to_tokens(
            build_supportive_prompt(ctx, q, opt, ca, coa), prepend_bos=True)

        lp_cor_conf, lp_rep_conf = run_logprobs(model, tok_conf, cor, rep)
        lp_cor_sup,  lp_rep_sup  = run_logprobs(model, tok_sup,  cor, rep)

        data.append({
            "tok_conf":      tok_conf,
            "tok_sup":       tok_sup,
            "cor":           cor,
            "rep":           rep,
            "conflict_type": case["conflict_type"],
            "case_id":       case["id"],
            "kc_severity":   case["kc_severity"],
            "lp_cor_conf":   lp_cor_conf,
            "lp_rep_conf":   lp_rep_conf,
            "lp_cor_sup":    lp_cor_sup,
            "lp_rep_sup":    lp_rep_sup,
            "has_kc":        lp_rep_conf > lp_cor_conf,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [baseline] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[baseline] 완료 ({time.time()-t0:.0f}s)", flush=True)
    return data


# ── Layer sweep ───────────────────────────────────────────────────────────────
def run_layer_sweep(model, case_data: List[Dict], cands: dict) -> Dict[int, List[Dict]]:
    """B × 각 레이어에 대해 4-delta + flip 측정."""
    all_results: Dict[int, List[Dict]] = {}
    n = len(case_data)

    for layer in SWEEP_LAYERS:
        dir_B = load_direction(cands, layer)
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn   = make_scale_hook(dir_B, ALPHA)
        fwd_hooks = [(hook_name, hook_fn)]

        print(f"\n[sweep] B × L{layer} ...", flush=True)
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
                print(f"  {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

        all_results[layer] = results
        kc_sup = sum(1 for r in results if r["kc_flipped"]) / n
        ie_kc  = np.mean([r["delta_cor_conf"] for r in results])
        ie_sup = np.mean([r["delta_cor_sup"]  for r in results])
        print(f"  L{layer}: IE_kc={ie_kc:+.3f}  IE_sup={ie_sup:+.3f}  KC_sup={kc_sup:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    return all_results


# ── 집계 ──────────────────────────────────────────────────────────────────────
def aggregate(all_results: Dict[int, List[Dict]]) -> Dict[int, Dict]:
    agg = {}
    for layer, results in all_results.items():
        n = len(results)

        def _s(vals):
            arr = np.array(vals)
            return {"mean": float(arr.mean()), "std": float(arr.std()),
                    "pos_ratio": float((arr > 0).mean())}

        summary = {
            "delta_cor_conf": _s([r["delta_cor_conf"] for r in results]),
            "delta_rep_conf": _s([r["delta_rep_conf"] for r in results]),
            "delta_cor_sup":  _s([r["delta_cor_sup"]  for r in results]),
            "delta_rep_sup":  _s([r["delta_rep_sup"]  for r in results]),
            "kc_suppression": sum(1 for r in results if r["kc_flipped"]) / n,
            "dissociation":   sum(1 for r in results
                                  if r["delta_cor_conf"] > r["delta_cor_sup"]) / n,
        }

        # severity 구간별
        sev_bins = []
        for lo, hi, label in SEVERITY_BINS:
            sub = [r for r in results if lo <= r["kc_severity"] < hi]
            if not sub:
                sev_bins.append({"label": label, "n": 0})
                continue
            sev_arr  = np.array([r["kc_severity"]    for r in sub])
            ie_arr   = np.array([r["delta_cor_conf"]  for r in sub])
            rep_arr  = np.array([r["delta_rep_conf"]  for r in sub])
            flipped  = sum(1 for r in sub if r["kc_flipped"])
            sev_bins.append({
                "label":      label,
                "range":      [lo, hi],
                "n":          len(sub),
                "sev_mean":   float(sev_arr.mean()),
                "ie_kc_mean": float(ie_arr.mean()),
                "ie_kc_std":  float(ie_arr.std()),
                "rep_mean":   float(rep_arr.mean()),
                "flip_rate":  flipped / len(sub),
                "flip_count": flipped,
            })
        summary["severity_bins"] = sev_bins

        # r(severity, IE_kc)
        sev_all = np.array([r["kc_severity"]   for r in results])
        ie_all  = np.array([r["delta_cor_conf"] for r in results])
        r_val, p_val = stats.pearsonr(sev_all, ie_all)
        summary["r_sev_ie"]  = float(r_val)
        summary["p_sev_ie"]  = float(p_val)

        agg[layer] = summary
    return agg


# ── 그래프 ────────────────────────────────────────────────────────────────────
# A × L16 참조값 (exp_supp Exp1/Exp2 결과)
A_REF = {
    "delta_cor_conf": +0.9164,
    "delta_rep_conf": -0.1284,
    "delta_cor_sup":  -0.0447,
    "delta_rep_sup":  +0.3862,
    "kc_suppression": 0.007,
}
A_SEV_REF = {
    "low":    {"sev_mean": 0.528, "ie_kc_mean": 0.198, "flip_rate": 0.353},
    "medium": {"sev_mean": 1.608, "ie_kc_mean": 0.409, "flip_rate": 0.008},
    "high":   {"sev_mean": 2.806, "ie_kc_mean": 0.774, "flip_rate": 0.000},
    "severe": {"sev_mean": 4.190, "ie_kc_mean": 1.273, "flip_rate": 0.000},
}


def plot_logit_tracking(agg: Dict[int, Dict], out_dir: Path) -> None:
    """Figure 1: 레이어별 4-delta 꺾은선 + KC_sup."""
    layers = SWEEP_LAYERS
    keys   = ["delta_cor_conf", "delta_rep_conf", "delta_cor_sup", "delta_rep_sup"]
    clrs   = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    lbls   = ["Δcor_conflict (IE_kc)", "Δrep_conflict", "Δcor_supp (IE_sup)", "Δrep_supp"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("B Direction Logit Tracking: L13-L19", fontsize=13, fontweight="bold")

    # Left: 4-delta lines
    ax = axes[0]
    for key, clr, lbl in zip(keys, clrs, lbls):
        vals = [agg[l][key]["mean"] for l in layers]
        ax.plot(layers, vals, color=clr, lw=2, marker="o", markersize=5, label=lbl)

    # A × L16 참조 점선
    for key, clr in zip(keys, clrs):
        ax.axhline(A_REF[key], color=clr, lw=1, ls="--", alpha=0.4)

    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Delta log-prob (mean)", fontsize=11)
    ax.set_title("4-delta by Layer\n(dashed = A×L16 reference)", fontsize=10.5)
    ax.set_xticks(layers)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # Right: KC_sup + dissociation
    ax2 = axes[1]
    kc_sups = [agg[l]["kc_suppression"] for l in layers]
    dissos  = [agg[l]["dissociation"]   for l in layers]
    ax2.bar([l - 0.2 for l in layers], kc_sups, width=0.35,
            color="#e74c3c", alpha=0.8, label="KC suppression (rank flip)")
    ax2.bar([l + 0.2 for l in layers], dissos,  width=0.35,
            color="#3498db", alpha=0.8, label="Dissociation ratio")
    ax2.axhline(A_REF["kc_suppression"], color="#e74c3c", lw=1.2, ls="--",
                alpha=0.6, label=f"A×L16 KC_sup={A_REF['kc_suppression']}")
    for l, v in zip(layers, kc_sups):
        ax2.text(l - 0.2, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color="#c0392b", fontweight="bold")
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Rate", fontsize=11)
    ax2.set_title("KC Suppression & Dissociation by Layer", fontsize=10.5)
    ax2.set_xticks(layers)
    ax2.legend(fontsize=8.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "figure1_logit_tracking.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] figure1_logit_tracking.png", flush=True)


def plot_severity(agg: Dict[int, Dict], out_dir: Path) -> None:
    """Figure 2: severity별 flip_rate heatmap (레이어 × 구간)."""
    layers    = SWEEP_LAYERS
    bin_labels = [b[2] for b in SEVERITY_BINS]

    # flip_rate matrix: rows=layers, cols=severity bins
    flip_mat = np.zeros((len(layers), len(bin_labels)))
    ie_mat   = np.zeros((len(layers), len(bin_labels)))
    n_mat    = np.zeros((len(layers), len(bin_labels)), dtype=int)

    for li, layer in enumerate(layers):
        for bi, blabel in enumerate(bin_labels):
            bins = agg[layer]["severity_bins"]
            b = next((x for x in bins if x["label"] == blabel), None)
            if b and b.get("n", 0) > 0:
                flip_mat[li, bi] = b["flip_rate"]
                ie_mat[li, bi]   = b["ie_kc_mean"]
                n_mat[li, bi]    = b["n"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("B Direction: Severity-based Analysis by Layer", fontsize=13, fontweight="bold")

    # Left: flip_rate heatmap
    ax = axes[0]
    im = ax.imshow(flip_mat, cmap="Reds", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=10)
    ax.set_xlabel("Severity Bin", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Flip Rate (KC Suppression)\nby Severity Bin × Layer", fontsize=10.5)
    for li in range(len(layers)):
        for bi in range(len(bin_labels)):
            v = flip_mat[li, bi]
            n = n_mat[li, bi]
            txt = f"{v:.2f}\n(n={n})" if n > 0 else "—"
            color = "white" if v > 0.3 else "black"
            ax.text(bi, li, txt, ha="center", va="center", fontsize=7.5, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Flip Rate")

    # Right: IE_kc by severity bin × layer (line plot)
    ax2 = axes[1]
    sev_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c", "severe": "#8e44ad"}
    for bi, blabel in enumerate(bin_labels):
        vals = ie_mat[:, bi]
        mask = n_mat[:, bi] > 0
        ax2.plot(np.array(layers)[mask], vals[mask],
                 color=sev_colors[blabel], lw=2, marker="o", markersize=5,
                 label=f"{blabel} (A×L16: {A_SEV_REF[blabel]['ie_kc_mean']:.3f})")
        # A reference dashed
        ax2.axhline(A_SEV_REF[blabel]["ie_kc_mean"], color=sev_colors[blabel],
                    lw=1, ls="--", alpha=0.4)

    ax2.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("IE_kc mean", fontsize=11)
    ax2.set_title("IE_kc by Severity Bin × Layer\n(dashed = A×L16 reference)", fontsize=10.5)
    ax2.set_xticks(layers)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "figure2_severity_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] figure2_severity_by_layer.png", flush=True)


def plot_summary(agg: Dict[int, Dict], out_dir: Path) -> None:
    """Figure 3: 레이어별 delta_cor_conf / delta_rep_conf / KC_sup 한눈에."""
    layers = SWEEP_LAYERS
    ie_kc  = [agg[l]["delta_cor_conf"]["mean"] for l in layers]
    d_rep  = [agg[l]["delta_rep_conf"]["mean"] for l in layers]
    ie_sup = [agg[l]["delta_cor_sup"]["mean"]  for l in layers]
    kc_sup = [agg[l]["kc_suppression"]          for l in layers]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle("B Direction Summary: L13-L19", fontsize=12, fontweight="bold")

    ax1.plot(layers, ie_kc,  "r-o", lw=2.2, markersize=6,  label="IE_kc (Δcor_conflict)")
    ax1.plot(layers, d_rep,  color="#e67e22", lw=1.8, marker="s", markersize=5,
             label="Δrep_conflict")
    ax1.plot(layers, ie_sup, "b--o", lw=1.5, markersize=5, label="IE_sup (Δcor_supp)")
    ax1.axhline(A_REF["delta_cor_conf"], color="red",    lw=1, ls=":", alpha=0.5,
                label=f"A×L16 IE_kc={A_REF['delta_cor_conf']:.3f}")
    ax1.axhline(A_REF["delta_rep_conf"], color="#e67e22", lw=1, ls=":", alpha=0.5)
    ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Delta log-prob", fontsize=11, color="black")
    ax1.set_xticks(layers)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar([l + 0.25 for l in layers], kc_sup, width=0.4,
            color="green", alpha=0.35, label="KC_sup (right)")
    for l, v in zip(layers, kc_sup):
        ax2.text(l + 0.25, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color="green", fontweight="bold")
    ax2.set_ylabel("KC Suppression (rank flip)", fontsize=10, color="green")
    ax2.set_ylim(0, 0.35)
    ax2.axhline(A_REF["kc_suppression"], color="green", lw=1, ls=":", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_dir / "figure3_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] figure3_summary.png", flush=True)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    with open(KC_PATH) as f:
        kc_cases = json.load(f)
    if args.test:
        kc_cases = kc_cases[:args.test]
        print(f"[test] {args.test}개 케이스만 사용", flush=True)
    elif args.n_cases:
        kc_cases = kc_cases[:args.n_cases]
    print(f"[data] {len(kc_cases)}개 케이스 / layers={SWEEP_LAYERS}", flush=True)

    # 후보 방향 로드
    cands = torch.load(CANDS_PATH, map_location="cpu", weights_only=False)

    # 모델 로드
    model = load_model()
    setup_letter_variants(model.tokenizer)

    # 베이스라인
    case_data = prepare_baselines(model, kc_cases)

    # Sweep
    all_results = run_layer_sweep(model, case_data, cands)

    # 집계
    agg = aggregate(all_results)

    # 결과 저장
    # per-case JSON: layer별로 분리
    per_case_out = {}
    for layer, results in all_results.items():
        per_case_out[f"L{layer}"] = results
    with open(OUT_DIR / "per_case.json", "w") as f:
        json.dump(per_case_out, f, indent=2)

    # summary JSON
    agg_json = {f"L{layer}": v for layer, v in agg.items()}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(agg_json, f, indent=2)

    # 요약 출력
    print("\n=== 결과 요약 ===", flush=True)
    print(f"{'Layer':>6} {'IE_kc':>8} {'d_rep':>8} {'IE_sup':>8} {'KC_sup':>8} {'disso':>6}")
    print("-" * 50)
    for layer in SWEEP_LAYERS:
        s = agg[layer]
        print(f"  L{layer:2d}  {s['delta_cor_conf']['mean']:+8.3f} "
              f"{s['delta_rep_conf']['mean']:+8.3f} "
              f"{s['delta_cor_sup']['mean']:+8.3f} "
              f"{s['kc_suppression']:8.3f} "
              f"{s['dissociation']:6.3f}", flush=True)

    # 그래프
    plot_logit_tracking(agg, OUT_DIR)
    plot_severity(agg, OUT_DIR)
    plot_summary(agg, OUT_DIR)

    print(f"\n[완료] → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exp2: Direction-Specific Causal Patching

IE[cand][layer][pos] = E[logP(correct | direction removed) - logP(correct | conflict)]
IE > 0  →  direction 제거 → KC 감소 (correct answer 더 가능성 높아짐)

핵심 비교: IE_D vs IE_A
  IE_D > IE_A  →  KC-specific direction(D)이 context direction(A)보다 KC와 더 인과적

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp2_causal_patching.py
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp2_causal_patching.py --test 20
  CUDA_VISIBLE_DEVICES=0 conda run -n apa python -u phase2/scripts/exp2_causal_patching.py --clean_target
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
from transformer_lens import HookedTransformer

# ── 상수 ──────────────────────────────────────────────────────────────────────
LAYERS     = list(range(6, 20))
POSITIONS  = ["final", "pre_answer"]
CANDIDATES = ["D", "C", "A"]          # 검증 우선순위 (B는 제외)

CAND_NAME_MAP: Dict[str, str] = {
    "A": "A_svd_conf_clean",
    "B": "B_svd_conf_support",
    "C": "C_cpca",
    "D": "D_regression",
}

MODEL_NAME         = "google/gemma-2-2b"
KC_PATH            = Path("data/phase0/kc_cases_1000.json")
CANDS_PATH         = Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt")
EXP1B_PATH         = Path("phase2/results/exp1b_candidates/summary.json")
RESULTS_DIR        = Path("phase2/results/exp2_causal_patching")
CHECKPOINT_EVERY   = 50
CLEAN_MEAN_N_CASES = 200
SEED               = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

LETTER_VARIANTS: Dict[str, List[int]] = {}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=int, default=None, metavar="N",
                   help="N개만 처리 (테스트용)")
    p.add_argument("--n_cases", type=int, default=None,
                   help="케이스 수 지정 (기본: 전체)")
    p.add_argument("--clean_target", action="store_true",
                   help="IE_clean_target (clean mean projection 기준 패치) 추가 계산")
    return p.parse_args()


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model() -> HookedTransformer:
    print("모델 로드 (bfloat16) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.bfloat16,
    )
    model.eval()
    return model


# ── 토크나이저 설정 ────────────────────────────────────────────────────────────
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


# ── 로그 확률 ──────────────────────────────────────────────────────────────────
def get_letter_logprob(logits: torch.Tensor, letter: str) -> float:
    """마지막 토큰 위치에서 letter의 log-probability (max over variants)."""
    log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    return max(log_probs[tid].item() for tid in LETTER_VARIANTS[letter])


# ── Position 인덱스 ────────────────────────────────────────────────────────────
def get_pos_idx(seq_len: int, pos_name: str) -> int:
    if pos_name == "final":
        return seq_len - 1
    elif pos_name == "pre_answer":
        return seq_len - 2    # "Answer:" 직전 토큰
    raise ValueError(f"Unknown position: {pos_name}")


# ── Hook 팩토리 ────────────────────────────────────────────────────────────────
def make_patch_hook(h_patched: torch.Tensor, pos_idx: int):
    """특정 위치의 residual을 h_patched로 교체하는 hook."""
    def _fn(val: torch.Tensor, hook) -> torch.Tensor:
        val[0, pos_idx, :] = h_patched
        return val
    return _fn


# ── 방향 로드 헬퍼 ────────────────────────────────────────────────────────────
def build_dir_cache(
    cands: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """candidates_for_exp2.pt에서 검증할 후보 방향을 GPU로 미리 로드."""
    dirs: Dict[str, Dict[int, torch.Tensor]] = {}
    for cand in CANDIDATES:
        name = CAND_NAME_MAP[cand]
        dirs[cand] = {}
        for l in LAYERS:
            for entry in cands[l]:
                if entry["name"] == name:
                    d = entry["dir"].to(device=device, dtype=torch.float32)
                    assert abs(d.norm().item() - 1.0) < 1e-3, \
                        f"L{l} {cand} not unit norm: {d.norm().item():.5f}"
                    dirs[cand][l] = d
                    break
    return dirs


# ── 빈 IE 구조 생성 ────────────────────────────────────────────────────────────
def make_ie_store() -> Dict:
    return {
        cand: {l: {pos: [] for pos in POSITIONS} for l in LAYERS}
        for cand in CANDIDATES
    }


# ── Clean mean projection 계산 ─────────────────────────────────────────────────
def compute_clean_mean_proj(
    model: HookedTransformer,
    kc_cases: list,
    dirs: Dict[str, Dict[int, torch.Tensor]],
    n_cases: int,
) -> Dict:
    """N개 neutral run으로 각 (candidate, layer, position) 평균 projection 계산."""
    print(f"\n[clean_mean] {n_cases}개 케이스 neutral run ...", flush=True)
    accum = {
        cand: {l: {pos: [] for pos in POSITIONS} for l in LAYERS}
        for cand in CANDIDATES
    }
    t0 = time.time()

    for i, case in enumerate(kc_cases[:n_cases]):
        prompt = build_neutral_prompt(case["question"], case["options"])
        tokens = model.to_tokens(prompt, prepend_bos=True)
        seq_len = tokens.shape[1]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name,
                return_type=None,
            )

        for cand in CANDIDATES:
            for l in LAYERS:
                direction = dirs[cand][l]
                for pos_name in POSITIONS:
                    pos_idx = get_pos_idx(seq_len, pos_name)
                    h = cache[f"blocks.{l}.hook_resid_post"][0, pos_idx, :].float()
                    accum[cand][l][pos_name].append((h @ direction).item())

        del cache
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  clean_mean {i+1}/{n_cases} ({time.time()-t0:.0f}s)", flush=True)

    result: Dict = {}
    for cand in CANDIDATES:
        result[cand] = {}
        for l in LAYERS:
            result[cand][l] = {}
            for pos_name in POSITIONS:
                result[cand][l][pos_name] = float(np.mean(accum[cand][l][pos_name]))

    print(f"  [clean_mean] 완료 ({time.time()-t0:.0f}s)", flush=True)
    return result


# ── 결과 집계 ──────────────────────────────────────────────────────────────────
def aggregate_ie(IE: Dict) -> Dict:
    summary: Dict = {}
    for cand in CANDIDATES:
        summary[cand] = {}
        for l in LAYERS:
            summary[cand][l] = {}
            for pos_name in POSITIONS:
                vals = np.array(IE[cand][l][pos_name], dtype=float)
                summary[cand][l][pos_name] = {
                    "mean":           float(np.mean(vals)),
                    "std":            float(np.std(vals)),
                    "median":         float(np.median(vals)),
                    "positive_ratio": float(np.mean(vals > 0)),
                }
    return summary


# ── 시각화 ────────────────────────────────────────────────────────────────────
CAND_COLORS = {"D": "#e74c3c", "C": "#2ecc71", "A": "#3498db"}
CAND_LABELS = {"D": "D (KC-specific)", "C": "C (cPCA)", "A": "A (context dir)"}


def plot_figure1_heatmap(summary: Dict, out_dir: Path) -> None:
    """Figure 1: 후보별 IE bar chart (final position)."""
    fig, axes = plt.subplots(len(CANDIDATES), 1, figsize=(12, 3.5 * len(CANDIDATES)), sharex=True)

    for ax, cand in zip(axes, CANDIDATES):
        vals = [summary[cand][l]["final"]["mean"] for l in LAYERS]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
        ax.bar(LAYERS, vals, color=colors, alpha=0.8, zorder=2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Candidate {CAND_LABELS[cand]} — IE (final, target_proj=0)", fontsize=11)
        ax.set_ylabel("IE mean")
        ax.set_xticks(LAYERS)
        ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=1)

    axes[-1].set_xlabel("Layer")
    plt.suptitle("Figure 1: IE Heatmap by Candidate x Layer (final position)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "figure1_IE_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure2_D_vs_A(summary: Dict, out_dir: Path) -> None:
    """Figure 2: IE_D vs IE_A 직접 비교 — 핵심 그래프."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

    for ax, pos_name in zip(axes, POSITIONS):
        ie_D   = np.array([summary["D"][l][pos_name]["mean"] for l in LAYERS])
        ie_A   = np.array([summary["A"][l][pos_name]["mean"] for l in LAYERS])
        ie_C   = np.array([summary["C"][l][pos_name]["mean"] for l in LAYERS])
        std_D  = np.array([summary["D"][l][pos_name]["std"]  for l in LAYERS])
        std_A  = np.array([summary["A"][l][pos_name]["std"]  for l in LAYERS])
        std_C  = np.array([summary["C"][l][pos_name]["std"]  for l in LAYERS])

        # IE_D > IE_A 구간 배경 하이라이트
        for i, l in enumerate(LAYERS):
            if ie_D[i] > ie_A[i]:
                ax.axvspan(l - 0.45, l + 0.45, alpha=0.10, color="#e74c3c", zorder=0)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        ax.errorbar(LAYERS, ie_D, yerr=std_D, label=CAND_LABELS["D"],
                    color=CAND_COLORS["D"], marker="o", linewidth=2.2, capsize=3, zorder=3)
        ax.errorbar(LAYERS, ie_A, yerr=std_A, label=CAND_LABELS["A"],
                    color=CAND_COLORS["A"], marker="s", linewidth=2.2, capsize=3, zorder=3,
                    linestyle="--")
        ax.errorbar(LAYERS, ie_C, yerr=std_C, label=CAND_LABELS["C"],
                    color=CAND_COLORS["C"], marker="^", linewidth=1.5, capsize=3, zorder=2,
                    linestyle=":", alpha=0.85)

        # D-A diff 수치 표기
        for i, l in enumerate(LAYERS):
            diff = ie_D[i] - ie_A[i]
            y_top = max(ie_D[i], ie_A[i]) + 0.008
            color = "#c0392b" if diff > 0 else "#7f8c8d"
            if abs(diff) > 0.005:
                ax.annotate(f"{diff:+.3f}", (l, y_top),
                            ha="center", va="bottom", fontsize=6.5, color=color)

        ax.set_title(f"Position: {pos_name}", fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("IE mean (log-prob increase)")
        ax.set_xticks(LAYERS)
        ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9, rotation=45)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.suptitle(
        "Figure 2: IE_D vs IE_A — KC-specific vs Context Direction\n"
        "(red shading: IE_D > IE_A regions)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "figure2_D_vs_A.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure3_testr_vs_ie(
    summary: Dict, exp1b_summary: Dict, out_dir: Path
) -> None:
    """Figure 3: Exp1b D test_r vs Exp2 IE_D (final position)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    test_r = [exp1b_summary.get(str(l), {}).get("D_test_r", float("nan")) for l in LAYERS]
    ie_D   = [summary["D"][l]["final"]["mean"] for l in LAYERS]

    ax.plot(LAYERS, test_r, color="#e74c3c", marker="o", linewidth=2,
            label="Exp1b D test_r (representational)")
    ax2.plot(LAYERS, ie_D,  color="#2ecc71", marker="s", linewidth=2,
             linestyle="--", label="Exp2 IE_D final (causal)")

    ax.axhline(0, color="#e74c3c", linewidth=0.5, linestyle=":")
    ax2.axhline(0, color="#2ecc71", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Exp1b D test_r", color="#e74c3c")
    ax2.set_ylabel("Exp2 IE_D mean", color="#2ecc71")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
    ax.tick_params(axis="y", colors="#e74c3c")
    ax2.tick_params(axis="y", colors="#2ecc71")

    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbs1 + lbs2, fontsize=9)

    plt.title("Figure 3: Exp1b test_r vs Exp2 IE_D\n"
              "(일치 → representational 예측력 = causal leverage)", fontsize=12)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = out_dir / "figure3_testr_vs_IE.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


def plot_figure4_IE_by_position(summary: Dict, out_dir: Path) -> None:
    """Figure 4: IE by position (final vs pre_answer) for candidate D."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ie_f   = [summary["D"][l]["final"]["mean"]       for l in LAYERS]
    ie_p   = [summary["D"][l]["pre_answer"]["mean"]  for l in LAYERS]
    std_f  = [summary["D"][l]["final"]["std"]        for l in LAYERS]
    std_p  = [summary["D"][l]["pre_answer"]["std"]   for l in LAYERS]

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.errorbar(LAYERS, ie_f, yerr=std_f, label="final (pos -1, ':')",
                color="#e74c3c", marker="o", linewidth=2.2, capsize=3)
    ax.errorbar(LAYERS, ie_p, yerr=std_p, label="pre_answer (pos -2, 'Answer')",
                color="#9b59b6", marker="s", linewidth=2.2, capsize=3, linestyle="--")

    ax.set_xlabel("Layer")
    ax.set_ylabel("IE mean — Candidate D")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.title("Figure 4: IE by Position — Candidate D", fontsize=13)
    plt.tight_layout()
    path = out_dir / "figure4_IE_by_position.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}", flush=True)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    kc_cases: list = json.load(open(KC_PATH))
    n_cases = args.test or args.n_cases or len(kc_cases)
    kc_cases = kc_cases[:n_cases]
    print(f"[exp2] N_CASES={n_cases}", flush=True)

    # 후보 방향 로드
    cands = torch.load(CANDS_PATH, map_location="cpu")
    print(f"[exp2] 후보: {[e['name'] for e in cands[LAYERS[0]]]}", flush=True)

    # 모델 로드
    model = load_model()
    setup_letter_variants(model.tokenizer)
    device = next(model.parameters()).device
    dtype  = torch.bfloat16

    # 방향 벡터 GPU 사전 로드 + 정규화 검증
    dirs = build_dir_cache(cands, device, dtype=torch.float32)
    print("[exp2] 방향 벡터 정규화 확인 완료", flush=True)

    # 체크포인트 확인
    start_idx = 0
    IE       = make_ie_store()
    IE_clean = make_ie_store() if args.clean_target else None

    ckpt_path = RESULTS_DIR / "IE_checkpoint_latest.pt"
    if ckpt_path.exists():
        ckpt      = torch.load(ckpt_path, map_location="cpu")
        IE        = ckpt["IE"]
        IE_clean  = ckpt.get("IE_clean")
        start_idx = ckpt["last_case_idx"] + 1
        print(f"[exp2] 체크포인트 재개: case {start_idx}부터", flush=True)

    # Clean mean projection 계산 (IE_clean_target 사용 시)
    clean_mean_proj: Optional[Dict] = None
    if args.clean_target:
        cmp_path = RESULTS_DIR / "clean_mean_proj.pt"
        if cmp_path.exists():
            clean_mean_proj = torch.load(cmp_path, map_location="cpu")
            print(f"[exp2] clean_mean_proj 로드 완료", flush=True)
        else:
            clean_n = min(CLEAN_MEAN_N_CASES, n_cases)
            clean_mean_proj = compute_clean_mean_proj(model, kc_cases, dirs, clean_n)
            torch.save(clean_mean_proj, cmp_path)

    # ── 메인 IE 루프 ──────────────────────────────────────────────────────────
    n_total = len(kc_cases)
    n_patches = len(CANDIDATES) * len(LAYERS) * len(POSITIONS)
    n_passes_per_case = n_patches * (2 if args.clean_target else 1)
    print(
        f"\n[IE loop] {start_idx}부터 {n_total}개 처리 "
        f"({n_patches} patches/case, "
        f"total ~{(n_total - start_idx) * n_passes_per_case} forward passes) ...",
        flush=True,
    )
    t0 = time.time()

    for case_idx, case in enumerate(kc_cases[start_idx:], start=start_idx):
        ctx  = case["conflict_context"]
        q    = case["question"]
        opts = case["options"]
        cor  = case["correct_option"]

        conflict_prompt = build_conflict_prompt(ctx, q, opts)
        tokens  = model.to_tokens(conflict_prompt, prepend_bos=True)
        seq_len = tokens.shape[1]

        # conflict forward with cache (모든 레이어 resid_post)
        with torch.no_grad():
            logits_conflict, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name,
            )

        logp_conflict = get_letter_logprob(logits_conflict, cor)
        del logits_conflict

        for cand in CANDIDATES:
            for l in LAYERS:
                direction = dirs[cand][l]   # float32, GPU

                for pos_name in POSITIONS:
                    pos_idx = get_pos_idx(seq_len, pos_name)
                    h_conf  = cache[f"blocks.{l}.hook_resid_post"][0, pos_idx, :].float()

                    # ── target_proj = 0 (방향 성분 완전 제거) ──────────────────
                    proj   = (h_conf @ direction).item()
                    h_p    = (h_conf - proj * direction).to(dtype=dtype)

                    with torch.no_grad():
                        logits_p = model.run_with_hooks(
                            tokens,
                            fwd_hooks=[(
                                f"blocks.{l}.hook_resid_post",
                                make_patch_hook(h_p, pos_idx),
                            )],
                        )
                    IE[cand][l][pos_name].append(
                        get_letter_logprob(logits_p, cor) - logp_conflict
                    )
                    del logits_p

                    # ── target_proj = clean_mean (선택적) ─────────────────────
                    if IE_clean is not None and clean_mean_proj is not None:
                        t_proj = clean_mean_proj[cand][l][pos_name]
                        h_c    = (h_conf - (proj - t_proj) * direction).to(dtype=dtype)

                        with torch.no_grad():
                            logits_c = model.run_with_hooks(
                                tokens,
                                fwd_hooks=[(
                                    f"blocks.{l}.hook_resid_post",
                                    make_patch_hook(h_c, pos_idx),
                                )],
                            )
                        IE_clean[cand][l][pos_name].append(
                            get_letter_logprob(logits_c, cor) - logp_conflict
                        )
                        del logits_c

        del cache
        gc.collect()

        if (case_idx + 1) % CHECKPOINT_EVERY == 0:
            torch.cuda.empty_cache()
            ckpt = {"IE": IE, "IE_clean": IE_clean, "last_case_idx": case_idx}
            torch.save(ckpt, RESULTS_DIR / "IE_checkpoint_latest.pt")
            torch.save(ckpt, RESULTS_DIR / f"IE_checkpoint_{case_idx+1}.pt")
            elapsed = time.time() - t0
            rate    = (case_idx + 1 - start_idx) / elapsed
            eta     = (n_total - case_idx - 1) / rate if rate > 0 else float("nan")
            print(
                f"  [ckpt] {case_idx+1}/{n_total} ({elapsed:.0f}s, "
                f"{rate:.2f} case/s, ETA {eta:.0f}s)",
                flush=True,
            )

    print(f"[IE loop] 완료 ({time.time()-t0:.0f}s)", flush=True)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    torch.save(IE, RESULTS_DIR / "IE_full.pt")
    print(f"[save] IE_full.pt", flush=True)

    if IE_clean is not None:
        torch.save(IE_clean, RESULTS_DIR / "IE_clean_target.pt")
        print(f"[save] IE_clean_target.pt", flush=True)

    # ── 집계 ──────────────────────────────────────────────────────────────────
    summary = aggregate_ie(IE)

    # IE_D - IE_A 차이값 추가
    for l in LAYERS:
        for pos_name in POSITIONS:
            d_minus_a = summary["D"][l][pos_name]["mean"] - summary["A"][l][pos_name]["mean"]
            summary["D"][l][pos_name]["IE_D_minus_A"] = float(d_minus_a)

    # JSON 직렬화 (int key → str)
    summary_json: Dict = {}
    for cand in CANDIDATES:
        summary_json[cand] = {}
        for l in LAYERS:
            summary_json[cand][str(l)] = summary[cand][l]

    with open(RESULTS_DIR / "IE_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[save] IE_summary.json", flush=True)

    # Exp3 입력: IE 높은 상위 10개
    top_combinations = []
    for cand in CANDIDATES:
        for l in LAYERS:
            for pos_name in POSITIONS:
                s = summary[cand][l][pos_name]
                top_combinations.append({
                    "candidate":      cand,
                    "layer":          l,
                    "position":       pos_name,
                    "IE_mean":        s["mean"],
                    "IE_std":         s["std"],
                    "positive_ratio": s["positive_ratio"],
                })
    top_combinations.sort(key=lambda x: x["IE_mean"], reverse=True)
    with open(RESULTS_DIR / "top_combinations_for_exp3.json", "w") as f:
        json.dump(top_combinations[:10], f, indent=2)
    print(f"[save] top_combinations_for_exp3.json", flush=True)

    # ── 핵심 결과 출력 ────────────────────────────────────────────────────────
    print("\n=== IE_D vs IE_A (final position) ===")
    print(f"{'Layer':<7} {'IE_D':>8} {'IE_A':>8} {'IE_C':>8} {'D-A diff':>10} {'pos_ratio_D':>12}")
    for l in LAYERS:
        ie_D  = summary["D"][l]["final"]["mean"]
        ie_A  = summary["A"][l]["final"]["mean"]
        ie_C  = summary["C"][l]["final"]["mean"]
        diff  = ie_D - ie_A
        pr_D  = summary["D"][l]["final"]["positive_ratio"]
        marker = " *" if diff > 0 else ""
        print(f"  L{l:<4} {ie_D:>8.4f} {ie_A:>8.4f} {ie_C:>8.4f} {diff:>10.4f} {pr_D:>12.3f}{marker}")

    print("\n=== 후보별 best (layer, position) ===")
    for cand in CANDIDATES:
        best_ie, best_l, best_pos = -float("inf"), None, None
        for l in LAYERS:
            for pos_name in POSITIONS:
                ie = summary[cand][l][pos_name]["mean"]
                if ie > best_ie:
                    best_ie, best_l, best_pos = ie, l, pos_name
        print(f"  {CAND_LABELS[cand]}: best IE={best_ie:.4f} at L{best_l}, {best_pos}")

    # Exp1b test_r vs IE_D 비교
    exp1b_summary: Dict = {}
    if EXP1B_PATH.exists():
        exp1b_summary = json.load(open(EXP1B_PATH))
        print("\n=== Exp1b test_r vs IE_D (representational vs causal) ===")
        for l in LAYERS:
            tr   = exp1b_summary.get(str(l), {}).get("D_test_r", float("nan"))
            ie_D = summary["D"][l]["final"]["mean"]
            print(f"  L{l}: test_r={tr:.3f}  IE_D={ie_D:.4f}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print("\n[exp2] 시각화 ...", flush=True)
    plot_figure1_heatmap(summary, RESULTS_DIR)
    plot_figure2_D_vs_A(summary, RESULTS_DIR)
    plot_figure3_testr_vs_ie(summary, exp1b_summary, RESULTS_DIR)
    plot_figure4_IE_by_position(summary, RESULTS_DIR)

    print(f"\n[exp2] 완료. 결과: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 4: Path Patching — Attention vs MLP KC 고착 컴포넌트 분석

방법: Component-level path patching
  1) Neutral run → hook_attn_out / hook_mlp_out / hook_resid_post 캐싱
     (레이어 13~25, final token 위치만)
  2) Corrupted run (문맥 임베딩 Gaussian 노이즈 σ=3×global_std) → P_corrupted[correct]
  3) Restored run × 3 조건 (레이어 13~25):
     A) hook_attn_out  교체 → IE_attn(l) = P_restored − P_corrupted
     B) hook_mlp_out   교체 → IE_mlp(l)
     C) hook_resid_post 교체 → IE_full(l)  (Step 3 검증용)

케이스당 전진 패스: 1 neutral + 1 corrupted + 13×3 = 41 회

실행:
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step4_path_patching.py
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step4_path_patching.py --test 10
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

# ── 경로 / 상수 ────────────────────────────────────────────────────────────────
KC_PATH    = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
STEP1_PATH = Path("/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_results.json")
OUT_DIR    = Path("/home/jaeik/xaimi/phase1/results/step4_path_patching")
MODEL_NAME = "google/gemma-2-2b"
SEED        = 42
MAX_SEQ_LEN = 2048
NOISE_MULT  = 3.0
CHECKPOINT_EVERY = 50
EARLY_THRESH = 3
LATE_THRESH  = 15

FOCUS_LAYERS = list(range(13, 26))   # 13, 14, ..., 25  (13개)
N_FOCUS      = len(FOCUS_LAYERS)

CTYPES = ("misinformation", "temporal", "semantic")
CTYPE_COLORS = {
    "misinformation": "#e74c3c",
    "temporal":       "#3498db",
    "semantic":       "#2ecc71",
}

LETTER_VARIANTS: Dict[str, List[int]] = {}

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=int, default=None, help="N개만 처리 (테스트용)")
    return p.parse_args()


# ── 토큰 variants ─────────────────────────────────────────────────────────────
def setup_letter_variants(tokenizer) -> None:
    global LETTER_VARIANTS
    for letter in "ABCD":
        ids: List[int] = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.append(toks[-1] if len(toks) > 1 else toks[0])
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
    print(f"[setup] LETTER_VARIANTS={LETTER_VARIANTS}")


# ── 프롬프트 빌더 ──────────────────────────────────────────────────────────────
def build_conflict_prompt(context: str, question: str, options: Dict[str, str]) -> str:
    lines = ["Context:", context, "", f"Question: {question}"]
    for l in "ABCD":
        lines.append(f"{l}. {options[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_neutral_prompt(question: str, options: Dict[str, str]) -> str:
    lines = [f"Question: {question}"]
    for l in "ABCD":
        lines.append(f"{l}. {options[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── 전역 노이즈 std ───────────────────────────────────────────────────────────
def compute_noise_std(model: HookedTransformer, cases: List[Dict]) -> float:
    sample = cases[:min(50, len(cases))]
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for c in sample:
            p   = build_conflict_prompt(c["conflict_context"], c["question"], c["options"])
            ids = model.to_tokens(p, prepend_bos=True)[0]
            if ids.shape[0] > MAX_SEQ_LEN:
                ids = ids[:MAX_SEQ_LEN]
            chunks.append(model.embed.W_E[ids].float().cpu())
    all_embs  = torch.cat(chunks, dim=0)
    std_val   = all_embs.std().item()
    noise_std = NOISE_MULT * std_val
    print(f"  embedding std={std_val:.4f},  noise σ={noise_std:.4f}")
    return noise_std


# ── 케이스 처리 ───────────────────────────────────────────────────────────────
def process_case(
    model:     HookedTransformer,
    case:      Dict,
    noise_std: float,
    sanity:    bool = False,
) -> Optional[Dict]:
    tokenizer = model.tokenizer
    ctx  = case["conflict_context"]
    q    = case["question"]
    opts = case["options"]
    cor  = case["correct_option"]

    conflict_p = build_conflict_prompt(ctx, q, opts)
    neutral_p  = build_neutral_prompt(q, opts)

    conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]
    neutral_ids  = model.to_tokens(neutral_p,  prepend_bos=True)[0]

    if conflict_ids.shape[0] > MAX_SEQ_LEN:
        conflict_ids = conflict_ids[:MAX_SEQ_LEN]

    n_c = conflict_ids.shape[0]
    n_n = neutral_ids.shape[0]
    dev = model.cfg.device
    d_m = model.cfg.d_model

    final_pos_c = n_c - 1   # conflict prompt의 마지막 토큰 (":")
    final_pos_n = n_n - 1   # neutral prompt의 마지막 토큰 (":")

    offset      = n_c - n_n
    ctx_indices = list(range(1, offset)) if offset > 1 else []

    # 케이스별 고정 노이즈 (context 위치에만)
    noise_vec = torch.zeros(n_c, d_m, dtype=torch.bfloat16, device=dev)
    if ctx_indices:
        ctx_t = torch.tensor(ctx_indices, dtype=torch.long, device=dev)
        noise_vec[ctx_t] = (
            torch.randn(len(ctx_indices), d_m, dtype=torch.bfloat16, device=dev)
            * noise_std
        )

    def noise_hook(val, hook):
        return val + noise_vec.unsqueeze(0)

    cor_tok_ids = LETTER_VARIANTS[cor]

    def get_prob(logits: torch.Tensor) -> float:
        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        return max(probs[tid].item() for tid in cor_tok_ids)

    # ── 1. Neutral run — final token의 3개 컴포넌트 캐싱 (레이어 13~25) ────────
    n_resid: Dict[int, torch.Tensor] = {}
    n_attn:  Dict[int, torch.Tensor] = {}
    n_mlp:   Dict[int, torch.Tensor] = {}

    def make_cache(store: dict, layer: int, pos: int):
        def _fn(val, hook):
            store[layer] = val[0, pos, :].detach().clone()
            return val
        return _fn

    neutral_hooks = []
    for l in FOCUS_LAYERS:
        neutral_hooks += [
            (f"blocks.{l}.hook_resid_post", make_cache(n_resid, l, final_pos_n)),
            (f"blocks.{l}.hook_attn_out",   make_cache(n_attn,  l, final_pos_n)),
            (f"blocks.{l}.hook_mlp_out",    make_cache(n_mlp,   l, final_pos_n)),
        ]

    with torch.no_grad():
        _ = model.run_with_hooks(neutral_ids.unsqueeze(0), fwd_hooks=neutral_hooks)

    # ── 2. Corrupted run → P_corrupted ────────────────────────────────────────
    with torch.no_grad():
        logits_c = model.run_with_hooks(
            conflict_ids.unsqueeze(0),
            fwd_hooks=[("hook_embed", noise_hook)],
        )
    p_corrupted = get_prob(logits_c)
    del logits_c
    torch.cuda.empty_cache()

    if sanity:
        with torch.no_grad():
            logits_n = model(neutral_ids.unsqueeze(0))
        p_neutral = get_prob(logits_n)
        del logits_n
        print(f"\n  [sanity] n_c={n_c}, n_n={n_n}, offset={offset}")
        print(f"    final_pos_c={final_pos_c}, final_pos_n={final_pos_n}")
        print(f"    P_neutral[{cor}]   = {p_neutral:.4f}")
        print(f"    P_corrupted[{cor}] = {p_corrupted:.4f}")
        print(f"    cached layers: {list(n_resid.keys())}")

    # ── 3. Restored runs (레이어 13~25, final token, 3조건) ────────────────────
    ie_attn = np.zeros(N_FOCUS, dtype=np.float32)
    ie_mlp  = np.zeros(N_FOCUS, dtype=np.float32)
    ie_full = np.zeros(N_FOCUS, dtype=np.float32)

    def make_patch_hook(pos: int, val: torch.Tensor):
        def _fn(v, hook):
            v[0, pos, :] = val
            return v
        return _fn

    for i, l in enumerate(FOCUS_LAYERS):
        attn_val  = n_attn[l]    # [d_model]
        mlp_val   = n_mlp[l]
        resid_val = n_resid[l]

        # 조건 A: hook_attn_out 교체
        with torch.no_grad():
            out_a = model.run_with_hooks(
                conflict_ids.unsqueeze(0),
                fwd_hooks=[
                    ("hook_embed", noise_hook),
                    (f"blocks.{l}.hook_attn_out",
                     make_patch_hook(final_pos_c, attn_val)),
                ],
            )
        ie_attn[i] = get_prob(out_a) - p_corrupted
        del out_a
        torch.cuda.empty_cache()

        # 조건 B: hook_mlp_out 교체
        with torch.no_grad():
            out_b = model.run_with_hooks(
                conflict_ids.unsqueeze(0),
                fwd_hooks=[
                    ("hook_embed", noise_hook),
                    (f"blocks.{l}.hook_mlp_out",
                     make_patch_hook(final_pos_c, mlp_val)),
                ],
            )
        ie_mlp[i] = get_prob(out_b) - p_corrupted
        del out_b
        torch.cuda.empty_cache()

        # 조건 C: hook_resid_post 교체 (검증용)
        with torch.no_grad():
            out_c = model.run_with_hooks(
                conflict_ids.unsqueeze(0),
                fwd_hooks=[
                    ("hook_embed", noise_hook),
                    (f"blocks.{l}.hook_resid_post",
                     make_patch_hook(final_pos_c, resid_val)),
                ],
            )
        ie_full[i] = get_prob(out_c) - p_corrupted
        del out_c
        torch.cuda.empty_cache()

    del n_resid, n_attn, n_mlp, noise_vec
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "id":                 case["id"],
        "conflict_type":      case["conflict_type"],
        "transition_group":   case.get("transition_group", "mid"),
        "transition_layer_A": case.get("transition_layer_A"),
        "correct_option":     cor,
        "P_corrupted":        float(p_corrupted),
        "IE_attn_by_layer":   ie_attn.tolist(),
        "IE_mlp_by_layer":    ie_mlp.tolist(),
        "IE_full_by_layer":   ie_full.tolist(),
        "kc_severity":        case.get("kc_severity", 0.0),
    }


# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def compute_stats(results: List[Dict]) -> Dict:
    def mean_arr(sub: List[Dict], key: str) -> List[float]:
        if not sub:
            return [0.0] * N_FOCUS
        return np.mean([np.array(r[key]) for r in sub], axis=0).tolist()

    def dominant_by_layer(attn_m: List[float], mlp_m: List[float]) -> Dict[str, str]:
        dom: Dict[str, str] = {}
        for i, l in enumerate(FOCUS_LAYERS):
            a, m = attn_m[i], mlp_m[i]
            if abs(a - m) < 1e-4:
                dom[str(l)] = "equal"
            elif a > m:
                dom[str(l)] = "attn"
            else:
                dom[str(l)] = "mlp"
        return dom

    def block_stats(sub: List[Dict]) -> Dict:
        attn = mean_arr(sub, "IE_attn_by_layer")
        mlp  = mean_arr(sub, "IE_mlp_by_layer")
        full = mean_arr(sub, "IE_full_by_layer")
        return {
            "IE_attn_mean":               attn,
            "IE_mlp_mean":                mlp,
            "IE_full_mean":               full,
            "dominant_component_by_layer": dominant_by_layer(attn, mlp),
        }

    return {
        "n":      len(results),
        "layers": FOCUS_LAYERS,
        "overall": block_stats(results),
        "by_type": {
            ct: block_stats([r for r in results if r["conflict_type"] == ct])
            for ct in CTYPES
        },
    }


# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
def print_analyses(results: List[Dict], stats: Dict) -> None:
    overall = stats["overall"]
    attn_m  = overall["IE_attn_mean"]
    mlp_m   = overall["IE_mlp_mean"]
    full_m  = overall["IE_full_mean"]
    dom     = overall["dominant_component_by_layer"]

    print(f"\n{'='*70}")
    print(f"레이어별 dominant component  (n={len(results)})")
    print(f"{'='*70}")
    print(f"  {'Layer':>5}  {'IE_attn':>10}  {'IE_mlp':>10}  {'IE_full':>10}  dominant")
    for i, l in enumerate(FOCUS_LAYERS):
        print(f"  {l:>5}  {attn_m[i]:>10.4f}  {mlp_m[i]:>10.4f}  "
              f"{full_m[i]:>10.4f}  {dom[str(l)]}")

    sum_attn = sum(attn_m)
    sum_mlp  = sum(mlp_m)
    if sum_attn > sum_mlp * 1.2:
        conclusion = "Attention"
    elif sum_mlp > sum_attn * 1.2:
        conclusion = "MLP"
    else:
        conclusion = "혼합 (Attn+MLP)"

    print(f"\n  결론: KC 고착의 주요 컴포넌트 = {conclusion}")
    print(f"  (Σ IE_attn={sum_attn:.4f},  Σ IE_mlp={sum_mlp:.4f})")

    print(f"\n{'='*70}")
    print("타입별 요약")
    print(f"{'='*70}")
    print(f"  {'type':<18}  {'Σ IE_attn':>10}  {'Σ IE_mlp':>10}  dominant")
    for ct in CTYPES:
        d    = stats["by_type"][ct]
        sa   = sum(d["IE_attn_mean"])
        sm   = sum(d["IE_mlp_mean"])
        dom_ct = "attn" if sa > sm * 1.2 else "mlp" if sm > sa * 1.2 else "혼합"
        n_ct = sum(1 for r in results if r["conflict_type"] == ct)
        print(f"  {ct:<18} (n={n_ct})  {sa:>10.4f}  {sm:>10.4f}  {dom_ct}")


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_overall(results: List[Dict], stats: Dict, out_path: Path) -> None:
    overall = stats["overall"]
    xs      = FOCUS_LAYERS
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(xs, overall["IE_attn_mean"], "b-o", label="IE_attn (attn_out)",
            linewidth=2, markersize=5)
    ax.plot(xs, overall["IE_mlp_mean"],  "r-o", label="IE_mlp  (mlp_out)",
            linewidth=2, markersize=5)
    ax.plot(xs, overall["IE_full_mean"], color="#888888", linestyle="--",
            label="IE_full (resid_post)", linewidth=1.5, markersize=3, marker="s")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("IE = P_restored − P_corrupted", fontsize=12)
    ax.set_title(f"Attention vs MLP — Path Patching (n={len(results)})", fontsize=13)
    ax.set_xticks(xs)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


def plot_by_type(results: List[Dict], stats: Dict, out_path: Path) -> None:
    xs = FOCUS_LAYERS
    # 공통 y 범위
    all_vals: List[float] = []
    for ct in CTYPES:
        d = stats["by_type"][ct]
        all_vals += d["IE_attn_mean"] + d["IE_mlp_mean"] + d["IE_full_mean"]
    ymin = min(-0.01, min(all_vals) * 1.1) if all_vals else -0.01
    ymax = max(0.01,  max(all_vals) * 1.1) if all_vals else 0.1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, ct in zip(axes, CTYPES):
        d    = stats["by_type"][ct]
        n_ct = sum(1 for r in results if r["conflict_type"] == ct)

        ax.plot(xs, d["IE_attn_mean"], "b-o", label="IE_attn", linewidth=2, markersize=5)
        ax.plot(xs, d["IE_mlp_mean"],  "r-o", label="IE_mlp",  linewidth=2, markersize=5)
        ax.plot(xs, d["IE_full_mean"], color="#888888", linestyle="--",
                label="IE_full", linewidth=1.5, markersize=3, marker="s")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{ct}  (n={n_ct})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("IE = P_restored − P_corrupted", fontsize=11)
    fig.suptitle("Attention vs MLP by Conflict Type", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t0   = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Path Patching] Attention vs MLP, final token, layers 13~25")

    # ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
    print("\n[1/5] 데이터 로드...")
    with KC_PATH.open()    as f: kc_cases  = json.load(f)
    with STEP1_PATH.open() as f: step1_res = json.load(f)

    s1_map = {r["id"]: r for r in step1_res}

    def get_tgroup(case_id: str) -> str:
        r    = s1_map.get(case_id, {})
        tl   = (r.get("positions") or {}).get("A", {}).get("transition_layer")
        b_tl = (r.get("positions") or {}).get("B", {}).get("transition_layer")
        if b_tl is None:
            return "no_b"
        if tl is None:
            return "mid"
        if tl <= EARLY_THRESH:
            return "early"
        if tl >= LATE_THRESH:
            return "late"
        return "mid"

    for c in kc_cases:
        c["transition_group"]   = get_tgroup(c["id"])
        r = s1_map.get(c["id"], {})
        c["transition_layer_A"] = (r.get("positions") or {}).get("A", {}).get("transition_layer")

    # ── 2. 전체 케이스 사용 ─────────────────────────────────────────────────────
    cases: List[Dict] = list(kc_cases)
    if args.test:
        cases = cases[:args.test]
    N = len(cases)
    print(f"  총 케이스: {N}개  (집중 레이어: {FOCUS_LAYERS[0]}~{FOCUS_LAYERS[-1]})")

    # ── 3. 모델 로드 ────────────────────────────────────────────────────────────
    print("\n[2/5] 모델 로드 (bfloat16)...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.bfloat16)
    model.eval()
    setup_letter_variants(model.tokenizer)
    print(f"  d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    # ── 4. 전역 노이즈 std ──────────────────────────────────────────────────────
    print("\n[3/5] 전역 노이즈 표준편차 계산...")
    noise_std = compute_noise_std(model, cases)

    # ── 5. Path Patching ────────────────────────────────────────────────────────
    n_passes = 1 + 1 + N_FOCUS * 3
    print(f"\n[4/5] Path Patching ({n_passes} passes/case: 1 neutral + 1 corrupted + "
          f"{N_FOCUS}×3 restored)...")

    results:  List[Dict] = []
    ckpt_path = OUT_DIR / "pp_checkpoint.json"

    if ckpt_path.exists() and not args.test:
        with ckpt_path.open() as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        cases    = [c for c in cases if c["id"] not in done_ids]
        print(f"  체크포인트 로드: {len(results)}개 완료, 남은 {len(cases)}개")

    t_ph      = time.time()
    do_sanity = (len(results) == 0)

    for idx, case in enumerate(cases):
        t_case = time.time()
        res    = process_case(model, case, noise_std, sanity=do_sanity)
        do_sanity = False

        if res is None:
            print(f"  [skip] {case['id']}")
            continue

        results.append(res)
        elapsed   = time.time() - t_ph
        case_time = time.time() - t_case
        rate      = (idx + 1) / max(elapsed, 1e-6)
        eta       = (len(cases) - idx - 1) / max(rate, 1e-6)

        if (idx + 1) % 10 == 0 or idx == 0:
            tg = case.get("transition_group", "?")
            print(f"  [{idx+1}/{len(cases)}] {case['id'][:30]} | "
                  f"tg={tg} | {case_time:.1f}s/case | ETA={eta:.0f}s")

        if (len(results) % CHECKPOINT_EVERY == 0) and not args.test:
            with ckpt_path.open("w") as f:
                json.dump(results, f, ensure_ascii=False)
            print(f"  [ckpt] {len(results)}개 저장")

    print(f"  PP 완료: {len(results)}개 성공")

    # ── 6. 통계 / 저장 / 시각화 ─────────────────────────────────────────────────
    print("\n[5/5] 통계 계산 및 저장...")
    stats = compute_stats(results)

    print_analyses(results, stats)

    with (OUT_DIR / "pp_results.json").open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "pp_stats.json").open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  pp_results.json ({len(results)} cases), pp_stats.json 저장")

    plot_overall(results, stats, OUT_DIR / "attn_vs_mlp_by_layer.png")
    plot_by_type(results, stats, OUT_DIR / "attn_vs_mlp_by_type.png")

    if ckpt_path.exists() and not args.test:
        ckpt_path.unlink()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Step 4 완료 (총 {elapsed/60:.1f}분, {len(results)}케이스)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

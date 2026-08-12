#!/usr/bin/env python3
"""
Step 5: Attention Head Ablation — KC 기여 헤드 탐색

Mean ablation 방식으로 레이어 13~25 × 헤드 0~7 중
KC 억제에 기여하는 헤드를 찾는다.

  suppression = baseline_diff - ablated_diff
    baseline_diff = P(replace) - P(correct)  [ablation 전, conflict 프롬프트]
    ablated_diff  = P(replace) - P(correct)  [해당 헤드 mean ablation 후]
  양수 → 해당 헤드가 KC를 강화하고 있음 (ablate하면 KC 감소)

mean ablation 대상: blocks.{layer}.attn.hook_z
  shape [batch, pos, n_heads, d_head]  (W_O 이전 per-head output, 기본 제공)

실행:
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step5_head_ablation.py
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step5_head_ablation.py --test
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
KC_PATH          = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
OUT_DIR          = Path("/home/jaeik/xaimi/phase1/results/step5_head_ablation")
MODEL_NAME       = "google/gemma-2-2b"
MEAN_VALUES_PATH = OUT_DIR / "mean_values.pt"
RESULTS_PATH     = OUT_DIR / "head_ablation_results.json"
S1_CKPT_PATH     = OUT_DIR / "stage1_checkpoint.json"
S2_CKPT_PATH     = OUT_DIR / "stage2_checkpoint.json"

SEED             = 42
FOCUS_LAYERS     = list(range(13, 26))   # 13 ~ 25 (13개)
N_FOCUS          = len(FOCUS_LAYERS)
STAGE1_N         = 300
CHECKPOINT_EVERY = 50
CTYPES           = ("misinformation", "temporal", "semantic")

LETTER_VARIANTS: Dict[str, List[int]] = {}

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="1단계 30개만 처리")
    return p.parse_args()


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model() -> HookedTransformer:
    print("모델 로드 (bfloat16) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    return model


# ── 토크나이저 설정 ───────────────────────────────────────────────────────────
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
def build_conflict_prompt(context: str, question: str, options: Dict[str, str]) -> str:
    lines = ["Context:", context, "", f"Question: {question}"]
    for l in "ABCD":
        lines.append(f"{l}. {options[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── 확률 계산 ─────────────────────────────────────────────────────────────────
def get_letter_prob(probs: torch.Tensor, letter: str) -> float:
    return max(probs[tid].item() for tid in LETTER_VARIANTS[letter])


def compute_diff(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    cor: str,
    rep: str,
    fwd_hooks: Optional[List] = None,
) -> float:
    """P(replace) - P(correct). fwd_hooks 없으면 표준 forward."""
    hooks = fwd_hooks or []
    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(input_ids.unsqueeze(0), fwd_hooks=hooks)
        else:
            logits = model(input_ids.unsqueeze(0))
    probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
    del logits
    return get_letter_prob(probs, rep) - get_letter_prob(probs, cor)


# ── 사전 작업: mean value 계산 ────────────────────────────────────────────────
def compute_mean_values(
    model: HookedTransformer,
    kc_cases: list,
    n_cases: int,
) -> Dict[int, torch.Tensor]:
    """
    각 레이어-헤드의 conflict run final token hook_z 평균.
    hook_z: [batch, pos, n_heads, d_head] — W_O 이전 per-head output
    Returns: {layer: tensor[n_heads, d_head]} (CPU, float32)
    """
    n_heads = model.cfg.n_heads
    d_head  = model.cfg.d_head
    use_cases = kc_cases[:n_cases]

    sum_vals = {
        lay: torch.zeros(n_heads, d_head, dtype=torch.float32)
        for lay in FOCUS_LAYERS
    }
    count = 0

    print(f"\n[사전 작업] mean value 계산 ({len(use_cases)}개) ...", flush=True)
    t0 = time.time()

    for i, case in enumerate(use_cases):
        ctx  = case["conflict_context"]
        q    = case["question"]
        opts = case["options"]

        conflict_p   = build_conflict_prompt(ctx, q, opts)
        conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]
        final_pos    = int(conflict_ids.shape[0]) - 1

        store: Dict[int, torch.Tensor] = {}

        def make_capture(lay: int, fp: int):
            def _fn(val: torch.Tensor, hook) -> torch.Tensor:
                # val: [1, seq, n_heads, d_head]
                store[lay] = val[0, fp, :, :].detach().float().cpu()
                return val
            return _fn

        hooks = [
            (f"blocks.{lay}.attn.hook_z", make_capture(lay, final_pos))
            for lay in FOCUS_LAYERS
        ]
        with torch.no_grad():
            model.run_with_hooks(conflict_ids.unsqueeze(0), fwd_hooks=hooks)

        for lay in FOCUS_LAYERS:
            sum_vals[lay] += store[lay]
        count += 1

        if (i + 1) % 100 == 0:
            print(f"  mean {i+1}/{len(use_cases)}  ({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()

    mean_vals = {lay: sum_vals[lay] / count for lay in FOCUS_LAYERS}
    torch.save(mean_vals, MEAN_VALUES_PATH)
    print(f"  → 저장: {MEAN_VALUES_PATH}  (n={count})", flush=True)
    return mean_vals


def load_or_compute_mean_values(
    model: HookedTransformer,
    kc_cases: list,
    test: bool,
) -> Dict[int, torch.Tensor]:
    if MEAN_VALUES_PATH.exists() and not test:
        print(f"[사전 작업] mean_values.pt 로드 (기존): {MEAN_VALUES_PATH}", flush=True)
        return torch.load(MEAN_VALUES_PATH, map_location="cpu")
    n = 30 if test else len(kc_cases)
    return compute_mean_values(model, kc_cases, n)


# ── Stage 1: 스크리닝 ─────────────────────────────────────────────────────────
def stage1_screening(
    model: HookedTransformer,
    kc_cases: list,
    mean_vals: Dict[int, torch.Tensor],
    test: bool,
) -> Tuple[np.ndarray, Dict[str, list]]:
    """
    Returns:
      suppression_matrix: [n_layers, n_heads]  (FOCUS_LAYERS 순서)
      top3: {"13": [{"head": h, "suppression": v}, ...], ...}
    """
    n_heads = model.cfg.n_heads
    dev     = next(model.parameters()).device
    dtype   = torch.bfloat16

    n_cases = 30 if test else STAGE1_N
    cases   = kc_cases[:n_cases]

    mean_gpu: Dict[int, torch.Tensor] = {
        lay: mv.to(dev, dtype=dtype) for lay, mv in mean_vals.items()
    }

    sup_sum  = np.zeros((N_FOCUS, n_heads), dtype=np.float64)
    n_valid  = np.zeros((N_FOCUS, n_heads), dtype=np.int64)

    print(f"\n[1단계] 스크리닝 ({n_cases}개, {N_FOCUS}레이어×{n_heads}헤드) ...", flush=True)
    t0 = time.time()

    for ci, case in enumerate(cases):
        ctx  = case["conflict_context"]
        q    = case["question"]
        opts = case["options"]
        cor  = case["correct_option"]
        rep  = case["replace_option"]

        conflict_p   = build_conflict_prompt(ctx, q, opts)
        conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]
        final_pos    = int(conflict_ids.shape[0]) - 1

        baseline_diff = compute_diff(model, conflict_ids, cor, rep)

        for li, layer in enumerate(FOCUS_LAYERS):
            layer_mean = mean_gpu[layer]   # [n_heads, d_head]

            for hi in range(n_heads):
                head_mean = layer_mean[hi]   # [d_head]

                def make_ablate(fp: int, hidx: int, hm: torch.Tensor):
                    def _fn(val: torch.Tensor, hook) -> torch.Tensor:
                        val[0, fp, hidx, :] = hm
                        return val
                    return _fn

                hook = (
                    f"blocks.{layer}.attn.hook_z",
                    make_ablate(final_pos, hi, head_mean),
                )
                ablated_diff = compute_diff(model, conflict_ids, cor, rep, [hook])
                sup_sum[li, hi]  += baseline_diff - ablated_diff
                n_valid[li, hi]  += 1

        if (ci + 1) % 10 == 0:
            print(f"  stage1 {ci+1}/{n_cases}  ({time.time()-t0:.0f}s)", flush=True)
        if (ci + 1) % CHECKPOINT_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()

    denom              = np.where(n_valid > 0, n_valid, 1)
    suppression_matrix = sup_sum / denom   # [n_layers, n_heads]

    # top3 per layer
    top3: Dict[str, list] = {}
    for li, layer in enumerate(FOCUS_LAYERS):
        row         = suppression_matrix[li]
        top_indices = np.argsort(row)[::-1][:3].tolist()
        top3[str(layer)] = [
            {"head": int(h), "suppression": float(row[h])}
            for h in top_indices
        ]

    print(f"\n레이어별 top 3 헤드:", flush=True)
    for layer in FOCUS_LAYERS:
        entries = top3[str(layer)]
        parts   = ", ".join(f"head {e['head']}({e['suppression']:+.4f})" for e in entries)
        print(f"  layer {layer:2d}: {parts}", flush=True)

    del mean_gpu
    gc.collect()
    torch.cuda.empty_cache()

    return suppression_matrix, top3


# ── Stage 2: 전체 검증 ────────────────────────────────────────────────────────
def stage2_validation(
    model: HookedTransformer,
    kc_cases: list,
    mean_vals: Dict[int, torch.Tensor],
    top3: Dict[str, list],
) -> list:
    """
    Returns: list of {layer, head, suppression_mean, suppression_std, by_type}
             내림차순 정렬
    """
    dev   = next(model.parameters()).device
    dtype = torch.bfloat16

    # top_heads 목록 (layer, head) 중복 없이
    seen: set = set()
    top_heads: List[Tuple[int, int]] = []
    for layer in FOCUS_LAYERS:
        for entry in top3[str(layer)]:
            key = (layer, entry["head"])
            if key not in seen:
                seen.add(key)
                top_heads.append(key)

    mean_gpu: Dict[int, torch.Tensor] = {
        lay: mv.to(dev, dtype=dtype) for lay, mv in mean_vals.items()
    }

    # 누적 {(layer, head): {ctype: [sup, ...]}}
    acc: Dict[Tuple[int, int], Dict[str, list]] = {
        lh: {ct: [] for ct in (CTYPES + ("all",))} for lh in top_heads
    }

    print(f"\n[2단계] 전체 검증 ({len(kc_cases)}개, {len(top_heads)}헤드) ...", flush=True)
    t0 = time.time()

    for ci, case in enumerate(kc_cases):
        ctx   = case["conflict_context"]
        q     = case["question"]
        opts  = case["options"]
        cor   = case["correct_option"]
        rep   = case["replace_option"]
        ctype = case["conflict_type"]

        conflict_p   = build_conflict_prompt(ctx, q, opts)
        conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]
        final_pos    = int(conflict_ids.shape[0]) - 1

        baseline_diff = compute_diff(model, conflict_ids, cor, rep)

        for (layer, hi) in top_heads:
            head_mean = mean_gpu[layer][hi]   # [d_head]

            def make_ablate(fp: int, hidx: int, hm: torch.Tensor):
                def _fn(val: torch.Tensor, hook) -> torch.Tensor:
                    val[0, fp, hidx, :] = hm
                    return val
                return _fn

            hook = (
                f"blocks.{layer}.attn.hook_z",
                make_ablate(final_pos, hi, head_mean),
            )
            ablated_diff = compute_diff(model, conflict_ids, cor, rep, [hook])
            sup = baseline_diff - ablated_diff

            acc[(layer, hi)]["all"].append(sup)
            if ctype in acc[(layer, hi)]:
                acc[(layer, hi)][ctype].append(sup)

        if (ci + 1) % 50 == 0:
            print(f"  stage2 {ci+1}/{len(kc_cases)}  ({time.time()-t0:.0f}s)", flush=True)
        if (ci + 1) % CHECKPOINT_EVERY == 0:
            # 중간 저장
            partial = _build_stage2_output(top_heads, acc)
            with S2_CKPT_PATH.open("w") as f:
                json.dump({"processed": ci + 1, "partial": partial}, f)
            gc.collect()
            torch.cuda.empty_cache()

    results = _build_stage2_output(top_heads, acc)

    print(f"\nKC 억제 효과 상위 10개 헤드:", flush=True)
    for rank, r in enumerate(results[:10], 1):
        print(
            f"  {rank:2d}위: layer {r['layer']:2d} head {r['head']}  "
            f"억제효과={r['suppression_mean']:+.4f} ± {r['suppression_std']:.4f}",
            flush=True,
        )

    del mean_gpu
    gc.collect()
    torch.cuda.empty_cache()

    return results


def _build_stage2_output(
    top_heads: List[Tuple[int, int]],
    acc: Dict[Tuple[int, int], Dict[str, list]],
) -> list:
    results = []
    for (layer, hi) in top_heads:
        vals = np.array(acc[(layer, hi)]["all"], dtype=np.float64)
        by_type: Dict[str, dict] = {}
        for ct in CTYPES:
            tv = acc[(layer, hi)][ct]
            by_type[ct] = {
                "mean": float(np.mean(tv)) if tv else 0.0,
                "std":  float(np.std(tv))  if tv else 0.0,
                "n":    len(tv),
            }
        results.append({
            "layer":            layer,
            "head":             hi,
            "suppression_mean": float(vals.mean()) if len(vals) > 0 else 0.0,
            "suppression_std":  float(vals.std())  if len(vals) > 0 else 0.0,
            "by_type":          by_type,
        })
    results.sort(key=lambda x: x["suppression_mean"], reverse=True)
    return results


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_heatmap(suppression_matrix: np.ndarray, n_heads: int) -> None:
    vmax = max(np.abs(suppression_matrix).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        suppression_matrix, aspect="auto", cmap="RdYlGn",
        vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Suppression (baseline_diff − ablated_diff)")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(N_FOCUS))
    ax.set_yticklabels([str(l) for l in FOCUS_LAYERS])
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Attention Head Suppression Heatmap (Stage 1 Mean Ablation)", fontsize=13)
    for li in range(N_FOCUS):
        for hi in range(n_heads):
            v = suppression_matrix[li, hi]
            ax.text(hi, li, f"{v:.3f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    fig.tight_layout()
    out_path = OUT_DIR / "head_suppression_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {out_path}", flush=True)


def plot_top_heads(stage2_results: list) -> None:
    top10  = stage2_results[:10]
    labels = [f"L{r['layer']}H{r['head']}" for r in top10]
    means  = [r["suppression_mean"] for r in top10]
    stds   = [r["suppression_std"]  for r in top10]
    colors = ["steelblue" if m >= 0 else "tomato" for m in means]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(top10))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, ecolor="gray")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Head (L=Layer, H=Head)", fontsize=12)
    ax.set_ylabel("Suppression (mean ± std)", fontsize=12)
    ax.set_title("Top 10 KC-Suppressing Attention Heads (Stage 2 Full Validation)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / "top_heads_profile.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {out_path}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with KC_PATH.open() as f:
        kc_cases = json.load(f)
    print(f"KC 케이스 로드: {len(kc_cases)}개", flush=True)

    model   = load_model()
    n_heads = model.cfg.n_heads
    setup_letter_variants(model.tokenizer)
    print(f"cfg: n_heads={n_heads}, d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}",
          flush=True)

    # ── 사전 작업: mean values ─────────────────────────────────────────────────
    mean_vals = load_or_compute_mean_values(model, kc_cases, args.test)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    suppression_matrix, top3 = stage1_screening(model, kc_cases, mean_vals, args.test)

    stage1_output = {
        "n_cases": 30 if args.test else STAGE1_N,
        "suppression_by_layer_head": {
            str(FOCUS_LAYERS[li]): suppression_matrix[li].tolist()
            for li in range(N_FOCUS)
        },
        "top3_heads_by_layer": top3,
    }

    # Stage 1 즉시 저장 (Stage 2 전 체크포인트)
    with S1_CKPT_PATH.open("w") as f:
        json.dump(stage1_output, f, indent=2)
    print(f"\nStage 1 체크포인트 저장: {S1_CKPT_PATH}", flush=True)

    # ── Stage 2 (test 모드 생략) ───────────────────────────────────────────────
    stage2_output: Optional[dict] = None
    if not args.test:
        stage2_results = stage2_validation(model, kc_cases, mean_vals, top3)
        stage2_output  = {
            "n_cases": len(kc_cases),
            "top_heads_validation": stage2_results,
        }

    # ── 최종 결과 저장 ─────────────────────────────────────────────────────────
    result: dict = {"stage1": stage1_output}
    if stage2_output:
        result["stage2"] = stage2_output

    with RESULTS_PATH.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {RESULTS_PATH}", flush=True)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print("\n시각화 저장 ...", flush=True)
    plot_heatmap(suppression_matrix, n_heads)
    if stage2_output:
        plot_top_heads(stage2_results)

    # 임시 체크포인트 삭제
    for ckpt in (S1_CKPT_PATH, S2_CKPT_PATH):
        if ckpt.exists():
            ckpt.unlink()

    print("\n완료!", flush=True)


if __name__ == "__main__":
    main()

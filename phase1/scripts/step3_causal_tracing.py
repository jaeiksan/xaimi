#!/usr/bin/env python3
"""
Step 3: Causal Tracing (Activation Patching) — KC 인과적 레이어 탐색

방법: Rome-style segment-level causal tracing
  1) Neutral run (문맥 없는 프롬프트) → 각 레이어 residual 캐싱
  2) Corrupted run (문맥 임베딩에 Gaussian 노이즈 σ=3×global_std) → P_corrupted[correct]
  3) Restored run (corrupted + layer×segment 패치 → neutral 값 복원) → P_restored[correct]
  IE(l, seg) = P_restored[correct] - P_corrupted[correct]

5개 세그먼트: context_first_sent, context_rest, question, options, final
26 레이어 × 5 세그먼트 = 130 패치/케이스

실행:
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step3_causal_tracing.py
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step3_causal_tracing.py --test 10
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

# ── 경로 / 상수 ────────────────────────────────────────────────────────────────
KC_PATH         = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
STEP1_PATH      = Path("/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_results.json")
OUT_DIR         = Path("/home/jaeik/xaimi/phase1/results/step3_causal_tracing")
MODEL_NAME      = "google/gemma-2-2b"
SEED            = 42
MAX_SEQ_LEN     = 2048
NOISE_MULT      = 3.0
CHECKPOINT_EVERY = 50
EARLY_THRESH    = 3
LATE_THRESH     = 15

CTYPES   = ("misinformation", "temporal", "semantic")
SEGMENTS = ("context_first_sent", "context_rest", "question", "options", "final")

SEG_COLORS = {
    "context_first_sent": "#c0392b",
    "context_rest":       "#e67e22",
    "question":           "#2980b9",
    "options":            "#27ae60",
    "final":              "#95a5a6",
}
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


# ── Context 분리 ───────────────────────────────────────────────────────────────
def split_claim_evidence(context: str) -> Tuple[str, str]:
    if "\n\n" in context:
        idx = context.index("\n\n")
        return context[:idx].strip(), context[idx + 2:].strip()
    for i, ch in enumerate(context):
        if ch == "." and i + 1 < len(context) and context[i + 1] in " \n":
            return context[:i + 1].strip(), context[i + 2:].strip()
    return context.strip(), ""


# ── 세그먼트 토큰 맵 ──────────────────────────────────────────────────────────
def get_segment_token_map(
    tokenizer,
    conflict_prompt: str,
    context: str,
    question: str,
    options: Dict[str, str],
) -> Dict[str, List[int]]:
    """
    conflict_prompt 토큰을 5개 세그먼트로 분류.
      context_first_sent : context의 첫 문장 (claim)
      context_rest       : context 나머지 (evidence)
      question           : 질문 텍스트
      options            : A~D 옵션 텍스트
      final              : BOS/헤더/"Answer:" 등 형식 토큰
    """
    enc     = tokenizer(conflict_prompt, return_offsets_mapping=True,
                        add_special_tokens=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()

    claim, _ = split_claim_evidence(context)
    p        = conflict_prompt

    ctx_hdr_end   = len("Context:\n")
    claim_end     = ctx_hdr_end + len(claim)

    q_marker      = "\n\nQuestion: "
    q_section_pos = p.rfind(q_marker)
    evidence_end  = q_section_pos if q_section_pos > claim_end else claim_end

    q_text_start  = q_section_pos + len(q_marker)
    q_text_end    = q_text_start + len(question)

    ans_pos   = p.rfind("\nAnswer:")
    opts_start = q_text_end + 1   # '\n' after question
    opts_end   = ans_pos

    groups: Dict[str, List[int]] = {s: [] for s in SEGMENTS}

    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:          # BOS / special
            groups["final"].append(i)
            continue

        mid = (s + e) / 2

        if mid < ctx_hdr_end:
            groups["final"].append(i)
        elif mid < claim_end:
            groups["context_first_sent"].append(i)
        elif mid < evidence_end:
            groups["context_rest"].append(i)
        elif mid < q_section_pos + 2:  # "\n\n"
            groups["final"].append(i)
        elif mid < q_text_start:       # "Question: " label
            groups["final"].append(i)
        elif mid < q_text_end:
            groups["question"].append(i)
        elif mid < opts_start:         # '\n' after question
            groups["final"].append(i)
        elif mid < opts_end:
            groups["options"].append(i)
        else:
            groups["final"].append(i)  # "\nAnswer:"

    return groups


# ── Neutral 위치 매핑 ─────────────────────────────────────────────────────────
def build_neutral_pos_map(
    segment_map: Dict[str, List[int]],
    n_conflict: int,
    n_neutral: int,
) -> Dict[int, int]:
    """
    conflict 토큰 위치 → neutral 토큰 위치.
    context 토큰: neutral pos 0 (BOS) 매핑.
    그 외: offset = n_conflict - n_neutral 만큼 시프트.
    """
    offset       = n_conflict - n_neutral
    ctx_pos_set  = set(
        segment_map["context_first_sent"] + segment_map["context_rest"]
    )
    pos_map: Dict[int, int] = {}
    for p in range(n_conflict):
        if p in ctx_pos_set:
            pos_map[p] = 0
        else:
            np_idx     = p - offset
            pos_map[p] = max(0, min(np_idx, n_neutral - 1))
    return pos_map


# ── 전역 노이즈 std ───────────────────────────────────────────────────────────
def compute_noise_std(model: HookedTransformer, cases: List[Dict]) -> float:
    """50개 케이스 임베딩 std → σ = NOISE_MULT × std"""
    sample = cases[:min(50, len(cases))]
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for c in sample:
            p   = build_conflict_prompt(c["conflict_context"], c["question"], c["options"])
            ids = model.to_tokens(p, prepend_bos=True)[0]
            if ids.shape[0] > MAX_SEQ_LEN:
                ids = ids[:MAX_SEQ_LEN]
            chunks.append(model.embed.W_E[ids].float().cpu())
    all_embs  = torch.cat(chunks, dim=0)   # [total_tokens, d_model]
    std_val   = all_embs.std().item()
    noise_std = NOISE_MULT * std_val
    print(f"  embedding std={std_val:.4f},  noise σ={noise_std:.4f}")
    return noise_std


# ── 케이스 처리 ───────────────────────────────────────────────────────────────
def process_case(
    model:     HookedTransformer,
    case:      Dict,
    noise_std: float,
    n_layers:  int,
    d_model:   int,
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

    segment_map = get_segment_token_map(tokenizer, conflict_p, ctx, q, opts)
    pos_map     = build_neutral_pos_map(segment_map, n_c, n_n)

    ctx_indices = (
        segment_map["context_first_sent"] + segment_map["context_rest"]
    )

    # 케이스별 고정 노이즈 (context 위치에만 적용)
    noise_vec = torch.zeros(n_c, d_model, dtype=torch.bfloat16, device=dev)
    if ctx_indices:
        ctx_t = torch.tensor(ctx_indices, dtype=torch.long, device=dev)
        noise_vec[ctx_t] = (
            torch.randn(len(ctx_indices), d_model, dtype=torch.bfloat16, device=dev)
            * noise_std
        )

    def noise_hook(val, hook):
        return val + noise_vec.unsqueeze(0)

    cor_tok_ids = LETTER_VARIANTS[cor]

    # ── 1. Neutral run → layer residual cache ─────────────────────────────────
    neutral_cache: Dict[int, torch.Tensor] = {}

    def make_cache_hook(layer: int):
        def _fn(val, hook):
            neutral_cache[layer] = val.detach()   # [1, n_n, d_model]
            return val
        return _fn

    with torch.no_grad():
        _ = model.run_with_hooks(
            neutral_ids.unsqueeze(0),
            fwd_hooks=[(f"blocks.{l}.hook_resid_post", make_cache_hook(l))
                       for l in range(n_layers)],
        )

    # ── 2. Corrupted run → P_corrupted ────────────────────────────────────────
    with torch.no_grad():
        logits_c = model.run_with_hooks(
            conflict_ids.unsqueeze(0),
            fwd_hooks=[("hook_embed", noise_hook)],
        )
    probs_c     = torch.softmax(logits_c[0, -1, :].float(), dim=-1)
    p_corrupted = max(probs_c[tid].item() for tid in cor_tok_ids)
    del logits_c
    torch.cuda.empty_cache()

    if sanity:
        print(f"\n  [sanity] n_conflict={n_c}, n_neutral={n_n}, offset={n_c - n_n}")
        for seg in SEGMENTS:
            idxs = segment_map[seg]
            if idxs:
                print(f"    [{seg}] {min(idxs)}~{max(idxs)} (n={len(idxs)})")
        with torch.no_grad():
            logits_n = model(neutral_ids.unsqueeze(0))
        probs_n   = torch.softmax(logits_n[0, -1, :].float(), dim=-1)
        p_neutral = max(probs_n[tid].item() for tid in cor_tok_ids)
        del logits_n
        print(f"    p_neutral[{cor}]   = {p_neutral:.4f}")
        print(f"    p_corrupted[{cor}] = {p_corrupted:.4f}")

    # ── 3. Restored runs → IE[layer, segment] ─────────────────────────────────
    ie_matrix = np.zeros((n_layers, len(SEGMENTS)), dtype=np.float32)

    for l in range(n_layers):
        nc_l = neutral_cache[l]   # [1, n_n, d_model]

        for seg_idx, seg_name in enumerate(SEGMENTS):
            seg_pos = segment_map[seg_name]
            if not seg_pos:
                continue

            seg_pos_t   = torch.tensor(seg_pos, dtype=torch.long, device=dev)
            neutral_pos = torch.tensor(
                [pos_map[p] for p in seg_pos], dtype=torch.long, device=dev
            )
            neutral_patch = nc_l[:, neutral_pos, :]    # [1, k, d_model]

            def make_patch_hook(_spt, _np):
                def _fn(val, hook):
                    val[:, _spt, :] = _np
                    return val
                return _fn

            with torch.no_grad():
                logits_r = model.run_with_hooks(
                    conflict_ids.unsqueeze(0),
                    fwd_hooks=[
                        ("hook_embed", noise_hook),
                        (f"blocks.{l}.hook_resid_post",
                         make_patch_hook(seg_pos_t, neutral_patch)),
                    ],
                )
            probs_r    = torch.softmax(logits_r[0, -1, :].float(), dim=-1)
            p_restored = max(probs_r[tid].item() for tid in cor_tok_ids)
            ie_matrix[l, seg_idx] = p_restored - p_corrupted

            del logits_r
            torch.cuda.empty_cache()

    del neutral_cache, noise_vec
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "id":                 case["id"],
        "conflict_type":      case["conflict_type"],
        "transition_group":   case.get("transition_group", "mid"),
        "transition_layer_A": case.get("transition_layer_A"),
        "correct_option":     cor,
        "p_corrupted":        float(p_corrupted),
        "ie_matrix":          ie_matrix.tolist(),   # [n_layers, n_segments]
        "kc_severity":        case.get("kc_severity", 0.0),
    }


# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def compute_stats(results: List[Dict], n_layers: int) -> Dict:
    def mean_ie(sub: List[Dict]) -> np.ndarray:
        if not sub:
            return np.zeros((n_layers, len(SEGMENTS)))
        return np.mean([np.array(r["ie_matrix"]) for r in sub], axis=0)

    overall_ie = mean_ie(results)

    by_type = {
        ct: mean_ie([r for r in results if r["conflict_type"] == ct]).tolist()
        for ct in CTYPES
    }
    by_tg = {
        tg: mean_ie([r for r in results if r["transition_group"] == tg]).tolist()
        for tg in ("early", "mid", "late", "no_b")
    }

    top_layer_per_seg: Dict[str, int] = {}
    for seg_idx, seg in enumerate(SEGMENTS):
        top_layer_per_seg[seg] = int(np.argmax(overall_ie[:, seg_idx]))

    return {
        "n":                          len(results),
        "segments":                   list(SEGMENTS),
        "overall_ie_matrix":          overall_ie.tolist(),
        "by_type_ie_matrix":          by_type,
        "by_transition_group_ie_matrix": by_tg,
        "top_layer_per_segment":      top_layer_per_seg,
        "layer_max_ie":               overall_ie.max(axis=1).tolist(),
        "p_corrupted_mean":           float(np.mean([r["p_corrupted"] for r in results])),
    }


# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
def print_analyses(results: List[Dict], stats: Dict, n_layers: int) -> None:
    N = len(results)
    overall = np.array(stats["overall_ie_matrix"])

    print(f"\n{'='*70}")
    print(f"분석 1: IE 행렬 (n={N}), mean P_corrupted={stats['p_corrupted_mean']:.4f}")
    print(f"{'='*70}")
    hdr = f"  {'L':>3}"
    for seg in SEGMENTS:
        hdr += f"  {seg[:14]:>14}"
    print(hdr)
    for l in range(n_layers):
        row = f"  {l:>3}"
        for si in range(len(SEGMENTS)):
            row += f"  {overall[l, si]:>14.4f}"
        print(row)

    print(f"\n{'='*70}")
    print("분석 2: 세그먼트별 최대 IE 레이어")
    print(f"{'='*70}")
    for seg, lyr in stats["top_layer_per_segment"].items():
        ie_v = overall[lyr, SEGMENTS.index(seg)]
        print(f"  {seg:<22}: layer {lyr:2d}  IE={ie_v:.4f}")

    print(f"\n{'='*70}")
    print("분석 3: 타입별 context_first_sent IE 최대 레이어")
    print(f"{'='*70}")
    cfs_idx = SEGMENTS.index("context_first_sent")
    for ct in CTYPES:
        mat  = np.array(stats["by_type_ie_matrix"][ct])
        n_ct = sum(1 for r in results if r["conflict_type"] == ct)
        top_l = int(np.argmax(mat[:, cfs_idx]))
        ie_v  = mat[top_l, cfs_idx]
        print(f"  {ct:<18} (n={n_ct}): layer {top_l:2d}  IE={ie_v:.4f}")


# ── 시각화 ────────────────────────────────────────────────────────────────────
def _heatmap(ie_matrix: np.ndarray, title: str, out_path: Path) -> None:
    vmax = max(0.05, float(np.percentile(ie_matrix, 98)))
    vmin = min(-0.02, float(np.percentile(ie_matrix, 2)))
    fig, ax = plt.subplots(figsize=(max(10, ie_matrix.shape[0] * 0.4), 5))
    im = ax.imshow(ie_matrix.T, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Segment", fontsize=12)
    ax.set_xticks(range(ie_matrix.shape[0]))
    ax.set_xticklabels(range(ie_matrix.shape[0]), fontsize=7)
    ax.set_yticks(range(len(SEGMENTS)))
    ax.set_yticklabels(SEGMENTS, fontsize=9)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, label="IE = P_restored − P_corrupted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


def plot_ie_heatmap_overall(results: List[Dict], stats: Dict, out_path: Path) -> None:
    _heatmap(np.array(stats["overall_ie_matrix"]),
             f"IE Heatmap — All Cases (n={len(results)})", out_path)


def plot_ie_heatmap_by_type(results: List[Dict], stats: Dict, out_path: Path) -> None:
    mats    = {ct: np.array(stats["by_type_ie_matrix"][ct]) for ct in CTYPES}
    all_v   = np.concatenate([m.flatten() for m in mats.values()])
    vmax    = max(0.05, float(np.percentile(all_v, 98))) if len(all_v) else 0.1
    vmin    = min(-0.02, float(np.percentile(all_v, 2))) if len(all_v) else -0.02

    fig, axes = plt.subplots(1, 3, figsize=(24, 5), sharey=True)
    for ax, ct in zip(axes, CTYPES):
        mat  = mats[ct]
        n_ct = sum(1 for r in results if r["conflict_type"] == ct)
        im   = ax.imshow(mat.T, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_xticks(range(mat.shape[0]))
        ax.set_xticklabels(range(mat.shape[0]), fontsize=6)
        ax.set_yticks(range(len(SEGMENTS)))
        ax.set_yticklabels(SEGMENTS, fontsize=9)
        ax.set_title(f"{ct}  (n={n_ct})", fontsize=12)
        plt.colorbar(im, ax=ax, label="IE")

    fig.suptitle("IE Heatmap by Conflict Type", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


def plot_ie_layer_profile(results: List[Dict], stats: Dict, out_path: Path) -> None:
    n_layers = len(stats["overall_ie_matrix"])
    fig, ax  = plt.subplots(figsize=(12, 5))

    for ct in CTYPES:
        mat      = np.array(stats["by_type_ie_matrix"][ct])
        n_ct     = sum(1 for r in results if r["conflict_type"] == ct)
        layer_ie = mat.sum(axis=1)
        ax.plot(range(n_layers), layer_ie, color=CTYPE_COLORS[ct],
                label=f"{ct} (n={n_ct})", linewidth=2, marker="o", markersize=4)

    overall  = np.array(stats["overall_ie_matrix"])
    ax.plot(range(n_layers), overall.sum(axis=1), "k--",
            label="overall", linewidth=2)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Σ IE (all segments)", fontsize=12)
    ax.set_title("IE Layer Profile by Conflict Type", fontsize=13)
    ax.set_xticks(range(n_layers))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t0   = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Causal Tracing] Segment-level activation patching")
    print(f"  세그먼트: {SEGMENTS}")

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
        c["transition_group"]    = get_tgroup(c["id"])
        r = s1_map.get(c["id"], {})
        c["transition_layer_A"]  = (r.get("positions") or {}).get("A", {}).get("transition_layer")

    # ── 2. 전체 케이스 사용 ─────────────────────────────────────────────────────
    sampled: List[Dict] = list(kc_cases)

    if args.test:
        sampled = sampled[:args.test]
    N = len(sampled)
    print(f"  총 케이스: {N}개")

    # ── 3. 모델 로드 ────────────────────────────────────────────────────────────
    print("\n[2/5] 모델 로드 (bfloat16)...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.bfloat16)
    model.eval()
    setup_letter_variants(model.tokenizer)
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    print(f"  d_model={d_model}, n_layers={n_layers}")

    # ── 4. 전역 노이즈 std ──────────────────────────────────────────────────────
    print("\n[3/5] 전역 노이즈 표준편차 계산...")
    noise_std = compute_noise_std(model, sampled)

    # ── 5. Causal Tracing ───────────────────────────────────────────────────────
    n_patches = n_layers * len(SEGMENTS)
    print(f"\n[4/5] Causal Tracing ({n_layers} layers × {len(SEGMENTS)} segs "
          f"= {n_patches} patches/case)...")

    results:  List[Dict] = []
    ckpt_path = OUT_DIR / "ct_checkpoint.json"

    if ckpt_path.exists() and not args.test:
        with ckpt_path.open() as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        sampled  = [c for c in sampled if c["id"] not in done_ids]
        print(f"  체크포인트 로드: {len(results)}개 완료, 남은 {len(sampled)}개")

    t_ph      = time.time()
    do_sanity = (len(results) == 0)

    for idx, case in enumerate(sampled):
        t_case = time.time()
        res    = process_case(model, case, noise_std, n_layers, d_model,
                              sanity=do_sanity)
        do_sanity = False

        if res is None:
            print(f"  [skip] {case['id']}")
            continue

        results.append(res)
        elapsed   = time.time() - t_ph
        case_time = time.time() - t_case
        rate      = (idx + 1) / max(elapsed, 1e-6)
        eta       = (len(sampled) - idx - 1) / max(rate, 1e-6)

        if (idx + 1) % 10 == 0 or idx == 0:
            tg = case.get("transition_group", "?")
            print(f"  [{idx+1}/{len(sampled)}] {case['id'][:30]} | "
                  f"tg={tg} | {case_time:.1f}s/case | ETA={eta:.0f}s")

        if (len(results) % CHECKPOINT_EVERY == 0) and not args.test:
            with ckpt_path.open("w") as f:
                json.dump(results, f, ensure_ascii=False)
            print(f"  [ckpt] {len(results)}개 저장")

    print(f"  CT 완료: {len(results)}개 성공")

    # ── 6. 통계 / 저장 / 시각화 ─────────────────────────────────────────────────
    print("\n[5/5] 통계 계산 및 저장...")
    stats = compute_stats(results, n_layers)

    print_analyses(results, stats, n_layers)

    with (OUT_DIR / "ct_results.json").open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "ct_stats.json").open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ct_results.json ({len(results)} cases), ct_stats.json 저장")

    plot_ie_heatmap_overall(results, stats, OUT_DIR / "ie_heatmap_overall.png")
    plot_ie_heatmap_by_type(results, stats, OUT_DIR / "ie_heatmap_by_type.png")
    plot_ie_layer_profile(results, stats,   OUT_DIR / "ie_layer_profile.png")

    if ckpt_path.exists() and not args.test:
        ckpt_path.unlink()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Step 3 완료 (총 {elapsed/60:.1f}분, {len(results)}케이스)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

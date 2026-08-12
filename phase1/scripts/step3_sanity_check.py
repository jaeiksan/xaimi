#!/usr/bin/env python3
"""
Step 3 Sanity Check — 3-run 수치 검증

kc_cases_1000.json 첫 번째 케이스로 3가지 run을 실행하여
causal tracing의 전제 조건이 성립하는지 확인한다.

정상 조건:
  P(correct): neutral > corrupted       (noise가 correct 확률을 낮춰야 함)
  P(replace): conflict > neutral        (KC가 실제로 발생해야 함)

실행:
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python phase1/scripts/step3_sanity_check.py
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from pathlib import Path
from typing import Dict, List

import torch
from transformer_lens import HookedTransformer

KC_PATH    = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
MODEL_NAME = "google/gemma-2-2b"
NOISE_MULT = 3.0
NOISE_SEED = 42

LETTER_VARIANTS: Dict[str, List[int]] = {}


def setup_letter_variants(tokenizer) -> None:
    global LETTER_VARIANTS
    for letter in "ABCD":
        ids: List[int] = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.append(toks[-1] if len(toks) > 1 else toks[0])
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
    print(f"[setup] LETTER_VARIANTS={LETTER_VARIANTS}\n")


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


def get_letter_prob(probs: torch.Tensor, letter: str) -> float:
    return max(probs[tid].item() for tid in LETTER_VARIANTS[letter])


def get_top_prediction(probs: torch.Tensor) -> str:
    best_letter, best_p = "?", -1.0
    for letter in "ABCD":
        p = get_letter_prob(probs, letter)
        if p > best_p:
            best_p, best_letter = p, letter
    return best_letter


def find_context_token_indices(
    tokenizer, conflict_prompt: str, neutral_prompt: str, conflict_ids: torch.Tensor
) -> List[int]:
    """conflict prompt에서 context 부분(neutral에 없는 부분)의 토큰 인덱스."""
    n_c = conflict_ids.shape[0]
    n_n = len(tokenizer.encode(neutral_prompt, add_special_tokens=True))
    offset = n_c - n_n
    # BOS(0) + context 토큰들(1 ~ offset-1)
    return list(range(1, offset))


def compute_noise_std(model: HookedTransformer, conflict_ids: torch.Tensor) -> float:
    with torch.no_grad():
        emb = model.embed.W_E[conflict_ids].float()
    std_val = emb.std().item()
    return NOISE_MULT * std_val


def main() -> None:
    # ── 데이터 로드 ───────────────────────────────────────────────────────────────
    with KC_PATH.open() as f:
        kc_cases = json.load(f)

    case = kc_cases[0]
    ctx  = case["conflict_context"]
    q    = case["question"]
    opts = case["options"]
    cor  = case["correct_option"]
    rep  = case["replace_option"]

    print("=" * 60)
    print(f"케이스 ID  : {case['id']}")
    print(f"conflict_type : {case['conflict_type']}")
    print(f"correct_option: {cor}  ({opts[cor]})")
    print(f"replace_option: {rep}  ({opts[rep]})")
    print(f"kc_severity   : {case['kc_severity']}")
    print("=" * 60)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────────
    print("\n모델 로드 (bfloat16)...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.bfloat16)
    model.eval()
    setup_letter_variants(model.tokenizer)

    conflict_p = build_conflict_prompt(ctx, q, opts)
    neutral_p  = build_neutral_prompt(q, opts)

    conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]
    neutral_ids  = model.to_tokens(neutral_p,  prepend_bos=True)[0]

    n_c = conflict_ids.shape[0]
    n_n = neutral_ids.shape[0]
    print(f"토큰 수 — conflict: {n_c}, neutral: {n_n}, offset: {n_c - n_n}")

    # noise std
    noise_std = compute_noise_std(model, conflict_ids)
    print(f"noise σ = {noise_std:.4f}  (NOISE_MULT={NOISE_MULT} × embedding std)\n")

    # context 토큰 인덱스 (noise 적용 대상)
    ctx_indices = find_context_token_indices(
        model.tokenizer, conflict_p, neutral_p, conflict_ids
    )
    print(f"context 토큰 인덱스: {ctx_indices[:5]} ... (총 {len(ctx_indices)}개)\n")

    # noise 벡터 생성 (context 위치에만)
    torch.manual_seed(NOISE_SEED)
    dev = model.cfg.device
    d_model = model.cfg.d_model
    noise_vec = torch.zeros(n_c, d_model, dtype=torch.bfloat16, device=dev)
    if ctx_indices:
        ctx_t = torch.tensor(ctx_indices, dtype=torch.long, device=dev)
        noise_vec[ctx_t] = (
            torch.randn(len(ctx_indices), d_model, dtype=torch.bfloat16, device=dev)
            * noise_std
        )

    def noise_hook(val, hook):
        return val + noise_vec.unsqueeze(0)

    # ── 1. Neutral run ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits_n = model(neutral_ids.unsqueeze(0))
    probs_n = torch.softmax(logits_n[0, -1, :].float(), dim=-1)
    p_n_cor = get_letter_prob(probs_n, cor)
    p_n_rep = get_letter_prob(probs_n, rep)
    top_n   = get_top_prediction(probs_n)
    del logits_n

    print(f"[neutral]   P(correct={cor}) = {p_n_cor:.4f}   "
          f"P(replace={rep}) = {p_n_rep:.4f}   → top={top_n}")

    # ── 2. Conflict run (noise 없이) ──────────────────────────────────────────────
    with torch.no_grad():
        logits_cf = model(conflict_ids.unsqueeze(0))
    probs_cf = torch.softmax(logits_cf[0, -1, :].float(), dim=-1)
    p_cf_cor = get_letter_prob(probs_cf, cor)
    p_cf_rep = get_letter_prob(probs_cf, rep)
    top_cf   = get_top_prediction(probs_cf)
    del logits_cf

    print(f"[conflict]  P(correct={cor}) = {p_cf_cor:.4f}   "
          f"P(replace={rep}) = {p_cf_rep:.4f}   → top={top_cf}")

    # ── 3. Corrupted run (noise 추가) ─────────────────────────────────────────────
    with torch.no_grad():
        logits_cr = model.run_with_hooks(
            conflict_ids.unsqueeze(0),
            fwd_hooks=[("hook_embed", noise_hook)],
        )
    probs_cr = torch.softmax(logits_cr[0, -1, :].float(), dim=-1)
    p_cr_cor = get_letter_prob(probs_cr, cor)
    p_cr_rep = get_letter_prob(probs_cr, rep)
    top_cr   = get_top_prediction(probs_cr)
    del logits_cr

    print(f"[corrupted] P(correct={cor}) = {p_cr_cor:.4f}   "
          f"P(replace={rep}) = {p_cr_rep:.4f}   → top={top_cr}")

    # ── 결과 검증 ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("검증 결과")
    print("=" * 60)

    ok1 = top_n == cor
    ok2 = top_cf == rep
    ok3 = p_n_cor > p_cr_cor
    ok4 = p_cf_rep > p_n_rep

    print(f"  {'✓' if ok1 else '✗'} neutral → correct 선택:    "
          f"top={top_n}  (기대={cor})")
    print(f"  {'✓' if ok2 else '✗'} conflict → replace 선택:   "
          f"top={top_cf}  (기대={rep})   KC {'발생' if ok2 else '미발생'}")
    print(f"  {'✓' if ok3 else '✗'} P(correct): neutral > corrupted:  "
          f"{p_n_cor:.4f} > {p_cr_cor:.4f}")
    print(f"  {'✓' if ok4 else '✗'} P(replace): conflict > neutral:   "
          f"{p_cf_rep:.4f} > {p_n_rep:.4f}")

    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"\n  전체 상태: {'모두 통과 ✓' if all_ok else '일부 실패 ✗'}")

    if not ok2:
        print("\n  [주의] KC가 이 케이스에서 발생하지 않음.")
        print("         passes_kc_filter=True이더라도 모델 버전(2b vs 9b) 차이일 수 있음.")
    if not ok3:
        print("\n  [주의] noise가 correct 확률을 오히려 높임.")
        print("         noise σ가 너무 작거나 context 인덱스 매핑 오류일 수 있음.")

    print()


if __name__ == "__main__":
    main()

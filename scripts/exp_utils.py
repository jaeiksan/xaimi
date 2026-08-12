#!/usr/bin/env python3
"""
exp_utils.py — 실험 1~3 공통 유틸리티

세 실험이 공유하는:
  - 모델/데이터 로딩
  - 프롬프트 빌더 (clean, conflict, supportive)
  - answer log-prob / logit-diff 계산
  - residual 캐싱 (hook_resid_post)
  - KC direction 패칭
  - 결과 저장
"""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# ── 공통 상수 ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/gemma-2-2b"
DATA_PATH    = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
RESULTS_ROOT = Path("/home/jaeik/xaimi/results")
LAYERS       = list(range(6, 20))   # 6, 7, ..., 19  (14개)
N_CASES      = 200
SEED         = 42

# 실험 3 대상 헤드 (파일럿 step5에서 확인된 고레버리지 헤드)
MED_LAYER = 17
MED_HEAD  = 3

LETTER_VARIANTS: Dict[str, List[int]] = {}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(device: str = "cuda") -> HookedTransformer:
    """gemma-2-2b를 TransformerLens로 로드한다."""
    print(f"[load_model] {MODEL_NAME} (bfloat16) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    setup_letter_variants(model.tokenizer)
    return model


# ── 토크나이저 variants ────────────────────────────────────────────────────────
def setup_letter_variants(tokenizer) -> None:
    """A/B/C/D 각 letter의 space-prefix / no-space variant token ID를 수집."""
    global LETTER_VARIANTS
    for letter in "ABCD":
        ids: List[int] = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.append(toks[-1] if len(toks) > 1 else toks[0])
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
    print(f"[setup] LETTER_VARIANTS={LETTER_VARIANTS}", flush=True)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_cases(n: int = N_CASES, seed: int = SEED) -> List[Dict]:
    """kc_cases_1000.json에서 passes_kc_filter=True인 케이스 n개를 로드."""
    with DATA_PATH.open() as f:
        cases = json.load(f)
    cases = [c for c in cases if c.get("passes_kc_filter", True)]
    rng = random.Random(seed)
    rng.shuffle(cases)
    selected = cases[:n]
    print(f"[load_cases] {len(selected)}개 로드 (전체 {len(cases)}개 중)", flush=True)
    return selected


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────
def build_clean_prompt(question: str, options: Dict[str, str]) -> str:
    lines = [f"Question: {question}"]
    for l in "ABCD":
        lines.append(f"{l}. {options[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_conflict_prompt(context: str, question: str, options: Dict[str, str]) -> str:
    lines = ["Context:", context, "", f"Question: {question}"]
    for l in "ABCD":
        lines.append(f"{l}. {options[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_supportive_prompt(
    question: str,
    options: Dict[str, str],
    correct_answer: str,
    conflict_context: Optional[str] = None,
    conflict_answer: Optional[str] = None,
) -> str:
    """
    Supportive context 생성 방식 (Step 2 IG 결과 반영):

    우선: conflict_context에서 conflict_answer → correct_answer 텍스트 교체.
      → evidence span의 위치·길이가 conflict_prompt와 동일하게 유지됨
      → D_sup = h_support - h_clean이 "context 추가 효과"만 포착
      → kc_sup_overlap이 낮을수록 KC-specific signal임을 더 깔끔하게 분리 가능.
    Fallback: 교체가 실패하면 "The answer is {correct_answer}." prepend 방식 사용.
    """
    if conflict_context and conflict_answer:
        supportive_ctx = conflict_context.replace(conflict_answer, correct_answer)
        if supportive_ctx != conflict_context:
            return build_conflict_prompt(supportive_ctx, question, options)
    return build_conflict_prompt(f"The answer is {correct_answer}.", question, options)


# ── 확률 / logit 계산 ─────────────────────────────────────────────────────────
def get_letter_lp(lp_all: torch.Tensor, letter: str) -> float:
    """log_softmax 벡터에서 해당 letter의 max log-prob을 반환."""
    return max(lp_all[tid].item() for tid in LETTER_VARIANTS[letter])


def get_letter_prob(probs: torch.Tensor, letter: str) -> float:
    """softmax 확률 벡터에서 해당 letter의 max 확률을 반환."""
    return max(probs[tid].item() for tid in LETTER_VARIANTS[letter])


def get_logit_diff(logits: torch.Tensor, replace_option: str, correct_option: str) -> float:
    """logP(replace) - logP(correct) at final token position."""
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    return get_letter_lp(lp, replace_option) - get_letter_lp(lp, correct_option)


def get_correct_prob(logits: torch.Tensor, correct_option: str) -> float:
    """P(correct) at final token position (linear probability)."""
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    return get_letter_prob(probs, correct_option)


# ── Position 계산 ─────────────────────────────────────────────────────────────
def get_pos_indices(model: HookedTransformer, prompt: str) -> Tuple[int, int]:
    """
    prompt에서 두 위치의 token index를 반환:
      final_pos:      마지막 token (Answer: 의 ":")
      pre_answer_pos: "\nAnswer:" 직전 token (options 마지막 줄 끝)
    """
    ids = model.to_tokens(prompt, prepend_bos=True)[0]
    final_pos = int(ids.shape[0]) - 1

    split = "\nAnswer:"
    idx = prompt.rfind(split)
    if idx != -1:
        prefix_ids = model.to_tokens(prompt[:idx], prepend_bos=True)[0]
        pre_answer_pos = int(prefix_ids.shape[0]) - 1
    else:
        pre_answer_pos = final_pos - 1

    return final_pos, pre_answer_pos


def get_evidence_span(model: HookedTransformer, conflict_context: str) -> Tuple[int, int]:
    """
    conflict_prompt에서 evidence (conflict_context)의 token position range를 근사.
    프롬프트 구조: BOS + "Context:\n" + {conflict_context} + "\n\n..."
    근사 방식: "Context:\n"을 tokenize해서 시작 위치를 구하고, context를 tokenize해서 끝 위치를 계산.
    """
    prefix = "Context:\n"
    prefix_toks = model.tokenizer.encode(prefix, add_special_tokens=False)
    ev_start = 1 + len(prefix_toks)  # 1 = BOS

    context_toks = model.tokenizer.encode(conflict_context, add_special_tokens=False)
    ev_end = ev_start + len(context_toks)

    return ev_start, ev_end


# ── Residual 수집 ─────────────────────────────────────────────────────────────
def collect_residuals(
    model: HookedTransformer,
    prompt: str,
    layers: List[int],
    pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    지정한 layers의 hook_resid_post를 position pos에서 수집.

    Returns:
        residuals: (len(layers), d_model) float32 CPU
        logits_final: (vocab_size,) float32 CPU — final token logits
    """
    ids = model.to_tokens(prompt, prepend_bos=True)
    seq_len = int(ids.shape[1])
    pos = min(pos, seq_len - 1)

    with torch.no_grad():
        logits_full, cache = model.run_with_cache(
            ids,
            names_filter=lambda n: n.endswith("hook_resid_post"),
        )

    residuals = torch.stack([
        cache[f"blocks.{l}.hook_resid_post"][0, pos].float().cpu()
        for l in layers
    ])  # (len(layers), d_model)

    logits_cpu = logits_full[0, -1].float().cpu()

    del cache, logits_full
    gc.collect()
    torch.cuda.empty_cache()

    return residuals, logits_cpu


def collect_residuals_two_pos(
    model: HookedTransformer,
    prompt: str,
    layers: List[int],
    final_pos: int,
    pre_answer_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    한 번의 forward pass로 두 position의 residual을 모두 수집.

    Returns:
        res_final:      (len(layers), d_model) float32 CPU — final_pos에서의 residual
        res_pre_answer: (len(layers), d_model) float32 CPU — pre_answer_pos에서의 residual
        logits_cpu:     (vocab_size,) float32 CPU — final token logits
    """
    ids = model.to_tokens(prompt, prepend_bos=True)
    seq_len = int(ids.shape[1])
    fp = min(final_pos, seq_len - 1)
    pp = min(pre_answer_pos, seq_len - 1)

    with torch.no_grad():
        logits_full, cache = model.run_with_cache(
            ids,
            names_filter=lambda n: n.endswith("hook_resid_post"),
        )

    res_final = torch.stack([
        cache[f"blocks.{l}.hook_resid_post"][0, fp].float().cpu()
        for l in layers
    ])
    res_pre = torch.stack([
        cache[f"blocks.{l}.hook_resid_post"][0, pp].float().cpu()
        for l in layers
    ])
    logits_cpu = logits_full[0, -1].float().cpu()

    del cache, logits_full
    gc.collect()
    torch.cuda.empty_cache()

    return res_final, res_pre, logits_cpu


# ── KC direction 패칭 ─────────────────────────────────────────────────────────
def patch_kc_direction(
    h_conflict: torch.Tensor,
    kc_dir: torch.Tensor,
    clean_mean_proj: float,
) -> torch.Tensor:
    """
    h_conflict: (d_model,) — conflict run의 residual
    kc_dir:     (d_model,) — exp1에서 구한 KC 주방향 (unit norm)
    clean_mean_proj: scalar — clean run 평균의 kc_dir 방향 projection

    KC 방향 성분만 clean 수준으로 교체한 residual 반환.
    h_patched = h_conflict - (proj_conflict - clean_mean_proj) * kc_dir
    """
    proj = (h_conflict @ kc_dir).item()
    return h_conflict - (proj - clean_mean_proj) * kc_dir


def make_patch_hook(h_patched: torch.Tensor, pos_idx: int, device: str):
    """hook_resid_post에 개입하는 hook function factory."""
    h_gpu = h_patched.to(device).to(torch.bfloat16)

    def hook_fn(val: torch.Tensor, hook) -> torch.Tensor:
        val[0, pos_idx, :] = h_gpu
        return val

    return hook_fn


# ── I/O 유틸 ─────────────────────────────────────────────────────────────────
def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {path}", flush=True)


def save_pt(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[save] {path}", flush=True)


def load_trajectory_stats(exp1_dir: Path) -> Dict:
    """exp1에서 저장한 trajectory_stats.pt를 로드."""
    path = exp1_dir / "trajectory_stats.pt"
    stats = torch.load(path, map_location="cpu")
    print(f"[load] trajectory_stats from {path}", flush=True)
    return stats

#!/usr/bin/env python3
"""
Step 2: Integrated Gradients — KC 유발 토큰 후보 탐색

방법:  Manual IG (TransformerLens + bfloat16 autograd)
       inseq는 text baseline 미지원 → 수동 IG 선택
베이스라인: neutral prompt embedding (zero-padding으로 conflict 길이에 맞춤)
타겟:  log_prob(replace_option) - log_prob(correct_option)
스텝:  50 (batch_sz=10씩 처리)
케이스: type별 100개 × 3 + no_b 케이스 전수 포함

실행:
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step2_ig.py           # 전체
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step2_ig.py --test 10 # 테스트
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer

# ── 경로 / 상수 ────────────────────────────────────────────────────────────────
KC_PATH      = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
STEP1_PATH   = Path("/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_results.json")
OUT_DIR      = Path("/home/jaeik/xaimi/phase1/results/step2_ig")
MODEL_NAME   = "google/gemma-2-2b"
SEED         = 42
MAX_SEQ_LEN  = 2048
IG_STEPS     = 50
IG_BATCH_SZ  = 10
SAMPLE_PER_TYPE = 100
EARLY_THRESH = 3
LATE_THRESH  = 15
NEAR_ZERO_THRESH = 1e-4
CHECKPOINT_EVERY = 50

CTYPES       = ("misinformation", "temporal", "semantic")
CTYPE_COLORS = {"misinformation": "#e74c3c", "temporal": "#3498db", "semantic": "#2ecc71"}
GROUP_COLORS = {
    "context_claim":    "#c0392b",
    "context_evidence": "#e67e22",
    "question":         "#2980b9",
    "options":          "#27ae60",
    "format_tokens":    "#95a5a6",
}
GROUPS       = ("context_claim", "context_evidence", "question", "options", "format_tokens")
TGROUP_COLORS = {"early": "#e74c3c", "mid": "#f39c12", "late": "#3498db", "no_b": "#9b59b6"}

FILTER_TOKENS = {
    "<bos>", "<eos>", "<pad>",
    ",", ".", "!", "?", ";", ":",
    "\n", "\n\n", " ", "  ",
    "*", "\"", "'", "(", ")", "[", "]", "-",
    "a", "A", "the", "The", "of", "in", "In",
    "is", "was", "and", "to", "for", "that",
    "It", "He", "She", "his", "her",
    "Context", "Question", "Answer",
}

LETTER_VARIANTS: Dict[str, List[int]] = {}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test",       type=int, default=None, help="테스트: N개만 처리")
    p.add_argument("--n_per_type", type=int, default=None, help="타입별 샘플 수 (기본: SAMPLE_PER_TYPE=100)")
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


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────
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


# ── Context 분리: claim / evidence ────────────────────────────────────────────
def split_claim_evidence(context: str) -> Tuple[str, str]:
    """
    build_context에서 f'{claim}\\n\\n{evidence}' 형태로 만들어짐.
    첫 번째 \\n\\n 기준으로 분리.
    없으면 첫 문장 기준으로 분리.
    """
    if "\n\n" in context:
        idx = context.index("\n\n")
        return context[:idx].strip(), context[idx + 2:].strip()
    for i, ch in enumerate(context):
        if ch == "." and i + 1 < len(context) and context[i + 1] in " \n":
            return context[:i + 1].strip(), context[i + 2:].strip()
    return context.strip(), ""


# ── 토큰 → 그룹 매핑 (offset_mapping 기반) ────────────────────────────────────
def get_group_token_map(
    tokenizer,
    conflict_prompt: str,
    context: str,
    question: str,
    options: Dict[str, str],
    replace_opt: str,
    correct_opt: str,
) -> Dict[str, List[int]]:
    """
    conflict_prompt의 각 토큰을 5개 그룹으로 분류.
    Returns {group_name: [token_indices]}
    """
    enc     = tokenizer(conflict_prompt, return_offsets_mapping=True,
                        add_special_tokens=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()  # [(start, end), ...]
    n_toks  = len(offsets)

    claim, evidence = split_claim_evidence(context)
    p = conflict_prompt

    # ── 문자 범위 계산 ─────────────────────────────────────────────────────────
    ctx_hdr_end   = len("Context:\n")                        # "Context:\n" 끝
    claim_start   = ctx_hdr_end
    claim_end     = claim_start + len(claim)

    # "\n\nQuestion: " 위치 (마지막 occurrence 사용 → 혹시 context에 같은 패턴이 있어도 대응)
    q_marker      = "\n\nQuestion: "
    q_section_pos = p.rfind(q_marker)
    evidence_end  = q_section_pos if q_section_pos >= claim_end else claim_end

    q_text_start  = q_section_pos + len(q_marker)
    q_text_end    = q_text_start + len(question)

    # "\nAnswer:" 위치
    ans_pos       = p.rfind("\nAnswer:")
    opts_start    = q_text_end + 1    # '\n' after question
    opts_end      = ans_pos

    # option 텍스트 위치: A~D를 순서대로 탐색 (A는 opts_start에서 직접 시작)
    option_text_ranges: Dict[str, Tuple[int, int]] = {}
    search_from = opts_start
    for l in "ABCD":
        if l == "A":
            marker = f"A. "                    # A 앞에는 \n 없음 (options 섹션 첫 줄)
            pos    = p.find(marker, search_from)
            off    = 0
        else:
            marker = f"\n{l}. "
            pos    = p.find(marker, search_from)
            off    = 1                          # \n 건너뜀
        if 0 <= pos < opts_end:
            ts = pos + off + len(f"{l}. ")
            te = ts + len(options[l])
            option_text_ranges[l] = (ts, te)
            search_from = te                   # 다음 옵션은 이후에서 탐색
        else:
            option_text_ranges[l] = (-1, -1)

    rep_text_s, rep_text_e = option_text_ranges.get(replace_opt, (-1, -1))
    cor_text_s, cor_text_e = option_text_ranges.get(correct_opt, (-1, -1))

    # ── 각 토큰을 그룹에 할당 ─────────────────────────────────────────────────
    groups: Dict[str, List[int]] = {g: [] for g in GROUPS}
    replace_opt_tokens: List[int] = []
    correct_opt_tokens: List[int] = []

    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:   # BOS or special token
            groups["format_tokens"].append(i)
            continue

        mid = (s + e) / 2      # 토큰 중심 문자 위치 기준으로 그룹 판단

        if mid < ctx_hdr_end:
            groups["format_tokens"].append(i)
        elif mid < claim_end:
            groups["context_claim"].append(i)
            if rep_text_s >= 0 and rep_text_s <= mid < rep_text_e:
                replace_opt_tokens.append(i)
            if cor_text_s >= 0 and cor_text_s <= mid < cor_text_e:
                correct_opt_tokens.append(i)
        elif mid < evidence_end:
            groups["context_evidence"].append(i)
        elif mid < q_section_pos + 2:   # "\n\n" 부분
            groups["format_tokens"].append(i)
        elif mid < q_text_start:        # "Question: " 레이블
            groups["format_tokens"].append(i)
        elif mid < q_text_end:
            groups["question"].append(i)
        elif mid < opts_start:          # '\n' after question
            groups["format_tokens"].append(i)
        elif mid < opts_end:
            groups["options"].append(i)
            if rep_text_s >= 0 and rep_text_s <= mid < rep_text_e:
                replace_opt_tokens.append(i)
            if cor_text_s >= 0 and cor_text_s <= mid < cor_text_e:
                correct_opt_tokens.append(i)
        else:
            groups["format_tokens"].append(i)  # "\nAnswer:"

    groups["_replace_opt"] = replace_opt_tokens
    groups["_correct_opt"]  = correct_opt_tokens
    return groups


# ── Integrated Gradients (batched alpha steps) ─────────────────────────────
def compute_ig(
    model:       HookedTransformer,
    conflict_ids: torch.Tensor,   # [seq_c]
    neutral_ids:  torch.Tensor,   # [seq_n]
    replace_opt:  str,
    correct_opt:  str,
    steps:       int = IG_STEPS,
    batch_sz:    int = IG_BATCH_SZ,
) -> torch.Tensor:
    """
    Returns per-token attribution: float32 [seq_c]
    Baseline: zero-padded neutral embeddings (front-padded to conflict length)
    """
    W_E = model.embed.W_E   # [vocab, d_model] bfloat16

    with torch.no_grad():
        conflict_emb = W_E[conflict_ids].detach()   # [seq_c, d_model]
        neutral_emb  = W_E[neutral_ids].detach()    # [seq_n, d_model]

    seq_c = conflict_emb.shape[0]
    seq_n = neutral_emb.shape[0]
    pad   = seq_c - seq_n

    if pad > 0:
        neutral_emb = torch.cat([
            torch.zeros(pad, neutral_emb.shape[1],
                        dtype=neutral_emb.dtype, device=neutral_emb.device),
            neutral_emb,
        ], dim=0)
    elif pad < 0:
        neutral_emb = neutral_emb[-seq_c:]

    delta   = conflict_emb - neutral_emb            # [seq_c, d_model]
    alphas  = torch.linspace(0., 1., steps, device=model.cfg.device)

    accum_g = torch.zeros(seq_c, model.cfg.d_model,
                          dtype=torch.float32, device=model.cfg.device)

    for i in range(0, steps, batch_sz):
        al  = alphas[i:i + batch_sz]                # [k]
        k   = al.shape[0]

        # interpolations: bfloat16 with requires_grad
        interps = (neutral_emb.unsqueeze(0) +
                   al[:, None, None] * delta.unsqueeze(0)).clone().requires_grad_(True)

        toks_b = conflict_ids.unsqueeze(0).expand(k, -1).contiguous()

        captured = [interps]

        def hook_fn(val, hook, _c=captured):
            return _c[0]

        with torch.enable_grad():
            logits = model.run_with_hooks(toks_b, fwd_hooks=[("hook_embed", hook_fn)])

        lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)  # [k, vocab]

        r_s = torch.stack([lp[:, tid] for tid in LETTER_VARIANTS[replace_opt]]).max(0).values
        c_s = torch.stack([lp[:, tid] for tid in LETTER_VARIANTS[correct_opt]]).max(0).values
        score = (r_s - c_s).sum()

        g = torch.autograd.grad(score, interps)[0].float()  # [k, seq_c, d_model]
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        # Trapezoidal weights
        w = torch.ones(k, 1, 1, dtype=torch.float32, device=g.device)
        if i == 0:
            w[0] *= 0.5
        if i + k == steps:
            w[-1] *= 0.5
        accum_g += (g * w).sum(0)

        del interps, logits, lp, g
        torch.cuda.empty_cache()

    integrated = accum_g / (steps - 1)                         # [seq_c, d_model]
    attr       = (delta.float() * integrated).norm(dim=-1)     # [seq_c]
    return attr.detach().cpu()


# ── 케이스 처리 ───────────────────────────────────────────────────────────────
def process_case(
    model,
    case: Dict,
    steps: int,
    batch_sz: int,
    sanity: bool = False,
) -> Optional[Dict]:
    tokenizer = model.tokenizer
    ctx     = case["conflict_context"]
    q       = case["question"]
    opts    = case["options"]
    rep     = case["replace_option"]
    cor     = case["correct_option"]

    conflict_p = build_conflict_prompt(ctx, q, opts)
    neutral_p  = build_neutral_prompt(q, opts)

    conflict_ids = model.to_tokens(conflict_p, prepend_bos=True)[0]
    neutral_ids  = model.to_tokens(neutral_p,  prepend_bos=True)[0]

    if conflict_ids.shape[0] > MAX_SEQ_LEN:
        conflict_ids = conflict_ids[:MAX_SEQ_LEN]

    attr = compute_ig(model, conflict_ids, neutral_ids, rep, cor, steps, batch_sz)

    total_attr = attr.sum().item()
    if not np.isfinite(total_attr) or total_attr < NEAR_ZERO_THRESH:
        return None  # NaN / 수치 불안정 케이스

    # 토큰 그룹 매핑
    group_map = get_group_token_map(tokenizer, conflict_p, ctx, q, opts, rep, cor)

    # 그룹별 attribution 합산
    group_attr: Dict[str, float] = {}
    for g in GROUPS:
        idxs = [i for i in group_map.get(g, []) if i < len(attr)]
        group_attr[g] = float(attr[idxs].sum().item()) if idxs else 0.0

    total = sum(group_attr.values()) + 1e-12
    group_ratio = {g: group_attr[g] / total for g in GROUPS}

    # replace / correct option attribution
    rep_idxs = [i for i in group_map.get("_replace_opt", []) if i < len(attr)]
    cor_idxs = [i for i in group_map.get("_correct_opt", []) if i < len(attr)]
    rep_attr = float(attr[rep_idxs].sum().item()) if rep_idxs else 0.0
    cor_attr = float(attr[cor_idxs].sum().item()) if cor_idxs else 0.0

    # Top tokens
    topk_n  = min(20, len(attr))
    topk_v, topk_i = attr.topk(topk_n)
    tok_strs = [tokenizer.decode([conflict_ids[idx].item()]) for idx in topk_i.tolist()]

    # 어느 그룹인지
    idx_to_group: Dict[int, str] = {}
    for g in GROUPS:
        for idx in group_map.get(g, []):
            idx_to_group[idx] = g

    top_tokens = [
        {
            "token":       tok_strs[k],
            "attribution": float(topk_v[k].item()),
            "group":       idx_to_group.get(topk_i[k].item(), "unknown"),
            "tok_idx":     int(topk_i[k].item()),
        }
        for k in range(topk_n)
    ]

    # ── sanity check 출력 ─────────────────────────────────────────────────────
    if sanity:
        print("\n  === Sanity Check: 첫 케이스 그룹 경계 ===")
        all_toks = [tokenizer.decode([conflict_ids[i].item()]) for i in range(len(attr))]
        for g in GROUPS:
            idxs = group_map.get(g, [])
            if idxs:
                first = min(idxs); last = max(idxs)
                fstr  = repr(all_toks[first])[:20]
                lstr  = repr(all_toks[last])[:20]
                print(f"  [{g}] 토큰 {first}~{last} (n={len(idxs)}): {fstr} ... {lstr}")
        rep_idxs_dbg = group_map.get("_replace_opt", [])
        cor_idxs_dbg = group_map.get("_correct_opt", [])
        print(f"  [replace_opt={rep}] 토큰 인덱스: {rep_idxs_dbg[:5]}")
        print(f"  [correct_opt={cor}] 토큰 인덱스: {cor_idxs_dbg[:5]}")
        print(f"\n  attribution 합계: {total_attr:.4f}")
        print(f"  replace_option attr: {rep_attr:.4f}")
        print(f"  correct_option attr: {cor_attr:.4f}")
        print(f"  비율 replace/correct: {rep_attr / (cor_attr + 1e-9):.2f}")
        print()

    return {
        "id":            case["id"],
        "conflict_type": case["conflict_type"],
        "transition_group":       case.get("transition_group", "mid"),
        "transition_layer_A":     case.get("transition_layer_A"),
        "group_attribution":      group_ratio,
        "group_attribution_abs":  {g: group_attr[g] for g in GROUPS},
        "options_attribution": {
            "replace_option_attr": rep_attr,
            "correct_option_attr": cor_attr,
            "replace_greater":     bool(rep_attr > cor_attr),
        },
        "top_tokens":    top_tokens,
        "kc_severity":   case.get("kc_severity", 0.0),
        "total_attr":    total_attr,
    }


# ── 시각화 ────────────────────────────────────────────────────────────────────
def _stacked_bar(data: Dict[str, Dict[str, float]], labels: List[str],
                 title: str, out_path: Path) -> None:
    keys   = list(data.keys())
    bottom = np.zeros(len(keys))
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.5), 5))
    for g in GROUPS:
        vals = [data[k].get(g, 0.0) for k in keys]
        ax.bar(keys, vals, bottom=bottom, label=g,
               color=GROUP_COLORS[g], alpha=0.85, edgecolor="white")
        bottom += np.array(vals)
    ax.set_ylabel("Attribution Ratio")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


def plot_group_by_type(results: List[Dict], out_path: Path) -> None:
    data: Dict[str, Dict[str, float]] = {}
    for ct in CTYPES:
        sub = [r for r in results if r["conflict_type"] == ct]
        if not sub:
            continue
        data[ct] = {g: float(np.mean([r["group_attribution"][g] for r in sub])) for g in GROUPS}
    _stacked_bar(data, CTYPES,
                 "Group Attribution by Conflict Type", out_path)


def plot_group_by_transition(results: List[Dict], out_path: Path) -> None:
    tgroups = ["early", "mid", "late", "no_b"]
    data: Dict[str, Dict[str, float]] = {}
    for tg in tgroups:
        sub = [r for r in results if r["transition_group"] == tg]
        if not sub:
            continue
        data[tg] = {g: float(np.mean([r["group_attribution"][g] for r in sub])) for g in GROUPS}
    _stacked_bar(data, tgroups,
                 "Group Attribution by Transition Group", out_path)


def plot_top_tokens_by_type(results: List[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, ct in zip(axes, CTYPES):
        sub = [r for r in results if r["conflict_type"] == ct]
        tok_sum:   Dict[str, float] = {}
        tok_cnt:   Dict[str, int]   = {}
        tok_group: Dict[str, str]   = {}
        for r in sub:
            for t in r["top_tokens"]:
                tok = t["token"]
                tok_sum[tok]   = tok_sum.get(tok, 0.0) + t["attribution"]
                tok_cnt[tok]   = tok_cnt.get(tok, 0) + 1
                tok_group[tok] = t.get("group", "format_tokens")
        tok_score = {
            tok: tok_sum[tok] / tok_cnt[tok]
            for tok in tok_sum
            if tok not in FILTER_TOKENS
        }
        top = sorted(tok_score.items(), key=lambda x: -x[1])[:15]
        if not top:
            continue
        toks, vals = zip(*top)
        colors = [GROUP_COLORS.get(tok_group.get(tk, "format_tokens"), "#95a5a6") for tk in toks]
        ax.barh(range(len(toks)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(toks)))
        ax.set_yticklabels([repr(t) for t in reversed(toks)], fontsize=8)
        ax.set_title(ct, fontsize=11)
        ax.set_xlabel("Avg Attribution (total/count)")
    fig.suptitle("Top Attribution Tokens by Conflict Type (filtered, avg)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


def plot_severity_vs_attribution(results: List[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    jitter = np.random.default_rng(42).uniform(-0.02, 0.02, len(results))
    for ct in CTYPES:
        sub = [r for r in results if r["conflict_type"] == ct]
        if not sub:
            continue
        xs = np.array([r["group_attribution"]["context_claim"] for r in sub])
        ys = np.array([r["kc_severity"] for r in sub])
        ax.scatter(xs, ys, color=CTYPE_COLORS[ct], alpha=0.45, s=18, label=ct)
        if len(xs) > 5:
            try:
                m, b = np.polyfit(xs, ys, 1)
                xr = np.linspace(xs.min(), xs.max(), 80)
                ax.plot(xr, m * xr + b, color=CTYPE_COLORS[ct], linewidth=1.5, alpha=0.7)
            except (np.linalg.LinAlgError, ValueError):
                pass

    all_x = np.array([r["group_attribution"]["context_claim"] for r in results])
    all_y = np.array([r["kc_severity"] for r in results])
    if len(all_x) > 5:
        try:
            r_all, p_all = scipy_stats.pearsonr(all_x, all_y)
            m, b = np.polyfit(all_x, all_y, 1)
            xr = np.linspace(all_x.min(), all_x.max(), 100)
            ax.plot(xr, m * xr + b, "k--", linewidth=1.8,
                    label=f"all  r={r_all:+.3f} p={p_all:.3f}")
        except (np.linalg.LinAlgError, ValueError):
            pass

    ax.set_xlabel("context_claim Attribution Ratio", fontsize=12)
    ax.set_ylabel("KC Severity", fontsize=12)
    ax.set_title("Severity vs Context-Claim Attribution", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def compute_stats(results: List[Dict]) -> Dict:
    def mean_groups(sub: List[Dict]) -> Dict[str, float]:
        if not sub:
            return {}
        return {g: float(np.mean([r["group_attribution"][g] for r in sub])) for g in GROUPS}

    # 전체 top tokens: score = total_attr / 등장횟수, FILTER_TOKENS 제외
    def top_tokens_agg(sub: List[Dict], n: int = 20) -> List[Dict]:
        tok_sum: Dict[str, float] = {}
        tok_cnt: Dict[str, int]   = {}
        for r in sub:
            for t in r["top_tokens"]:
                tok = t["token"]
                tok_sum[tok] = tok_sum.get(tok, 0.0) + t["attribution"]
                tok_cnt[tok] = tok_cnt.get(tok, 0) + 1
        tok_score = {
            tok: tok_sum[tok] / tok_cnt[tok]
            for tok in tok_sum
            if tok not in FILTER_TOKENS
        }
        top = sorted(tok_score.items(), key=lambda x: -x[1])[:n]
        return [{"token": k, "score": v, "total_attr": tok_sum[k], "count": tok_cnt[k]} for k, v in top]

    # severity correlation
    def pearson_info(xs, ys) -> str:
        if len(xs) < 3:
            return "n/a"
        r, p = scipy_stats.pearsonr(xs, ys)
        return f"r = {r:+.4f}, p = {p:.4f}"

    all_x = [r["group_attribution"]["context_claim"] for r in results]
    all_y = [r["kc_severity"] for r in results]

    stats = {
        "n":       len(results),
        "n_unstable": 0,
        "overall": {
            "group_attribution_mean": mean_groups(results),
        },
        "by_type": {
            ct: {"group_attribution_mean": mean_groups([r for r in results if r["conflict_type"] == ct])}
            for ct in CTYPES
        },
        "by_transition_group": {
            tg: {"group_attribution_mean": mean_groups([r for r in results if r["transition_group"] == tg])}
            for tg in ["early", "mid", "late", "no_b"]
        },
        "top_tokens_by_type": {
            ct: top_tokens_agg([r for r in results if r["conflict_type"] == ct])
            for ct in CTYPES
        },
        "severity_correlation": {
            "context_claim_vs_severity": pearson_info(all_x, all_y),
        },
    }
    return stats


# ── 콘솔 분석 출력 ────────────────────────────────────────────────────────────
def print_analyses(results: List[Dict], stats: Dict) -> None:
    N = len(results)
    print(f"\n{'='*65}")
    print(f"분석 1: 전체 그룹별 attribution (n={N})")
    print(f"{'='*65}")
    print(f"  {'그룹':<22} {'전체':>8}", end="")
    for ct in CTYPES:
        print(f"  {ct[:8]:>10}", end="")
    print()
    for g in GROUPS:
        ov  = stats["overall"]["group_attribution_mean"].get(g, 0)
        row = f"  {g:<22} {ov:>8.3f}"
        for ct in CTYPES:
            v = stats["by_type"][ct]["group_attribution_mean"].get(g, 0)
            row += f"  {v:>10.3f}"
        print(row)

    print(f"\n{'='*65}")
    print("분석 2: 이른/늦은/전환없음 그룹 비교")
    print(f"{'='*65}")
    header = f"  {'그룹':<8}"
    for g in GROUPS:
        header += f"  {g[:10]:>12}"
    print(header)
    for tg in ["early", "mid", "late", "no_b"]:
        d   = stats["by_transition_group"][tg]["group_attribution_mean"]
        n_g = sum(1 for r in results if r["transition_group"] == tg)
        row = f"  {tg:<8}(n={n_g})"
        for g in GROUPS:
            row += f"  {d.get(g, 0):>12.3f}"
        print(row)

    print(f"\n{'='*65}")
    print("분석 3: 상위 attribution 토큰 (type별 top 10, 필터 적용, avg score)")
    print(f"{'='*65}")
    for ct in CTYPES:
        tops = stats["top_tokens_by_type"][ct][:10]
        print(f"\n  [{ct}]")
        for i, t in enumerate(tops, 1):
            print(f"    {i:2d}. {repr(t['token']):<30} score={t['score']:.4f}  (total={t['total_attr']:.2f}, n={t['count']})")

    print(f"\n{'='*65}")
    print("분석 4: severity vs attribution 상관관계")
    print(f"{'='*65}")
    print(f"  context_claim vs severity: {stats['severity_correlation']['context_claim_vs_severity']}")
    for ct in CTYPES:
        sub = [r for r in results if r["conflict_type"] == ct]
        xs  = [r["group_attribution"]["context_claim"] for r in sub]
        ys  = [r["kc_severity"] for r in sub]
        if len(xs) > 2:
            r_val, p_val = scipy_stats.pearsonr(xs, ys)
            print(f"  {ct:15}: r = {r_val:+.4f}  (p = {p_val:.4f}, n={len(xs)})")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t0   = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[IG 방법] Manual IG (TransformerLens bfloat16 autograd)")
    print(f"         inseq는 text baseline 미지원 → 수동 IG 사용")

    # ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
    print("\n[1/5] 데이터 로드...")
    with KC_PATH.open()   as f: kc_cases  = json.load(f)
    with STEP1_PATH.open() as f: step1_res = json.load(f)

    # Step 1 결과로 transition_group 레이블 부여
    s1_map = {r["id"]: r for r in step1_res}
    def get_tgroup(case_id: str) -> str:
        r = s1_map.get(case_id, {})
        tl = (r.get("positions") or {}).get("A", {}).get("transition_layer")
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

    # ── 2. 케이스 샘플링 ────────────────────────────────────────────────────────
    no_b_cases = [c for c in kc_cases if c["transition_group"] == "no_b"]
    no_b_ids   = {c["id"] for c in no_b_cases}

    n_per_type = args.n_per_type if args.n_per_type is not None else SAMPLE_PER_TYPE
    sampled: List[Dict] = []
    rng = random.Random(SEED)
    for ct in CTYPES:
        pool = [c for c in kc_cases if c["conflict_type"] == ct and c["id"] not in no_b_ids]
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_type])

    # no_b 중 아직 미포함 케이스 추가
    sampled_ids = {c["id"] for c in sampled}
    extra_no_b  = [c for c in no_b_cases if c["id"] not in sampled_ids]
    sampled.extend(extra_no_b)
    rng.shuffle(sampled)

    if args.test:
        sampled = sampled[:args.test]
    N = len(sampled)
    print(f"  총 케이스: {N}개  (no_b 추가분: {len(extra_no_b)}개)")

    # ── 3. 모델 로드 ────────────────────────────────────────────────────────────
    print("\n[2/5] 모델 로드 (bfloat16)...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda", dtype=torch.bfloat16)
    model.eval()
    setup_letter_variants(model.tokenizer)
    print(f"  d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    # ── 4. IG 계산 ───────────────────────────────────────────────────────────
    print(f"\n[3/5] IG 계산 (steps={IG_STEPS}, batch_sz={IG_BATCH_SZ})...")
    results:   List[Dict] = []
    unstable:  List[str]  = []
    ckpt_path  = OUT_DIR / "ig_checkpoint.json"

    # 기존 체크포인트 로드
    if ckpt_path.exists() and not args.test:
        with ckpt_path.open() as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        sampled  = [c for c in sampled if c["id"] not in done_ids]
        print(f"  체크포인트 로드: {len(results)}개 완료, 남은 {len(sampled)}개")

    t_ph   = time.time()
    do_sanity = (len(results) == 0)

    for idx, case in enumerate(sampled):
        t_case = time.time()
        res    = process_case(model, case,
                              steps=IG_STEPS, batch_sz=IG_BATCH_SZ,
                              sanity=do_sanity)
        do_sanity = False

        if res is None:
            unstable.append(case["id"])
            print(f"  [skip] {case['id']} — attribution near-zero (unstable)")
            continue

        results.append(res)
        elapsed    = time.time() - t_ph
        case_time  = time.time() - t_case
        done_total = len(results) + len(unstable)
        rate       = done_total / elapsed
        eta        = (len(sampled) - idx - 1) / max(rate, 1e-6)

        if (idx + 1) % 10 == 0 or idx == 0:
            tg = case.get("transition_group", "?")
            print(f"  [{idx+1}/{len(sampled)}] {case['id'][:30]} | "
                  f"tg={tg} | {case_time:.1f}s/case | ETA={eta:.0f}s")

        # 체크포인트 저장
        if (len(results) % CHECKPOINT_EVERY == 0) and not args.test:
            with ckpt_path.open("w") as f:
                json.dump(results, f, ensure_ascii=False)
            print(f"  [ckpt] {len(results)}개 저장")

    print(f"  IG 완료: {len(results)}개 성공, {len(unstable)}개 불안정")

    # ── 5. 통계 및 저장 ─────────────────────────────────────────────────────────
    print("\n[4/5] 통계 계산...")
    stats = compute_stats(results)
    stats["n_unstable"] = len(unstable)
    stats["unstable_ids"] = unstable

    print_analyses(results, stats)

    print(f"\n[5/5] 결과 저장 및 시각화...")
    with (OUT_DIR / "ig_results.json").open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "ig_stats.json").open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ig_results.json ({len(results)} cases), ig_stats.json 저장")

    plot_group_by_type(results,       OUT_DIR / "group_attribution_by_type.png")
    plot_group_by_transition(results, OUT_DIR / "group_attribution_by_transition.png")
    plot_top_tokens_by_type(results,  OUT_DIR / "top_tokens_by_type.png")
    plot_severity_vs_attribution(results, OUT_DIR / "severity_vs_attribution.png")

    if ckpt_path.exists():
        ckpt_path.unlink()

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"Step 2 완료 (총 {elapsed/60:.1f}분, {len(results)}케이스)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

"""
Phase 4 — 공통 평가 유틸리티

모든 실험에서 재사용하는 hook factory, 평가 함수, 지표 계산 함수를 제공한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

VEC_DIR  = ROOT / "phase4/vectors"
DATA_DIR = ROOT / "phase4/data"
DEVICE   = "cuda"

CANDIDATE_LAYERS = list(range(10, 23))  # L10 ~ L22

# ── 벡터 로딩 ────────────────────────────────────────────────────────────────

def load_vectors():
    """사전계산된 방향 벡터, u_PK, residuals, logprobs 로드."""
    dir_A  = torch.load(VEC_DIR / "direction_A_train.pt",  map_location="cpu", weights_only=False)
    dir_B  = torch.load(VEC_DIR / "direction_B_train.pt",  map_location="cpu", weights_only=False)
    u_PK   = torch.load(VEC_DIR / "u_PK_train.pt",         map_location="cpu", weights_only=False)
    proj_c = torch.load(VEC_DIR / "train_proj_clean.pt",   map_location="cpu", weights_only=False)
    bl     = torch.load(VEC_DIR / "baseline_logprobs.pt",  map_location="cpu", weights_only=False)
    res    = torch.load(VEC_DIR / "residuals.pt",          map_location="cpu", weights_only=False)
    return dir_A, dir_B, u_PK, proj_c, bl, res


def load_split(fname: str):
    with open(DATA_DIR / fname) as f:
        return json.load(f)


def compute_tau(proj_clean_train: torch.Tensor, k: float) -> float:
    """tau = mean + k * std  (train clean projections에서 계산)."""
    return float(proj_clean_train.mean() + k * proj_clean_train.std())


# ── Hook Factory ─────────────────────────────────────────────────────────────

def make_patching_hook(clean_residual: torch.Tensor, alpha: float):
    """Exp0 all-position: 모든 position에 clean residual을 interpolate. 강한 개입."""
    def hook_fn(value, hook):
        v = value.float()
        c = clean_residual.to(v.device).float()   # [d_model]
        return ((1 - alpha) * v + alpha * c.unsqueeze(0).unsqueeze(0)).to(value.dtype)
    return hook_fn


def make_patching_hook_last(clean_residual: torch.Tensor, alpha: float):
    """Exp0 last-position: 마지막 토큰만 교체. 다른 Exp와 일관된 보수적 upper bound."""
    def hook_fn(value, hook):
        v = value.float()
        c = clean_residual.to(v.device).float()   # [d_model]
        v[:, -1, :] = (1 - alpha) * v[:, -1, :] + alpha * c
        return v.to(value.dtype)
    return hook_fn


def make_steering_hook(direction: torch.Tensor, sign: float, alpha: float):
    """Exp1: h' = h + sign * alpha * d  (마지막 position)."""
    def hook_fn(value, hook):
        v = value.float()
        d = F.normalize(direction, dim=0).to(v.device)
        v[:, -1, :] = v[:, -1, :] + sign * alpha * d
        return v.to(value.dtype)
    return hook_fn


def make_std_steering_hook(direction: torch.Tensor, sign: float, alpha: float,
                           proj_std: float):
    """Exp1 대안: h' = h + sign * alpha * std_proj * d."""
    def hook_fn(value, hook):
        v = value.float()
        d = F.normalize(direction, dim=0).to(v.device)
        v[:, -1, :] = v[:, -1, :] + sign * alpha * proj_std * d
        return v.to(value.dtype)
    return hook_fn


def make_clamp_hook(direction: torch.Tensor, tau: float, beta: float):
    """Exp2: projection이 tau를 초과하는 만큼 제거 (마지막 position)."""
    def hook_fn(value, hook):
        v  = value.float()
        d  = F.normalize(direction, dim=0).to(v.device)
        p  = v[:, -1, :] @ d                   # [batch]
        excess = (p - tau).clamp(min=0)         # [batch]
        v[:, -1, :] = v[:, -1, :] - beta * excess.unsqueeze(-1) * d
        return v.to(value.dtype)
    return hook_fn


def make_gated_restoration_hook(
    gate_dir: torch.Tensor,
    tau: float,
    gamma: float,
    u_PK: torch.Tensor,
    alpha: float,
):
    """Exp3/4: gated PK restoration (마지막 position).
    h' = h + sigmoid(gamma * (h @ d_gate - tau)) * alpha * u_PK
    """
    def hook_fn(value, hook):
        v  = value.float()
        gd = F.normalize(gate_dir, dim=0).to(v.device)
        up = u_PK.to(v.device)
        proj = v[:, -1, :] @ gd               # [batch]
        gate = torch.sigmoid(gamma * (proj - tau))  # [batch]
        v[:, -1, :] = v[:, -1, :] + gate.unsqueeze(-1) * alpha * up
        return v.to(value.dtype)
    return hook_fn


def make_ungated_restoration_hook(u_PK: torch.Tensor, alpha: float):
    """Exp3 baseline: gate 없이 항상 u_PK 주입."""
    def hook_fn(value, hook):
        v  = value.float()
        up = u_PK.to(v.device)
        v[:, -1, :] = v[:, -1, :] + alpha * up
        return v.to(value.dtype)
    return hook_fn


def make_head_restoration_hook(u_PK_head: torch.Tensor, alpha: float,
                               gate_fn=None):
    """Exp5: attention head output에 복원 벡터 주입.
    hook_name = blocks.{layer}.attn.hook_result (shape: [batch, seq, n_heads, d_head])
    head_idx 기준으로 특정 head에만 적용한다.
    """
    def hook_fn(value, hook):
        # value: [batch, seq, n_heads, d_head]
        v  = value.float()
        up = u_PK_head.to(v.device)   # [n_heads, d_head] or [d_head]
        # Apply to last position
        if gate_fn is not None:
            gate = gate_fn(v)  # scalar or [batch]
            v[:, -1, :, :] = v[:, -1, :, :] + gate * alpha * up
        else:
            v[:, -1, :, :] = v[:, -1, :, :] + alpha * up
        return v.to(value.dtype)
    return hook_fn


# ── 평가 함수 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_with_hooks(model, prompt: str, hooks: list) -> torch.Tensor:
    """hook이 적용된 forward pass → last token logits [vocab_size]."""
    ids = model.to_tokens(prompt, prepend_bos=True)
    with model.hooks(fwd_hooks=hooks):
        logits = model(ids)
    return logits[0, -1, :].float().cpu()


def lp_pair(logits: torch.Tensor, cor_id: int, rep_id: int) -> Tuple[float, float]:
    return logits[cor_id].item(), logits[rep_id].item()


@torch.no_grad()
def evaluate_cases(
    model,
    cases: List[dict],
    hooks_fn,           # callable(case_idx) → List[Tuple[str, callable]]
    conditions: Tuple[str, ...] = ("conflict", "clean", "supportive"),
) -> Dict:
    """
    각 케이스를 지정 조건 × hooks으로 평가한다.

    hooks_fn(i) → hooks list for case i (may vary per case, e.g. patching)
    조건별 lp_correct, lp_conflict 반환.
    """
    from exp_utils import (
        build_clean_prompt,
        build_conflict_prompt,
        build_supportive_prompt,
        get_pos_indices,
    )

    N = len(cases)
    result = {
        c: {"lp_correct": torch.zeros(N), "lp_conflict": torch.zeros(N)}
        for c in conditions
    }

    for i, case in enumerate(cases):
        q   = case["question"]
        opt = case["options"]
        ctx = case["conflict_context"]
        cor = case["correct_option"]
        rep = case["replace_option"]
        ca  = case["conflict_answer"]
        coa = case["correct_answer"]

        cor_id = model.tokenizer.encode(f" {cor}", add_special_tokens=False)[-1]
        rep_id = model.tokenizer.encode(f" {rep}", add_special_tokens=False)[-1]

        hooks = hooks_fn(i)

        prompt_map = {
            "conflict":   build_conflict_prompt(ctx, q, opt),
            "clean":      build_clean_prompt(q, opt),
            "supportive": build_supportive_prompt(q, opt, coa, ctx, ca),
        }
        for cond in conditions:
            if cond not in prompt_map:
                continue
            logits = run_with_hooks(model, prompt_map[cond], hooks)
            lpc, lpr = lp_pair(logits, cor_id, rep_id)
            result[cond]["lp_correct"][i]  = lpc
            result[cond]["lp_conflict"][i] = lpr

    return result


# ── 지표 계산 ────────────────────────────────────────────────────────────────

def accuracy(lp_correct: torch.Tensor, lp_conflict: torch.Tensor) -> float:
    return float((lp_correct > lp_conflict).float().mean())


def logit_margin(lp_correct: torch.Tensor, lp_conflict: torch.Tensor) -> float:
    return float((lp_correct - lp_conflict).mean())


def compute_metrics(
    baseline: Dict,   # {cond: {"lp_correct":..., "lp_conflict":...}}
    after:    Dict,
) -> Dict:
    """전후 비교 지표 계산."""
    metrics = {}
    for cond in ("conflict", "clean", "supportive"):
        if cond not in baseline:
            continue
        acc_before = accuracy(baseline[cond]["lp_correct"], baseline[cond]["lp_conflict"])
        acc_after  = accuracy(after[cond]["lp_correct"],    after[cond]["lp_conflict"])
        margin_before = logit_margin(baseline[cond]["lp_correct"], baseline[cond]["lp_conflict"])
        margin_after  = logit_margin(after[cond]["lp_correct"],    after[cond]["lp_conflict"])
        metrics[f"{cond}_acc_before"]  = acc_before
        metrics[f"{cond}_acc_after"]   = acc_after
        metrics[f"{cond}_acc_gain"]    = acc_after - acc_before
        metrics[f"{cond}_margin_delta"] = margin_after - margin_before

    if all(k in metrics for k in ("conflict_acc_gain", "clean_acc_gain", "supportive_acc_gain")):
        conflict_gain = metrics["conflict_acc_gain"]
        clean_drop    = -metrics["clean_acc_gain"]
        sup_drop      = -metrics["supportive_acc_gain"]
        metrics["selective_gain"] = conflict_gain - (clean_drop + sup_drop) / 2

    return metrics


def compute_gate_stats(gate_values: Dict[str, torch.Tensor]) -> Dict:
    """gate_values = {cond: tensor[N]}."""
    stats = {}
    for cond, vals in gate_values.items():
        stats[f"gate_{cond}_mean"] = float(vals.mean())
        stats[f"gate_{cond}_std"]  = float(vals.std())
    if "conflict" in gate_values and "clean" in gate_values:
        stats["gate_separation"] = float(
            gate_values["conflict"].mean() - gate_values["clean"].mean()
        )
    return stats


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC (sklearn 없이 직접 계산)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = 0
    for p in pos:
        count += (p > neg).sum() + 0.5 * (p == neg).sum()
    return count / (len(pos) * len(neg))


def compute_projection_auroc(
    residuals: Dict,   # {cond: {layer: tensor[N, d_model]}}
    dir_A: Dict,       # {layer: tensor[d_model]}
    dir_B: Dict,
    layer: int,
    all_ids: List[int],
) -> Dict:
    """
    Conflict vs Clean+Supportive 분리 AUROC.
    labels: conflict=1, clean/supportive=0
    """
    h_conf = residuals["conflict"][layer][all_ids].float()
    h_cln  = residuals["clean"][layer][all_ids].float()
    h_sup  = residuals["supportive"][layer][all_ids].float()

    auroc = {}
    for dname, d_dict in [("A", dir_A), ("B", dir_B)]:
        d = d_dict[layer]
        proj_conf = (h_conf @ d).numpy()
        proj_cln  = (h_cln  @ d).numpy()
        proj_sup  = (h_sup  @ d).numpy()

        # conflict(1) vs clean(0)
        scores = np.concatenate([proj_conf, proj_cln])
        labels = np.concatenate([np.ones(len(proj_conf)), np.zeros(len(proj_cln))])
        auroc[f"auroc_{dname}_vs_clean"] = compute_auroc(scores, labels)

        # conflict(1) vs supportive(0)
        scores2 = np.concatenate([proj_conf, proj_sup])
        labels2 = np.concatenate([np.ones(len(proj_conf)), np.zeros(len(proj_sup))])
        auroc[f"auroc_{dname}_vs_supportive"] = compute_auroc(scores2, labels2)

    return auroc


# ── 결과 출력 ────────────────────────────────────────────────────────────────

def print_metrics(label: str, metrics: Dict) -> None:
    cfl = metrics.get("conflict_acc_after",    metrics.get("conflict_acc_gain", float("nan")))
    sel = metrics.get("selective_gain",         float("nan"))
    cln = metrics.get("clean_acc_after",        metrics.get("clean_acc_gain",    float("nan")))
    sup = metrics.get("supportive_acc_after",   metrics.get("supportive_acc_gain", float("nan")))
    print(f"  {label:40s}  conflict={cfl:.3f}  selective={sel:+.3f}  clean={cln:.3f}  sup={sup:.3f}")

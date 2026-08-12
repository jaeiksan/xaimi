"""
Phase 4 Exp3: Intervention Method Hook Factories

5가지 B-gated action hook과 공통 gate reader hook을 구현한다.
모든 hook은 gate_store를 통해 gate 값을 공유한다.

Hook 등록 원칙:
  Residual 방법 (1, 2):
    gate_reader: blocks.{layer}.hook_resid_post  (먼저 등록)
    action:      blocks.{layer}.hook_resid_post  (나중 등록)

  Attention 방법 (3, 4):
    gate_reader: blocks.{layer}.hook_resid_pre   (attention 계산 전)
    action:      blocks.{layer}.attn.hook_attn_scores

  MLP 방법 (5):
    gate_reader: blocks.{layer}.hook_resid_pre   (MLP 계산 전)
    action:      blocks.{layer}.mlp.hook_post
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


# ── Gate Reader (hook_resid_post 버전: Residual 방법 1, 2) ────────────────────

def make_gate_reader_resid_post(gate_dir: torch.Tensor, tau: float, gamma: float,
                                 gate_store: list):
    """
    hook_name: blocks.{layer}.hook_resid_post
    마지막 position의 B projection으로 gate 값 계산 → gate_store[0] 저장.
    residual stream은 수정 없이 반환.
    """
    nd = F.normalize(gate_dir, dim=0)
    def hook_fn(value, hook):
        v    = value.float()
        proj = v[:, -1, :] @ nd.to(v.device)
        gate_store[0] = torch.sigmoid(gamma * (proj - tau)).detach()
        return value
    return hook_fn


# ── Gate Reader (hook_resid_pre 버전: Attention / MLP 방법 3, 4, 5) ──────────

def make_gate_reader_resid_pre(gate_dir: torch.Tensor, tau: float, gamma: float,
                                gate_store: list):
    """
    hook_name: blocks.{layer}.hook_resid_pre
    attention / MLP 계산 전 residual로 gate 값 계산 → gate_store[0] 저장.
    residual stream은 수정 없이 반환.
    """
    nd = F.normalize(gate_dir, dim=0)
    def hook_fn(value, hook):
        v    = value.float()
        proj = v[:, -1, :] @ nd.to(v.device)
        gate_store[0] = torch.sigmoid(gamma * (proj - tau)).detach()
        return value
    return hook_fn


# ── 방법 1: B-gated + u_PK ───────────────────────────────────────────────────

def make_upk_hook(gate_store: list, u_PK: torch.Tensor, alpha: float):
    """
    h' = h + gate * alpha * u_PK
    u_PK: mean(h_clean - h_conflict), clean 상태 복원 방향
    """
    def hook_fn(value, hook):
        v    = value.float()
        up   = u_PK.to(v.device)
        gate = gate_store[0].to(v.device) if gate_store[0] is not None else torch.ones(v.shape[0], device=v.device)
        v[:, -1, :] = v[:, -1, :] + gate.unsqueeze(-1) * alpha * up
        return v.to(value.dtype)
    return hook_fn


# ── 방법 2: B-gated + Contrastive Direction ──────────────────────────────────

def make_contrast_hook(gate_store: list, d_contrast: torch.Tensor, alpha: float):
    """
    h' = h + gate * alpha * d_contrast
    d_contrast: normalize(mean(h_conflict|correct) - mean(h_conflict|wrong))
    """
    def hook_fn(value, hook):
        v    = value.float()
        dc   = d_contrast.to(v.device)
        gate = gate_store[0].to(v.device) if gate_store[0] is not None else torch.ones(v.shape[0], device=v.device)
        v[:, -1, :] = v[:, -1, :] + gate.unsqueeze(-1) * alpha * dc
        return v.to(value.dtype)
    return hook_fn


# ── 방법 3: B-gated + Attention Score Re-weighting ───────────────────────────

def make_attn_reweight_hook(gate_store: list, conflict_positions: List[int],
                             suppress_value: float):
    """
    hook_name: blocks.{layer}.attn.hook_attn_scores
    shape: [batch, n_heads, seq_len, seq_len]

    conflict token position에 대한 attention score를 gate 비례로 감소.
    fallback: conflict_positions가 빈 리스트면 아무것도 하지 않음.
    """
    def hook_fn(value, hook):
        if not conflict_positions or gate_store[0] is None:
            return value
        v = value.float()
        g = gate_store[0].to(v.device)     # [batch]
        for pos in conflict_positions:
            if pos < v.shape[-1]:
                v[:, :, :, pos] -= g.view(-1, 1, 1) * suppress_value
        return v.to(value.dtype)
    return hook_fn


# ── 방법 4: B-gated + Attention Redirect ─────────────────────────────────────

def make_attn_redirect_hook(gate_store: list,
                             conflict_positions: List[int],
                             question_positions: List[int],
                             suppress_value: float,
                             boost_value: float):
    """
    hook_name: blocks.{layer}.attn.hook_attn_scores
    shape: [batch, n_heads, seq_len, seq_len]

    conflict → suppress, question → boost
    fallback: conflict_positions가 빈 리스트면 아무것도 하지 않음.
    """
    def hook_fn(value, hook):
        if not conflict_positions or gate_store[0] is None:
            return value
        v = value.float()
        g = gate_store[0].to(v.device)     # [batch]
        for pos in conflict_positions:
            if pos < v.shape[-1]:
                v[:, :, :, pos] -= g.view(-1, 1, 1) * suppress_value
        for pos in question_positions:
            if pos < v.shape[-1]:
                v[:, :, :, pos] += g.view(-1, 1, 1) * boost_value
        return v.to(value.dtype)
    return hook_fn


def make_dynamic_attn_suppress_hook(gate_store: list, top_k: int,
                                    suppress_value: float):
    """
    마지막 token이 head 평균 기준으로 가장 많이 attend하는 상위 k개 position을 억제.
    BOS 위치는 항상 높은 경향이 있어 제외한다.
    """
    def hook_fn(value, hook):
        if gate_store[0] is None:
            return value
        v = value.float()
        g = gate_store[0].to(v.device)  # [batch]

        last_attn = v[:, :, -1, :]      # [batch, n_heads, seq_len]
        mean_attn = last_attn.mean(dim=1)
        mean_attn[:, 0] = -1e9
        k = min(top_k, int(mean_attn.shape[-1]))
        top_k_ids = mean_attn.topk(k, dim=-1).indices

        for b in range(v.shape[0]):
            for pos in top_k_ids[b]:
                pos_idx = pos.item()
                if pos_idx < v.shape[-1]:
                    v[b, :, :, pos_idx] -= g[b] * suppress_value
        return v.to(value.dtype)
    return hook_fn


def make_dynamic_attn_redirect_hook(gate_store: list, top_k_suppress: int,
                                    top_k_boost: int, suppress_value: float,
                                    boost_value: float, question_positions: List[int]):
    """
    상위 attend position을 동적으로 suppress하고 question positions를 boost.
    top_k_boost는 현재 구현에선 question position 전체에 동일 boost를 주되,
    향후 확장을 위해 인자로 유지한다.
    """
    _ = top_k_boost

    def hook_fn(value, hook):
        if gate_store[0] is None:
            return value
        v = value.float()
        g = gate_store[0].to(v.device)

        last_attn = v[:, :, -1, :]
        mean_attn = last_attn.mean(dim=1)
        mean_attn[:, 0] = -1e9
        k = min(top_k_suppress, int(mean_attn.shape[-1]))
        top_k_ids = mean_attn.topk(k, dim=-1).indices

        for b in range(v.shape[0]):
            for pos in top_k_ids[b]:
                pos_idx = pos.item()
                if pos_idx < v.shape[-1]:
                    v[b, :, :, pos_idx] -= g[b] * suppress_value
            for pos in question_positions:
                if pos < v.shape[-1]:
                    v[b, :, :, pos] += g[b] * boost_value
        return v.to(value.dtype)
    return hook_fn


# ── 방법 5: B-gated + MLP Neuron Suppression ─────────────────────────────────

def make_mlp_suppression_hook(gate_store: list, neuron_ids: List[int], beta: float):
    """
    hook_name: blocks.{layer}.mlp.hook_post
    shape: [batch, seq_len, d_mlp]

    B direction과 양수 aligned된 neuron의 activation을 gate 비례로 억제.
    h_mlp[:, -1, neuron_ids] -= gate * beta * relu(activation)
    """
    neuron_ids_tensor = torch.tensor(neuron_ids, dtype=torch.long)
    def hook_fn(value, hook):
        if gate_store[0] is None:
            return value
        v    = value.float()
        g    = gate_store[0].to(v.device)          # [batch]
        nids = neuron_ids_tensor.to(v.device)
        acts = v[:, -1, :][:, nids]                # [batch, k]
        relu_acts = acts.clamp(min=0)
        v[:, -1, :][:, nids] = acts - g.unsqueeze(-1) * beta * relu_acts
        return v.to(value.dtype)
    return hook_fn


def get_conflict_neurons_activation(
    mlp_activations: dict,
    layer: int,
    train_ids: list,
    top_k: int = 50,
):
    """
    conflict에서 많이 켜지고 clean에서 덜 켜지는 neuron을 activation 기반으로 선택.
    """
    act_conf = mlp_activations["conflict"][layer][train_ids].float()
    act_cln  = mlp_activations["clean"][layer][train_ids].float()

    diff = act_conf.mean(0) - act_cln.mean(0)
    diff_positive = diff.clamp(min=0)
    k = min(top_k, int(diff_positive.shape[0]))
    top_ids = diff_positive.topk(k).indices.tolist()

    print(f"  L{layer} activation-based: "
          f"n_positive={int((diff > 0).sum())}, "
          f"mean_diff_top={float(diff[top_ids].mean()):.4f}")
    return top_ids


def get_conflict_neurons_weight(model, dir_B: dict, layer: int, top_k: int = 50):
    """
    W_out @ d_B가 양수인 neuron 중 상위 top_k를 선택.
    """
    W_out = model.blocks[layer].mlp.W_out.float().cpu()
    d_B_l = F.normalize(dir_B[layer], dim=0).float().cpu()

    alignment = W_out @ d_B_l
    alignment_positive = alignment.clamp(min=0)
    k = min(top_k, int(alignment_positive.shape[0]))
    top_ids = alignment_positive.topk(k).indices.tolist()

    print(f"  L{layer} weight-based: "
          f"n_positive={int((alignment > 0).sum())}, "
          f"mean_align_top={float(alignment[top_ids].mean()):.4f}")
    return top_ids


# ── Position 탐색 유틸 ────────────────────────────────────────────────────────

def find_subsequence_positions(full_ids: torch.Tensor,
                                sub_ids: torch.Tensor) -> List[int]:
    """full_ids에서 sub_ids가 처음 등장하는 위치를 반환. 못 찾으면 []."""
    n_full = len(full_ids)
    n_sub  = len(sub_ids)
    for i in range(n_full - n_sub + 1):
        if (full_ids[i:i + n_sub] == sub_ids).all():
            return list(range(i, i + n_sub))
    return []


def get_false_entity_positions(model, prompt: str, false_entity: str) -> List[int]:
    """
    prompt에서 false entity (= conflict_answer 텍스트)가 차지하는 token position 목록.
    못 찾으면 [] (no-op fallback).
    """
    if not false_entity:
        return []
    full_ids   = model.to_tokens(prompt, prepend_bos=True)[0]
    entity_ids = model.to_tokens(false_entity, prepend_bos=False)[0]
    return find_subsequence_positions(full_ids, entity_ids)


def get_false_sentence_positions(model, prompt: str, false_sentence: str) -> List[int]:
    """
    prompt에서 false sentence가 차지하는 token position 목록.
    false_sentence가 빈 문자열이면 [] 반환 (caller가 fallback 처리).
    """
    if not false_sentence:
        return []
    full_ids = model.to_tokens(prompt, prepend_bos=True)[0]
    sent_ids = model.to_tokens(false_sentence, prepend_bos=False)[0]
    return find_subsequence_positions(full_ids, sent_ids)


def get_full_context_positions(model, prompt: str, context: str) -> List[int]:
    """prompt에서 conflict_context 전체가 차지하는 token position 목록."""
    if not context:
        return []
    full_ids = model.to_tokens(prompt, prepend_bos=True)[0]
    ctx_ids  = model.to_tokens(context, prepend_bos=False)[0]
    return find_subsequence_positions(full_ids, ctx_ids)


def get_question_positions(model, prompt: str, question: str) -> List[int]:
    """prompt에서 question이 차지하는 token position 목록."""
    full_ids = model.to_tokens(prompt, prepend_bos=True)[0]
    q_ids    = model.to_tokens(question, prepend_bos=False)[0]
    return find_subsequence_positions(full_ids, q_ids)


def find_false_sentence(conflict_context: str, conflict_answer: str) -> str:
    """
    conflict_context에서 conflict_answer를 포함하는 첫 번째 문장을 반환.
    못 찾으면 빈 문자열 (caller가 full_context로 fallback).
    """
    if not conflict_answer or not conflict_context:
        return ""
    for sent in conflict_context.replace(".\n", ". ").split(". "):
        if conflict_answer.lower() in sent.lower():
            return sent.strip()
    return ""

#!/usr/bin/env python3
"""
Phase 4 — Step 1: 벡터 사전계산

Train 케이스에서 방향 벡터 A, B와 복원 벡터 u_PK를 추출하고,
전체 케이스의 residual stream과 baseline logprob을 저장한다.

저장 파일:
  phase4/vectors/residuals.pt
    {cond: {layer: tensor[N, d_model]}}  # cond ∈ {conflict, clean, supportive}

  phase4/vectors/baseline_logprobs.pt
    {cond: {"lp_correct": tensor[N], "lp_conflict": tensor[N]}}

  phase4/vectors/direction_A_train.pt    {layer: unit_tensor[d_model]}
  phase4/vectors/direction_B_train.pt    {layer: unit_tensor[d_model]}
  phase4/vectors/u_PK_train.pt           {layer: tensor[d_model]}  (not normalized)
  phase4/vectors/train_proj_clean.pt     {dir: {layer: tensor[n_train]}}

실행 (약 40~60분):
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 python -u phase4/scripts/setup_vectors.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "phase4/scripts"))

from exp_utils import (
    build_clean_prompt,
    build_conflict_prompt,
    build_supportive_prompt,
    collect_residuals,
    get_pos_indices,
    load_model,
)
from intervention_methods import (
    get_conflict_neurons_activation,
    get_conflict_neurons_weight,
)

DEVICE     = "cuda"
VEC_DIR    = ROOT / "phase4/vectors"
DATA_DIR   = ROOT / "phase4/data"
VEC_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_LAYERS = list(range(10, 23))  # L10 ~ L22
CHECKPOINT       = 100


def load_split(fname: str):
    with open(DATA_DIR / fname) as f:
        return json.load(f)


# ── Residuals + Logprobs 수집 ─────────────────────────────────────────────────

def collect_all(model, cases, layers, label=""):
    N       = len(cases)
    d_model = model.cfg.d_model

    residuals = {
        cond: {l: torch.zeros(N, d_model) for l in layers}
        for cond in ("conflict", "clean", "supportive")
    }
    lp_correct  = {c: torch.zeros(N) for c in ("conflict", "clean", "supportive")}
    lp_conflict = {c: torch.zeros(N) for c in ("conflict", "clean", "supportive")}

    t0 = time.time()
    for i, case in enumerate(cases):
        q   = case["question"]
        opt = case["options"]
        ctx = case["conflict_context"]
        cor = case["correct_option"]
        rep = case["replace_option"]
        ca  = case["conflict_answer"]
        coa = case["correct_answer"]

        prompts = {
            "conflict":   build_conflict_prompt(ctx, q, opt),
            "clean":      build_clean_prompt(q, opt),
            "supportive": build_supportive_prompt(q, opt, coa, ctx, ca),
        }

        for cond, prompt in prompts.items():
            pos, _ = get_pos_indices(model, prompt)
            res, logits = collect_residuals(model, prompt, layers, pos)
            for li, l in enumerate(layers):
                residuals[cond][l][i] = res[li].cpu()

            probs = logits.softmax(-1)
            # letter token ids (single-token)
            cor_id  = model.tokenizer.encode(f" {cor}",  add_special_tokens=False)[-1]
            rep_id  = model.tokenizer.encode(f" {rep}",  add_special_tokens=False)[-1]
            lp_correct[cond][i]  = logits[cor_id].item()
            lp_conflict[cond][i] = logits[rep_id].item()

        if (i + 1) % CHECKPOINT == 0:
            print(f"  {label}[{i+1}/{N}] {time.time()-t0:.0f}s", flush=True)

    return residuals, lp_correct, lp_conflict


# ── Direction A: Mean Difference ─────────────────────────────────────────────

def extract_direction_A(h_conflict, h_clean, layers):
    dirs = {}
    for l in layers:
        diff = (h_conflict[l] - h_clean[l]).float()
        dirs[l] = F.normalize(diff.mean(0), dim=0)
    return dirs


# ── Direction B: SVD (conflict − supportive) ──────────────────────────────────

def extract_direction_B(h_conflict, h_supportive, layers):
    dirs = {}
    for l in layers:
        D = (h_conflict[l] - h_supportive[l]).float()
        _, _, Vt = torch.linalg.svd(D, full_matrices=False)
        dirs[l] = F.normalize(Vt[0], dim=0)
    return dirs


# ── u_PK: mean(h_clean − h_conflict) ─────────────────────────────────────────

def compute_u_PK(h_conflict, h_clean, layers):
    u_pk = {}
    for l in layers:
        u_pk[l] = (h_clean[l] - h_conflict[l]).float().mean(0)
    return u_pk


# ── Train clean projection (for tau 계산용) ───────────────────────────────────

def compute_train_proj_clean(h_clean, dir_A, dir_B, layers, train_ids):
    proj = {"A": {}, "B": {}}
    for l in layers:
        h = h_clean[l][train_ids].float()
        proj["A"][l] = (h @ dir_A[l]).cpu()
        proj["B"][l] = (h @ dir_B[l]).cpu()
    return proj


# ── Contrastive Direction: mean(h_conflict_correct) - mean(h_conflict_wrong) ──

def compute_contrastive_direction_from_new_cases(
    model, conflict_correct_cases, h_conflict_wrong, layers
):
    """
    conflict_correct_cases에서 conflict condition residual을 수집해
    d_contrast = normalize(mean(h_correct) - mean(h_wrong)) 계산.

    conflict_correct_cases: conflict prompt → model still outputs correct answer
    h_conflict_wrong: residuals["conflict"][l][train_ids] — KC train cases
    """
    from exp_utils import build_conflict_prompt

    N_correct = len(conflict_correct_cases)
    d_model   = model.cfg.d_model
    h_correct = {l: torch.zeros(N_correct, d_model) for l in layers}

    print(f"  collecting residuals for {N_correct} conflict_correct cases ...")
    for i, case in enumerate(conflict_correct_cases):
        prompt = build_conflict_prompt(
            case["conflict_context"], case["question"], case["options"]
        )
        ids = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                ids,
                names_filter=lambda n: any(
                    n == f"blocks.{l}.hook_resid_post" for l in layers
                ),
            )
        for l in layers:
            h_correct[l][i] = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float().cpu()
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{N_correct}]", flush=True)

    d_contrast = {}
    for l in layers:
        hc = h_correct[l].float()            # [N_correct, d_model]
        hw = h_conflict_wrong[l].float()     # [N_wrong, d_model]
        diff = hc.mean(0) - hw.mean(0)
        d_contrast[l] = F.normalize(diff, dim=0)
        print(f"  L{l}: |d_contrast|={diff.norm():.2f}  "
              f"n_correct={N_correct}  n_wrong={hw.shape[0]}")
    return d_contrast, h_correct


def collect_mlp_activations(model, cases, layers, label=""):
    """
    conflict / clean 조건에서 마지막 token의 mlp.hook_post activation 수집.
    """
    N = len(cases)
    d_mlp = model.cfg.d_mlp
    activations = {
        cond: {l: torch.zeros(N, d_mlp) for l in layers}
        for cond in ("conflict", "clean")
    }

    t0 = time.time()
    for i, case in enumerate(cases):
        prompts = {
            "conflict": build_conflict_prompt(
                case["conflict_context"], case["question"], case["options"]
            ),
            "clean": build_clean_prompt(case["question"], case["options"]),
        }

        for cond, prompt in prompts.items():
            ids = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    ids,
                    names_filter=lambda n: any(
                        n == f"blocks.{l}.mlp.hook_post" for l in layers
                    ),
                )
            for l in layers:
                activations[cond][l][i] = cache[f"blocks.{l}.mlp.hook_post"][0, -1, :].float().cpu()

        if (i + 1) % CHECKPOINT == 0:
            print(f"  {label}[{i+1}/{N}] {time.time()-t0:.0f}s", flush=True)

    return activations


# ── Conflict-aligned MLP Neurons (signed positive alignment) ─────────────────

def compute_conflict_neurons(model, dir_B, layers, top_k_list=(10, 20, 50, 100)):
    """
    W_out[d_mlp, d_model] 기준으로 B direction과 양수 방향으로 aligned된 neuron index.
    alignment = W_out @ d_B (부호 유지, abs 미사용).
    양수 alignment가 큰 neuron만 선택 — 이 neuron을 suppression하면 B direction 투영 감소.

    반환 형식: {f"L{l}_top{top_k}": [neuron_ids]} — 복수의 top_k 지원.
    """
    neurons = {}
    for l in layers:
        W_out    = model.blocks[l].mlp.W_out.float().cpu()  # [d_mlp, d_model]
        d_B      = dir_B[l].float()
        alignment = W_out @ d_B                              # [d_mlp], signed

        # positive alignment만 고려 (음수는 B방향 반대 → suppression이 역효과)
        pos_alignment = alignment.clamp(min=0)
        max_k = int(pos_alignment.shape[0])

        for top_k in top_k_list:
            k = min(top_k, max_k)
            ids = pos_alignment.topk(k).indices.tolist()
            neurons[f"L{l}_top{top_k}"] = ids

        print(f"  L{l}: max_pos_align={float(pos_alignment.max()):.4f}  "
              f"top50={neurons[f'L{l}_top50'][:5]}")
    return neurons


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    train_cases = load_split("train_800.json")
    dev_cases   = load_split("dev_200.json")
    all_cases   = train_cases + dev_cases   # consistent ordering

    # Index bookkeeping
    N_train = len(train_cases)
    N_all   = len(all_cases)
    train_ids = list(range(N_train))
    dev_ids   = list(range(N_train, N_all))

    print(f"Train: {N_train}, Dev: {len(dev_cases)}, Total: {N_all}")

    model = load_model(device=DEVICE)

    # ── 1. Collect residuals + logprobs ────────────────────────────────────────
    print("\n[collect] residuals + logprobs for all cases ...")
    residuals, lp_correct, lp_conflict = collect_all(
        model, all_cases, CANDIDATE_LAYERS, label="all-"
    )
    print(f"[collect] done  {time.time()-t0:.0f}s")

    # ── 2. Directions from train only ──────────────────────────────────────────
    print("\n[vectors] computing from train subset ...")
    h_conf_tr = {l: residuals["conflict"][l][train_ids] for l in CANDIDATE_LAYERS}
    h_cln_tr  = {l: residuals["clean"][l][train_ids]    for l in CANDIDATE_LAYERS}
    h_sup_tr  = {l: residuals["supportive"][l][train_ids] for l in CANDIDATE_LAYERS}

    dir_A = extract_direction_A(h_conf_tr, h_cln_tr,  CANDIDATE_LAYERS)
    dir_B = extract_direction_B(h_conf_tr, h_sup_tr,  CANDIDATE_LAYERS)
    u_PK  = compute_u_PK(h_conf_tr, h_cln_tr,         CANDIDATE_LAYERS)

    for l in CANDIDATE_LAYERS:
        ov = float(torch.dot(dir_A[l], dir_B[l]).abs().item())
        print(f"  L{l}: |cos(A,B)|={ov:.3f}  |u_PK|={u_PK[l].norm():.1f}")

    # Train clean projections (for tau)
    proj_clean = compute_train_proj_clean(h_cln_tr, dir_A, dir_B, CANDIDATE_LAYERS, train_ids)

    # Contrastive direction (from conflict_correct_cases)
    print("\n[vectors] contrastive direction (from conflict_correct_cases) ...")
    cc_path = DATA_DIR / "conflict_correct_cases.json"
    if cc_path.exists():
        with open(cc_path) as f:
            conflict_correct_cases = json.load(f)
        print(f"  loaded {len(conflict_correct_cases)} conflict_correct_cases")

        h_conflict_wrong = {l: residuals["conflict"][l][train_ids] for l in CANDIDATE_LAYERS}
        d_contrast, h_correct = compute_contrastive_direction_from_new_cases(
            model, conflict_correct_cases, h_conflict_wrong, CANDIDATE_LAYERS
        )

        print("\n[contrastive] checking direction separation ...")
        for l in CANDIDATE_LAYERS:
            d = d_contrast[l].float()
            h_correct_l = h_correct[l].float()
            h_wrong_l   = h_conflict_wrong[l].float()
            proj_correct = h_correct_l @ d
            proj_wrong   = h_wrong_l @ d
            separation = float(proj_correct.mean() - proj_wrong.mean())
            print(f"  L{l}: separation={separation:.4f}  "
                  f"(correct_mean={float(proj_correct.mean()):.4f}, "
                  f"wrong_mean={float(proj_wrong.mean()):.4f})")
            if abs(separation) < 0.5:
                print(f"  [WARNING] L{l}: separation too small, "
                      f"contrastive direction may be noisy")
    else:
        print(f"  [warn] {cc_path} not found — skipping contrastive direction")
        d_contrast = {l: torch.zeros(model.cfg.d_model) for l in CANDIDATE_LAYERS}
        h_correct = {l: torch.zeros(0, model.cfg.d_model) for l in CANDIDATE_LAYERS}

    # MLP activation collection
    print("\n[mlp_activations] collecting MLP post-activation ...")
    mlp_activations = collect_mlp_activations(
        model, all_cases, CANDIDATE_LAYERS, label="mlp-"
    )

    # Conflict neurons: weight + activation based
    print("\n[neurons] computing conflict neurons (weight + activation based) ...")
    conflict_neurons = {}
    for l in CANDIDATE_LAYERS:
        for top_k in (10, 20, 50, 100):
            ids_weight = get_conflict_neurons_weight(model, dir_B, l, top_k)
            conflict_neurons[f"L{l}_weight_top{top_k}"] = ids_weight

            ids_act = get_conflict_neurons_activation(
                mlp_activations, l, train_ids, top_k
            )
            conflict_neurons[f"L{l}_activation_top{top_k}"] = ids_act

    # ── 3. 저장 ────────────────────────────────────────────────────────────────
    torch.save(residuals, VEC_DIR / "residuals.pt")
    print(f"[save] residuals.pt")

    torch.save({
        "lp_correct":  lp_correct,
        "lp_conflict": lp_conflict,
        "n_train":     N_train,
        "n_dev":       len(dev_cases),
        "train_ids":   train_ids,
        "dev_ids":     dev_ids,
    }, VEC_DIR / "baseline_logprobs.pt")
    print(f"[save] baseline_logprobs.pt")

    torch.save(dir_A,           VEC_DIR / "direction_A_train.pt")
    torch.save(dir_B,           VEC_DIR / "direction_B_train.pt")
    torch.save(u_PK,            VEC_DIR / "u_PK_train.pt")
    torch.save(proj_clean,      VEC_DIR / "train_proj_clean.pt")
    torch.save(d_contrast,      VEC_DIR / "direction_contrast_train.pt")
    torch.save(mlp_activations, VEC_DIR / "mlp_activations.pt")
    torch.save(conflict_neurons, VEC_DIR / "conflict_neurons.pt")
    print(f"[save] direction_A/B_train.pt, u_PK_train.pt, train_proj_clean.pt")
    print(f"[save] direction_contrast_train.pt, mlp_activations.pt, conflict_neurons.pt")

    # ── 4. 통계 요약 ──────────────────────────────────────────────────────────
    for cond in ("conflict", "clean", "supportive"):
        acc = float((lp_correct[cond] > lp_conflict[cond]).float().mean())
        print(f"  baseline {cond:12s} accuracy: {acc:.3f}")

    print(f"\n[done] total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 4 Exp3 v2: selected best-config quick full-dev check

목적:
  - 오래 걸리는 full grid 대신,
  - 파일럿에서 찾은 best config 2개만
  - dev 전체 197개에서 빠르게 재평가한다.

평가 방법:
  1. B_gated_attn_dynamic_rw
  2. B_gated_mlp_activation

저장:
  phase4/results/exp3_comparison_v2/quick_full_dev_selected_results.json

실행 예시:
  cd /home/jaeik/xaimi
  source /home/jaeik/miniconda3/etc/profile.d/conda.sh
  conda activate apa
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    python -u phase4/scripts/exp3_quick_full_dev_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "phase4/scripts"))

from eval_utils import accuracy, compute_metrics, compute_tau, load_split, load_vectors
from exp_utils import (
    build_clean_prompt,
    build_conflict_prompt,
    build_supportive_prompt,
    load_model,
)
from intervention_methods import (
    get_question_positions,
    make_dynamic_attn_suppress_hook,
    make_gate_reader_resid_pre,
    make_mlp_suppression_hook,
)

OUT_PATH = ROOT / "phase4/results/exp3_comparison_v2/quick_full_dev_selected_results.json"
BASE_BEST_PATH = ROOT / "phase4/results/exp3_comparison_v2/best_configs.json"
MERGED_BEST_PATH = ROOT / "phase4/results/exp3_comparison_v2/best_configs_7plus2.json"
MERGED_TABLE_PATH = ROOT / "phase4/results/exp3_comparison_v2/ACCURACY_COMPARISON_TABLE_7PLUS2.md"
VEC_DIR = ROOT / "phase4/vectors"

# 파일럿/이전 실험에서 유망했던 config
SELECTED_CONFIGS = {
    "B_gated_attn_dynamic_rw": {
        "layer": 17,
        "gamma": 10.0,
        "k": 1.0,
        "top_k_positions": 10,
        "suppress_value": 5.0,
    },
    "B_gated_mlp_activation": {
        "layer": 17,
        "gamma": 5.0,
        "k": 0.5,
        "top_k_neurons": 100,
        "beta": 8.0,
    },
}


def evaluate_method(model, cases, hooks_fn,
                    conditions=("conflict", "clean", "supportive")) -> dict:
    N = len(cases)
    result = {
        c: {"lp_correct": torch.zeros(N), "lp_conflict": torch.zeros(N)}
        for c in conditions
    }

    for j, case in enumerate(cases):
        q = case["question"]
        opt = case["options"]
        ctx = case["conflict_context"]
        cor = case["correct_option"]
        rep = case["replace_option"]
        ca = case["conflict_answer"]
        coa = case["correct_answer"]
        cor_id = model.tokenizer.encode(f" {cor}", add_special_tokens=False)[-1]
        rep_id = model.tokenizer.encode(f" {rep}", add_special_tokens=False)[-1]
        hooks = hooks_fn(case)

        for cond, prompt in [
            ("conflict", build_conflict_prompt(ctx, q, opt)),
            ("clean", build_clean_prompt(q, opt)),
            ("supportive", build_supportive_prompt(q, opt, coa, ctx, ca)),
        ]:
            with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                ids = model.to_tokens(prompt, prepend_bos=True)
                logits = model(ids)[0, -1, :].float().cpu()
            result[cond]["lp_correct"][j] = logits[cor_id].item()
            result[cond]["lp_conflict"][j] = logits[rep_id].item()

    return result


def main():
    t0 = time.time()
    _, dir_B, _, proj_clean, bl, _ = load_vectors()
    conflict_neurons = torch.load(VEC_DIR / "conflict_neurons.pt", map_location="cpu", weights_only=False)

    dev_cases = load_split("dev_200.json")
    dev_ids = bl["dev_ids"]
    print(f"Using full dev: {len(dev_cases)} cases")

    baseline = {
        c: {
            "lp_correct": bl["lp_correct"][c][dev_ids],
            "lp_conflict": bl["lp_conflict"][c][dev_ids],
        }
        for c in ("conflict", "clean", "supportive")
    }
    print(f"Baseline conflict acc: {accuracy(baseline['conflict']['lp_correct'], baseline['conflict']['lp_conflict']):.3f}")

    model = load_model(device="cuda")
    results = {}

    # 1) Dynamic attention RW
    cfg = SELECTED_CONFIGS["B_gated_attn_dynamic_rw"]
    layer = cfg["layer"]
    gamma = cfg["gamma"]
    k = cfg["k"]
    top_k = cfg["top_k_positions"]
    suppress_value = cfg["suppress_value"]
    tau = compute_tau(proj_clean["B"][layer], k)
    d_B_l = dir_B[layer].to("cuda")
    print(f"\n[eval] B_gated_attn_dynamic_rw L{layer} gamma={gamma} k={k} top_k={top_k} suppress={suppress_value}")

    def mk_dynamic_rw(l=layer, g=gamma, t=tau, top_k_=top_k, sv_=suppress_value, db=d_B_l):
        def fn(case):
            gs = [None]
            return [
                (f"blocks.{l}.hook_resid_pre", make_gate_reader_resid_pre(db, t, g, gs)),
                (f"blocks.{l}.attn.hook_attn_scores", make_dynamic_attn_suppress_hook(gs, top_k_, sv_)),
            ]
        return fn

    after = evaluate_method(model, dev_cases, mk_dynamic_rw())
    results["B_gated_attn_dynamic_rw"] = {
        "method": "B_gated_attn_dynamic_rw",
        **cfg,
        "tau": tau,
        **compute_metrics(baseline, after),
    }

    # 2) MLP activation
    cfg = SELECTED_CONFIGS["B_gated_mlp_activation"]
    layer = cfg["layer"]
    gamma = cfg["gamma"]
    k = cfg["k"]
    top_k_neurons = cfg["top_k_neurons"]
    beta = cfg["beta"]
    tau = compute_tau(proj_clean["B"][layer], k)
    d_B_l = dir_B[layer].to("cuda")
    neuron_ids = conflict_neurons[f"L{layer}_activation_top{top_k_neurons}"]
    print(f"\n[eval] B_gated_mlp_activation L{layer} gamma={gamma} k={k} top_k={top_k_neurons} beta={beta}")

    def mk_mlp_activation(l=layer, g=gamma, t=tau, db=d_B_l, ni=neuron_ids, b_=beta):
        def fn(case):
            gs = [None]
            return [
                (f"blocks.{l}.hook_resid_pre", make_gate_reader_resid_pre(db, t, g, gs)),
                (f"blocks.{l}.mlp.hook_post", make_mlp_suppression_hook(gs, ni, b_)),
            ]
        return fn

    after = evaluate_method(model, dev_cases, mk_mlp_activation())
    results["B_gated_mlp_activation"] = {
        "method": "B_gated_mlp_activation",
        **cfg,
        "tau": tau,
        **compute_metrics(baseline, after),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    merged = {}
    if BASE_BEST_PATH.exists():
        with open(BASE_BEST_PATH) as f:
            merged = json.load(f)
    merged.update(results)
    with open(MERGED_BEST_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    order = [
        ("B_gated_upk", "B-gated Residual"),
        ("B_gated_contrast", "B-gated Contrast"),
        ("B_gated_attn_rw_sentence", "Attention Reweight"),
        ("B_gated_attn_rd_sentence", "Attention Redirect"),
        ("B_gated_attn_dynamic_rw", "Attention Dynamic Reweight"),
        ("B_gated_mlp_activation", "MLP Activation"),
    ]
    lines = [
        "# Accuracy Comparison Table (7 full-dev + 2 quick full-dev)",
        "",
        "아래 표는 기존 full-dev 7개 결과에, selected best config로 빠르게 재평가한 2개 방법을 붙인 표이다.",
        "",
        "| 표시 이름 | 내부 method | Conflict Accuracy | Clean Accuracy | Supportive Accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for key, label in order:
        if key not in merged:
            continue
        r = merged[key]
        lines.append(
            f"| {label} | `{key}` | "
            f"{r.get('conflict_acc_after', r.get('conflict_acc_before', 0.0)):.3f} | "
            f"{r.get('clean_acc_after', r.get('clean_acc_before', 0.0)):.3f} | "
            f"{r.get('supportive_acc_after', r.get('supportive_acc_before', 0.0)):.3f} |"
        )
    with open(MERGED_TABLE_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[save] {OUT_PATH}")
    print(f"[save] {MERGED_BEST_PATH}")
    print(f"[save] {MERGED_TABLE_PATH}")
    for name, r in results.items():
        print(
            f"{name:28s} "
            f"conflict={r['conflict_acc_after']:.3f} "
            f"clean={r['clean_acc_after']:.3f} "
            f"supportive={r['supportive_acc_after']:.3f} "
            f"selective={r['selective_gain']:.3f} "
            f"margin={r['conflict_margin_delta']:.3f}"
        )
    print(f"[done] total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 4 Exp3: Localized Intervention 개선 비교

저장:
  phase4/results/exp3_comparison_v2/

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 python -u phase4/scripts/exp3_intervention_comparison.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "phase4/scripts"))

from eval_utils import (
    accuracy,
    compute_metrics,
    compute_tau,
    load_split,
    load_vectors,
    print_metrics,
)
from exp_utils import (
    build_clean_prompt,
    build_conflict_prompt,
    build_supportive_prompt,
    load_model,
)
from intervention_methods import (
    find_false_sentence,
    get_false_entity_positions,
    get_false_sentence_positions,
    get_full_context_positions,
    get_question_positions,
    make_attn_redirect_hook,
    make_attn_reweight_hook,
    make_contrast_hook,
    make_dynamic_attn_redirect_hook,
    make_dynamic_attn_suppress_hook,
    make_gate_reader_resid_post,
    make_gate_reader_resid_pre,
    make_mlp_suppression_hook,
    make_upk_hook,
)

VEC_DIR = ROOT / "phase4/vectors"
OUT_DIR = ROOT / "phase4/results/exp3_comparison_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42

# Residual / MLP methods
CANDIDATE_LAYERS = list(range(14, 23))  # L14 ~ L22
# Attention methods
ATTN_CANDIDATE_LAYERS = list(range(10, 23))  # L10 ~ L22

ALPHAS = [1.0, 2.0, 4.0, 6.0]
GAMMAS = [2.0, 5.0, 10.0]
K_VALS = [0.5, 1.0, 1.5]

TOP_K_POSITIONS = [3, 5, 10, 20]
SUPPRESS_VALUES = [1.0, 2.0, 5.0, 10.0]
BOOST_VALUES = [0.5, 1.0, 2.0]
MULTI_LAYER_COMBOS = [
    [10, 11, 12],
    [11, 12, 13, 14],
    [10, 11, 12, 13, 14],
    [12, 13, 14],
]

TOP_K_NEURONS = [10, 20, 50, 100]
BETA_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]

DEFAULT_METHODS = [
    "no_intervention",
    "ungated_upk",
    "shuffled_B_gated_upk",
    "B_gated_upk",
    "B_gated_contrast",
    "B_gated_attn_rw_sentence",
    "B_gated_attn_rd_sentence",
    "B_gated_attn_dynamic_rw",
    "B_gated_attn_dynamic_rd",
    "B_gated_attn_multilayer",
    "B_gated_mlp_weight",
    "B_gated_mlp_activation",
]

DEV_CASE_LIMIT = int(os.environ.get("PHASE4_DEV_LIMIT", "40"))
ENABLED_METHODS = set(
    m.strip() for m in os.environ.get("PHASE4_METHODS", ",".join(DEFAULT_METHODS)).split(",")
    if m.strip()
)


def save_results(all_results: list, best_configs: dict) -> None:
    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(OUT_DIR / "best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)


def pick_dev_subset(dev_cases: list, dev_ids: list[int]) -> tuple[list[dict], list[int]]:
    if DEV_CASE_LIMIT <= 0 or DEV_CASE_LIMIT >= len(dev_cases):
        return dev_cases, dev_ids
    rng = random.Random(SEED)
    keep = sorted(rng.sample(range(len(dev_cases)), DEV_CASE_LIMIT))
    return [dev_cases[i] for i in keep], [dev_ids[i] for i in keep]


def method_enabled(name: str) -> bool:
    return name in ENABLED_METHODS


def update_best(current_best, candidate):
    if current_best is None:
        return candidate
    current_key = (
        current_best.get("selective_gain", -99.0),
        current_best.get("conflict_margin_delta", -99.0),
    )
    candidate_key = (
        candidate.get("selective_gain", -99.0),
        candidate.get("conflict_margin_delta", -99.0),
    )
    return candidate if candidate_key > current_key else current_best


def get_positions_for_case(model, case: dict, version: str, prompt: str):
    ctx = case["conflict_context"]
    ca = case["conflict_answer"]

    if version == "entity":
        return get_false_entity_positions(model, prompt, ca)
    if version == "sentence":
        false_sent = find_false_sentence(ctx, ca)
        pos = get_false_sentence_positions(model, prompt, false_sent) if false_sent else []
        return pos or get_full_context_positions(model, prompt, ctx)
    return get_full_context_positions(model, prompt, ctx)


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


def evaluate_multi_layer_attn(model, cases, layer_list, dir_B, proj_clean,
                              gamma, k, top_k, suppress_value,
                              conditions=("conflict", "clean", "supportive")):
    N = len(cases)
    result = {
        c: {"lp_correct": torch.zeros(N), "lp_conflict": torch.zeros(N)}
        for c in conditions
    }
    tau_dict = {layer: compute_tau(proj_clean["B"][layer], k) for layer in layer_list}

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

        hooks = []
        for layer in layer_list:
            gate_store = [None]
            hooks.append((
                f"blocks.{layer}.hook_resid_pre",
                make_gate_reader_resid_pre(dir_B[layer].to("cuda"), tau_dict[layer], gamma, gate_store),
            ))
            hooks.append((
                f"blocks.{layer}.attn.hook_attn_scores",
                make_dynamic_attn_suppress_hook(gate_store, top_k, suppress_value),
            ))

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


def compute_shuffled_gate_result(model, cases, layer, d_B, tau, gamma, u_PK, alpha):
    torch.manual_seed(SEED)
    nd = F.normalize(d_B, dim=0).to("cuda")
    gate_vals = torch.zeros(len(cases))
    for j, case in enumerate(cases):
        prompt = build_conflict_prompt(case["conflict_context"], case["question"], case["options"])
        ids = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                ids,
                names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post",
            )
        h = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].float()
        proj = float(h @ nd)
        gate_vals[j] = float(torch.sigmoid(torch.tensor(gamma * (proj - tau))))

    shuffled = gate_vals[torch.randperm(len(gate_vals))]
    up = u_PK.to("cuda")

    def hooks_fn(case):
        j = cases.index(case)
        g = float(shuffled[j])

        def hook_fn(value, hook):
            v = value.float()
            v[:, -1, :] = v[:, -1, :] + g * alpha * up.to(v.device)
            return v.to(value.dtype)

        return [(f"blocks.{layer}.hook_resid_post", hook_fn)]

    return evaluate_method(model, cases, hooks_fn)


def main():
    t0 = time.time()

    dir_A, dir_B, u_PK, proj_clean, bl, residuals = load_vectors()
    del dir_A, residuals
    d_contrast = torch.load(VEC_DIR / "direction_contrast_train.pt", map_location="cpu", weights_only=False)
    conflict_neurons = torch.load(VEC_DIR / "conflict_neurons.pt", map_location="cpu", weights_only=False)

    dev_cases = load_split("dev_200.json")
    dev_ids = bl["dev_ids"]
    dev_cases, dev_ids = pick_dev_subset(dev_cases, dev_ids)
    print(f"Using dev subset: {len(dev_cases)} cases")
    print(f"Enabled methods: {sorted(ENABLED_METHODS)}")

    baseline = {
        c: {
            "lp_correct": bl["lp_correct"][c][dev_ids],
            "lp_conflict": bl["lp_conflict"][c][dev_ids],
        }
        for c in ("conflict", "clean", "supportive")
    }
    baseline_acc = accuracy(baseline["conflict"]["lp_correct"], baseline["conflict"]["lp_conflict"])
    print(f"Baseline conflict acc (dev): {baseline_acc:.3f}")

    model = load_model(device="cuda")
    all_results = []
    best_configs = {}

    if method_enabled("no_intervention"):
        no_int = {
            "method": "no_intervention",
            "conflict_acc_before": baseline_acc,
            "conflict_acc_after": baseline_acc,
            "conflict_acc_gain": 0.0,
            "clean_acc_before": accuracy(baseline["clean"]["lp_correct"], baseline["clean"]["lp_conflict"]),
            "clean_acc_after": accuracy(baseline["clean"]["lp_correct"], baseline["clean"]["lp_conflict"]),
            "clean_acc_gain": 0.0,
            "supportive_acc_before": accuracy(baseline["supportive"]["lp_correct"], baseline["supportive"]["lp_conflict"]),
            "supportive_acc_after": accuracy(baseline["supportive"]["lp_correct"], baseline["supportive"]["lp_conflict"]),
            "supportive_acc_gain": 0.0,
            "selective_gain": 0.0,
            "conflict_margin_delta": 0.0,
        }
        all_results.append(no_int)
        best_configs["no_intervention"] = no_int
        print_metrics("no_intervention", no_int)
        save_results(all_results, best_configs)

    if method_enabled("ungated_upk"):
        print("\n[ungated_upk]")
        best = None
        total = len(CANDIDATE_LAYERS) * len(ALPHAS)
        done = 0
        for layer, alpha in product(CANDIDATE_LAYERS, ALPHAS):
            done += 1
            print(f"  config {done}/{total}: L{layer} alpha={alpha}", flush=True)
            up = u_PK[layer].to("cuda")

            def mk(l=layer, a=alpha, up_=up):
                def fn(case):
                    def h(value, hook):
                        v = value.float()
                        v[:, -1, :] = v[:, -1, :] + a * up_.to(v.device)
                        return v.to(value.dtype)
                    return [(f"blocks.{l}.hook_resid_post", h)]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {"method": "ungated_upk", "layer": layer, "alpha": alpha, **compute_metrics(baseline, after)}
            all_results.append(r)
            best = update_best(best, r)
        best_configs["ungated_upk"] = best
        print_metrics(f"ungated_upk best (L{best['layer']})", best)
        save_results(all_results, best_configs)

    if method_enabled("B_gated_upk"):
        print("\n[B_gated_upk]")
        best = None
        total = len(CANDIDATE_LAYERS) * len(ALPHAS) * len(GAMMAS) * len(K_VALS)
        done = 0
        for layer, alpha, gamma, k in product(CANDIDATE_LAYERS, ALPHAS, GAMMAS, K_VALS):
            done += 1
            print(f"  config {done}/{total}: L{layer} alpha={alpha} gamma={gamma} k={k}", flush=True)
            d_B_l = dir_B[layer].to("cuda")
            up = u_PK[layer].to("cuda")
            tau = compute_tau(proj_clean["B"][layer], k)

            def mk(l=layer, a=alpha, g=gamma, t=tau, db=d_B_l, up_=up):
                def fn(case):
                    gs = [None]
                    return [
                        (f"blocks.{l}.hook_resid_post", make_gate_reader_resid_post(db, t, g, gs)),
                        (f"blocks.{l}.hook_resid_post", make_upk_hook(gs, up_, a)),
                    ]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {"method": "B_gated_upk", "layer": layer, "alpha": alpha, "gamma": gamma, "k": k, "tau": tau, **compute_metrics(baseline, after)}
            all_results.append(r)
            best = update_best(best, r)
        best_configs["B_gated_upk"] = best
        print_metrics(f"B_gated_upk best (L{best['layer']})", best)
        save_results(all_results, best_configs)

        if method_enabled("shuffled_B_gated_upk"):
            print("\n[shuffled_B_gated_upk]")
            after = compute_shuffled_gate_result(
                model, dev_cases, best["layer"],
                dir_B[best["layer"]].to("cuda"), best["tau"], best["gamma"],
                u_PK[best["layer"]].to("cuda"), best["alpha"],
            )
            r = {
                "method": "shuffled_B_gated_upk",
                "layer": best["layer"],
                "alpha": best["alpha"],
                "gamma": best["gamma"],
                "tau": best["tau"],
                **compute_metrics(baseline, after),
            }
            all_results.append(r)
            best_configs["shuffled_B_gated_upk"] = r
            print_metrics("shuffled_B_gated_upk", r)
            save_results(all_results, best_configs)

    if method_enabled("B_gated_contrast"):
        print("\n[B_gated_contrast]")
        best = None
        total = len(CANDIDATE_LAYERS) * len(ALPHAS) * len(GAMMAS) * len(K_VALS)
        done = 0
        for layer, alpha, gamma, k in product(CANDIDATE_LAYERS, ALPHAS, GAMMAS, K_VALS):
            done += 1
            print(f"  config {done}/{total}: L{layer} alpha={alpha} gamma={gamma} k={k}", flush=True)
            d_B_l = dir_B[layer].to("cuda")
            dc = d_contrast[layer].to("cuda")
            tau = compute_tau(proj_clean["B"][layer], k)

            def mk(l=layer, a=alpha, g=gamma, t=tau, db=d_B_l, dc_=dc):
                def fn(case):
                    gs = [None]
                    return [
                        (f"blocks.{l}.hook_resid_post", make_gate_reader_resid_post(db, t, g, gs)),
                        (f"blocks.{l}.hook_resid_post", make_contrast_hook(gs, dc_, a)),
                    ]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {"method": "B_gated_contrast", "layer": layer, "alpha": alpha, "gamma": gamma, "k": k, "tau": tau, **compute_metrics(baseline, after)}
            all_results.append(r)
            best = update_best(best, r)
        best_configs["B_gated_contrast"] = best
        print_metrics(f"B_gated_contrast best (L{best['layer']})", best)
        save_results(all_results, best_configs)

    if method_enabled("B_gated_attn_rw_sentence"):
        print("\n[B_gated_attn_rw_sentence]")
        best = None
        total = len(ATTN_CANDIDATE_LAYERS) * len(GAMMAS) * len(K_VALS) * len(SUPPRESS_VALUES)
        done = 0
        for layer, gamma, k, sv in product(ATTN_CANDIDATE_LAYERS, GAMMAS, K_VALS, SUPPRESS_VALUES):
            done += 1
            print(f"  config {done}/{total}: L{layer} gamma={gamma} k={k} suppress={sv}", flush=True)
            d_B_l = dir_B[layer].to("cuda")
            tau = compute_tau(proj_clean["B"][layer], k)

            def mk(l=layer, g=gamma, t=tau, sv_=sv, db=d_B_l):
                def fn(case):
                    gs = [None]
                    conf_prompt = build_conflict_prompt(case["conflict_context"], case["question"], case["options"])
                    conf_pos = get_positions_for_case(model, case, "sentence", conf_prompt)
                    return [
                        (f"blocks.{l}.hook_resid_pre", make_gate_reader_resid_pre(db, t, g, gs)),
                        (f"blocks.{l}.attn.hook_attn_scores", make_attn_reweight_hook(gs, conf_pos, sv_)),
                    ]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {"method": "B_gated_attn_rw_sentence", "layer": layer, "gamma": gamma, "k": k, "suppress_value": sv, "tau": tau, "position_version": "sentence", **compute_metrics(baseline, after)}
            all_results.append(r)
            best = update_best(best, r)
        best_configs["B_gated_attn_rw_sentence"] = best
        print_metrics(f"B_gated_attn_rw_sentence best (L{best['layer']})", best)
        save_results(all_results, best_configs)

    if method_enabled("B_gated_attn_rd_sentence"):
        print("\n[B_gated_attn_rd_sentence]")
        best = None
        total = len(ATTN_CANDIDATE_LAYERS) * len(GAMMAS) * len(K_VALS) * len(SUPPRESS_VALUES) * len(BOOST_VALUES)
        done = 0
        for layer, gamma, k, sv, bv in product(ATTN_CANDIDATE_LAYERS, GAMMAS, K_VALS, SUPPRESS_VALUES, BOOST_VALUES):
            done += 1
            print(f"  config {done}/{total}: L{layer} gamma={gamma} k={k} suppress={sv} boost={bv}", flush=True)
            d_B_l = dir_B[layer].to("cuda")
            tau = compute_tau(proj_clean["B"][layer], k)

            def mk(l=layer, g=gamma, t=tau, sv_=sv, bv_=bv, db=d_B_l):
                def fn(case):
                    gs = [None]
                    conf_prompt = build_conflict_prompt(case["conflict_context"], case["question"], case["options"])
                    conf_pos = get_positions_for_case(model, case, "sentence", conf_prompt)
                    q_pos = get_question_positions(model, conf_prompt, case["question"])
                    return [
                        (f"blocks.{l}.hook_resid_pre", make_gate_reader_resid_pre(db, t, g, gs)),
                        (f"blocks.{l}.attn.hook_attn_scores", make_attn_redirect_hook(gs, conf_pos, q_pos, sv_, bv_)),
                    ]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {"method": "B_gated_attn_rd_sentence", "layer": layer, "gamma": gamma, "k": k, "suppress_value": sv, "boost_value": bv, "tau": tau, "position_version": "sentence", **compute_metrics(baseline, after)}
            all_results.append(r)
            best = update_best(best, r)
        best_configs["B_gated_attn_rd_sentence"] = best
        print_metrics(f"B_gated_attn_rd_sentence best (L{best['layer']})", best)
        save_results(all_results, best_configs)

    if method_enabled("B_gated_attn_dynamic_rw"):
        print("\n[B_gated_attn_dynamic_rw]")
        best = None
        total = len(ATTN_CANDIDATE_LAYERS) * len(GAMMAS) * len(K_VALS) * len(TOP_K_POSITIONS) * len(SUPPRESS_VALUES)
        done = 0
        for layer, gamma, k, top_k, sv in product(ATTN_CANDIDATE_LAYERS, GAMMAS, K_VALS, TOP_K_POSITIONS, SUPPRESS_VALUES):
            done += 1
            print(f"  config {done}/{total}: L{layer} gamma={gamma} k={k} top_k={top_k} suppress={sv}", flush=True)
            d_B_l = dir_B[layer].to("cuda")
            tau = compute_tau(proj_clean["B"][layer], k)

            def mk(l=layer, g=gamma, t=tau, top_k_=top_k, sv_=sv, db=d_B_l):
                def fn(case):
                    gs = [None]
                    return [
                        (f"blocks.{l}.hook_resid_pre", make_gate_reader_resid_pre(db, t, g, gs)),
                        (f"blocks.{l}.attn.hook_attn_scores", make_dynamic_attn_suppress_hook(gs, top_k_, sv_)),
                    ]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {"method": "B_gated_attn_dynamic_rw", "layer": layer, "gamma": gamma, "k": k, "top_k_positions": top_k, "suppress_value": sv, "tau": tau, **compute_metrics(baseline, after)}
            all_results.append(r)
            best = update_best(best, r)
        best_configs["B_gated_attn_dynamic_rw"] = best
        print_metrics(f"B_gated_attn_dynamic_rw best (L{best['layer']})", best)
        save_results(all_results, best_configs)

    if method_enabled("B_gated_attn_dynamic_rd"):
        print("\n[B_gated_attn_dynamic_rd]")
        best = None
        total = len(ATTN_CANDIDATE_LAYERS) * len(GAMMAS) * len(K_VALS) * len(TOP_K_POSITIONS) * len(SUPPRESS_VALUES) * len(BOOST_VALUES)
        done = 0
        for layer, gamma, k, top_k, sv, bv in product(ATTN_CANDIDATE_LAYERS, GAMMAS, K_VALS, TOP_K_POSITIONS, SUPPRESS_VALUES, BOOST_VALUES):
            done += 1
            print(f"  config {done}/{total}: L{layer} gamma={gamma} k={k} top_k={top_k} suppress={sv} boost={bv}", flush=True)
            d_B_l = dir_B[layer].to("cuda")
            tau = compute_tau(proj_clean["B"][layer], k)

            def mk(l=layer, g=gamma, t=tau, top_k_=top_k, sv_=sv, bv_=bv, db=d_B_l):
                def fn(case):
                    gs = [None]
                    conf_prompt = build_conflict_prompt(case["conflict_context"], case["question"], case["options"])
                    q_pos = get_question_positions(model, conf_prompt, case["question"])
                    return [
                        (f"blocks.{l}.hook_resid_pre", make_gate_reader_resid_pre(db, t, g, gs)),
                        (f"blocks.{l}.attn.hook_attn_scores", make_dynamic_attn_redirect_hook(gs, top_k_, top_k_, sv_, bv_, q_pos)),
                    ]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {"method": "B_gated_attn_dynamic_rd", "layer": layer, "gamma": gamma, "k": k, "top_k_positions": top_k, "suppress_value": sv, "boost_value": bv, "tau": tau, **compute_metrics(baseline, after)}
            all_results.append(r)
            best = update_best(best, r)
        best_configs["B_gated_attn_dynamic_rd"] = best
        print_metrics(f"B_gated_attn_dynamic_rd best (L{best['layer']})", best)
        save_results(all_results, best_configs)

    if method_enabled("B_gated_attn_multilayer"):
        print("\n[B_gated_attn_multilayer]")
        best = None
        total = len(MULTI_LAYER_COMBOS) * len(GAMMAS) * len(K_VALS) * len(TOP_K_POSITIONS) * len(SUPPRESS_VALUES)
        done = 0
        for layer_list, gamma, k, top_k, sv in product(MULTI_LAYER_COMBOS, GAMMAS, K_VALS, TOP_K_POSITIONS, SUPPRESS_VALUES):
            done += 1
            print(f"  config {done}/{total}: layers={layer_list} gamma={gamma} k={k} top_k={top_k} suppress={sv}", flush=True)
            after = evaluate_multi_layer_attn(
                model, dev_cases, layer_list, dir_B, proj_clean, gamma, k, top_k, sv
            )
            r = {
                "method": "B_gated_attn_multilayer",
                "layer_list": layer_list,
                "gamma": gamma,
                "k": k,
                "top_k_positions": top_k,
                "suppress_value": sv,
                **compute_metrics(baseline, after),
            }
            all_results.append(r)
            best = update_best(best, r)
        best_configs["B_gated_attn_multilayer"] = best
        print_metrics("B_gated_attn_multilayer best", best)
        save_results(all_results, best_configs)

    for neuron_select in ("weight", "activation"):
        method_name = f"B_gated_mlp_{neuron_select}"
        if not method_enabled(method_name):
            continue
        print(f"\n[{method_name}]")
        best = None
        total = len(CANDIDATE_LAYERS) * len(GAMMAS) * len(K_VALS) * len(TOP_K_NEURONS) * len(BETA_VALUES)
        done = 0
        for layer, gamma, k, top_k, beta in product(CANDIDATE_LAYERS, GAMMAS, K_VALS, TOP_K_NEURONS, BETA_VALUES):
            done += 1
            print(f"  config {done}/{total}: L{layer} gamma={gamma} k={k} top_k={top_k} beta={beta}", flush=True)
            d_B_l = dir_B[layer].to("cuda")
            tau = compute_tau(proj_clean["B"][layer], k)
            nkey = f"L{layer}_{neuron_select}_top{top_k}"
            n_ids = conflict_neurons[nkey]

            def mk(l=layer, g=gamma, t=tau, db=d_B_l, ni=n_ids, b_=beta):
                def fn(case):
                    gs = [None]
                    return [
                        (f"blocks.{l}.hook_resid_pre", make_gate_reader_resid_pre(db, t, g, gs)),
                        (f"blocks.{l}.mlp.hook_post", make_mlp_suppression_hook(gs, ni, b_)),
                    ]
                return fn

            after = evaluate_method(model, dev_cases, mk())
            r = {
                "method": method_name,
                "neuron_select": neuron_select,
                "layer": layer,
                "gamma": gamma,
                "k": k,
                "top_k_neurons": top_k,
                "beta": beta,
                "tau": tau,
                **compute_metrics(baseline, after),
            }
            all_results.append(r)
            best = update_best(best, r)
        best_configs[method_name] = best
        print_metrics(f"{method_name} best (L{best['layer']})", best)
        save_results(all_results, best_configs)

    save_results(all_results, best_configs)
    print("[save] all_results.json, best_configs.json")

    method_print_order = [
        "no_intervention",
        "ungated_upk",
        "shuffled_B_gated_upk",
        "B_gated_upk",
        "B_gated_contrast",
        "B_gated_attn_rw_sentence",
        "B_gated_attn_rd_sentence",
        "B_gated_attn_dynamic_rw",
        "B_gated_attn_dynamic_rd",
        "B_gated_attn_multilayer",
        "B_gated_mlp_weight",
        "B_gated_mlp_activation",
    ]
    print("\n" + "=" * 100)
    print(f"{'Method':<32} {'Conflict↑':>10} {'Clean':>8} {'Sup':>8} {'Selective↑':>12} {'Margin↑':>9}")
    print("=" * 100)
    for name in method_print_order:
        bc = best_configs.get(name, {})
        print(f"  {name:<30} {bc.get('conflict_acc_after', bc.get('conflict_acc_before', 0)):>10.3f} "
              f"{bc.get('clean_acc_after', bc.get('clean_acc_before', 0)):>8.3f} "
              f"{bc.get('supportive_acc_after', bc.get('supportive_acc_before', 0)):>8.3f} "
              f"{bc.get('selective_gain', 0.0):>12.4f} "
              f"{bc.get('conflict_margin_delta', 0.0):>9.4f}")
    print("=" * 100)
    print(f"\n[done] total={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

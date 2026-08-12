#!/usr/bin/env python3
"""
Step3: Flipped vs Non-flipped Case 비교

Phase2 grid search 최우선 config로 개입 후 rank flip이 일어난 케이스(성공)와
안 일어난 케이스(실패)를 분리해, conflict projection이 강한 케이스에서
실제로 rank flip도 더 잘 일어나는지 확인한다.

grid_results에 케이스별 flip 정보가 없으므로,
exp_grid 모듈을 임포트해 모델 재실행으로 flip 여부를 직접 판정한다.

저장: phase3/results/step3/flipped_ids.json
      phase3/results/step3/group_stats.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

TARGET_LAYERS = list(range(10, 23))
DIRECTIONS    = ["A", "B", "C"]
RESULTS       = ROOT / "phase3/results"
PROJ_PATH     = RESULTS / "projections_raw.pt"
GRID_JSON     = ROOT / "phase2/results/exp_grid/all_results_ranked.json"
EXP_GRID_PY   = ROOT / "phase2/scripts/exp_grid.py"

PREFERRED_CONFIGS = [
    "L16:A+L17:C+L19:B",
    "L16:A+L17:C+L18:B+L19:B",
    "L17:C+L18:B+L19:B",
]
MIN_FLIPPED = 50


def import_exp_grid():
    spec   = importlib.util.spec_from_file_location("phase2_exp_grid", EXP_GRID_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_config(ranked: list, preferred: list) -> dict:
    by_label = {r["config"]: r for r in ranked}
    for label in preferred:
        if label in by_label:
            return by_label[label]
    return ranked[0]


def get_flipped_indices(grid_mod, model, cases: list, layer_dir_pairs: list) -> list:
    """각 케이스에 대해 개입 후 rank flip 여부를 직접 판정."""
    cands = torch.load(ROOT / "phase2/results/exp1b_candidates/candidates_for_exp2.pt",
                       map_location="cpu")
    hooks = grid_mod.make_fwd_hooks(
        [(int(l), str(d)) for l, d in layer_dir_pairs],
        cands,
    )
    flipped = []
    for idx, case in enumerate(cases):
        if not case.get("passes_kc_filter", True):
            continue
        cor = case["correct_option"]
        rep = case["replace_option"]
        tok = model.to_tokens(
            grid_mod.build_conflict_prompt(
                case["conflict_context"], case["question"], case["options"]
            ),
            prepend_bos=True,
        )
        lp_cor, lp_rep = grid_mod.run_logprobs_hooked(model, tok, cor, rep, hooks)
        if lp_cor > lp_rep:
            flipped.append(idx)
    return flipped


def main() -> None:
    t0 = time.time()

    with open(GRID_JSON) as f:
        ranked = json.load(f)

    chosen = choose_config(ranked, PREFERRED_CONFIGS)
    print(f"[config] {chosen['config']}")

    grid_mod = import_exp_grid()
    model    = grid_mod.load_model()
    grid_mod.setup_letter_variants(model.tokenizer)

    sys.path.insert(0, str(ROOT / "scripts"))
    from exp_utils import load_cases
    cases = load_cases(n=1000)

    flipped_ids = get_flipped_indices(grid_mod, model, cases, chosen["layer_dir_pairs"])
    # fallback
    if len(flipped_ids) < MIN_FLIPPED:
        fallback = choose_config(ranked, ["L17:C+L18:B+L19:B"])
        if fallback["config"] != chosen["config"]:
            chosen = fallback
            print(f"[fallback] {chosen['config']}")
            flipped_ids = get_flipped_indices(grid_mod, model, cases, chosen["layer_dir_pairs"])

    N = len(cases)
    all_ids       = set(range(N))
    flipped_set   = set(flipped_ids)
    non_flipped_ids = sorted(all_ids - flipped_set)
    flipped_ids     = sorted(flipped_ids)

    print(f"[flip] flipped={len(flipped_ids)}  non_flipped={len(non_flipped_ids)}")

    projections = torch.load(PROJ_PATH, map_location="cpu")

    group_stats: dict = {}
    for d in DIRECTIONS:
        group_stats[d] = {}
        for layer in TARGET_LAYERS:
            proj_conflict = projections[d]["conflict"][layer].float()

            fp  = proj_conflict[flipped_ids].numpy()
            nfp = proj_conflict[non_flipped_ids].numpy()

            t_stat, p_val = scipy_stats.ttest_ind(fp, nfp)
            pooled_std    = float(proj_conflict.std().item())
            effect_size   = float((fp.mean() - nfp.mean()) / (pooled_std + 1e-8))

            group_stats[d][layer] = {
                "flipped_mean":     float(fp.mean())  if len(fp)  else None,
                "non_flipped_mean": float(nfp.mean()) if len(nfp) else None,
                "effect_size":      effect_size,
                "t_stat":           float(t_stat),
                "p_val":            float(p_val),
                "n_flipped":        len(fp),
                "n_non_flipped":    len(nfp),
            }

    with open(RESULTS / "step3/flipped_ids.json", "w") as f:
        json.dump({
            "config":        chosen["config"],
            "n_flipped":     len(flipped_ids),
            "n_non_flipped": len(non_flipped_ids),
            "flipped":       flipped_ids,
            "non_flipped":   non_flipped_ids,
        }, f, indent=2)

    with open(RESULTS / "step3/group_stats.json", "w") as f:
        json.dump(group_stats, f, indent=2)

    print(f"[save] step3/flipped_ids.json  step3/group_stats.json")
    print(f"[done] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

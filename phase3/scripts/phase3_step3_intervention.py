#!/usr/bin/env python3
"""
Step3 revised — Direction별 단독 개입으로 Flipped/Non-flipped case 생성

기존 Step3의 한계(mixed config로 A/B/C 개입을 함께 써서 어느 direction 덕분인지
모름)를 해결하기 위해, 각 direction의 단독 최선 config를 사용해 flip 여부를 판정한다.

선택된 config:
  A_solo  : L17:A+L19:A         (kc_flipped=31)
  B_solo  : L18:B               (kc_flipped=173)
  B_multi : L16:B+L17:B+L18:B   (kc_flipped=205)

출력: phase3/results/step3_revised/intervention_results.json
  {
    "<config_name>": {
      "config_label": "...",
      "n_total": 1000,
      "n_kc": <int>,
      "n_flipped": <int>,
      "kc_suppression": <float>,
      "cases": [
        {"baseline_kc": bool, "rank_flipped": bool}, ...
      ]
    }, ...
  }
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

GRID_JSON      = ROOT / "phase2/results/exp_grid/all_results_ranked.json"
EXP_GRID_PY    = ROOT / "phase2/scripts/exp_grid.py"
CANDS_PATH     = ROOT / "phase2/results/exp1b_candidates/candidates_for_exp2.pt"
RESULTS_DIR    = ROOT / "phase3/results/step3_revised"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIGS = ["L17:A+L19:A", "L18:B", "L16:B+L17:B+L18:B"]
CHECKPOINT     = 200


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def import_exp_grid():
    spec   = importlib.util.spec_from_file_location("phase2_exp_grid", EXP_GRID_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_conflict_tok(grid_mod, model, case) -> torch.Tensor:
    return model.to_tokens(
        grid_mod.build_conflict_prompt(
            case["conflict_context"], case["question"], case["options"]
        ),
        prepend_bos=True,
    )


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    with open(GRID_JSON) as f:
        ranked = json.load(f)
    by_label = {r["config"]: r for r in ranked}

    grid_mod = import_exp_grid()
    model    = grid_mod.load_model()
    grid_mod.setup_letter_variants(model.tokenizer)

    from exp_utils import load_cases
    cases = load_cases(n=1000)
    N     = len(cases)
    cands = torch.load(CANDS_PATH, map_location="cpu", weights_only=False)

    # ── 1. Baseline KC 상태 계산 ─────────────────────────────────────────────
    print("\n[baseline] KC 상태 계산 ...")
    baseline_kc = []
    for i, case in enumerate(cases):
        cor = case["correct_option"]
        rep = case["replace_option"]
        tok = build_conflict_tok(grid_mod, model, case)
        lp_cor, lp_rep = grid_mod.run_logprobs_hooked(model, tok, cor, rep, [])
        baseline_kc.append(lp_rep > lp_cor)  # KC = conflict > correct
        if (i + 1) % CHECKPOINT == 0:
            n_kc = sum(baseline_kc)
            print(f"  [{i+1}/{N}] kc_so_far={n_kc}  ({time.time()-t0:.0f}s)", flush=True)

    n_kc_total = sum(baseline_kc)
    kc_indices = [i for i, k in enumerate(baseline_kc) if k]
    print(f"[baseline] KC cases: {n_kc_total}/{N}  ({time.time()-t0:.0f}s)")

    # ── 2. 각 config로 개입 후 flip 판정 ────────────────────────────────────
    output = {}

    for cfg_label in TARGET_CONFIGS:
        if cfg_label not in by_label:
            print(f"[skip] {cfg_label} not in grid results")
            continue

        r     = by_label[cfg_label]
        pairs = [(int(l), str(d)) for l, d in r["layer_dir_pairs"]]
        hooks = grid_mod.make_fwd_hooks(pairs, cands)

        print(f"\n[config] {cfg_label}  layers={r['layers']}  expected_flipped={r['kc_flipped']}")

        case_results = [{"baseline_kc": False, "rank_flipped": False}] * N
        flipped_count = 0
        t1 = time.time()

        for idx in kc_indices:
            case = cases[idx]
            cor  = case["correct_option"]
            rep  = case["replace_option"]
            tok  = build_conflict_tok(grid_mod, model, case)
            lp_cor, lp_rep = grid_mod.run_logprobs_hooked(model, tok, cor, rep, hooks)
            flipped = bool(lp_cor > lp_rep)
            case_results[idx] = {"baseline_kc": True, "rank_flipped": flipped}
            if flipped:
                flipped_count += 1

        # non-KC cases: 기본값(baseline_kc=False, rank_flipped=False)은 위에서 설정됨
        # 위의 list multiplication은 같은 dict 참조를 공유하므로 리스트 컴프로 재생성
        case_results = [
            {"baseline_kc": bool(baseline_kc[i]), "rank_flipped": bool(baseline_kc[i] and case_results[i]["rank_flipped"])}
            for i in range(N)
        ]

        kc_supp = flipped_count / n_kc_total if n_kc_total else 0.0
        print(f"  KC cases={n_kc_total}  flipped={flipped_count}  kc_suppression={kc_supp:.3f}  "
              f"elapsed={time.time()-t1:.0f}s")

        output[cfg_label] = {
            "config_label":    cfg_label,
            "layer_dir_pairs": r["layer_dir_pairs"],
            "n_total":         N,
            "n_kc":            n_kc_total,
            "n_flipped":       flipped_count,
            "kc_suppression":  round(kc_supp, 4),
            "cases":           case_results,
        }

    out_path = RESULTS_DIR / "intervention_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\n[save] {out_path}")
    print(f"[done] total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

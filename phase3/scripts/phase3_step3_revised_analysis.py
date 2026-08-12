#!/usr/bin/env python3
"""
Step3 revised — Direction별 Flipped vs Non-flipped Projection 분석

A config → A direction projection 비교
B config → B direction projection 비교

출력: phase3/results/step3_revised/group_stats_revised.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "phase3/results/step3_revised"
PROJ_PATH   = ROOT / "phase3/results/projections_raw.pt"

TARGET_LAYERS = list(range(10, 23))

# 각 config가 분석할 direction
CONFIG_TARGET_DIR = {
    "L17:A+L19:A":           "A",
    "L18:B":                 "B",
    "L16:B+L17:B+L18:B":    "B",
}


def main() -> None:
    with open(RESULTS_DIR / "intervention_results.json") as f:
        inter = json.load(f)

    projections = torch.load(PROJ_PATH, map_location="cpu", weights_only=False)

    group_stats: dict = {}

    for cfg_label, data in inter.items():
        target_dir = CONFIG_TARGET_DIR.get(cfg_label)
        if target_dir is None:
            print(f"[skip] no target direction defined for {cfg_label}")
            continue

        cases     = data["cases"]
        kc_ids    = [i for i, c in enumerate(cases) if c["baseline_kc"]]
        flip_ids  = [i for i in kc_ids if cases[i]["rank_flipped"]]
        nflip_ids = [i for i in kc_ids if not cases[i]["rank_flipped"]]

        n_kc   = len(kc_ids)
        n_flip = len(flip_ids)
        print(f"\n[config] {cfg_label}  dir={target_dir}  "
              f"KC={n_kc}  flipped={n_flip}  non-flipped={len(nflip_ids)}")

        if n_flip < 5:
            print(f"  Warning: too few flipped cases ({n_flip}), skipping")
            continue

        layer_stats: dict = {}
        for layer in TARGET_LAYERS:
            proj = projections[target_dir]["conflict"][layer].float()

            fp  = proj[flip_ids].numpy()
            nfp = proj[nflip_ids].numpy()

            # pooled std over KC cases
            pooled_std = float(proj[kc_ids].std().item()) + 1e-8
            effect_size = float((fp.mean() - nfp.mean()) / pooled_std)

            t_stat, p_val = scipy_stats.ttest_ind(fp, nfp)

            layer_stats[str(layer)] = {
                "flipped_mean":     float(fp.mean()),
                "non_flipped_mean": float(nfp.mean()),
                "effect_size":      effect_size,
                "t_stat":           float(t_stat),
                "p_val":            float(p_val),
                "n_flipped":        n_flip,
                "n_non_flipped":    len(nflip_ids),
            }
            sig = "*" if p_val < 0.05 else ""
            print(f"  L{layer:2d}: flip={fp.mean():.2f}  nflip={nfp.mean():.2f}  "
                  f"d={effect_size:.3f}  p={p_val:.3f}{sig}")

        group_stats[cfg_label] = {
            "target_direction": target_dir,
            "n_kc":             n_kc,
            "n_flipped":        n_flip,
            "n_non_flipped":    len(nflip_ids),
            "kc_suppression":   data["kc_suppression"],
            "layers":           layer_stats,
        }

    out_path = RESULTS_DIR / "group_stats_revised.json"
    with open(out_path, "w") as f:
        json.dump(group_stats, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()

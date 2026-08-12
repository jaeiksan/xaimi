#!/usr/bin/env python3
"""
Step2: Ratio 비교 (Δ_sc / Δ_cc)

각 케이스에서:
  Δ_cc = conflict projection - clean projection
  Δ_sc = supportive projection - clean projection

ratio = mean(Δ_sc) / mean(Δ_cc) per layer

판단 기준:
  ratio ≈ 0  → supportive가 clean에서 거의 안 올라감 → conflict detection
  ratio ≈ 1  → supportive도 conflict만큼 올라감      → context following

저장: phase3/results/step2/ratio_results.json
      phase3/results/step2/delta_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT      = Path(__file__).resolve().parents[2]
RESULTS   = ROOT / "phase3/results"
PROJ_PATH = RESULTS / "projections_raw.pt"

TARGET_LAYERS = list(range(10, 23))
DIRECTIONS    = ["A", "B", "C"]


def main() -> None:
    projections = torch.load(PROJ_PATH, map_location="cpu")

    ratio_results: dict = {}
    delta_results: dict = {}

    for d in DIRECTIONS:
        ratio_results[d] = {}
        delta_results[d] = {}
        for layer in TARGET_LAYERS:
            proj_clean      = projections[d]["clean"][layer].float()
            proj_supportive = projections[d]["supportive"][layer].float()
            proj_conflict   = projections[d]["conflict"][layer].float()

            delta_cc = proj_conflict   - proj_clean   # [N]
            delta_sc = proj_supportive - proj_clean   # [N]

            mean_cc = delta_cc.mean().item()
            mean_sc = delta_sc.mean().item()

            ratio = float("nan") if abs(mean_cc) < 1e-6 else mean_sc / mean_cc

            ratio_results[d][layer] = ratio
            delta_results[d][layer] = {
                "mean_delta_cc": mean_cc,
                "mean_delta_sc": mean_sc,
                "std_delta_cc":  delta_cc.std().item(),
                "std_delta_sc":  delta_sc.std().item(),
                "n":             int(delta_cc.numel()),
            }

    with open(RESULTS / "step2/ratio_results.json", "w") as f:
        json.dump(ratio_results, f, indent=2)
    with open(RESULTS / "step2/delta_results.json", "w") as f:
        json.dump(delta_results, f, indent=2)
    print(f"[save] step2/ratio_results.json  step2/delta_results.json")


if __name__ == "__main__":
    main()

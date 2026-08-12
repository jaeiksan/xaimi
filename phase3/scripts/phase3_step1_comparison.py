#!/usr/bin/env python3
"""
Step1: 세 조건 Projection 직접 비교

각 direction × layer에서 clean / supportive / conflict projection 값의
평균과 표준편차를 계산하고 저장한다.

저장: phase3/results/step1/projection_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT       = Path(__file__).resolve().parents[2]
RESULTS    = ROOT / "phase3/results"
PROJ_PATH  = RESULTS / "projections_raw.pt"
OUT_PATH   = RESULTS / "step1/projection_stats.json"

TARGET_LAYERS = list(range(10, 23))
DIRECTIONS    = ["A", "B", "C"]
CONDITIONS    = ["clean", "supportive", "conflict"]


def main() -> None:
    projections = torch.load(PROJ_PATH, map_location="cpu")

    stats: dict = {}
    for d in DIRECTIONS:
        stats[d] = {}
        for layer in TARGET_LAYERS:
            stats[d][layer] = {}
            for c in CONDITIONS:
                vals = projections[d][c][layer].float()
                stats[d][layer][c] = {
                    "mean": vals.mean().item(),
                    "std":  vals.std().item(),
                    "n":    int(vals.numel()),
                }

    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[save] {OUT_PATH}")


if __name__ == "__main__":
    main()

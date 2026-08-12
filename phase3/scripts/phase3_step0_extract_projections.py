#!/usr/bin/env python3
"""
Step0: Projection 추출 (공통 전처리)

세 조건(clean / supportive / conflict)에서 forward pass를 돌려
마지막 토큰 위치의 residual stream을 추출하고,
각 direction으로 projection 값을 계산해 저장한다.

저장: phase3/results/projections_raw.pt
  dict[direction: str][condition: str][layer: int] -> Tensor[N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from exp_utils import (
    build_clean_prompt,
    build_conflict_prompt,
    build_supportive_prompt,
    load_cases,
    load_model,
)

# ── 설정 ────────────────────────────────────────────────────────────────────
TARGET_LAYERS = list(range(10, 23))   # L10 ~ L22
DIRECTIONS    = ["A", "B", "C"]
CONDITIONS    = ["clean", "supportive", "conflict"]

EXP1B_DIR  = ROOT / "phase2/results/exp1b_direction_extraction"
RESULTS    = ROOT / "phase3/results"
OUT_PATH   = RESULTS / "projections_raw.pt"

CHECKPOINT_EVERY = 100


def load_directions(device: str = "cuda") -> dict:
    """방향 벡터 로드 및 정규화. 반환: dict[name][layer] = unit_vector[d_model] on device"""
    raw = {
        "A": torch.load(EXP1B_DIR / "method_A_mean_diff.pt",    map_location="cpu"),
        "B": torch.load(EXP1B_DIR / "method_B_svd_conf_sup.pt", map_location="cpu"),
        "C": torch.load(EXP1B_DIR / "method_C_cpca.pt",         map_location="cpu"),
    }
    key_map = {"A": "mean_dir", "B": "conf_sup_dir", "C": "best_dir"}

    dirs: dict = {}
    for name, data in raw.items():
        dirs[name] = {}
        for layer in TARGET_LAYERS:
            vec = data[layer][key_map[name]].float()
            dirs[name][layer] = F.normalize(vec, dim=0).to(device)
    print(f"[directions] loaded A/B/C for layers {TARGET_LAYERS[0]}~{TARGET_LAYERS[-1]} on {device}")
    return dirs


def main() -> None:
    t0 = time.time()

    model  = load_model(device="cuda")
    cases  = load_cases(n=1000)
    N      = len(cases)
    dirs   = load_directions()

    # 결과 버퍼
    projections: dict = {
        d: {c: {l: [] for l in TARGET_LAYERS} for c in CONDITIONS}
        for d in DIRECTIONS
    }

    for idx, case in enumerate(cases):
        q   = case["question"]
        opt = case["options"]

        prompts = {
            "clean":      build_clean_prompt(q, opt),
            "conflict":   build_conflict_prompt(case["conflict_context"], q, opt),
            "supportive": build_supportive_prompt(
                q, opt, case["correct_answer"],
                case["conflict_context"], case["conflict_answer"],
            ),
        }

        for condition, prompt in prompts.items():
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda n: n.endswith("hook_resid_post"),
                )
            for layer in TARGET_LAYERS:
                # 마지막 토큰 위치의 residual stream
                resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1].float()
                for d in DIRECTIONS:
                    proj = (resid @ dirs[d][layer]).item()
                    projections[d][condition][layer].append(proj)

            del cache
        torch.cuda.empty_cache()

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [{idx+1}/{N}] elapsed={time.time()-t0:.0f}s", flush=True)

    # list -> Tensor
    for d in DIRECTIONS:
        for c in CONDITIONS:
            for l in TARGET_LAYERS:
                projections[d][c][l] = torch.tensor(projections[d][c][l])

    torch.save(projections, OUT_PATH)
    print(f"[save] {OUT_PATH}")
    print(f"[done] total={time.time()-t0:.1f}s  N={N}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 4 — Step 0: Seed Question 단위 Train/Dev Split

340개 고유 seed question을 80/20으로 분리한다.
같은 question의 모든 variant(다른 conflict_context)는 반드시 같은 split에 속해야 한다.

출력:
  phase4/data/train_800.json  (~800 케이스, seed question 272개)
  phase4/data/dev_200.json    (~200 케이스, seed question 68개)
  phase4/data/split_info.json  (split 메타데이터)

실행:
  cd /home/jaeik/xaimi
  python phase4/scripts/setup_data.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from exp_utils import load_cases

SEED       = 42
DEV_RATIO  = 0.2
DATA_DIR   = ROOT / "phase4/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cases = load_cases(n=1000)

    # ── 1. seed question 단위로 그룹화 ────────────────────────────────────────
    q_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        q_to_indices[c["question"]].append(i)

    questions = sorted(q_to_indices.keys())   # deterministic order
    rng = random.Random(SEED)
    rng.shuffle(questions)

    n_dev  = max(1, round(len(questions) * DEV_RATIO))
    n_train = len(questions) - n_dev

    dev_questions   = set(questions[:n_dev])
    train_questions = set(questions[n_dev:])

    train_cases = [c for c in cases if c["question"] in train_questions]
    dev_cases   = [c for c in cases if c["question"] in dev_questions]

    print(f"Seed questions:  {len(questions)} total  → {n_train} train, {n_dev} dev")
    print(f"Cases:           {len(cases)} total  → {len(train_cases)} train, {len(dev_cases)} dev")

    # ── 2. 저장 ────────────────────────────────────────────────────────────────
    for fname, data in [("train_800.json", train_cases), ("dev_200.json", dev_cases)]:
        out = DATA_DIR / fname
        with open(out, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[save] {out}  ({len(data)} cases)")

    split_info = {
        "seed":            SEED,
        "n_seed_questions": len(questions),
        "n_train_questions": n_train,
        "n_dev_questions":  n_dev,
        "n_train_cases":   len(train_cases),
        "n_dev_cases":     len(dev_cases),
    }
    with open(DATA_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("[save] split_info.json")


if __name__ == "__main__":
    main()

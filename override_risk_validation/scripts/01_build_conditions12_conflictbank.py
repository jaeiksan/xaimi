#!/usr/bin/env python3
"""Make disjoint ConflictBank candidates for paired conditions ① and ②.

The builder uses ConflictBank's default (fact-aligned) evidence as the
supportive-context view. Behavioural labels are intentionally deferred to
`05_screen_conditions12_conflictbank.py`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
PHASE0_PATH = PROJECT_ROOT / "data/phase0/kc_cases_1000.json"
COND3_PATH = EXPERIMENT_ROOT / "data/processed/cond3_beneficial_override_300.jsonl"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data/processed/cond12_conflictbank_candidates.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def forbidden_questions() -> set[str]:
    phase0 = json.loads(PHASE0_PATH.read_text())
    cond3 = read_jsonl(COND3_PATH)
    return {row["question"] for row in phase0} | {row["question"] for row in cond3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--offset", type=int, default=0, help="skip this many valid seed-shuffled candidates")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset("Warrieryes/CB_qa", split="train", download_mode="reuse_dataset_if_exists")
    excluded = forbidden_questions()
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    candidates = []
    skipped = 0
    for index in indices:
        row = dataset[index]
        if row["question"] in excluded or len(row["options"]) != 4:
            continue
        if skipped < args.offset:
            skipped += 1
            continue
        options = {letter: value for letter, value in zip("ABCD", row["options"])}
        candidates.append({
            "id": f"conflictbank_default_{index}",
            "source": "ConflictBank CB_qa default evidence",
            "source_row_index": index,
            "question": row["question"],
            "options": options,
            "correct_option": row["correct_option"],
            "correct_answer": options[row["correct_option"]],
            "supportive_context": row["default_evidence"],
            "supportive_claim": row["default_claim"],
            "relation": row["relation"],
            "selection_seed": args.seed,
            "screening_status": "pending_model_screen",
        })
        if len(candidates) == args.n:
            break
    if len(candidates) < args.n:
        raise RuntimeError(f"Only found {len(candidates)} disjoint candidates")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"[saved] {len(candidates)} candidates (offset={args.offset}) → {args.output}")


if __name__ == "__main__":
    main()

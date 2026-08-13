#!/usr/bin/env python3
"""Freeze disjoint held-out evaluation sets for conditions ①–④.

Condition ①: 300 paired safe-context cases sampled from screened ConflictBank.
Condition ②: every available persistent-error case (currently expected: 105).
Condition ③: the already frozen 300-case TempLAMA beneficial-override set.
Condition ④: Phase 4 dev cases only, never the vector-training cases.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DATA = EXPERIMENT_ROOT / "data/processed"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def assert_unique(rows: list[dict], label: str) -> None:
    questions = [row["question"] for row in rows]
    if len(questions) != len(set(questions)):
        raise ValueError(f"{label} contains duplicate questions")


def one_case_per_question(rows: list[dict]) -> list[dict]:
    """Phase 4 dev has conflict-type variants; keep one deterministic case each."""
    selected: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (item["question"], item.get("id", ""))):
        selected.setdefault(row["question"], row)
    return list(selected.values())


def main() -> None:
    c1_all = read_jsonl(DATA / "cond1_safe_paired.jsonl") + read_jsonl(DATA / "cond1_safe_paired_merged.jsonl")
    c2_all = read_jsonl(DATA / "cond2_persistent_error_paired.jsonl") + read_jsonl(DATA / "cond2_persistent_error_paired_merged.jsonl")
    c3 = read_jsonl(DATA / "cond3_beneficial_override_300.jsonl")
    c4 = one_case_per_question(json.loads((PROJECT_ROOT / "phase4/data/dev_200.json").read_text()))
    train = json.loads((PROJECT_ROOT / "phase4/data/train_800.json").read_text())

    for label, rows in (("cond1_all", c1_all), ("cond2_all", c2_all), ("cond3", c3), ("cond4", c4)):
        assert_unique(rows, label)

    c1 = random.Random(SEED).sample(c1_all, 300)
    c2 = c2_all
    train_questions = {row["question"] for row in train}
    groups = {"cond1": c1, "cond2": c2, "cond3": c3, "cond4": c4}
    for label, rows in groups.items():
        overlap = {row["question"] for row in rows} & train_questions
        if label == "cond4" and overlap:
            # Phase 4's split is seed-question based; dev questions should still
            # be distinct from the training questions.
            raise ValueError(f"{label} overlaps Phase 4 vector-training questions")
    labels = list(groups)
    for i, left in enumerate(labels):
        left_questions = {row["question"] for row in groups[left]}
        for right in labels[i + 1:]:
            overlap = left_questions & {row["question"] for row in groups[right]}
            if overlap:
                raise ValueError(f"Question overlap between {left} and {right}: {len(overlap)}")

    for label, rows in groups.items():
        output_rows = [{**row, "evaluation_condition": label, "selection_seed": SEED} for row in rows]
        write_jsonl(DATA / f"final_{label}.jsonl", output_rows)
    manifest = {
        "selection_seed": SEED,
        "counts": {label: len(rows) for label, rows in groups.items()},
        "condition4_source": "phase4/data/dev_200.json",
        "phase4_train_questions_excluded": True,
        "all_pairwise_question_overlap": 0,
    }
    (DATA / "final_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[saved] " + ", ".join(f"{label}={len(rows)}" for label, rows in groups.items()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Screen paired ConflictBank candidates into condition ① or ②.

Condition ①: the same question is correct in closed-book and supportive views.
Condition ②: the same question is incorrect in both views.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from evaluation_utils import is_correct, option_logprobs, predict_option  # noqa: E402
from exp_utils import build_clean_prompt, build_conflict_prompt, load_model  # noqa: E402


DEFAULT_INPUT = EXPERIMENT_ROOT / "data/processed/cond12_conflictbank_candidates.jsonl"
DEFAULT_SCREENED = EXPERIMENT_ROOT / "data/processed/cond12_conflictbank_screened.jsonl"
DEFAULT_COND1 = EXPERIMENT_ROOT / "data/processed/cond1_safe_paired.jsonl"
DEFAULT_COND2 = EXPERIMENT_ROOT / "data/processed/cond2_persistent_error_paired.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify(model, row: dict) -> dict:
    options = row["options"]
    correct = row["correct_option"]
    clean = option_logprobs(model, build_clean_prompt(row["question"], options), options)
    support = option_logprobs(
        model,
        build_conflict_prompt(row["supportive_context"], row["question"], options),
        options,
    )
    clean_ok = is_correct(clean, correct)
    support_ok = is_correct(support, correct)
    condition = "cond1" if clean_ok and support_ok else "cond2" if not clean_ok and not support_ok else "excluded"
    return {
        **row,
        "closed_book_prediction": predict_option(clean),
        "supportive_prediction": predict_option(support),
        "closed_book_option_logprobs": clean,
        "supportive_option_logprobs": support,
        "closed_book_correct": clean_ok,
        "supportive_correct": support_ok,
        "assigned_condition": condition,
        "screening_status": "qualified" if condition != "excluded" else "excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--screened-output", type=Path, default=DEFAULT_SCREENED)
    parser.add_argument("--cond1-output", type=Path, default=DEFAULT_COND1)
    parser.add_argument("--cond2-output", type=Path, default=DEFAULT_COND2)
    parser.add_argument("--start-index", type=int, default=0, help="0-based index into the candidate JSONL")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    all_rows = read_jsonl(args.input)
    if args.start_index < 0 or args.start_index > len(all_rows):
        raise ValueError(f"start-index must be in [0, {len(all_rows)}]")
    rows = all_rows[args.start_index:]
    if args.limit:
        rows = rows[:args.limit]
    print(
        f"[screen] candidates {args.start_index + 1}–{args.start_index + len(rows)} "
        f"of {len(all_rows)}",
        flush=True,
    )
    model = load_model(device="cuda")
    screened = []
    for i, row in enumerate(rows, 1):
        screened.append(classify(model, row))
        if i % args.checkpoint_every == 0:
            save_jsonl(args.screened_output, screened)
            print(f"[screen] {args.start_index + i}/{len(all_rows)}", flush=True)
    save_jsonl(args.screened_output, screened)
    save_jsonl(args.cond1_output, [row for row in screened if row["assigned_condition"] == "cond1"])
    save_jsonl(args.cond2_output, [row for row in screened if row["assigned_condition"] == "cond2"])
    print(
        f"[saved] screened={len(screened)} cond1={sum(r['assigned_condition']=='cond1' for r in screened)} "
        f"cond2={sum(r['assigned_condition']=='cond2' for r in screened)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

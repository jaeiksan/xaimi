#!/usr/bin/env python3
"""Behaviourally screen TempLAMA candidates into factual beneficial overrides.

Qualifying rule (fixed by the study specification):
  1. closed-book full-choice prediction is historical answer M;
  2. context-prompt full-choice prediction is newer answer C.

No directions, thresholds, or interventions are used here. This is purely an
evaluation-data filter, so its output remains held out from Phase 4 tuning.
Run in the existing `apa` Conda environment with a working CUDA driver.
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

from evaluation_utils import option_logprobs, predict_option  # noqa: E402
from exp_utils import build_clean_prompt, build_conflict_prompt, load_model  # noqa: E402


DEFAULT_INPUT = EXPERIMENT_ROOT / "data/processed/cond3_templama_candidates.jsonl"
DEFAULT_ALL = EXPERIMENT_ROOT / "data/processed/cond3_templama_screened.jsonl"
DEFAULT_QUALIFIED = EXPERIMENT_ROOT / "data/processed/cond3_beneficial_override.jsonl"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def screen_candidate(model, candidate: dict) -> dict:
    question = candidate["question"]
    options = candidate["options"]
    clean_scores = option_logprobs(model, build_clean_prompt(question, options), options)
    context_scores = option_logprobs(
        model, build_conflict_prompt(candidate["context"], question, options), options
    )
    closed_book_prediction = predict_option(clean_scores)
    context_prediction = predict_option(context_scores)
    old_option = candidate["parametric_option_M"]
    new_option = candidate["correct_option"]
    qualified = closed_book_prediction == old_option and context_prediction == new_option
    return {
        **candidate,
        "closed_book_prediction": closed_book_prediction,
        "context_prediction": context_prediction,
        "closed_book_option_logprobs": clean_scores,
        "context_option_logprobs": context_scores,
        "passes_beneficial_override_filter": qualified,
        "screening_status": "qualified" if qualified else "excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--screened-output", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--qualified-output", type=Path, default=DEFAULT_QUALIFIED)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every candidate")
    parser.add_argument("--resume", action="store_true", help="continue after existing screened rows")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    all_candidates = list(read_jsonl(args.input))
    screened: list[dict] = list(read_jsonl(args.screened_output)) if args.resume and args.screened_output.exists() else []
    start_index = len(screened)
    candidates = all_candidates[start_index:]
    if args.limit:
        candidates = candidates[:args.limit]
    print(
        f"[screen] candidates {start_index + 1}–{start_index + len(candidates)} "
        f"of {len(all_candidates)} (existing={start_index})",
        flush=True,
    )

    model = load_model(device="cuda")
    for index, candidate in enumerate(candidates, 1):
        screened.append(screen_candidate(model, candidate))
        if index % args.checkpoint_every == 0:
            write_jsonl(args.screened_output, screened)
            qualified = sum(row["passes_beneficial_override_filter"] for row in screened)
            print(
                f"[screen] {start_index + index}/{len(all_candidates)}; qualified={qualified}",
                flush=True,
            )

    write_jsonl(args.screened_output, screened)
    qualified_rows = [row for row in screened if row["passes_beneficial_override_filter"]]
    write_jsonl(args.qualified_output, qualified_rows)
    print(
        f"[saved] screened={len(screened)}/{len(all_candidates)}, qualified={len(qualified_rows)}\n"
        f"  {args.screened_output}\n  {args.qualified_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

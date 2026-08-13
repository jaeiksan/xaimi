#!/usr/bin/env python3
"""Collect frozen B-gate signals without intervention for Experiment B.

Each row stores the L17 B-direction projection, frozen gate value, gate
activation, and full-choice prediction. The primary context view is used for
the harmful-risk comparison; paired closed-book views are also collected for
conditions ① and ②.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from evaluation_utils import option_logprobs_from_logits, predict_option  # noqa: E402
from exp_utils import build_clean_prompt, build_conflict_prompt, load_model  # noqa: E402


LAYER = 17
GAMMA = 10.0
TAU = 19.256011962890625
DATA = EXPERIMENT_ROOT / "data/processed"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/exp_B_per_case.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_views(condition: str, row: dict) -> list[tuple[str, str]]:
    question, options = row["question"], row["options"]
    clean = build_clean_prompt(question, options)
    if condition in {"cond1", "cond2"}:
        context = build_conflict_prompt(row["supportive_context"], question, options)
        return [("closed_book", clean), ("context", context)]
    if condition == "cond3":
        context = build_conflict_prompt(row["context"], question, options)
        return [("closed_book", clean), ("context", context)]
    if condition == "cond4":
        context = build_conflict_prompt(row["conflict_context"], question, options)
        return [("context", context)]
    raise ValueError(f"Unknown condition: {condition}")


@torch.no_grad()
def evaluate_view(model, direction: torch.Tensor, row: dict, condition: str, view: str, prompt: str) -> dict:
    ids = model.to_tokens(prompt, prepend_bos=True)
    logits, cache = model.run_with_cache(
        ids,
        names_filter=lambda name: name == f"blocks.{LAYER}.hook_resid_post",
    )
    hidden = cache[f"blocks.{LAYER}.hook_resid_post"][0, -1, :].float()
    projection = float(hidden @ direction)
    gate_value = float(torch.sigmoid(torch.tensor(GAMMA * (projection - TAU))))
    scores = option_logprobs_from_logits(logits[0, -1, :], row["options"])
    del cache, logits
    prediction = predict_option(scores)
    return {
        "id": row["id"],
        "question": row["question"],
        "condition": condition,
        "view": view,
        "correct_option": row["correct_option"],
        "prediction": prediction,
        "is_correct": prediction == row["correct_option"],
        "projection": projection,
        "gate_value": gate_value,
        "gate_activated": projection >= TAU,
        "option_logprobs": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue after rows already present in --output")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    cases = {
        condition: read_jsonl(DATA / f"final_{condition}.jsonl")
        for condition in ("cond1", "cond2", "cond3", "cond4")
    }
    tasks = [(condition, row, view, prompt) for condition, rows in cases.items() for row in rows for view, prompt in prompt_views(condition, row)]
    completed = read_jsonl(args.output) if args.resume and args.output.exists() else []
    start_index = len(completed)
    if start_index > len(tasks):
        raise ValueError(f"Output has {start_index} rows, but the task set has {len(tasks)}")
    tasks = tasks[start_index:]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[collect] prompt views {start_index + 1}–{start_index + len(tasks)}", flush=True)

    direction = F.normalize(
        torch.load(PROJECT_ROOT / "phase4/vectors/direction_B_train.pt", map_location="cpu", weights_only=False)[LAYER].float(),
        dim=0,
    ).to("cuda")
    model = load_model(device="cuda")
    results = completed
    for index, (condition, row, view, prompt) in enumerate(tasks, 1):
        results.append(evaluate_view(model, direction, row, condition, view, prompt))
        if index % args.checkpoint_every == 0:
            write_jsonl(args.output, results)
            print(f"[collect] {start_index + index}/{start_index + len(tasks)}", flush=True)
    write_jsonl(args.output, results)
    print(f"[saved] {len(results)} rows → {args.output}", flush=True)


if __name__ == "__main__":
    main()

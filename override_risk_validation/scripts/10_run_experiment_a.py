#!/usr/bin/env python3
"""Experiment A: test whether the frozen gate preserves beneficial overrides.

On the frozen TempLAMA condition ③ set, compare:
  - no intervention,
  - ungated parametric restoration,
  - L17 B-gated parametric restoration.
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
sys.path.insert(0, str(PROJECT_ROOT / "phase4/scripts"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

from evaluation_utils import option_logprobs_from_logits, predict_option  # noqa: E402
from exp_utils import build_conflict_prompt, load_model  # noqa: E402
from intervention_methods import make_upk_hook  # noqa: E402


LAYER = 17
ALPHA = 6.0
GAMMA = 10.0
TAU = 19.256011962890625
DATA_PATH = EXPERIMENT_ROOT / "data/processed/final_cond3.jsonl"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/exp_A_per_case.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def run_one(model, direction: torch.Tensor, restoration: torch.Tensor, row: dict, intervention: str) -> dict:
    prompt = build_conflict_prompt(row["context"], row["question"], row["options"])
    ids = model.to_tokens(prompt, prepend_bos=True)
    gate_store = [None]
    signal_store = {"projection": None}
    normalized_direction = F.normalize(direction, dim=0)

    def reader_hook(value, hook):
        projection = value[:, -1, :].float() @ normalized_direction.to(value.device)
        signal_store["projection"] = projection.detach()
        gate_store[0] = torch.sigmoid(GAMMA * (projection - TAU)).detach()
        return value

    hooks = []
    if intervention == "gated":
        hooks = [
            (f"blocks.{LAYER}.hook_resid_post", reader_hook),
            (f"blocks.{LAYER}.hook_resid_post", make_upk_hook(gate_store, restoration, ALPHA)),
        ]
    elif intervention == "ungated":
        # Reader is kept to report the same pre-intervention gate signal for
        # all intervention arms; the action itself always applies restoration.
        def ungated_hook(value, hook):
            v = value.float()
            v[:, -1, :] = v[:, -1, :] + ALPHA * restoration.to(v.device)
            return v.to(value.dtype)
        hooks = [
            (f"blocks.{LAYER}.hook_resid_post", reader_hook),
            (f"blocks.{LAYER}.hook_resid_post", ungated_hook),
        ]
    elif intervention == "none":
        hooks = [(f"blocks.{LAYER}.hook_resid_post", reader_hook)]
    else:
        raise ValueError(intervention)
    with model.hooks(fwd_hooks=hooks):
        logits = model(ids)[0, -1, :].float()
    scores = option_logprobs_from_logits(logits, row["options"])
    prediction = predict_option(scores)
    gate_value = float(gate_store[0][0].item())
    projection = float(signal_store["projection"][0].item())
    return {
        "id": row["id"],
        "question": row["question"],
        "condition": "cond3_beneficial_override",
        "intervention_type": intervention,
        "correct_option": row["correct_option"],
        "prediction": prediction,
        "override_success": prediction == row["correct_option"],
        "projection": projection,
        "gate_value": gate_value,
        "gate_activated": gate_value >= 0.5,
        "option_logprobs": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue after complete three-arm case groups")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    cases = read_jsonl(DATA_PATH)
    existing = read_jsonl(args.output) if args.resume and args.output.exists() else []
    if len(existing) % 3:
        raise ValueError("Resume output must contain a multiple of three rows")
    completed_cases = len(existing) // 3
    if completed_cases > len(cases):
        raise ValueError("Resume output has more cases than the frozen condition ③ set")
    cases = cases[completed_cases:]
    if args.limit:
        cases = cases[:args.limit]
    direction = F.normalize(
        torch.load(PROJECT_ROOT / "phase4/vectors/direction_B_train.pt", map_location="cpu", weights_only=False)[LAYER].float(),
        dim=0,
    ).to("cuda")
    restoration = torch.load(PROJECT_ROOT / "phase4/vectors/u_PK_train.pt", map_location="cpu", weights_only=False)[LAYER].float().to("cuda")
    model = load_model(device="cuda")
    results = existing
    for index, row in enumerate(cases, 1):
        for intervention in ("none", "ungated", "gated"):
            results.append(run_one(model, direction, restoration, row, intervention))
        if index % args.checkpoint_every == 0:
            write_jsonl(args.output, results)
            print(f"[experiment A] {completed_cases + index}/{completed_cases + len(cases)}", flush=True)
    write_jsonl(args.output, results)
    print(f"[saved] {len(results)} rows → {args.output}", flush=True)


if __name__ == "__main__":
    main()

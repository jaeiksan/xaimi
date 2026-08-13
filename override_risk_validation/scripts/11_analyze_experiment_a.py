#!/usr/bin/env python3
"""Summarize Experiment A and compute paired-bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_ROOT / "results/exp_A_per_case.jsonl"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/exp_A_beneficial_override.csv"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_ci(differences: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    samples = np.empty(n)
    for i in range(n):
        samples[i] = rng.choice(differences, len(differences), replace=True).mean()
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.input):
        grouped[row["intervention_type"]].append(row)
    expected = {"none", "ungated", "gated"}
    if set(grouped) != expected:
        raise ValueError(f"Expected {expected}, found {set(grouped)}")
    by_id = {method: {row["id"]: row for row in rows} for method, rows in grouped.items()}
    ids = sorted(by_id["none"])
    if any(set(by_id[method]) != set(ids) for method in expected):
        raise ValueError("Intervention rows are not paired by case ID")

    rng = np.random.default_rng(SEED)
    summary = []
    baseline = np.array([by_id["none"][case_id]["override_success"] for case_id in ids], dtype=float)
    for method in ("none", "ungated", "gated"):
        rows = [by_id[method][case_id] for case_id in ids]
        success = np.array([row["override_success"] for row in rows], dtype=float)
        diff = success - baseline
        lower, upper = paired_ci(diff, rng, args.bootstrap_n)
        summary.append({
            "condition": "cond3_beneficial_override",
            "intervention_type": method,
            "override_success_rate": success.mean(),
            "mean_projection": np.mean([row["projection"] for row in rows]),
            "mean_gate_value": np.mean([row["gate_value"] for row in rows]),
            "gate_activation_rate": np.mean([row["gate_activated"] for row in rows]),
            "paired_delta_vs_no_intervention": diff.mean(),
            "paired_delta_ci_lower": lower,
            "paired_delta_ci_upper": upper,
            "n_samples": len(rows),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()

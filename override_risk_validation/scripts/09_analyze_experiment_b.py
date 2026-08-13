#!/usr/bin/env python3
"""Create Experiment B summaries, AUROC values, and bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_ROOT / "results/exp_B_per_case.jsonl"
DEFAULT_RESULTS = EXPERIMENT_ROOT / "results"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a positive score exceeds a negative score; ties get 0.5."""
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def bootstrap_auroc(positive: np.ndarray, negative: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    values = np.empty(n)
    for i in range(n):
        values[i] = auroc(rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def bootstrap_mean_difference(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    values = np.empty(n)
    for i in range(n):
        values[i] = rng.choice(left, len(left), replace=True).mean() - rng.choice(right, len(right), replace=True).mean()
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["view"])].append(row)

    summaries = []
    for (condition, view), group in sorted(grouped.items()):
        projection = np.array([row["projection"] for row in group])
        gate = np.array([row["gate_value"] for row in group])
        summaries.append({
            "condition": condition,
            "view": view,
            "mean_projection": projection.mean(),
            "std_projection": projection.std(ddof=1),
            "mean_gate_value": gate.mean(),
            "gate_activation_rate": np.mean([row["gate_activated"] for row in group]),
            "full_choice_accuracy": np.mean([row["is_correct"] for row in group]),
            "n_samples": len(group),
        })
    write_csv(args.results_dir / "exp_B_specificity.csv", summaries)

    rng = np.random.default_rng(SEED)
    signals = {(condition, view): np.array([row["projection"] for row in group]) for (condition, view), group in grouped.items()}
    required = {
        ("cond1", "closed_book"), ("cond1", "context"),
        ("cond2", "closed_book"), ("cond2", "context"),
        ("cond3", "context"), ("cond4", "context"),
    }
    missing = required - set(signals)
    if missing:
        raise ValueError(f"Incomplete Experiment B collection; missing views: {sorted(missing)}")
    comparisons = [
        ("cond4_context_vs_cond1_context", signals[("cond4", "context")], signals[("cond1", "context")]),
        ("cond4_context_vs_cond3_context", signals[("cond4", "context")], signals[("cond3", "context")]),
        ("cond2_closed_book_vs_cond1_closed_book", signals[("cond2", "closed_book")], signals[("cond1", "closed_book")]),
        ("cond2_context_vs_cond1_context", signals[("cond2", "context")], signals[("cond1", "context")]),
    ]
    aurocs = []
    for name, positive, negative in comparisons:
        lower, upper = bootstrap_auroc(positive, negative, rng, args.bootstrap_n)
        diff_lower, diff_upper = bootstrap_mean_difference(positive, negative, rng, args.bootstrap_n)
        aurocs.append({
            "comparison_pair": name,
            "auroc": auroc(positive, negative),
            "auroc_ci_lower": lower,
            "auroc_ci_upper": upper,
            "mean_projection_difference": positive.mean() - negative.mean(),
            "difference_ci_lower": diff_lower,
            "difference_ci_upper": diff_upper,
            "n_positive": len(positive),
            "n_negative": len(negative),
        })
    write_csv(args.results_dir / "exp_B_auroc.csv", aurocs)
    print(f"[saved] {args.results_dir / 'exp_B_specificity.csv'}")
    print(f"[saved] {args.results_dir / 'exp_B_auroc.csv'}")


if __name__ == "__main__":
    main()

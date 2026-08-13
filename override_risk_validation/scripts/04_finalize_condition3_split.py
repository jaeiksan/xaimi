#!/usr/bin/env python3
"""Select a fixed, relation-stratified 300-case condition ③ evaluation split."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_ROOT / "data/processed/cond3_beneficial_override.jsonl"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data/processed/cond3_beneficial_override_300.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def proportional_quotas(groups: dict[str, list[dict]], n: int) -> dict[str, int]:
    total = sum(map(len, groups.values()))
    base = {key: int(n * len(rows) / total) for key, rows in groups.items()}
    remainder = n - sum(base.values())
    fractions = sorted(
        groups,
        key=lambda key: (n * len(groups[key]) / total - base[key], key),
        reverse=True,
    )
    for key in fractions[:remainder]:
        base[key] += 1
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.n > len(rows):
        raise ValueError(f"Requested {args.n} rows but only {len(rows)} are available")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["relation"]].append(row)
    quotas = proportional_quotas(groups, args.n)
    rng = random.Random(args.seed)
    selected = []
    for relation in sorted(groups):
        selected.extend(rng.sample(groups[relation], quotas[relation]))
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps({**row, "selection_seed": args.seed}, ensure_ascii=False) + "\n")
    print(f"[saved] {len(selected)} rows → {args.output}")
    print("[quotas] " + ", ".join(f"{key}={quotas[key]}" for key in sorted(quotas)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and merge independently screened condition ①/② shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = EXPERIMENT_ROOT / "data/processed"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [row for path in args.shards for row in read_jsonl(path)]
    ids = [row["id"] for row in rows]
    questions = [row["question"] for row in rows]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError("Shard overlap detected: IDs and questions must be unique before merging")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "cond12_conflictbank_screened_merged.jsonl", rows)
    cond1 = [row for row in rows if row["assigned_condition"] == "cond1"]
    cond2 = [row for row in rows if row["assigned_condition"] == "cond2"]
    write_jsonl(args.output_dir / "cond1_safe_paired_merged.jsonl", cond1)
    write_jsonl(args.output_dir / "cond2_persistent_error_paired_merged.jsonl", cond2)
    print(f"[saved] screened={len(rows)} cond1={len(cond1)} cond2={len(cond2)}")


if __name__ == "__main__":
    main()

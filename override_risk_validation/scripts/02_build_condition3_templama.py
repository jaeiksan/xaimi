#!/usr/bin/env python3
"""Build held-out beneficial-override candidates from official TempLAMA JSONL.

This is deliberately a *candidate* builder. It creates factual old-value (M)
and newer-value (C) pairs from TempLAMA, but does not label a row as a
beneficial override until the separate model screen verifies M in closed-book
mode and C after the newer-value context is supplied.

The output uses the existing XAIMI multiple-choice prompt schema while retaining
the original TempLAMA fields needed for provenance and later screening.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXPERIMENT_ROOT / "data/raw/templama_test.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data/processed/cond3_templama_candidates.jsonl"
SEED = 42


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error


def answer_name(value: Any) -> str | None:
    """Return a single entity name; multi-answer facts are excluded."""
    if isinstance(value, dict):
        name = value.get("name")
        return name if isinstance(name, str) and name.strip() else None
    if isinstance(value, list) and len(value) == 1:
        return answer_name(value[0])
    return None


def numeric_date(value: str) -> tuple[int, str]:
    """Use the leading year for chronological selection, preserving raw date."""
    try:
        return int(value[:4]), value
    except (TypeError, ValueError):
        return -1, str(value)


def choose_distractors(pool: list[str], excluded: set[str], rng: random.Random) -> list[str]:
    candidates = sorted({name for name in pool if name not in excluded})
    if len(candidates) < 2:
        return []
    return rng.sample(candidates, 2)


def build_candidates(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Produce one old-vs-new candidate per (query, relation) pair."""
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relation_pool: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        old = answer_name(row.get("answer"))
        new = answer_name(row.get("most_recent_answer"))
        query = row.get("query")
        relation = row.get("relation")
        if old and new and isinstance(query, str) and isinstance(relation, str):
            relation_pool[relation].extend((old, new))
            if old != new:
                by_query[(query, relation)].append(row)

    candidates: list[dict[str, Any]] = []
    for (query, relation), versions in sorted(by_query.items()):
        # The latest available historical row with an answer distinct from C is
        # the strongest old-value M candidate for this query.
        selected = max(versions, key=lambda row: numeric_date(str(row.get("date", ""))))
        old = answer_name(selected["answer"])
        new = answer_name(selected["most_recent_answer"])
        assert old is not None and new is not None and old != new

        distractors = choose_distractors(relation_pool[relation], {old, new}, rng)
        if len(distractors) != 2:
            continue
        option_values = [old, new, *distractors]
        rng.shuffle(option_values)
        options = {letter: value for letter, value in zip("ABCD", option_values)}
        old_option = next(letter for letter, value in options.items() if value == old)
        new_option = next(letter for letter, value in options.items() if value == new)

        rendered_fact = query.replace("_X_", new)
        candidates.append({
            "id": f"templama_{selected['id']}",
            "source": "TempLAMA test",
            "source_id": selected["id"],
            "relation": relation,
            "question": query.replace("_X_", "____"),
            "options": options,
            "parametric_answer_M": old,
            "parametric_option_M": old_option,
            "context_answer_C": new,
            "correct_option": new_option,
            "old_value_date": selected["date"],
            "context": (
                "A later TempLAMA/Wikidata record states: "
                f"{rendered_fact}"
            ),
            "most_recent_answer": selected["most_recent_answer"],
            "screening_status": "pending_model_screen",
        })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = list(jsonl(args.input))
    candidates = build_candidates(rows, random.Random(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"[saved] {len(candidates)} TempLAMA candidates → {args.output}")


if __name__ == "__main__":
    main()

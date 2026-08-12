#!/usr/bin/env python3
"""
Phase 4: conflict_correct_cases 수집

CB_qa 데이터셋에서 모델이 false context에서도 parametric knowledge를 유지하는 케이스를 수집한다.
조건: clean→correct AND conflict→correct AND supportive→correct

기존 1000개 KC 케이스의 source_row_index는 제외한다 (leakage 방지).
목표: 최소 300개.

저장:
  phase4/data/conflict_correct_cases.json
  phase4/data/conflict_correct_stats.json

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 python -u phase4/scripts/collect_conflict_correct_cases.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "phase4/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from exp_utils import (
    build_clean_prompt,
    build_conflict_prompt,
    build_supportive_prompt,
    load_model,
)

CB_QA_PATH = Path(
    "/home/jaeik/.cache/huggingface/hub/datasets--Warrieryes--CB_qa/snapshots/"
    "b3b12e1dba7097151312c29c13e2a76f47d726c9/QA_dataset.json"
)
KC_CASES_PATH = ROOT / "data/phase0/kc_cases_1000.json"

TARGET_N        = 300
MAX_ROWS_SCAN   = 60_000
CHECKPOINT      = 1_000
CONFLICT_TYPES  = ("misinformation", "temporal", "semantic")
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def option_from_letter(options: list, letter: str) -> Optional[str]:
    idx = LETTER_TO_INDEX.get((letter or "").strip().upper())
    if idx is None or idx >= len(options):
        return None
    return options[idx]


def build_conflict_context(row: dict, conflict_type: str) -> str:
    if conflict_type == "misinformation":
        evidence = row.get("misinformation_conflict_evidence_evidence", "").strip()
        claim    = row.get("misinformation_conflict_claim", "").strip()
    elif conflict_type == "temporal":
        evidence = row.get("temporal_conflict_evidence", "").strip()
        claim    = row.get("temporal_conflict_claim", "").strip()
    else:
        evidence = row.get("semantic_conflict_evidence", "").strip()
        claim    = row.get("semantic_conflict_claim", "").strip()

    if claim and claim not in evidence:
        return f"{claim}\n\n{evidence}".strip()
    return evidence or claim


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_logits_last(model, prompt: str) -> torch.Tensor:
    ids    = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = model(ids)[0, -1, :].float()
    return logits


def is_correct(logits: torch.Tensor, cor_id: int, rep_id: int) -> bool:
    return float(logits[cor_id]) > float(logits[rep_id])


def check_case(model, case: dict) -> bool:
    """
    True iff clean→correct AND conflict→correct AND supportive→correct.
    Early-exit on first failure.
    """
    q   = case["question"]
    opt = case["options"]
    ctx = case["conflict_context"]
    cor = case["correct_option"]
    rep = case["replace_option"]
    coa = case["correct_answer"]
    ca  = case["conflict_answer"]

    cor_id = model.tokenizer.encode(f" {cor}", add_special_tokens=False)[-1]
    rep_id = model.tokenizer.encode(f" {rep}", add_special_tokens=False)[-1]

    # 1. clean
    logits = get_logits_last(model, build_clean_prompt(q, opt))
    if not is_correct(logits, cor_id, rep_id):
        return False

    # 2. conflict
    logits = get_logits_last(model, build_conflict_prompt(ctx, q, opt))
    if not is_correct(logits, cor_id, rep_id):
        return False

    # 3. supportive
    logits = get_logits_last(model, build_supportive_prompt(q, opt, coa, ctx, ca))
    if not is_correct(logits, cor_id, rep_id):
        return False

    return True


def main():
    t0 = time.time()

    # ── 제외 목록 구축 ──────────────────────────────────────────────────────────
    with open(KC_CASES_PATH) as f:
        kc_cases = json.load(f)

    excluded_row_indices = {c["source_row_index"] for c in kc_cases}
    excluded_questions   = {c["question"].strip() for c in kc_cases}
    print(f"[exclude] {len(excluded_row_indices)} KC rows, {len(excluded_questions)} KC questions")

    # ── 모델 로딩 ──────────────────────────────────────────────────────────────
    model = load_model(device="cuda")

    # ── CB_qa 스트리밍 ─────────────────────────────────────────────────────────
    collected  = []
    rows_scanned = 0
    rows_skipped = 0
    rows_tested  = 0
    type_counts  = {t: 0 for t in CONFLICT_TYPES}
    type_tried   = {t: 0 for t in CONFLICT_TYPES}
    clean_fail   = 0

    print(f"\n[collect] scanning CB_qa (max {MAX_ROWS_SCAN} rows) ...")

    for raw_idx, row in enumerate(iter_jsonl(CB_QA_PATH)):
        if len(collected) >= TARGET_N:
            break
        if rows_scanned >= MAX_ROWS_SCAN:
            break

        rows_scanned += 1

        if raw_idx in excluded_row_indices:
            rows_skipped += 1
            continue

        options_list = row.get("options") or []
        if len(options_list) < 4:
            continue

        options_dict = {l: options_list[i] for i, l in enumerate("ABCD")}
        correct_opt = (row.get("correct_option") or "").strip().upper()
        replace_opt = (row.get("replace_option") or "").strip().upper()
        question    = (row.get("question") or "").strip()

        if correct_opt not in "ABCD" or replace_opt not in "ABCD":
            continue
        if not question:
            continue
        if question in excluded_questions:
            continue

        correct_answer  = option_from_letter(options_list, correct_opt)
        conflict_answer = option_from_letter(options_list, replace_opt)
        if not correct_answer or not conflict_answer:
            continue

        # clean pass (shared across conflict types for this row)
        clean_prompt = build_clean_prompt(question, options_dict)
        cor_id = model.tokenizer.encode(f" {correct_opt}",  add_special_tokens=False)[-1]
        rep_id = model.tokenizer.encode(f" {replace_opt}", add_special_tokens=False)[-1]

        clean_logits = get_logits_last(model, clean_prompt)
        if not is_correct(clean_logits, cor_id, rep_id):
            clean_fail += 1
            continue

        # per conflict type
        for conflict_type in CONFLICT_TYPES:
            if len(collected) >= TARGET_N:
                break

            conflict_context = build_conflict_context(row, conflict_type)
            if not conflict_context:
                continue

            case = {
                "id":                 f"cc_{raw_idx:06d}_{conflict_type}",
                "source_row_index":   raw_idx,
                "conflict_type":      conflict_type,
                "question":           question,
                "options":            options_dict,
                "correct_option":     correct_opt,
                "replace_option":     replace_opt,
                "correct_answer":     correct_answer,
                "conflict_answer":    conflict_answer,
                "conflict_context":   conflict_context,
                "conflict_correct":   True,
            }

            type_tried[conflict_type] += 1
            rows_tested += 1

            if check_case(model, case):
                collected.append(case)
                type_counts[conflict_type] += 1
                print(
                    f"  [+{len(collected):3d}] {case['id']}  "
                    f"type={conflict_type}  row={raw_idx}",
                    flush=True,
                )

        if rows_scanned % CHECKPOINT == 0:
            elapsed = time.time() - t0
            print(
                f"  [scan] {rows_scanned}/{MAX_ROWS_SCAN} rows, "
                f"{len(collected)}/{TARGET_N} collected, "
                f"{elapsed:.0f}s",
                flush=True,
            )

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "conflict_correct_cases.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    stats = {
        "collected":        len(collected),
        "target":           TARGET_N,
        "rows_scanned":     rows_scanned,
        "rows_skipped_kc":  rows_skipped,
        "rows_clean_fail":  clean_fail,
        "rows_tested":      rows_tested,
        "by_type":          type_counts,
        "tried_by_type":    type_tried,
        "elapsed_s":        round(time.time() - t0, 1),
    }

    with open(DATA_DIR / "conflict_correct_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[done] collected={len(collected)}, elapsed={time.time()-t0:.0f}s")
    print(f"       by_type={type_counts}")
    print(f"       saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check Knowledge Conflict (KC) rates across multiple HF causal language models.

This script uses the same 4-choice exact-match criterion as phase0_collect_kc.py:

  clean prompt    -> model should choose correct_option
  conflict prompt -> model should choose replace_option

It reports whether each model has enough KC cases on ConflictBank before running
the heavier mechanistic experiments.

Example:
  conda run -n apa python -u scripts/check_kc_rates_multi_model.py --max_rows 300

  conda run -n apa python -u scripts/check_kc_rates_multi_model.py \
    --models google/gemma-3-1b-it Qwen/Qwen3-4B \
    --max_rows 1000 --local_files_only
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoTokenizer = None


DATA_PATH = Path(
    "/home/jaeik/.cache/huggingface/hub/datasets--Warrieryes--CB_qa"
    "/snapshots/b3b12e1dba7097151312c29c13e2a76f47d726c9/QA_dataset.json"
)
OUT_DIR = Path("/home/jaeik/xaimi/results/kc_rate_cross_model")

CONFLICT_TYPES = ("misinformation", "temporal", "semantic")
LETTERS = "ABCD"
SEED = 42
MAX_SEQ_LEN = 2048

DEFAULT_MODELS = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it", 
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure ConflictBank KC rates for Gemma 3 and Qwen models."
    )
    p.add_argument("--input", type=Path, default=DATA_PATH)
    p.add_argument("--output", type=Path, default=OUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--max_rows", type=int, default=1000)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    p.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use models already available in the local HF cache.",
    )
    p.add_argument(
        "--save_cases",
        action="store_true",
        help="Save per-model KC case examples in addition to summary statistics.",
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Write per-model partial results every N raw rows.",
    )
    return p.parse_args()


def resolve_dtype(dtype_arg: str) -> Any:
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_clean_prompt(question: str, options: List[str]) -> str:
    lines = [f"Question: {question}"]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_conflict_prompt(context: str, question: str, options: List[str]) -> str:
    lines = ["Context:", context, "", f"Question: {question}"]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_context(row: Dict[str, Any], ctype: str) -> Optional[str]:
    if ctype == "misinformation":
        claim = (row.get("misinformation_conflict_claim") or "").strip()
        evidence = (row.get("misinformation_conflict_evidence_evidence") or "").strip()
    elif ctype == "temporal":
        claim = (row.get("temporal_conflict_claim") or "").strip()
        evidence = (row.get("temporal_conflict_evidence") or "").strip()
    else:
        claim = (row.get("semantic_conflict_claim") or "").strip()
        evidence = (row.get("semantic_conflict_evidence") or "").strip()

    if not evidence and not claim:
        return None
    if claim and claim not in evidence:
        return f"{claim}\n\n{evidence}"
    return evidence or claim


def setup_letter_variants(tokenizer: Any) -> Dict[str, List[int]]:
    variants: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids: List[int] = []
        for text in (f" {letter}", letter, f"{letter}."):
            toks = tokenizer.encode(text, add_special_tokens=False)
            if toks:
                ids.append(toks[-1])
        variants[letter] = list(dict.fromkeys(ids))
    return variants


def load_model_and_tokenizer(
    model_name: str,
    dtype: Any,
    device_map: str,
    local_files_only: bool,
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    if device_map == "cpu":
        kwargs["device_map"] = None
    else:
        kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception:
        if AutoModelForImageTextToText is None:
            raise
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    if device_map == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def mc_forward(
    model: Any,
    tokenizer: Any,
    prompt: str,
    letter_variants: Dict[str, List[int]],
    max_seq_len: int,
) -> Tuple[str, Dict[str, float]]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=max_seq_len,
        truncation=True,
    )
    dev = model_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    lp_all = torch.log_softmax(logits[0, -1].float(), dim=-1)

    scores: Dict[str, float] = {}
    for letter, token_ids in letter_variants.items():
        scores[letter] = max(lp_all[tid].item() for tid in token_ids)

    pred = max(scores, key=scores.get)
    return pred, scores


def init_stats() -> Dict[str, Any]:
    return {
        "rows_seen": 0,
        "valid_rows": 0,
        "clean_correct_rows": 0,
        "query_kc_rows": 0,
        "clean_accuracy_rows": 0.0,
        "conflict_trials": {t: 0 for t in CONFLICT_TYPES},
        "kc_cases": {t: 0 for t in CONFLICT_TYPES},
        "query_kc_by_type": {t: 0 for t in CONFLICT_TYPES},
        "query_kc_num_types": Counter(),
        "query_kc_type_combo": Counter(),
        "clean_correct_trials": {t: 0 for t in CONFLICT_TYPES},
        "skip_empty_context": {t: 0 for t in CONFLICT_TYPES},
        "severity": {t: [] for t in CONFLICT_TYPES},
        "clean_pred_dist": Counter(),
        "conflict_pred_dist": {t: Counter() for t in CONFLICT_TYPES},
    }


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    total_trials = sum(stats["conflict_trials"].values())
    total_kc = sum(stats["kc_cases"].values())
    valid_rows = stats["valid_rows"]
    clean_rows = stats["clean_correct_rows"]
    query_kc_rows = stats["query_kc_rows"]

    out = {
        "rows_seen": stats["rows_seen"],
        "valid_rows": valid_rows,
        "clean_correct_rows": clean_rows,
        "clean_accuracy_rows": clean_rows / valid_rows if valid_rows else None,
        "query_kc_rows": query_kc_rows,
        "query_kc_rate_over_valid_rows": (
            query_kc_rows / valid_rows if valid_rows else None
        ),
        "query_kc_rate_after_clean_correct_rows": (
            query_kc_rows / clean_rows if clean_rows else None
        ),
        "query_kc_num_types": dict(stats["query_kc_num_types"]),
        "query_kc_type_combo": dict(stats["query_kc_type_combo"]),
        "total_conflict_trials_after_clean_correct": total_trials,
        "total_kc_cases": total_kc,
        "kc_rate_after_clean_correct": total_kc / total_trials if total_trials else None,
        "kc_rate_per_valid_row_x_type": (
            total_kc / (valid_rows * len(CONFLICT_TYPES)) if valid_rows else None
        ),
        "by_type": {},
        "clean_pred_dist": dict(stats["clean_pred_dist"]),
        "conflict_pred_dist": {
            t: dict(stats["conflict_pred_dist"][t]) for t in CONFLICT_TYPES
        },
    }

    for t in CONFLICT_TYPES:
        tried = stats["conflict_trials"][t]
        kc = stats["kc_cases"][t]
        sev = stats["severity"][t]
        out["by_type"][t] = {
            "conflict_trials_after_clean_correct": tried,
            "kc_cases": kc,
            "kc_rate_after_clean_correct": kc / tried if tried else None,
            "query_kc_rows": stats["query_kc_by_type"][t],
            "query_kc_rate_over_valid_rows": (
                stats["query_kc_by_type"][t] / valid_rows if valid_rows else None
            ),
            "query_kc_rate_after_clean_correct_rows": (
                stats["query_kc_by_type"][t] / clean_rows if clean_rows else None
            ),
            "skip_empty_context": stats["skip_empty_context"][t],
            "severity": {
                "mean": float(np.mean(sev)) if sev else None,
                "median": float(np.median(sev)) if sev else None,
                "std": float(np.std(sev)) if sev else None,
                "min": float(np.min(sev)) if sev else None,
                "max": float(np.max(sev)) if sev else None,
            },
        }
    return out


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def evaluate_model(
    model_name: str,
    rows: List[Tuple[int, Dict[str, Any]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t0 = time.time()
    print(f"\n{'=' * 80}")
    print(f"[model] {model_name}")
    print(f"{'=' * 80}", flush=True)

    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    letter_variants = setup_letter_variants(tokenizer)
    print(f"[letters] {letter_variants}", flush=True)

    stats = init_stats()
    kc_examples: List[Dict[str, Any]] = []
    model_dir = args.output / safe_model_id(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, (row_idx, row) in enumerate(rows, start=1):
        stats["rows_seen"] += 1

        question = (row.get("question") or "").strip()
        options = row.get("options") or []
        correct = (row.get("correct_option") or "").strip().upper()
        replace = (row.get("replace_option") or "").strip().upper()

        if not question or len(options) < 4:
            continue
        if correct not in LETTERS or replace not in LETTERS:
            continue

        stats["valid_rows"] += 1
        clean_prompt = build_clean_prompt(question, options)
        clean_pred, _ = mc_forward(
            model, tokenizer, clean_prompt, letter_variants, args.max_seq_len
        )
        stats["clean_pred_dist"][clean_pred] += 1

        if clean_pred != correct:
            if i % args.checkpoint_every == 0:
                save_json(model_dir / "partial_summary.json", finalize_stats(stats))
            continue

        stats["clean_correct_rows"] += 1
        opt_dict = {l: o for l, o in zip(LETTERS, options)}
        row_kc_types: List[str] = []

        for ctype in CONFLICT_TYPES:
            context = build_context(row, ctype)
            if not context:
                stats["skip_empty_context"][ctype] += 1
                continue

            stats["conflict_trials"][ctype] += 1
            conflict_prompt = build_conflict_prompt(context, question, options)
            conflict_pred, lp_conf = mc_forward(
                model, tokenizer, conflict_prompt, letter_variants, args.max_seq_len
            )
            stats["conflict_pred_dist"][ctype][conflict_pred] += 1

            if conflict_pred == replace:
                severity = lp_conf[replace] - lp_conf[correct]
                stats["kc_cases"][ctype] += 1
                row_kc_types.append(ctype)
                stats["severity"][ctype].append(severity)
                if args.save_cases:
                    kc_examples.append(
                        {
                            "source_row_index": row_idx,
                            "conflict_type": ctype,
                            "question": question,
                            "options": opt_dict,
                            "correct_option": correct,
                            "replace_option": replace,
                            "correct_answer": opt_dict[correct],
                            "conflict_answer": opt_dict[replace],
                            "clean_pred": clean_pred,
                            "conflict_pred": conflict_pred,
                            "kc_severity": severity,
                            "conflict_context": context,
                        }
                    )

        if row_kc_types:
            stats["query_kc_rows"] += 1
            unique_types = sorted(set(row_kc_types))
            stats["query_kc_num_types"][str(len(unique_types))] += 1
            stats["query_kc_type_combo"]["+".join(unique_types)] += 1
            for ctype in unique_types:
                stats["query_kc_by_type"][ctype] += 1

        if i % args.checkpoint_every == 0:
            summary = finalize_stats(stats)
            save_json(model_dir / "partial_summary.json", summary)
            print(
                f"[{model_name}] rows={i}/{len(rows)} "
                f"clean={summary['clean_accuracy_rows']:.3f} "
                f"query_kc={summary['query_kc_rate_over_valid_rows']}",
                flush=True,
            )

    summary = finalize_stats(stats)
    summary["model"] = model_name
    summary["letter_variants"] = letter_variants
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    save_json(model_dir / "summary.json", summary)
    if args.save_cases:
        save_json(model_dir / "kc_examples.json", kc_examples)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def print_table(summaries: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 110)
    print("KC rate summary")
    print("=" * 110)
    print(
        f"{'model':42s} {'valid':>7s} {'clean':>8s} {'KC-q':>7s} "
        f"{'KC-q%':>8s} {'cond%':>8s} {'trials':>8s} {'KC/t%':>8s}"
    )
    for s in summaries:
        def pct(x: Optional[float]) -> str:
            return "n/a" if x is None else f"{x * 100:.1f}%"

        print(
            f"{s['model'][:42]:42s} "
            f"{s['valid_rows']:7d} "
            f"{pct(s['clean_accuracy_rows']):>8s} "
            f"{s['query_kc_rows']:7d} "
            f"{pct(s['query_kc_rate_over_valid_rows']):>8s} "
            f"{pct(s['query_kc_rate_after_clean_correct_rows']):>8s} "
            f"{s['total_conflict_trials_after_clean_correct']:8d} "
            f"{pct(s['kc_rate_after_clean_correct']):>8s}"
        )


def write_summary_tables(output_dir: Path, summaries: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        by = s["by_type"]
        rows.append(
            {
                "model": s["model"],
                "valid_rows": s["valid_rows"],
                "clean_correct_rows": s["clean_correct_rows"],
                "clean_accuracy_pct": (
                    round(s["clean_accuracy_rows"] * 100, 3)
                    if s["clean_accuracy_rows"] is not None
                    else None
                ),
                "query_kc_rows": s["query_kc_rows"],
                "query_kc_rate_pct": (
                    round(s["query_kc_rate_over_valid_rows"] * 100, 3)
                    if s["query_kc_rate_over_valid_rows"] is not None
                    else None
                ),
                "query_kc_rate_after_clean_correct_pct": (
                    round(s["query_kc_rate_after_clean_correct_rows"] * 100, 3)
                    if s["query_kc_rate_after_clean_correct_rows"] is not None
                    else None
                ),
                "misinformation_query_kc_rate_pct": (
                    round(by["misinformation"]["query_kc_rate_over_valid_rows"] * 100, 3)
                    if by["misinformation"]["query_kc_rate_over_valid_rows"] is not None
                    else None
                ),
                "temporal_query_kc_rate_pct": (
                    round(by["temporal"]["query_kc_rate_over_valid_rows"] * 100, 3)
                    if by["temporal"]["query_kc_rate_over_valid_rows"] is not None
                    else None
                ),
                "semantic_query_kc_rate_pct": (
                    round(by["semantic"]["query_kc_rate_over_valid_rows"] * 100, 3)
                    if by["semantic"]["query_kc_rate_over_valid_rows"] is not None
                    else None
                ),
                "total_conflict_trials_after_clean_correct": s[
                    "total_conflict_trials_after_clean_correct"
                ],
                "total_kc_cases": s["total_kc_cases"],
                "trial_kc_rate_after_clean_correct_pct": (
                    round(s["kc_rate_after_clean_correct"] * 100, 3)
                    if s["kc_rate_after_clean_correct"] is not None
                    else None
                ),
            }
        )

    if not rows:
        return

    csv_path = output_dir / "model_query_kc_rates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "model_query_kc_rates.md"
    headers = list(rows[0].keys())
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Model Query-Level KC Rates\n\n")
        f.write(
            "KC query = clean prompt is answered correctly, and at least one "
            "conflict context makes the model choose the replacement option.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


def main() -> None:
    args = parse_args()
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ModuleNotFoundError(
            "torch/transformers are required. Run this inside the experiment "
            "environment, e.g. `conda run -n apa python -u "
            "scripts/check_kc_rates_multi_model.py --max_rows 300`."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    rows = list(enumerate(iter_jsonl(args.input)))
    if args.start_index:
        rows = [(idx, row) for idx, row in rows if idx >= args.start_index]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_rows]
    print(f"[data] loaded {len(rows)} rows from {args.input}")

    summaries: List[Dict[str, Any]] = []
    failures: Dict[str, str] = {}
    for model_name in args.models:
        if "gemma-3-27b" in model_name.lower():
            print(f"[skip] excluding Gemma 3 27B model: {model_name}")
            continue
        try:
            summary = evaluate_model(model_name, rows, args)
            summaries.append(summary)
            save_json(args.output / "all_model_summary.json", summaries)
        except Exception as exc:
            failures[model_name] = repr(exc)
            print(f"[error] {model_name}: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "input": str(args.input),
        "max_rows": args.max_rows,
        "seed": args.seed,
        "models_requested": args.models,
        "summaries": summaries,
        "failures": failures,
    }
    save_json(args.output / "run_summary.json", payload)
    write_summary_tables(args.output, summaries)
    print_table(summaries)
    if failures:
        print("\nFailures:")
        for model_name, err in failures.items():
            print(f"  {model_name}: {err}")
    print(f"\n[done] saved to {args.output}")


if __name__ == "__main__":
    main()

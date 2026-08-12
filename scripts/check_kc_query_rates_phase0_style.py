#!/usr/bin/env python3
"""
Check query-level Knowledge Conflict (KC) rates across models using the same
letter-logit exact-match criterion as phase0_collect_kc.py.

KC definition for one query:
  1. Clean prompt predicts correct_option.
  2. At least one conflict context predicts replace_option.

This differs from phase0_collect_kc.py: it does not collect until 1000 KC cases.
Instead, it evaluates a fixed sample of N ConflictBank rows and reports what
percentage of queries show KC for each model.

Example:
  conda run -n apa python -u scripts/check_kc_query_rates_phase0_style.py \
    --max_rows 1000 --save_cases --local_files_only --offline
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


DATA_PATH = Path(
    "/home/jaeik/.cache/huggingface/hub/datasets--Warrieryes--CB_qa"
    "/snapshots/b3b12e1dba7097151312c29c13e2a76f47d726c9/QA_dataset.json"
)
OUT_DIR = Path("/home/jaeik/xaimi/results/kc_query_rates_phase0_style")

CONFLICT_TYPES = ("misinformation", "temporal", "semantic")
LETTERS = "ABCD"
SEED = 42
MAX_SEQ_LEN = 2048
INSTRUCT_TAGS = ("-it", "-instruct", "-chat", "qwen3")

DEFAULT_MODELS = [
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure query-level ConflictBank KC rates with the phase0 "
            "letter-logit exact-match criterion."
        )
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
        default="float16",
    )
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Force Transformers/HF Hub offline mode before loading models.",
    )
    p.add_argument("--save_cases", action="store_true")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument(
        "--sanity_rows",
        type=int,
        default=50,
        help="Check for one-letter prediction collapse every N valid rows.",
    )
    p.add_argument(
        "--max_letter_frac",
        type=float,
        default=0.85,
        help="Maximum allowed fraction for one clean-predicted letter at sanity check.",
    )
    p.add_argument(
        "--abort_on_sanity_fail",
        action="store_true",
        help="Abort when clean predictions collapse to one letter instead of warning.",
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


def is_instruct_model(model_name: str) -> bool:
    """Detect instruction-tuned models by common name conventions."""
    lower = model_name.lower()
    return any(tag in lower for tag in INSTRUCT_TAGS)


def build_chat_prompt(tokenizer: Any, user_content: str) -> Optional[str]:
    """Apply tokenizer chat template for instruction-tuned models.

    Tries enable_thinking=False first (for Qwen3) to suppress thinking tokens,
    then falls back to standard template. Returns None if template unavailable.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    messages = [{"role": "user", "content": user_content}]
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        print(f"[warn] apply_chat_template failed ({exc}); falling back to plain text", flush=True)
        return None


def setup_letter_variants(tokenizer: Any) -> Dict[str, List[int]]:
    variants: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids: List[int] = []
        for text in (f" {letter}", letter):
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

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
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
    add_special_tokens: bool = True,
) -> Tuple[str, Dict[str, float]]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
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
        "conflict_trials_after_clean_correct": {t: 0 for t in CONFLICT_TYPES},
        "kc_trials": {t: 0 for t in CONFLICT_TYPES},
        "query_kc_by_type": {t: 0 for t in CONFLICT_TYPES},
        "query_kc_num_types": Counter(),
        "query_kc_type_combo": Counter(),
        "skip_empty_context": {t: 0 for t in CONFLICT_TYPES},
        "clean_pred_dist": Counter(),
        "conflict_pred_dist": {t: Counter() for t in CONFLICT_TYPES},
        "severity": {t: [] for t in CONFLICT_TYPES},
    }


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    valid = stats["valid_rows"]
    clean = stats["clean_correct_rows"]
    query_kc = stats["query_kc_rows"]
    total_trials = sum(stats["conflict_trials_after_clean_correct"].values())
    total_kc_trials = sum(stats["kc_trials"].values())

    out: Dict[str, Any] = {
        "rows_seen": stats["rows_seen"],
        "valid_rows": valid,
        "clean_correct_rows": clean,
        "clean_accuracy": clean / valid if valid else None,
        "query_kc_rows": query_kc,
        "query_kc_rate_over_valid_rows": query_kc / valid if valid else None,
        "query_kc_rate_after_clean_correct": query_kc / clean if clean else None,
        "total_conflict_trials_after_clean_correct": total_trials,
        "total_kc_trials": total_kc_trials,
        "trial_kc_rate_after_clean_correct": (
            total_kc_trials / total_trials if total_trials else None
        ),
        "query_kc_num_types": dict(stats["query_kc_num_types"]),
        "query_kc_type_combo": dict(stats["query_kc_type_combo"]),
        "clean_pred_dist": dict(stats["clean_pred_dist"]),
        "conflict_pred_dist": {
            t: dict(stats["conflict_pred_dist"][t]) for t in CONFLICT_TYPES
        },
        "by_type": {},
    }

    for ctype in CONFLICT_TYPES:
        trials = stats["conflict_trials_after_clean_correct"][ctype]
        kc = stats["kc_trials"][ctype]
        sev = stats["severity"][ctype]
        out["by_type"][ctype] = {
            "conflict_trials_after_clean_correct": trials,
            "kc_trials": kc,
            "trial_kc_rate_after_clean_correct": kc / trials if trials else None,
            "query_kc_rows": stats["query_kc_by_type"][ctype],
            "query_kc_rate_over_valid_rows": (
                stats["query_kc_by_type"][ctype] / valid if valid else None
            ),
            "query_kc_rate_after_clean_correct": (
                stats["query_kc_by_type"][ctype] / clean if clean else None
            ),
            "skip_empty_context": stats["skip_empty_context"][ctype],
            "severity": {
                "mean": float(np.mean(sev)) if sev else None,
                "median": float(np.median(sev)) if sev else None,
                "std": float(np.std(sev)) if sev else None,
                "min": float(np.min(sev)) if sev else None,
                "max": float(np.max(sev)) if sev else None,
            },
        }
    return out


def add_run_metadata(
    summary: Dict[str, Any],
    model_name: str,
    args: argparse.Namespace,
    letter_variants: Dict[str, List[int]],
    t0: float,
) -> Dict[str, Any]:
    summary["model"] = model_name
    summary["criterion"] = "phase0_letter_logit_exact_match"
    summary["use_chat_template"] = is_instruct_model(model_name)
    summary["letter_variants"] = letter_variants
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    summary["run_metadata"] = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "output": str(args.output),
        "max_rows": args.max_rows,
        "seed": args.seed,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
        "offline": args.offline,
        "script_path": str(Path(__file__).resolve()),
        "script_mtime": Path(__file__).stat().st_mtime,
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else None,
        "cuda_device_count": torch.cuda.device_count() if torch is not None else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return summary


def maybe_warn_on_letter_collapse(
    stats: Dict[str, Any],
    model_name: str,
    args: argparse.Namespace,
) -> None:
    """Warn when predictions collapse; optionally abort if requested.

    Fires at valid_rows == sanity_rows and every sanity_rows interval thereafter,
    so collapse late in a run is also caught.
    """
    valid = stats["valid_rows"]
    if valid < args.sanity_rows or valid % args.sanity_rows != 0:
        return
    dist = stats["clean_pred_dist"]
    if not dist:
        return
    top_letter, top_count = max(dist.items(), key=lambda item: item[1])
    top_frac = top_count / valid
    if top_frac <= args.max_letter_frac:
        return
    msg = (
        f"[sanity] {model_name}: clean predictions collapsed to "
        f"'{top_letter}'={top_count}/{valid} ({top_frac:.1%}) — "
        "results may not be comparable to phase0."
    )
    print(msg, flush=True)
    if args.abort_on_sanity_fail:
        raise RuntimeError(msg)


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
    print(f"[letter variants] {letter_variants}", flush=True)

    use_chat = is_instruct_model(model_name)
    print(f"[prompt_mode] {'chat template' if use_chat else 'plain text'}", flush=True)

    stats = init_stats()
    kc_examples: List[Dict[str, Any]] = []
    model_dir = args.output / safe_model_id(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, (row_idx, row) in enumerate(rows, start=1):
        stats["rows_seen"] += 1
        question = (row.get("question") or "").strip()
        options_raw = row.get("options") or []
        # Handle both list and dict options formats from ConflictBank
        if isinstance(options_raw, dict):
            options: List[str] = [options_raw.get(letter, "") for letter in LETTERS]
        else:
            options = list(options_raw)
        correct = (row.get("correct_option") or "").strip().upper()
        replace = (row.get("replace_option") or "").strip().upper()

        if not question or len(options) < 4:
            continue
        if correct not in LETTERS or replace not in LETTERS:
            continue

        stats["valid_rows"] += 1

        clean_text = build_clean_prompt(question, options)
        if use_chat:
            formatted = build_chat_prompt(tokenizer, clean_text)
            clean_prompt = formatted if formatted is not None else clean_text
            clean_add_special = formatted is None
        else:
            clean_prompt = clean_text
            clean_add_special = True

        clean_pred, _ = mc_forward(
            model, tokenizer, clean_prompt, letter_variants, args.max_seq_len,
            add_special_tokens=clean_add_special,
        )
        stats["clean_pred_dist"][clean_pred] += 1

        if clean_pred != correct:
            if i % args.checkpoint_every == 0:
                partial = finalize_stats(stats)
                add_run_metadata(partial, model_name, args, letter_variants, t0)
                save_json(model_dir / "partial_summary.json", partial)
            maybe_warn_on_letter_collapse(stats, model_name, args)
            continue

        stats["clean_correct_rows"] += 1
        opt_dict = {letter: opt for letter, opt in zip(LETTERS, options)}
        row_kc_types: List[str] = []

        for ctype in CONFLICT_TYPES:
            context = build_context(row, ctype)
            if not context:
                stats["skip_empty_context"][ctype] += 1
                continue

            stats["conflict_trials_after_clean_correct"][ctype] += 1

            conflict_text = build_conflict_prompt(context, question, options)
            if use_chat:
                formatted = build_chat_prompt(tokenizer, conflict_text)
                conflict_prompt = formatted if formatted is not None else conflict_text
                conflict_add_special = formatted is None
            else:
                conflict_prompt = conflict_text
                conflict_add_special = True

            conflict_pred, lp_conf = mc_forward(
                model, tokenizer, conflict_prompt, letter_variants, args.max_seq_len,
                add_special_tokens=conflict_add_special,
            )
            stats["conflict_pred_dist"][ctype][conflict_pred] += 1

            if conflict_pred == replace:
                severity = lp_conf[replace] - lp_conf[correct]
                stats["kc_trials"][ctype] += 1
                stats["severity"][ctype].append(severity)
                row_kc_types.append(ctype)
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
            add_run_metadata(summary, model_name, args, letter_variants, t0)
            save_json(model_dir / "partial_summary.json", summary)
            print(
                f"[{model_name}] rows={i}/{len(rows)} "
                f"clean={format_pct(summary['clean_accuracy'])} "
                f"query_kc={format_pct(summary['query_kc_rate_over_valid_rows'])}",
                flush=True,
            )
        maybe_warn_on_letter_collapse(stats, model_name, args)

    summary = finalize_stats(stats)
    add_run_metadata(summary, model_name, args, letter_variants, t0)
    save_json(model_dir / "summary.json", summary)
    if args.save_cases:
        save_json(model_dir / "kc_examples.json", kc_examples)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def format_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def print_table(summaries: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print("Query-level KC rates with phase0 letter-logit criterion")
    print("=" * 112)
    print(
        f"{'model':42s} {'valid':>7s} {'clean':>8s} {'KC-q':>7s} "
        f"{'KC-q%':>8s} {'cond%':>8s} {'trial%':>8s} {'elapsed':>9s}"
    )
    for s in summaries:
        print(
            f"{s['model'][:42]:42s} "
            f"{s['valid_rows']:7d} "
            f"{format_pct(s['clean_accuracy']):>8s} "
            f"{s['query_kc_rows']:7d} "
            f"{format_pct(s['query_kc_rate_over_valid_rows']):>8s} "
            f"{format_pct(s['query_kc_rate_after_clean_correct']):>8s} "
            f"{format_pct(s['trial_kc_rate_after_clean_correct']):>8s} "
            f"{s['elapsed_sec']:9.1f}"
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
                "clean_accuracy_pct": pct_number(s["clean_accuracy"]),
                "query_kc_rows": s["query_kc_rows"],
                "query_kc_rate_pct": pct_number(s["query_kc_rate_over_valid_rows"]),
                "query_kc_rate_after_clean_correct_pct": pct_number(
                    s["query_kc_rate_after_clean_correct"]
                ),
                "misinformation_query_kc_rate_pct": pct_number(
                    by["misinformation"]["query_kc_rate_over_valid_rows"]
                ),
                "temporal_query_kc_rate_pct": pct_number(
                    by["temporal"]["query_kc_rate_over_valid_rows"]
                ),
                "semantic_query_kc_rate_pct": pct_number(
                    by["semantic"]["query_kc_rate_over_valid_rows"]
                ),
                "total_conflict_trials_after_clean_correct": s[
                    "total_conflict_trials_after_clean_correct"
                ],
                "total_kc_trials": s["total_kc_trials"],
                "trial_kc_rate_after_clean_correct_pct": pct_number(
                    s["trial_kc_rate_after_clean_correct"]
                ),
                "elapsed_sec": s["elapsed_sec"],
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
            "Criterion: same letter-logit exact-match filter as "
            "`phase0_collect_kc.py`. A KC query means the clean prompt predicts "
            "the correct option and at least one conflict context predicts the "
            "replacement option.\n\n"
        )
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


def pct_number(value: Optional[float]) -> Optional[float]:
    return round(value * 100, 3) if value is not None else None


def main() -> None:
    args = parse_args()
    if args.offline or args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ModuleNotFoundError(
            "torch/transformers are required. Run inside the experiment "
            "environment, e.g. `conda run -n apa python -u "
            "scripts/check_kc_query_rates_phase0_style.py --max_rows 1000`."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    rows = list(enumerate(iter_jsonl(args.input)))
    if args.start_index:
        rows = [(idx, row) for idx, row in rows if idx >= args.start_index]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_rows]
    print(f"[data] loaded {len(rows)} sampled rows from {args.input}")

    summaries: List[Dict[str, Any]] = []
    failures: Dict[str, str] = {}
    for model_name in args.models:
        if "gemma-3-27b" in model_name.lower() or "gemma-2-27b" in model_name.lower():
            print(f"[skip] excluding 27B model: {model_name}")
            continue
        try:
            summary = evaluate_model(model_name, rows, args)
            summaries.append(summary)
            save_json(args.output / "all_model_summary.json", summaries)
            write_summary_tables(args.output, summaries)
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
        "criterion": "phase0_letter_logit_exact_match",
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

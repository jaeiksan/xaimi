#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
CONFLICT_TYPES = ("misinformation", "temporal", "semantic")
DEFAULT_CB_QA_PATH = Path(
    "/home/jaeik/.cache/huggingface/hub/datasets--Warrieryes--CB_qa/snapshots/"
    "b3b12e1dba7097151312c29c13e2a76f47d726c9/QA_dataset.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare KC cases from ConflictBank CB_qa."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CB_QA_PATH,
        help="Path to ConflictBank CB_qa JSONL file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="conflictbank",
        help="Dataset name kept for AGENTS.md interface compatibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for KC cases.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=9,
        help="Maximum number of output KC cases.",
    )
    parser.add_argument(
        "--per_type",
        type=int,
        default=None,
        help="Optional cap per conflict type.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=list(CONFLICT_TYPES),
        choices=list(CONFLICT_TYPES),
        help="Conflict types to include.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting raw row offset.",
    )
    parser.add_argument(
        "--verify_model",
        type=str,
        default=None,
        help="Optional HF model name or local path for actual KC verification.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for verification model. Defaults to cuda if available else cpu.",
    )
    parser.add_argument(
        "--max_rows_to_scan",
        type=int,
        default=None,
        help="Optional cap on raw rows scanned before stopping.",
    )
    return parser.parse_args()


def option_from_letter(options: List[str], letter: str) -> Optional[str]:
    idx = LETTER_TO_INDEX.get((letter or "").strip().upper())
    if idx is None or idx >= len(options):
        return None
    return options[idx]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_conflict_context(row: Dict[str, Any], conflict_type: str) -> str:
    if conflict_type == "misinformation":
        evidence = row.get("misinformation_conflict_evidence_evidence", "").strip()
        claim = row.get("misinformation_conflict_claim", "").strip()
    elif conflict_type == "temporal":
        evidence = row.get("temporal_conflict_evidence", "").strip()
        claim = row.get("temporal_conflict_claim", "").strip()
    else:
        evidence = row.get("semantic_conflict_evidence", "").strip()
        claim = row.get("semantic_conflict_claim", "").strip()

    if claim and claim not in evidence:
        return f"{claim}\n\n{evidence}".strip()
    return evidence or claim


def convert_row(
    row: Dict[str, Any],
    raw_index: int,
    conflict_type: str,
) -> Optional[Dict[str, Any]]:
    options = row.get("options") or []
    parametric_answer = option_from_letter(options, row.get("correct_option", ""))
    conflict_answer = option_from_letter(options, row.get("replace_option", ""))
    conflict_context = build_conflict_context(row, conflict_type)

    if not parametric_answer or not conflict_answer or not conflict_context:
        return None

    return {
        "id": f"cb_{raw_index:06d}_{conflict_type}",
        "source_dataset": "Warrieryes/CB_qa",
        "source_row_index": raw_index,
        "conflict_type": conflict_type,
        "relation": row.get("relation"),
        "question": row.get("question", "").strip(),
        "parametric_answer": parametric_answer.strip(),
        "conflict_answer": conflict_answer.strip(),
        "conflict_context": conflict_context,
        "default_claim": row.get("default_claim", "").strip(),
        "conflict_claim": (
            row.get(f"{conflict_type}_conflict_claim", "").strip()
            if conflict_type != "misinformation"
            else row.get("misinformation_conflict_claim", "").strip()
        ),
        "options": options,
        "correct_option": row.get("correct_option"),
        "replace_option": row.get("replace_option"),
    }


def select_cases(
    rows: Iterable[Dict[str, Any]],
    types: List[str],
    n_samples: int,
    per_type: Optional[int],
    start_index: int,
    max_rows_to_scan: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts = {conflict_type: 0 for conflict_type in types}

    for raw_index, row in enumerate(rows):
        if raw_index < start_index:
            continue
        if max_rows_to_scan is not None and raw_index >= start_index + max_rows_to_scan:
            break

        for conflict_type in types:
            if per_type is not None and counts[conflict_type] >= per_type:
                continue
            case = convert_row(row, raw_index, conflict_type)
            if case is None:
                continue
            selected.append(case)
            counts[conflict_type] += 1
            if len(selected) >= n_samples:
                return selected

        if per_type is not None and all(counts[t] >= per_type for t in types):
            break

    return selected


def resolve_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_local_model_path(model_name: str) -> str:
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)

    if "/" not in model_name:
        return model_name

    org, repo = model_name.split("/", 1)
    base = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{org}--{repo}"
    )
    refs_main = base / "refs" / "main"
    snapshots_dir = base / "snapshots"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots_dir / revision
        if snapshot.exists():
            return str(snapshot)
    if snapshots_dir.exists():
        snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return model_name


def load_verifier(model_name: str, device: str) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_local_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def build_clean_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def build_conflict_prompt(conflict_context: str, question: str) -> str:
    return f"Context:\n{conflict_context.strip()}\n\nQuestion: {question.strip()}\nAnswer:"


def score_completion(tokenizer: Any, model: Any, prompt: str, completion: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ]
    completion_ids = tokenizer(
        " " + completion.strip(),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits

    prompt_len = prompt_ids.shape[1]
    completion_len = completion_ids.shape[1]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    total = 0.0
    for offset in range(completion_len):
        pos = prompt_len - 1 + offset
        token_id = input_ids[0, prompt_len + offset]
        total += log_probs[0, pos, token_id].item()
    return total


def verify_case(case: Dict[str, Any], tokenizer: Any, model: Any) -> Dict[str, Any]:
    clean_prompt = build_clean_prompt(case["question"])
    conflict_prompt = build_conflict_prompt(case["conflict_context"], case["question"])

    clean_param = score_completion(tokenizer, model, clean_prompt, case["parametric_answer"])
    clean_conflict = score_completion(
        tokenizer, model, clean_prompt, case["conflict_answer"]
    )
    conflict_param = score_completion(
        tokenizer, model, conflict_prompt, case["parametric_answer"]
    )
    conflict_conflict = score_completion(
        tokenizer, model, conflict_prompt, case["conflict_answer"]
    )

    passes = clean_param > clean_conflict and conflict_conflict > conflict_param
    return {
        "passes_kc_filter": passes,
        "clean_scores": {
            "parametric": clean_param,
            "conflict": clean_conflict,
        },
        "conflict_scores": {
            "parametric": conflict_param,
            "conflict": conflict_conflict,
        },
    }


def verify_cases(
    cases: List[Dict[str, Any]],
    model_name: str,
    device: str,
    n_samples: int,
    per_type: Optional[int],
) -> List[Dict[str, Any]]:
    tokenizer, model = load_verifier(model_name, device)
    verified: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    for idx, case in enumerate(cases, start=1):
        info = verify_case(case, tokenizer, model)
        case["verification"] = info
        if info["passes_kc_filter"]:
            conflict_type = case["conflict_type"]
            if per_type is not None and counts.get(conflict_type, 0) >= per_type:
                continue
            verified.append(case)
            counts[conflict_type] = counts.get(conflict_type, 0) + 1
            print(
                f"[pass {len(verified)}] {case['id']} "
                f"type={conflict_type} clean={info['clean_scores']} "
                f"conflict={info['conflict_scores']}"
            )
            if len(verified) >= n_samples:
                break
        elif idx <= 5:
            print(
                f"[fail] {case['id']} clean={info['clean_scores']} "
                f"conflict={info['conflict_scores']}"
            )

    return verified


def main() -> None:
    args = parse_args()
    if args.dataset.lower() != "conflictbank":
        raise ValueError("Only --dataset conflictbank is currently supported.")
    if not args.input.exists():
        raise FileNotFoundError(f"ConflictBank source file not found: {args.input}")

    rows = iter_jsonl(args.input)
    candidate_target = args.n_samples
    if args.verify_model:
        per_type_multiplier = len(args.types) if args.per_type else 1
        candidate_target = max(args.n_samples * 10, 30 * per_type_multiplier)

    selected = select_cases(
        rows=rows,
        types=args.types,
        n_samples=candidate_target,
        per_type=(None if args.verify_model else args.per_type),
        start_index=args.start_index,
        max_rows_to_scan=args.max_rows_to_scan,
    )
    if not selected:
        raise ValueError("No KC cases were selected from the source data.")

    if args.verify_model:
        device = resolve_device(args.device)
        print(f"[info] verifying with model={args.verify_model} on device={device}")
        selected = verify_cases(
            cases=selected,
            model_name=args.verify_model,
            device=device,
            n_samples=args.n_samples,
            per_type=args.per_type,
        )
        if not selected:
            raise ValueError("No cases passed the actual-model KC verification filter.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"[done] saved {len(selected)} KC cases to {args.output}")
    by_type: Dict[str, int] = {}
    for case in selected:
        by_type[case["conflict_type"]] = by_type.get(case["conflict_type"], 0) + 1
    print(f"[info] by_type={by_type}")


if __name__ == "__main__":
    main()

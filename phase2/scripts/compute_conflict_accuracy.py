#!/usr/bin/env python3
"""
Phase2 exp_grid 결과에 대해 true conflict accuracy 재계산.

- 기존 kc_suppression은 correct vs replace의 rank flip 기준
- 이 스크립트는 A/B/C/D 전체 선택지 중 argmax로 conflict accuracy를 계산

기본값:
- phase3_results.json 상위 3개 config 평가
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformer_lens import HookedTransformer

from exp_grid import (
    KC_PATH,
    OUT_DIR,
    CHECKPOINT_EVERY,
    build_conflict_prompt,
    make_fwd_hooks,
    setup_letter_variants,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(OUT_DIR / "phase3_results.json"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--config", action="append", default=None,
                   help="특정 config label만 평가. 여러 번 지정 가능")
    return p.parse_args()


def load_cases() -> list:
    return json.loads(KC_PATH.read_text())


def load_candidates():
    return torch.load(Path("phase2/results/exp1b_candidates/candidates_for_exp2.pt"),
                      map_location="cpu", weights_only=False)


def load_model_for_eval() -> HookedTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"모델 로드 ({device}, {dtype}) ...", flush=True)
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def serialize_layer_dir_pairs(raw_pairs) -> List[Tuple[int, str]]:
    return [(int(layer), str(dir_name)) for layer, dir_name in raw_pairs]


def run_choice_logprobs(model, tokens, option_letters: List[str], fwd_hooks=None):
    if fwd_hooks is None:
        ctx = torch.no_grad()
        with ctx:
            logits = model(tokens)
    else:
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
            logits = model(tokens)

    lp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    scores = {}
    for letter in option_letters:
        ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        ids2 = model.tokenizer.encode(letter, add_special_tokens=False)
        cand_ids = [ids[-1] if len(ids) > 1 else ids[0], ids2[-1] if len(ids2) > 1 else ids2[0]]
        cand_ids = list(dict.fromkeys(cand_ids))
        scores[letter] = max(lp[tid].item() for tid in cand_ids)
    pred = max(scores, key=scores.get)
    return pred, scores


def evaluate_conflict_accuracy(model, cases: list, fwd_hooks, label: str) -> Dict:
    n = len(cases)
    correct = 0
    t0 = time.time()

    for i, case in enumerate(cases):
        prompt = build_conflict_prompt(case["conflict_context"], case["question"], case["options"])
        tokens = model.to_tokens(prompt, prepend_bos=True)
        pred, _ = run_choice_logprobs(model, tokens, ["A", "B", "C", "D"], fwd_hooks=fwd_hooks)
        if pred == case["correct_option"]:
            correct += 1

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  [{label}] {i+1}/{n}  acc={correct/(i+1):.3f}  ({time.time()-t0:.0f}s)", flush=True)

    return {
        "conflict_accuracy": correct / n,
        "conflict_correct": correct,
        "n_cases": n,
    }


def main():
    args = parse_args()
    rows = json.loads(Path(args.input).read_text())
    if args.config is not None:
        wanted = set(args.config)
        rows = [r for r in rows if r["config"] in wanted]
    else:
        rows = rows[:args.top_k]

    print(f"[load] {len(rows)} configs from {args.input}", flush=True)
    cases = load_cases()
    cands = load_candidates()
    model = load_model_for_eval()
    setup_letter_variants(model.tokenizer)

    out_rows = []
    for idx, row in enumerate(rows, start=1):
        layer_dir_pairs = serialize_layer_dir_pairs(row["layer_dir_pairs"])
        label = row["config"]
        print(f"\n[{idx}/{len(rows)}] {label}", flush=True)
        fwd_hooks = make_fwd_hooks(layer_dir_pairs, cands)
        acc = evaluate_conflict_accuracy(model, cases, fwd_hooks, label)
        merged = {
            "config": label,
            "layer_dir_pairs": layer_dir_pairs,
            "kc_suppression": row.get("kc_suppression"),
            "clean_accuracy_pairwise": row.get("clean_accuracy"),
            **acc,
        }
        out_rows.append(merged)
        print(
            f"  -> conflict_acc={acc['conflict_accuracy']:.3f}  "
            f"kc_sup={row.get('kc_suppression'):.3f}  "
            f"clean_pairwise={row.get('clean_accuracy'):.3f}",
            flush=True,
        )

    out_path = OUT_DIR / "conflict_accuracy_check.json"
    out_path.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()

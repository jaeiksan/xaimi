#!/usr/bin/env python3
"""
Phase 0: 4지선다 Exact Match 방식으로 KC 케이스 수집

KC 정의:
  clean run  → 모델이 correct_option 선택
  conflict run → 모델이 replace_option 선택
  두 조건 모두 충족해야 KC 케이스

실행:
  conda run -n apa python scripts/phase0_collect_kc.py --test 20
  conda run -n apa python scripts/phase0_collect_kc.py
  conda run -n apa python scripts/phase0_collect_kc.py --resume
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 경로 / 상수 ───────────────────────────────────────────────────────────────
DATA_PATH = Path(
    "/home/jaeik/.cache/huggingface/hub/datasets--Warrieryes--CB_qa"
    "/snapshots/b3b12e1dba7097151312c29c13e2a76f47d726c9/QA_dataset.json"
)
OUTPUT_DIR   = Path("/home/jaeik/xaimi/data/phase0")
MODEL_NAME   = "google/gemma-2-2b"
SEED         = 42
MAX_SEQ_LEN  = 2048

CONFLICT_TYPES = ("misinformation", "temporal", "semantic")
TARGETS_FULL   = {"misinformation": 334, "temporal": 333, "semantic": 333}

# Gemma tokenizer 확인 결과:
#   " A" → [586], " B" → [599], " C" → [585], " D" → [608]  (space-prefixed, single token)
#   "A"  → [235280], "B" → [235305] 등 (no-space variant)
# "Answer:" 다음에는 space-prefixed 형태(" A")가 자연스러우므로 이를 primary로 사용.
# 두 variant 모두 체크해서 더 높은 log_prob을 선택 → 더 robust한 평가.
LETTER_VARIANTS: Dict[str, List[int]] = {}  # 런타임에 채워짐


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 0: 4지선다 Exact Match KC 케이스 수집"
    )
    p.add_argument("--test",   type=int,           help="N개만 수집 (테스트용)")
    p.add_argument("--resume", action="store_true", help="이전 진행 이어하기")
    return p.parse_args()


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────
def build_clean_prompt(question: str, options: List[str]) -> str:
    lines = [f"Question: {question}"]
    for letter, opt in zip("ABCD", options):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_conflict_prompt(context: str, question: str, options: List[str]) -> str:
    lines = ["Context:", context, "", f"Question: {question}"]
    for letter, opt in zip("ABCD", options):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_context(row: dict, ctype: str) -> Optional[str]:
    if ctype == "misinformation":
        claim    = (row.get("misinformation_conflict_claim") or "").strip()
        evidence = (row.get("misinformation_conflict_evidence_evidence") or "").strip()
    elif ctype == "temporal":
        claim    = (row.get("temporal_conflict_claim") or "").strip()
        evidence = (row.get("temporal_conflict_evidence") or "").strip()
    else:  # semantic
        claim    = (row.get("semantic_conflict_claim") or "").strip()
        evidence = (row.get("semantic_conflict_evidence") or "").strip()

    if not evidence and not claim:
        return None
    if claim and claim not in evidence:
        return f"{claim}\n\n{evidence}"
    return evidence or claim


# ── 토큰 ID 설정 ──────────────────────────────────────────────────────────────
def setup_letter_variants(tokenizer) -> None:
    """A/B/C/D 각 letter에 대해 space-prefix 및 no-space variant token ID를 수집."""
    global LETTER_VARIANTS
    print("\n=== Answer 토큰 ID 확인 ===")
    for letter in "ABCD":
        ids: List[int] = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
            else:
                # 멀티 토큰이면 마지막 토큰만 사용 (fallback)
                ids.append(toks[-1])
                print(f"  ⚠  '{text}' → multi-token {toks}, last 사용")
        # 중복 제거 (혹시 같은 id인 경우)
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
        decoded = [repr(tokenizer.decode([i])) for i in LETTER_VARIANTS[letter]]
        print(f"  '{letter}' variant ids: {LETTER_VARIANTS[letter]} → {decoded}")


# ── 추론 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mc_forward(
    model,
    tokenizer,
    prompt: str,
    device: str,
) -> Tuple[str, Dict[str, float]]:
    """
    프롬프트의 마지막 위치 logits에서 A/B/C/D log_prob을 계산.
    각 letter의 variant token id 중 최댓값을 해당 letter의 점수로 사용.
    greedy 선택 (argmax) → 예측 선택지 반환.

    Returns:
        (predicted_letter, {letter: best_log_prob})
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=MAX_SEQ_LEN,
        truncation=True,
    ).to(device)

    outputs  = model(**inputs)
    lp_all   = torch.log_softmax(outputs.logits[0, -1], dim=-1)  # [vocab_size]

    # 각 letter별 best log_prob (variant 중 max)
    lp: Dict[str, float] = {}
    for letter, ids in LETTER_VARIANTS.items():
        lp[letter] = max(lp_all[tid].item() for tid in ids)

    predicted = max(lp, key=lp.get)
    return predicted, lp


# ── 저장 헬퍼 ─────────────────────────────────────────────────────────────────
def save_checkpoint(
    kc_cases: List[dict],
    collected: Dict[str, int],
    processed_set: set,
    kc_path: Path,
    prog_path: Path,
) -> None:
    with kc_path.open("w", encoding="utf-8") as f:
        json.dump(kc_cases, f, indent=2, ensure_ascii=False)
    with prog_path.open("w") as f:
        json.dump(
            {
                "processed_row_indices": sorted(processed_set),
                "collected": dict(collected),
            },
            f, indent=2,
        )


# ── 진행 출력 ─────────────────────────────────────────────────────────────────
def print_progress(
    collected: Dict[str, int],
    total_target: int,
    rows_processed: int,
    stat_tried: Dict[str, int],
    stat_kc: Dict[str, int],
) -> None:
    total = sum(collected.values())
    rates = {
        t: f"{100 * stat_kc[t] / max(stat_tried[t], 1):.1f}%"
        for t in CONFLICT_TYPES
    }
    denom = rows_processed * 3
    eff   = f"{total} / {denom:,} = {100 * total / max(denom, 1):.1f}%"
    print(
        f"[수집 {total}/{total_target}] "
        f"mis: {collected['misinformation']} | "
        f"tmp: {collected['temporal']} | "
        f"sem: {collected['semantic']}\n"
        f" 처리 row: {rows_processed:,} | "
        f"KC 발생률: mis {rates['misinformation']} "
        f"tmp {rates['temporal']} "
        f"sem {rates['semantic']}\n"
        f" 실질 효율: {eff}"
    )


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 수집 목표
    if args.test:
        q, r = divmod(args.test, 3)
        targets: Dict[str, int] = {
            t: q + (1 if i < r else 0)
            for i, t in enumerate(CONFLICT_TYPES)
        }
    else:
        targets = dict(TARGETS_FULL)
    total_target = sum(targets.values())

    # ── 원본 데이터 로드 ──────────────────────────────────────────────────────
    print(f"원본 데이터 로딩: {DATA_PATH}")
    all_rows: Dict[int, dict] = {}
    with DATA_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                all_rows[i] = json.loads(line)
    print(f"  총 {len(all_rows):,}개 row 로드")

    shuffled_idx = list(all_rows.keys())
    random.Random(SEED).shuffle(shuffled_idx)
    print(f"  seed={SEED} 랜덤 셔플 완료")

    # ── 이어하기 ──────────────────────────────────────────────────────────────
    kc_path   = OUTPUT_DIR / "kc_cases_1000.json"
    prog_path = OUTPUT_DIR / "collection_progress.json"

    processed_set: set = set()
    collected: Dict[str, int] = {t: 0 for t in CONFLICT_TYPES}
    kc_cases:  List[dict] = []
    kc_id_ctr: int = 1

    if args.resume and prog_path.exists():
        with prog_path.open() as f:
            prog = json.load(f)
        processed_set = set(prog["processed_row_indices"])
        collected     = {t: prog["collected"].get(t, 0) for t in CONFLICT_TYPES}
        if kc_path.exists():
            with kc_path.open() as f:
                kc_cases = json.load(f)
        kc_id_ctr = len(kc_cases) + 1
        print(f"이어하기: {sum(collected.values())}/{total_target} 케이스 로드됨")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print(f"\n모델 로딩: {MODEL_NAME}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  로드 완료 (dtype=float16, device={device})")

    setup_letter_variants(tokenizer)

    # ── 샘플 프롬프트 / 응답 확인 ─────────────────────────────────────────────
    _verify_sample(model, tokenizer, all_rows, shuffled_idx, device)

    # ── 통계 추적 ─────────────────────────────────────────────────────────────
    stat_tried: Dict[str, int] = {t: 0 for t in CONFLICT_TYPES}
    stat_kc:    Dict[str, int] = {t: 0 for t in CONFLICT_TYPES}
    stat_skip:  Dict[str, int] = {t: 0 for t in CONFLICT_TYPES}  # context 빈 케이스
    rows_processed = len(processed_set)
    severities: List[float] = [c["kc_severity"] for c in kc_cases]

    print("\n" + "═" * 60)
    print(f" KC 케이스 수집 시작  (목표: {total_target}개)")
    print("═" * 60)

    # ── 수집 루프 ─────────────────────────────────────────────────────────────
    for row_idx in shuffled_idx:
        if sum(collected.values()) >= total_target:
            break
        if row_idx in processed_set:
            continue

        row = all_rows[row_idx]
        processed_set.add(row_idx)
        rows_processed += 1

        question       = (row.get("question") or "").strip()
        options        = row.get("options") or []
        correct_option = (row.get("correct_option") or "").strip().upper()
        replace_option = (row.get("replace_option") or "").strip().upper()

        if not question or len(options) < 4:
            continue
        if correct_option not in "ABCD" or replace_option not in "ABCD":
            continue

        # Clean run (세 타입 공통 → 한 번만 실행)
        clean_prompt  = build_clean_prompt(question, options)
        clean_pred, _ = mc_forward(model, tokenizer, clean_prompt, device)

        # 조건 1 실패 → 이 row는 어떤 type도 KC 불가
        if clean_pred != correct_option:
            continue

        opt_dict = {l: o for l, o in zip("ABCD", options)}

        for ctype in CONFLICT_TYPES:
            if sum(collected.values()) >= total_target:
                break
            if collected[ctype] >= targets[ctype]:
                continue

            context = build_context(row, ctype)
            if not context:
                stat_skip[ctype] += 1
                continue

            stat_tried[ctype] += 1

            conflict_prompt          = build_conflict_prompt(context, question, options)
            conflict_pred, lp_conf   = mc_forward(model, tokenizer, conflict_prompt, device)

            # 조건 2: 모델이 replace_option을 선택해야 KC
            if conflict_pred != replace_option:
                continue

            # KC severity: conflict run에서 replace - correct log_prob 차이
            severity = lp_conf[replace_option] - lp_conf[correct_option]
            severities.append(severity)
            stat_kc[ctype]    += 1
            collected[ctype]  += 1

            kc_cases.append({
                "id":                    f"kc_{kc_id_ctr:06d}_{ctype}",
                "source_row_index":      row_idx,
                "conflict_type":         ctype,
                "question":              question,
                "options":               opt_dict,
                "correct_option":        correct_option,
                "replace_option":        replace_option,
                "correct_answer":        opt_dict[correct_option],
                "conflict_answer":       opt_dict[replace_option],
                "conflict_context":      context,
                "clean_answer":          clean_pred,
                "conflict_answer_model": conflict_pred,
                "passes_kc_filter":      True,
                "kc_severity":           round(severity, 4),
            })
            kc_id_ctr += 1

            # 100개마다 중간 저장 + 진행 출력
            if sum(collected.values()) % 100 == 0:
                print_progress(collected, total_target, rows_processed, stat_tried, stat_kc)
                save_checkpoint(kc_cases, collected, processed_set, kc_path, prog_path)

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    save_checkpoint(kc_cases, collected, processed_set, kc_path, prog_path)

    # ── collection_stats.json ─────────────────────────────────────────────────
    sev = np.array(severities) if severities else np.zeros(1)
    stats_out = {
        "total_rows_processed": rows_processed,
        "total_cases_tried":    sum(stat_tried.values()),
        "kc_cases_collected":   sum(collected.values()),
        "kc_rate": {
            t: f"{100 * stat_kc[t] / max(stat_tried[t], 1):.1f}%"
            for t in CONFLICT_TYPES
        },
        "skip_empty_count": dict(stat_skip),
        "parse_fail_rate":  "0.0%",  # argmax는 항상 ABCD 중 하나 반환
        "severity_stats": {
            "mean":   round(float(sev.mean()),       4),
            "median": round(float(np.median(sev)),   4),
            "std":    round(float(sev.std()),         4),
            "min":    round(float(sev.min()),         4),
            "max":    round(float(sev.max()),         4),
        },
    }
    with (OUTPUT_DIR / "collection_stats.json").open("w") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)

    # ── 완료 메시지 ───────────────────────────────────────────────────────────
    r = stats_out["kc_rate"]
    print("\n" + "━" * 40)
    print(" Phase 0 완료")
    print("━" * 40)
    print(f" 수집 케이스: {sum(collected.values()):,}개")
    print(f" 총 처리 row: {rows_processed:,}개")
    print(f" KC 발생률: mis {r['misinformation']} / tmp {r['temporal']} / sem {r['semantic']}")
    print(f" 파싱 실패율: {stats_out['parse_fail_rate']}")
    print(f" 평균 severity: {stats_out['severity_stats']['mean']:.4f}")
    print(f" 저장: {kc_path}")
    print("━" * 40)


# ── 샘플 확인 헬퍼 ────────────────────────────────────────────────────────────
def _verify_sample(model, tokenizer, all_rows, shuffled_idx, device) -> None:
    """실행 전 프롬프트 구조 및 모델 응답을 눈으로 확인."""
    row = all_rows[shuffled_idx[0]]
    question = (row.get("question") or "").strip()
    options  = row.get("options") or []
    if not question or len(options) < 4:
        return

    prompt = build_clean_prompt(question, options)
    print("\n=== 샘플 Clean 프롬프트 ===")
    print(prompt)

    pred, lp = mc_forward(model, tokenizer, prompt, device)
    lp_str   = "  ".join(f"{l}={lp[l]:.3f}" for l in "ABCD")
    print("\n=== 샘플 모델 응답 ===")
    print(f"  예측: {pred}  ({lp_str})")
    print(f"  정답: {row.get('correct_option')}")

    # conflict 프롬프트도 하나 확인
    context = build_context(row, "misinformation")
    if context:
        conf_prompt = build_conflict_prompt(context, question, options)
        conf_pred, conf_lp = mc_forward(model, tokenizer, conf_prompt, device)
        conf_lp_str = "  ".join(f"{l}={conf_lp[l]:.3f}" for l in "ABCD")
        print("\n=== 샘플 Conflict 프롬프트 (misinformation) — 앞 200자 ===")
        print(conf_prompt[:200] + "...")
        print(f"\n=== 샘플 Conflict 모델 응답 ===")
        print(f"  예측: {conf_pred}  ({conf_lp_str})")
        print(f"  replace: {row.get('replace_option')}")
    print()


if __name__ == "__main__":
    main()

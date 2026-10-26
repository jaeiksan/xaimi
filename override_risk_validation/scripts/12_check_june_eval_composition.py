#!/usr/bin/env python3
"""Post-hoc audit of the June Phase 4 30-case supportive evaluation.

This recreates the original Exp3 30-case seed-42 subset and its B_gated_upk
configuration. It does not tune or create new data.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "phase4/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from evaluation_utils import option_logprobs_from_logits, predict_option
from exp_utils import build_clean_prompt, build_supportive_prompt, load_model
from intervention_methods import make_gate_reader_resid_post, make_upk_hook


SEED = 42
N_CASES = 30
LAYER = 17
ALPHA = 6.0
GAMMA = 2.0
TAU = 13.041983604431152
OUT_DIR = ROOT / "override_risk_validation/results"
OUT_REPORT = OUT_DIR / "june_eval_composition_check.md"
OUT_CASES = OUT_DIR / "june_eval_composition_per_case.jsonl"


@torch.no_grad()
def forward(model, prompt: str, options: dict[str, str], hooks=None) -> tuple[dict[str, float], str]:
    ids = model.to_tokens(prompt, prepend_bos=True)
    if hooks:
        with model.hooks(fwd_hooks=hooks):
            logits = model(ids)[0, -1, :].float()
    else:
        logits = model(ids)[0, -1, :].float()
    scores = option_logprobs_from_logits(logits, options)
    return scores, predict_option(scores)


def pair_correct(scores: dict[str, float], correct: str, replacement: str) -> bool:
    """Historical Phase 4 metric: correct option logit > replacement logit."""
    return scores[correct] > scores[replacement]


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}% ({numerator}/{denominator})"


def main() -> None:
    dev = json.loads((ROOT / "phase4/data/dev_200.json").read_text())
    subset_indices = sorted(random.Random(SEED).sample(range(len(dev)), N_CASES))
    cases = [dev[index] for index in subset_indices]

    direction = torch.load(ROOT / "phase4/vectors/direction_B_train.pt", map_location="cpu", weights_only=False)[LAYER].to("cuda")
    upk = torch.load(ROOT / "phase4/vectors/u_PK_train.pt", map_location="cpu", weights_only=False)[LAYER].to("cuda")
    model = load_model(device="cuda")

    records = []
    for local_index, (source_index, case) in enumerate(zip(subset_indices, cases)):
        q, opt = case["question"], case["options"]
        correct, replacement = case["correct_option"], case["replace_option"]
        clean_prompt = build_clean_prompt(q, opt)
        supportive_prompt = build_supportive_prompt(q, opt, case["correct_answer"], case["conflict_context"], case["conflict_answer"])
        clean_scores, clean_pred = forward(model, clean_prompt, opt)
        support_scores, support_pred = forward(model, supportive_prompt, opt)
        gate_store = [None]
        hooks = [
            (f"blocks.{LAYER}.hook_resid_post", make_gate_reader_resid_post(direction, TAU, GAMMA, gate_store)),
            (f"blocks.{LAYER}.hook_resid_post", make_upk_hook(gate_store, upk, ALPHA)),
        ]
        gated_scores, gated_pred = forward(model, supportive_prompt, opt, hooks)
        record = {
            "source_dev_index": source_index,
            "id": case["id"],
            "question": q,
            "correct_option": correct,
            "replace_option": replacement,
            "clean_prediction": clean_pred,
            "supportive_prediction": support_pred,
            "gated_supportive_prediction": gated_pred,
            "clean_full_choice_correct": clean_pred == correct,
            "supportive_full_choice_correct": support_pred == correct,
            "gated_supportive_full_choice_correct": gated_pred == correct,
            "clean_historical_pair_correct": pair_correct(clean_scores, correct, replacement),
            "supportive_historical_pair_correct": pair_correct(support_scores, correct, replacement),
            "gated_supportive_historical_pair_correct": pair_correct(gated_scores, correct, replacement),
            "gate_value": float(gate_store[0][0].item()),
        }
        record["beneficial_override_type_full_choice"] = (not record["clean_full_choice_correct"] and record["supportive_full_choice_correct"])
        record["beneficial_override_type_historical_metric"] = (not record["clean_historical_pair_correct"] and record["supportive_historical_pair_correct"])
        records.append(record)
        print(f"[audit] {local_index + 1}/{N_CASES}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CASES.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Report the historical pair metric as the primary comparison because that
    # is what generated the June 0.833 supportive accuracy. Full-choice values
    # are included as an audit cross-check.
    b_hist = [r for r in records if r["beneficial_override_type_historical_metric"]]
    b_full = [r for r in records if r["beneficial_override_type_full_choice"]]
    hist_no = sum(r["supportive_historical_pair_correct"] for r in records)
    hist_gate = sum(r["gated_supportive_historical_pair_correct"] for r in records)
    full_no = sum(r["supportive_full_choice_correct"] for r in records)
    full_gate = sum(r["gated_supportive_full_choice_correct"] for r in records)
    b_no = sum(r["supportive_historical_pair_correct"] for r in b_hist)
    b_gate = sum(r["gated_supportive_historical_pair_correct"] for r in b_hist)
    full_b_no = sum(r["supportive_full_choice_correct"] for r in b_full)
    full_b_gate = sum(r["gated_supportive_full_choice_correct"] for r in b_full)

    lines = [
        "# 6월 Phase 4 일반 QA 평가셋 구성 사후 점검",
        "",
        "## 확인한 원래 평가",
        "",
        "6월의 ‘일반 QA 성능 80%대’는 `phase4/results/exp3_comparison/best_configs.json`의 `B_gated_upk` 결과를 가리킨다.",
        "해당 실험은 `dev_200.json`에서 seed 42로 뽑은 30-case subset을 사용했고, 일반 QA는 `supportive` 조건(정답을 지지하는 합성 context)에서의 historical two-option metric으로 측정했다.",
        "원래 frozen configuration은 L17, alpha=6.0, gamma=2.0, k=1.0, tau=13.041983604431152이다.",
        "",
        "`supportive` context는 원 conflict context에서 conflict answer 문자열을 correct answer 문자열로 치환해 만든 합성 context다. 따라서 이 평가셋은 처음부터 clean에서 정답인 KC 사례를 중심으로 구성됐으며, 실제 최신 사실에 의해 outdated 지식이 갱신되는 TempLAMA식 beneficial override 평가와는 다르다.",
        "",
        "## 핵심 결과: historical Phase 4 metric",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 전체 평가셋 크기 | {N_CASES} |",
        f"| beneficial-override형 문항 수 (closed-book pair 오류 → supportive pair 정답) | {len(b_hist)} |",
        f"| beneficial-override형 문항 비율 | {pct(len(b_hist), N_CASES)} |",
        f"| 해당 문항에서 no-intervention 정확도 | {pct(b_no, len(b_hist)) if b_hist else '해당 없음'} |",
        f"| 해당 문항에서 B-gated intervention 정확도 | {pct(b_gate, len(b_hist)) if b_hist else '해당 없음'} |",
        f"| 전체 평가셋에서 no-intervention 정확도 | {pct(hist_no, N_CASES)} |",
        f"| 전체 평가셋에서 B-gated intervention 정확도 | {pct(hist_gate, N_CASES)} |",
        "",
        "## Full-choice 교차 확인",
        "",
        "원래 6월 지표는 correct option과 replacement option의 두 logit만 비교했다. 전체 네 선택지 중 정답을 고르는 full-choice 기준으로도 같은 구성을 다시 계산했다.",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| beneficial-override형 문항 수 (closed-book full-choice 오류 → supportive full-choice 정답) | {len(b_full)} |",
        f"| beneficial-override형 문항 비율 | {pct(len(b_full), N_CASES)} |",
        f"| 해당 문항에서 no-intervention 정확도 | {pct(full_b_no, len(b_full)) if b_full else '해당 없음'} |",
        f"| 해당 문항에서 B-gated intervention 정확도 | {pct(full_b_gate, len(b_full)) if b_full else '해당 없음'} |",
        f"| 전체 평가셋에서 no-intervention 정확도 | {pct(full_no, N_CASES)} |",
        f"| 전체 평가셋에서 B-gated intervention 정확도 | {pct(full_gate, N_CASES)} |",
        "",
        "## 해석",
        "",
        "위 primary 표의 beneficial-override형 문항이 0개라면, 6월 supportive accuracy는 이 유형의 failure를 원천적으로 탐지할 수 없었다. 0개보다 크더라도 표본 수가 매우 작으면 전체 30-case 평균에 묻힐 수 있다.",
        "",
        "이번 TempLAMA 조건 ③은 실제 temporal old/new answer와 M→C 행동을 명시적으로 요구해 300개를 구성했다. 따라서 6월의 합성 supportive 평가와 달리 beneficial update 보존을 직접 측정한다.",
        "",
        "## 재현 파일",
        "",
        "- 원래 결과: `phase4/results/exp3_comparison/best_configs.json`",
        "- 원래 평가 코드: `phase4/scripts/exp3_intervention_comparison.py`",
        "- 원래 dev 데이터: `phase4/data/dev_200.json`",
        "- 본 사후 점검 코드: `override_risk_validation/scripts/12_check_june_eval_composition.py`",
        "- case별 audit: `results/june_eval_composition_per_case.jsonl`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {OUT_REPORT}")


if __name__ == "__main__":
    main()

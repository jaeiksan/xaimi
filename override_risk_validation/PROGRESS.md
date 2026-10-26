# Override-Risk Validation — 진행 기록

이 파일은 실험의 결정, 완료 작업, 실행 중인 작업, 차단 요인을 한국 표준시(KST) 기준으로 기록한다.

## 2026-08-12 16:55 KST — 연구 구조와 기존 파이프라인 고정

- 기존 `phase1/`~`phase4/`는 재구성하지 않고, 후속 검증 실험만 `override_risk_validation/`에 분리했다.
- 기존 Phase 4 전체-dev V2의 고정 설정을 [`notes_existing_pipeline.md`](notes_existing_pipeline.md)에 기록했다.
  - 모델: `google/gemma-2-2b`
  - gate: B direction
  - 개입 위치: L17 `hook_resid_post`
  - `alpha=6.0`, `gamma=10.0`, `tau=19.256011962890625`
- 이번 검증 데이터에서는 위 direction, layer, threshold, hyperparameter를 재튜닝하지 않는다.
- 기존 Phase 0의 `kc_cases_1000.json`은 조건 ④ natural harmful override 데이터로 유지한다.

## 2026-08-12 17:00 KST — 조건 ③ 데이터 소스 확정

- ConflictBank temporal은 실제 최신 사실을 보장하지 않으므로 조건 ③ factual beneficial override에는 사용하지 않기로 했다.
- 조건 ③의 1차 소스로 TempLAMA를 확정했다.
- TempLAMA의 old value를 M 후보, `most_recent_answer`를 C 후보로 사용하고, 모델 행동 필터를 적용하기로 했다.
  - closed-book 예측 = M
  - newer-value context 예측 = C
  - 두 조건을 모두 만족할 때만 최종 조건 ③으로 채택

## 2026-08-12 17:05 KST — TempLAMA 후보 및 조건 ③ 확정

- 공식 TempLAMA test JSONL을 `data/raw/templama_test.json`에 받았다.
- [`scripts/02_build_condition3_templama.py`](scripts/02_build_condition3_templama.py)로 old/new 값이 다른 후보 3,232개를 생성했다.
- 기존 `apa` Conda 환경(Python 3.10.19, torch 2.9.1+cu128, TransformerLens)을 사용해 GPU 0(A100 80GB)에서 전체 screening을 완료했다.
- screening 결과 1,134개가 M → C behavioural filter를 통과했다. 사후 검증에서 규칙 위반은 0개였다.
- [`scripts/04_finalize_condition3_split.py`](scripts/04_finalize_condition3_split.py)로 relation 비율을 보존해 300개를 seed 42로 고정했다.
- 최종 조건 ③ 파일: `data/processed/cond3_beneficial_override_300.jsonl`

## 2026-08-12 17:10 KST — 조건 ①·② 첫 번째 screening 완료

- 로컬 Hugging Face 캐시에 ConflictBank `CB_qa` 원본 553,117개가 이미 있음을 확인했다.
- 조건 ③ 및 조건 ④ 질문과 중복되지 않는 ConflictBank 후보 5,000개를 생성했다.
- 같은 질문에 대해 closed-book view와 fact-aligned supportive-context view를 모두 평가했다.
  - 조건 ①: 두 view 모두 정답
  - 조건 ②: 두 view 모두 오답
- 첫 5,000개 결과:
  - 조건 ①: 1,603개
  - 조건 ②: 18개
  - 나머지 mixed behaviour: 3,379개
- 조건 ②는 의도적으로 엄격한 정의 때문에 희소함(0.36%)을 확인했다.

## 2026-08-12 17:15 KST — 조건 ② 확보를 위한 추가 screening 전환

- 기존 5,000개와 질문이 겹치지 않는 ConflictBank 후보 25,000개를 추가 생성했다.
- 대규모 GPU 실행은 사용자가 직접 수행하도록 방식을 전환했다. 제가 시작한 GPU 0 작업은
  체크포인트만 남긴 채 중지했다.
- `scripts/05_screen_conditions12_conflictbank.py`에 `--start-index`와 `--limit`을 추가해
  두 GPU가 서로 다른 후보를 안전하게 처리할 수 있게 했다.
- `scripts/06_merge_condition12_shards.py`를 추가했다. shard 간 ID·질문 중복을 검증한 뒤
  조건 ①·② 결과를 병합한다.
- 정확한 GPU 0·1 분할 실행 및 병합 커맨드는 `README.md`에 기록했다.

## 2026-08-13 09:00 KST — 추가 GPU screening 및 병합 완료

- GPU 0·1에서 실행한 두 shard가 각각 12,500개씩, 총 25,000개를 모두 완료했다.
- `scripts/06_merge_condition12_shards.py`로 shard 간 ID·질문 중복 검증을 통과한 뒤 병합했다.
- 추가 블록 결과:
  - 조건 ①: 8,963개
  - 조건 ②: 87개
- 첫 5,000개 block과는 질문이 겹치지 않는다. 다음 단계에서는 두 block을 통합하고,
  조건 ①·②의 최종 고정 평가 표본을 구성한다.

## 2026-08-13 11:01 KST — 조건 ①·②·③·④ 최종 평가 세트 고정

- `scripts/07_finalize_evaluation_sets.py`를 추가해 조건별 screening 결과를 통합하고,
  seed 42 기반 평가 세트를 고정했다.
- 최종 구성:
  - 조건 ① paired safe context: 300개
  - 조건 ② paired persistent error: 105개 (현재 확보된 전수)
  - 조건 ③ TempLAMA beneficial override: 300개
  - 조건 ④ Phase 4 dev 기반 natural harmful override: 68개
- 조건 ④는 Phase 4 dev에 동일 질문의 conflict-type variant가 있어, 질문 단위 독립성을
  보장하도록 질문당 하나의 case만 deterministic하게 선택했다.
- 네 조건은 질문 단위로 서로 겹치지 않으며, 조건 ①·②·③은 Phase 4 vector-training
  질문과도 겹치지 않음을 자동 검증했다.
- manifest: `data/processed/final_evaluation_manifest.json`

## 2026-08-13 11:10 KST — 실험 B 코드 작성 및 소규모 검증 완료

- `scripts/08_collect_experiment_b.py`를 추가했다. 개입 없이 각 prompt view에서 다음을
  수집한다.
  - L17 B-direction projection
  - frozen gate value 및 activation 여부
  - 전체 4지선다 기반 prediction/accuracy
- 조건 ①·②는 closed-book/context paired view를 모두 수집하고, 조건 ③은 M→C 검증용
  closed-book/context view, 조건 ④는 harmful conflict context view를 수집한다.
- `scripts/09_analyze_experiment_b.py`를 추가했다. 조건별 summary CSV, 주 비교
  (④ context vs ① context) 및 보조 비교의 AUROC, 1,000회 bootstrap 95% CI를 계산한다.
- GPU 0에서 5개 prompt view pilot을 완료했고, activation·projection·gate·full-choice
  출력 형식이 정상임을 확인했다.
- 전체 실험 B 수집(1,478 prompt view)은 사용자가 README의 명령으로 실행한다. 중단 시
  `--resume`으로 checkpoint 이후부터 재개할 수 있다.

## 2026-08-13 11:15 KST — 실험 B 수집·분석 완료

- 전체 1,478 prompt view 수집과 summary/AUROC/bootstrap 분석이 정상 완료됐다.
- 주요 결과:
  - 조건 ④ harmful context vs 조건 ①b safe context: AUROC 0.752,
    95% CI [0.690, 0.812], projection 평균 차이 +34.45,
    95% CI [+26.18, +41.79].
  - 조건 ④ harmful context vs 조건 ③ beneficial context: AUROC 0.821,
    95% CI [0.746, 0.888], projection 평균 차이 +20.93,
    95% CI [+13.25, +27.57].
  - 조건 ② closed-book error vs 조건 ①a closed-book correct: AUROC 0.487,
    95% CI [0.425, 0.554]. 단순 closed-book 오답에는 신호가 반응하지 않는 결과다.
  - 조건 ② supportive-context error vs 조건 ①b safe context: AUROC 0.549,
    95% CI [0.497, 0.599], 평균 차이 +10.28, 95% CI [+4.24, +16.18].
- 해석: B direction은 harmful override를 safe/beneficial context보다 구분하지만,
  context가 있는데도 계속 오답인 사례에서 약한 양의 반응도 보인다. 따라서 현재 결과는
  "harmful-override 위험 신호"에 유리하지만 "오답과 완전히 무관한 순수 harmful signal"이라는
  가장 강한 주장은 아직 지지하지 않는다.
- 결과 파일: `results/exp_B_per_case.jsonl`, `results/exp_B_specificity.csv`,
  `results/exp_B_auroc.csv`.

## 2026-08-13 11:25 KST — 실험 A 코드 작성 및 pilot 완료

- `scripts/10_run_experiment_a.py`를 추가했다. 조건 ③ 300개에 대해 개입 없음,
  ungated restoration, 고정 L17 B-gated restoration을 case별 paired하게 평가한다.
- `scripts/11_analyze_experiment_a.py`를 추가했다. beneficial override 성공률, raw
  projection, gate value/activation 및 no-intervention 대비 paired bootstrap 95% CI를 기록한다.
- GPU 0에서 5개 사례 pilot(총 15 forward pass)을 완료했다. 코드·출력·paired 분석은 정상이다.
- pilot의 다섯 사례는 모두 gate가 활성화됐고 gated/ungated arm 모두 beneficial override를
  막았다. 이는 표본이 매우 작아 최종 결론은 아니지만, full run에서 확인해야 할 중요한
  negative-result 가능성을 보여준다.
- 전체 300-case run은 README의 명령으로 사용자가 실행한다. 중단 시 `--resume`을 사용한다.

## 2026-08-13 11:35 KST — 실험 A 전체 실행 및 negative result 확정

- 조건 ③ 300개 × 3 intervention arm(총 900 forward pass)의 실행과 paired-bootstrap
  분석이 정상 완료됐다.
- beneficial override 성공률:
  - 개입 없음: 100.0%
  - ungated restoration: 15.3% (개입 없음 대비 -84.7%p, 95% CI [-88.7, -80.3])
  - fixed L17 B-gated restoration: 33.7% (개입 없음 대비 -66.3%p,
    95% CI [-71.7, -61.0])
- gated 방식은 ungated보다 18.3%p 더 많은 beneficial override를 보존했지만, 사전
  성공 기준인 "개입 없음과 통계적으로 차이 없음"을 명확히 충족하지 못했다.
- 원인 확인: 300개 중 199개(66.3%)에서 gate가 활성화됐다. 활성화된 199개에서 gated
  beneficial override 성공률은 0.5%(1/199)였고, 비활성 101개에서는 99.0%(100/101)였다.
- 해석: 현 frozen L17 B gate는 harmful override와 beneficial override를 충분히
  구분하지 못한다. 즉 현재 설정은 harmful-risk-only gate가 아니라, 상당수 context-induced
  override를 감지하고 parametric restoration을 적용하는 gate에 가깝다.
- 결과 파일: `results/exp_A_per_case.jsonl`,
  `results/exp_A_beneficial_override.csv`.

## 2026-08-13 11:40 KST — 6월 Phase 4 일반 QA 평가셋 구성 사후 점검

- 6월의 “일반 QA 성능 80%대”는 `phase4/results/exp3_comparison/best_configs.json`의
  `B_gated_upk` 결과이며, `dev_200.json`에서 seed 42로 뽑은 30문항의 `supportive`
  조건을 historical two-option metric으로 평가한 것이다.
- 당시 실제 B-gated 설정(L17, alpha=6.0, gamma=2.0, k=1.0,
  tau=13.041983604431152)을 그대로 재현했다. `supportive` context는 conflict
  context의 오답 문자열을 정답 문자열로 치환한 합성 context임도 확인했다.
- closed-book 오답 → supportive-context 정답으로 바뀌는 beneficial-override형 문항은
  원래 two-option 지표와 네 선택지 full-choice 교차 확인 모두에서 **0/30 (0.0%)**였다.
- 따라서 6월의 일반 QA/supportive 정확도는 beneficial override 실패를 탐지할 수 없는
  구성이다. 이 평가는 대부분 closed-book에서 이미 정답인 KC 사례의 context 지지 여부를
  측정했으며, 실제 temporal old→new 갱신을 요구하는 TempLAMA 조건 ③과 다르다.
- 전체 정확도는 원래 historical two-option 기준으로 no-intervention 28/30 (93.3%),
  B-gated 25/30 (83.3%)였다. full-choice 기준 B-gated는 16/30 (53.3%)으로 더 낮았는데,
  historical 지표가 정답과 replacement 두 선택지만 비교해 제3의 오답 선택을 놓칠 수
  있음을 확인했다.
- 상세 보고서: `results/june_eval_composition_check.md`; case별 audit:
  `results/june_eval_composition_per_case.jsonl`; 재현 코드:
  `scripts/12_check_june_eval_composition.py`.

# XAIMI 연구/실험 전체 정리

작성 기준: `/home/jaeik/xaimi`의 현재 코드와 `results/` 산출물  
주 모델: `google/gemma-2-2b`  
주 데이터: `data/phase0/kc_cases_1000.json`  
핵심 연구 질문: **Knowledge Conflict(KC)가 모델 내부에서 어떤 signal/vector/circuit으로 나타나며, 이를 선택적으로 제거해 KC를 완화할 수 있는가?**

---

## 0. 전체 연구 흐름

현재 프로젝트는 크게 4단계로 진행되었다.

```text
1. Pilot / Observation & Diagnosis
   - ConflictBank에서 KC case 구축
   - 모델이 실제로 conflict context에 끌려가는지 확인
   - tuned lens, IG, causal tracing, path patching, head ablation으로 어디서/무엇을 보는지 진단

2. KC 관련 signal/vector 추출
   - conflict-clean / conflict-supportive residual 차이 분석
   - SVD, mean difference, cPCA, regression 등으로 방향 후보 A/B/C/D 추출
   - 후보 방향의 구조, 안정성, type 간 일관성 확인

3. Causal intervention
   - 단일 vector 제거
   - layer-wise IE 측정
   - double dissociation으로 KC-specificity 확인
   - 단일 vector 제거가 rank flip으로 이어지는지 평가

4. Multi-vector / multi-layer intervention
   - 단일 방향 개입의 한계를 확인
   - 방향 × layer grid search
   - A/B/C 혼합 방향 조합으로 KC suppression >30%, clean >80% 목표에 접근
```

현재까지의 핵심 결론은 다음이다.

```text
KC는 하나의 단일 KC-specific vector로 완전히 설명되지 않는다.
초기에는 context-following 방향(A)이 강하게 나타나지만,
rank flip 수준의 KC 억제를 위해서는 후반 layer의 conflict-supportive 방향(B)과
L17의 cPCA 방향(C)을 조합해야 한다.
```

---

## 1. 데이터 구축 및 모델별 KC 발생률 검증

### 1.1 Phase0: KC case 수집

관련 코드:

- `scripts/phase0_collect_kc.py`
- `scripts/prepare_data.py`

결과:

- `data/phase0/kc_cases_1000.json`
- `data/phase0/collection_stats.json`

수집 결과:

| 항목 | 값 |
|---|---:|
| processed rows | 900 |
| tried cases | 1010 |
| collected KC cases | 1000 |
| misinformation KC rate | 97.9% |
| temporal KC rate | 99.7% |
| semantic KC rate | 99.4% |
| severity mean | 3.1669 |
| severity median | 3.2197 |

여기서 KC는 대략 다음 조건으로 수집되었다.

```text
clean prompt에서는 parametric/correct answer를 맞힘
conflict context가 주어지면 replace/conflict answer 쪽으로 답이 뒤집힘
```

즉 이후 실험은 이미 강한 KC가 일어난 1000개 case를 기반으로 한다.

### 1.2 여러 모델에서 KC가 실제로 충분히 발생하는가

관련 코드:

- `scripts/check_kc_query_rates_phase0_style.py`
- `scripts/check_kc_rates_multi_model.py`

결과:

- `results/kc_rates_*`
- `MODEL_KC_RATES_TABLE.md`
- `MODEL_KC_RATES_TABLE.csv`

핵심 지표는 `Query KC / clean-correct`이다. 이는 clean에서 정답을 맞힌 query만 놓고 conflict context에서 replace option으로 뒤집힌 비율이다.

| Model | Clean acc | Query KC / clean-correct | 해석 |
|---|---:|---:|---|
| `google/gemma-2-2b` | 37.4% | 99.7% | 주 실험 모델 |
| `google/gemma-2-9b` | 46.5% | 100.0% | 강한 재현 |
| `google/gemma-3-1b-it` | 29.2% | 99.0% | 가능 |
| `Qwen/Qwen2.5-3B` | 50.2% | 99.8% | 강한 재현 |
| `Qwen/Qwen2.5-7B` | 49.4% | 99.8% | 강한 재현 |
| `Qwen/Qwen3-4B` | 35.9% | 99.7% | 강한 재현 |
| `Qwen/Qwen3-8B` | 26.1% | 94.3% | 가능, 약간 A 쏠림 |

주의:

| Model | 문제 |
|---|---|
| `gemma-2-2b-it` | clean acc 13.4%, 보조 결과로만 적합 |
| `gemma-2-9b-it` | clean acc 19.9%, 보조 결과로만 적합 |
| `gemma-3-4b-it` | A-collapse, 현재 letter-logit 평가 부적합 |
| `gemma-3-12b-it` | A-collapse, 현재 letter-logit 평가 부적합 |

Gemma-3 4B/12B IT는 clean prediction이 `A: 1000/1000`으로 붕괴했다. 이는 KC가 없다는 결과가 아니라 **평가 방식이 모델 출력 형식과 맞지 않은 실패**로 해석해야 한다. 필요하면 generation + parsing 방식으로 재평가해야 한다.

---

## 2. Pilot / Observation & Diagnosis 단계

이 단계의 목적은 “KC가 어느 layer/토큰/컴포넌트에서 나타나는가?”를 탐색하는 것이었다.

### 2.1 Step1: Tuned Lens / Transition Layer 분석

관련 코드:

- `phase1/scripts/step1_tuned_lens.py`
- `phase1/scripts/step1_case_analysis.py`

결과:

- `phase1/results/step1_tuned_lens/tuned_lens_stats.json`
- `phase1/results/step1_tuned_lens/preference_trajectory.png`
- `phase1/results/step1_tuned_lens/transition_layer_distribution.png`

핵심 결과:

| Position / target | transition mean layer | median |
|---|---:|---:|
| A | 6.744 | 7 |
| B | 10.405 | 11 |
| C | 1.727 | 1 |

해석:

- 모델 내부 표현에서 conflict answer 쪽 선호가 초중반 layer부터 나타난다.
- B transition은 평균 L10-L11로, 이후 실험에서 L11 이후가 중요해지는 패턴과 맞물린다.
- type별로도 큰 차이는 없고 misinformation/temporal/semantic 모두 유사한 transition 구조를 보인다.

### 2.2 Step2: Integrated Gradients

관련 코드:

- `phase1/scripts/step2_ig.py`
- `phase1/scripts/step2_filtered_token_plot.py`

결과:

- `phase1/results/step2_ig/ig_stats.json`
- `STEP2_IG_1000_ANALYSIS_KO.md`

전체 attribution:

| Group | Attribution mean |
|---|---:|
| context_claim | 0.0624 |
| context_evidence | 0.8479 |
| question | 0.0000 |
| options | 0.0000 |
| format_tokens | 0.0897 |

type별:

| Type | claim | evidence | format |
|---|---:|---:|---:|
| misinformation | 0.0528 | 0.8550 | 0.0922 |
| temporal | 0.0819 | 0.8306 | 0.0874 |
| semantic | 0.0525 | 0.8580 | 0.0895 |

해석:

- conflict context의 **evidence/body 부분**이 logit 변화에 가장 크게 기여한다.
- question/options attribution이 0인 것은 baseline prompt에 question/options가 포함되어 있어 delta가 거의 0이기 때문이다. 즉 이 IG는 “전체 prompt 중요도”가 아니라 “conflict context 추가 효과”를 측정한 것이다.
- temporal type은 claim attribution이 약간 높다.

### 2.3 Step3: Causal Tracing

관련 코드:

- `phase1/scripts/step3_causal_tracing.py`
- `phase1/scripts/step3_redraw_heatmap.py`
- `phase1/scripts/step3_sanity_check.py`

결과:

- `phase1/results/step3_causal_tracing/ct_stats.json`
- `phase1/results/step3_causal_tracing/ie_heatmap_overall.png`
- `phase1/results/step3_causal_tracing/ie_layer_profile.png`

실험 설계:

- conflict prompt에 noise를 넣고, 특정 layer/segment residual을 clean/support 상태로 restore
- segment:
  - `context_first_sent`
  - `context_rest`
  - `question`
  - `options`
  - `final`

주요 관찰:

- context 관련 segment 복원이 가장 큰 causal effect를 보인다.
- final segment는 후반 layer에서 양의 IE가 커진다.
- question/options는 상대적으로 약하다.

해석:

- KC는 context token processing을 통해 들어오고, 후반 layer에서 answer prediction에 반영된다.
- 이 결과는 Step2 IG의 “context evidence 우세”와 일관된다.

### 2.4 Step4: Path Patching

관련 코드:

- `phase1/scripts/step4_path_patching.py`
- `phase1/scripts/step4_analysis.py`

결과:

- `phase1/results/step4_path_patching/pp_stats.json`
- `phase1/results/step4_path_patching/attn_vs_mlp_by_layer.png`
- `phase1/results/step4_path_patching/component_effects_by_layer.png`

핵심 수치:

| Layer | full IE 대략 | dominant component |
|---|---:|---|
| L16 | 0.075 | MLP |
| L17 | 0.116 | MLP |
| L18 | 0.117 | Attention |
| L19 | 0.116 | MLP |
| L21 | 0.129 | MLP |

해석:

- full restoration IE는 후반 layer에서 강하다.
- attention과 MLP가 모두 관여하지만, layer에 따라 dominant component가 바뀐다.
- 이후 L17/L18/L19에 관심을 두게 된 근거가 된다.

### 2.5 Step5: Head Ablation

관련 코드:

- `phase1/scripts/step5_head_ablation.py`

결과:

- `phase1/results/step5_head_ablation/head_ablation_results.json`
- `phase1/results/step5_head_ablation/head_suppression_heatmap.png`

핵심 발견:

| Layer | Top head | suppression |
|---|---:|---:|
| L17 | H3 | 0.333 |
| L18 | H6 | 0.095 |
| L17 | H4/H6 | 약 0.026-0.028 |

해석:

- L17H3가 매우 강한 KC suppression head로 관찰되었다.
- 하지만 이후 Phase3에서 L17H3 ablation은 KC는 줄이지만 clean accuracy를 크게 망가뜨린다.

---

## 3. Pilot Intervention: Phase3

관련 코드:

- `phase1/scripts/step6_intervention.py`
- `phase1/scripts/step6_redraw_plots.py`

결과:

- `phase1/results/step6_intervention/intervention_results.json`
- `phase1/STEP6_INTERVENTION_ANALYSIS_KO.md`

목적:

- Step5에서 발견한 L17H3를 직접 ablation하거나 조절해 KC를 억제할 수 있는지 확인

결과:

| Strategy | KC suppression | Clean accuracy | 해석 |
|---|---:|---:|---|
| Baseline | 0.0% | 93.0% | 개입 없음 |
| Strategy A: L17H3 ablation | 57.5% | 25.9% | KC 억제 강하지만 clean 붕괴 |
| Strategy B: mild alpha search | 약 0.0% | 최대 94.5% | clean 보존하지만 KC 억제 없음 |

type별 Strategy A:

| Type | KC suppression | clean |
|---|---:|---:|
| misinformation | 57.8% | 25.7% |
| temporal | 58.3% | 26.1% |
| semantic | 56.6% | 25.8% |

해석:

- L17H3는 conflict context processing에 강하게 관여한다.
- 그러나 clean QA에도 필수적이다.
- 따라서 head ablation은 **non-selective intervention**이다.
- 이 실패가 이후 vector-based selective intervention으로 넘어간 직접적인 동기다.

---

## 4. Signal / Vector 추출 단계

### 4.1 Exp1: KC Trajectory

관련 코드:

- `phase2/scripts/exp1_trajectory.py`

결과:

- `phase2/results/exp1_trajectory/trajectory_stats_summary.json`
- `phase2/results/exp1_trajectory/trajectory_plot.png`
- `phase2/results/exp1_trajectory/cosine_similarity_only.png`

실험 설계:

세 조건의 residual을 비교했다.

```text
clean
conflict
supportive
```

주 방향:

```text
KC direction = SVD(conflict - clean)
Support direction = SVD(supportive - clean)
```

핵심 결과:

| Layer | KC strength | Support strength | KC-support cosine |
|---|---:|---:|---:|
| L14 | 2105.7 | 2056.9 | 0.974 |
| L16 | 2428.3 | 2372.7 | 0.977 |
| L17 | 2845.8 | 2512.3 | 0.914 |
| L18 | 2947.6 | 2563.7 | 0.892 |
| L19 | 3050.8 | 2669.2 | 0.881 |

결론:

```text
SVD(conflict-clean) 방향은 KC-specific이라기보다 context-following 방향에 가깝다.
```

즉 Exp1은 “KC direction을 찾았다”가 아니라, 오히려 “conflict와 supportive가 거의 같은 context-following 방향을 공유한다”는 음성 결과였다.

### 4.2 Exp1b: KC-specific Direction Extraction

관련 코드:

- `phase2/scripts/exp1b_direction_extraction.py`
- `phase2/scripts/exp1b_candidates.py`
- `scripts/exp1b_cohens_d.py`
- `scripts/exp1b_candidate_C_analysis.py`

결과:

- `phase2/results/exp1b_direction_extraction/`
- `phase2/results/exp1b_candidates/`
- `EXP1B_DIRECTION_EXTRACTION_ANALYSIS_KO.md`

Exp1의 실패를 해결하기 위해 baseline을 clean이 아니라 supportive로 바꾸었다.

```text
conflict - supportive
```

후보 방향:

| 기호 | 정의 | 의도 |
|---|---|---|
| A | SVD(conflict - clean) | context direction / Exp1 방향 |
| B | SVD(conflict - supportive) | conflict-supportive 차이 |
| C | cPCA | conflict에서 크고 supportive에서 작은 variance |
| D | regression | KC severity/excess를 예측하는 방향 |

Exp1b direction extraction의 자동 결론:

```text
conclusion = case2_weak_direction
best_dir_key = D_reg_excess
best_layers = [18, 15, 19]
```

하지만 후속 분석에서 더 중요한 해석은 다음이다.

#### A/B/C와 D가 분리된다

- A/B/C는 conflict-supportive 구조적 차이를 잡는다.
- D는 severity를 예측하지만 A/B/C와 거의 직교한다.
- 즉 “단일 KC-specific direction”으로 수렴하지 않는다.

#### B 방향은 Exp1 방향과 분리된다

Method B의 `overlap_with_exp1_dir`는 중간 layer에서 매우 낮다.

| Layer | B vs Exp1 direction |
|---|---:|
| L10 | 0.005 |
| L11 | 0.001 |
| L12 | 0.056 |
| L13 | 0.074 |
| L15 | 0.080 |

#### L14 이후 구조적 전환

| Layer | B explained ratio | effective rank |
|---|---:|---:|
| L13 | 0.197 | 76.85 |
| L14 | 0.442 | 20.63 |
| L17 | 0.452 | 10.12 |
| L18 | 0.410 | 8.60 |
| L19 | 0.406 | 9.39 |

해석:

```text
L14 이후 conflict-supportive 차이가 저차원 구조로 모인다.
```

---

## 5. Direction Causality: 단일 방향 IE 실험

### 5.1 Exp2: Direction-Specific Causal Patching

관련 코드:

- `phase2/scripts/exp2_causal_patching.py`
- `scripts/exp2_direction_patching.py`는 이전 버전

결과:

- `phase2/results/exp2_causal_patching/`
- `phase2/results/exp2_causal_patching/exp2_analysis.md`

실험 설계:

각 후보 방향 성분을 conflict residual에서 제거하고 correct log-prob 증가량을 IE로 측정했다.

```text
IE = logP(correct | direction removed) - logP(correct | conflict)
```

초기 Exp2에서는 A/C/D 중심으로 평가했다.

핵심 결과:

| Layer | IE_D | IE_A | IE_C | D pos ratio |
|---|---:|---:|---:|---:|
| L15 | +0.144 | +0.559 | -0.670 | 0.895 |
| L16 | +0.058 | +0.883 | -0.390 | 0.674 |
| L17 | +0.025 | +0.996 | +0.449 | 0.505 |
| L18 | +0.016 | +0.885 | +0.498 | 0.538 |
| L19 | -0.006 | +1.110 | +0.550 | 0.491 |

결론:

- D regression direction은 severity를 예측하지만 causal effect는 약하다.
- A direction은 correct logP를 크게 올린다.
- C는 layer에 따라 부호가 바뀌며 불안정하다.
- 효과는 final answer position에서 집중되고 pre-answer position에서는 약하다.

### 5.2 Exp IE Dissociation: 4방향 × 전 layer 확장

관련 코드:

- `phase2/scripts/exp_ie_dissociation.py`

결과:

- `phase2/results/exp_ie_dissociation/`
- `phase2/results/exp_ie_dissociation/exp_ie_dissociation_analysis.md`

실험 설계:

```text
layers = L6-L19
directions = A, B, C, D
metrics:
  IE_kc = logP(correct | patched conflict) - logP(correct | conflict)
  IE_sup = logP(correct | patched supportive) - logP(correct | supportive)
  dissociation = P(IE_kc > IE_sup)
```

핵심 표:

| Layer | A IE_kc | B IE_kc | C IE_kc | D IE_kc |
|---|---:|---:|---:|---:|
| L11 | +0.557 | +0.196 | +0.396 | +0.094 |
| L12 | +0.436 | +0.277 | +0.573 | +0.006 |
| L14 | +0.652 | +0.402 | -0.487 | +0.031 |
| L16 | +0.916 | +0.588 | -0.339 | +0.054 |
| L17 | +0.987 | +1.139 | +0.447 | +0.032 |
| L18 | +0.882 | +1.171 | +0.505 | +0.016 |
| L19 | +1.093 | +1.186 | +0.555 | -0.007 |

L14-L19 평균:

| Direction | IE_kc mean | IE_sup mean | dissociation mean |
|---|---:|---:|---:|
| A | +0.849 | -0.040 | 0.897 |
| B | +0.772 | -0.185 | 0.834 |
| C | +0.033 | -0.056 | 0.463 |
| D | +0.040 | -0.011 | 0.668 |

결론:

```text
A와 B만 강하게 causal하다.
C는 일부 layer에서 역효과를 낸다.
D는 severity probe로는 의미 있지만 causal pathway는 아니다.
```

### 5.3 A vs B: specificity-power tradeoff

| 특성 | A | B |
|---|---:|---:|
| IE_kc max | +1.093 | +1.186 |
| IE_sup | 약 -0.045 | 약 -0.314 |
| dissociation peak | 0.994 | 0.873 |
| 안정적 high dissociation layer 수 | 많음 | 적음 |
| 역할 | stable KC-specific/context direction | stronger conflict-supportive direction |

해석:

- A는 supportive를 거의 건드리지 않는 안정적 방향이다.
- B는 후반 layer에서 KC logit을 더 강하게 움직이지만 supportive도 일부 손상시킨다.
- 따라서 A는 “선택성 증명”에 좋고, B는 “강한 intervention”에 좋다.

---

## 6. Exp3: Double Dissociation + L17H3 Mediation

관련 코드:

- `phase2/scripts/exp3_dissociation_mediation.py`
- `phase2/scripts/exp3_mediation.py`

결과:

- `phase2/results/exp3/dissociation_stats.json`
- `phase2/results/exp3/mediation_stats.json`
- `phase2/results/exp3/exp3_analysis.md`

### 6.1 Double Dissociation: A × L16

결과:

| Metric | Value |
|---|---:|
| IE_kc mean | +0.9164 |
| IE_sup mean | -0.0447 |
| dissociation ratio | 0.994 |
| IE_kc positive ratio | 0.979 |
| IE_sup positive ratio | 0.324 |

type별 IE_kc:

| Type | IE_kc | IE_sup |
|---|---:|---:|
| misinformation | +0.920 | -0.034 |
| temporal | +0.895 | -0.051 |
| semantic | +0.934 | -0.048 |

해석:

```text
A × L16 제거는 conflict prompt에서는 correct logP를 크게 올리지만,
supportive prompt에서는 거의 효과가 없다.
```

따라서 A는 “KC-specific causal effect”를 갖는다고 볼 수 있다. 단, 이 효과는 log-prob IE 기준이지 rank flip 기준은 아니다.

### 6.2 L17H3 mediation 검증

결과:

| Metric | Value |
|---|---:|
| delta_attn_mean | +0.0298 |
| attn_decrease_ratio | 0.271 |
| delta_logit_diff_mean | -1.0448 |
| logit_decrease_ratio | 0.980 |
| chain_ratio | 0.269 |
| r_mediation | 0.034 |
| p_mediation | 0.281 |

해석:

- A 제거는 logit diff를 거의 항상 줄인다.
- 그러나 L17H3 evidence attention 감소를 매개로 하지는 않는다.
- 즉 A direction의 효과는 L17H3 attention pathway보다는 residual stream의 직접 변화에 가깝다.

---

## 7. Exp4: 단일 벡터 제거 intervention

관련 코드:

- `phase2/scripts/exp4_selective_intervention.py`

결과:

- `phase2/results/exp4/alpha_results.json`
- `phase2/results/exp4/layer_results.json`
- `phase2/results/exp4/type_results.json`
- `phase2/results/exp4/exp4_analysis.md`

목적:

```text
IE가 큰 A 방향 제거가 실제 answer rank flip, 즉 KC suppression으로 이어지는가?
```

### 7.1 Alpha scaling: A × L16

| alpha | KC suppression | clean acc | supportive acc | flipped |
|---:|---:|---:|---:|---:|
| 0.0 | 0.000 | 0.961 | 0.921 | 0 |
| 0.5 | 0.000 | 0.943 | 0.921 | 0 |
| 0.7 | 0.001 | 0.928 | 0.920 | 1 |
| 1.0 | 0.007 | 0.916 | 0.918 | 7 |

해석:

- clean/supportive는 잘 보존된다.
- 그러나 KC suppression은 거의 없다.
- log-prob IE가 커도 rank flip으로 이어지지 않는다.

### 7.2 Layer sweep: A direction

| Layer | KC suppression | clean acc | supportive acc |
|---|---:|---:|---:|
| L14 | 0.000 | 0.907 | 0.918 |
| L15 | 0.000 | 0.913 | 0.913 |
| L16 | 0.007 | 0.916 | 0.918 |
| L17 | 0.009 | 0.747 | 0.910 |
| L18 | 0.001 | 0.723 | 0.920 |
| L19 | 0.016 | 0.937 | 0.907 |

결론:

```text
A 단일 vector 제거는 selective하지만 insufficient하다.
```

---

## 8. Exp Supp: 왜 IE가 큰데 rank flip은 안 되는가

관련 코드:

- `phase2/scripts/exp_supp.py`

결과:

- `phase2/results/exp_supp/exp_supp_analysis.md`
- `phase2/results/exp_supp/exp1_results.json`
- `phase2/results/exp_supp/exp2_results.json`
- `phase2/results/exp_supp/exp3_results.json`

### 8.1 Logit tracking: A × L16

| Delta | Mean | 의미 |
|---|---:|---|
| delta_cor_conf | +0.916 | conflict에서 correct 상승 |
| delta_rep_conf | -0.128 | conflict에서 replace 하락 |
| delta_cor_sup | -0.045 | supportive에서 correct 미세 하락 |
| delta_rep_sup | +0.386 | supportive에서 wrong 상승 |

해석:

- conflict prompt에서는 correct↑ + replace↓로 KC 완화 방향.
- supportive prompt에서는 correct↓ + wrong↑로 반대 방향.
- 이것이 double dissociation의 mechanistic 근거다.

### 8.2 Severity analysis

`kc_severity = logP(replace) - logP(correct)`

| Bin | N | severity mean | IE_kc mean | flip rate |
|---|---:|---:|---:|---:|
| low [0,1) | 17 | 0.528 | +0.198 | 0.353 |
| medium [1,2) | 119 | 1.608 | +0.409 | 0.008 |
| high [2,3.5) | 472 | 2.806 | +0.774 | 0.000 |
| severe [3.5,∞) | 392 | 4.190 | +1.273 | 0.000 |

상관:

| Correlation | Value |
|---|---:|
| r(severity, IE_kc) | +0.630 |
| p | < 1e-100 |

핵심 역설:

```text
severity가 높을수록 IE_kc도 커진다.
하지만 severity가 높을수록 초기 logit gap이 너무 커져서 rank flip은 더 어렵다.
```

즉 A × L16의 평균 이동량은 약 1.04 정도이다.

```text
correct 상승 + replace 하락 ≈ 0.916 + 0.128 = 1.044
```

하지만 대부분 case의 초기 gap은 2-4 이상이다. 그래서 rank flip이 거의 일어나지 않는다.

### 8.3 A multi-layer patch

| Config | KC suppression | clean acc | supportive acc |
|---|---:|---:|---:|
| L16 only | 0.007 | 0.916 | 0.918 |
| L19 only | 0.016 | 0.937 | 0.907 |
| L16 + L19 | 0.018 | 0.931 | 0.904 |
| L16 + L17 + L19 | 0.019 | 0.904 | 0.901 |

결론:

```text
A 방향만 여러 layer에서 제거해도 KC suppression은 1.9%에 그친다.
동일한 context direction 반복 제거로는 강한 KC를 뒤집기 어렵다.
```

---

## 9. B direction 심층: Exp B Logit Severity

관련 코드:

- `phase2/scripts/exp_logit_severity.py`

결과:

- `phase2/results/exp_logit_severity/summary.json`
- `phase2/results/exp_logit_severity/run_log.txt`

목적:

```text
B = SVD(conflict - supportive)가 A보다 더 직접적인 KC direction인가?
```

layer sweep 결과:

| Layer | IE_kc | IE_sup | KC suppression | dissociation |
|---|---:|---:|---:|---:|
| L13 | -0.002 | -0.095 | 0.000 | 0.590 |
| L14 | +0.402 | -0.040 | 0.000 | 0.898 |
| L15 | +0.145 | -0.086 | 0.001 | 0.759 |
| L16 | +0.588 | -0.075 | 0.000 | 0.873 |
| L17 | +1.139 | -0.299 | 0.009 | 0.789 |
| L18 | +1.171 | -0.293 | 0.173 | 0.845 |
| L19 | +1.186 | -0.313 | 0.160 | 0.837 |

핵심:

- B는 L18/L19에서 rank flip을 크게 만든다.
- 단일 layer 기준으로 A의 최고 KC suppression 0.016보다 약 10배 크다.
- 그러나 supportive condition도 더 크게 건드린다.

---

## 10. Exp Grid: 다중 벡터/다중 레이어 intervention

관련 코드:

- `phase2/scripts/exp_grid.py`

결과:

- `phase2/results/exp_grid/`
- `phase2/results/exp_grid/exp_grid_analysis.md`
- `phase2/results/exp_grid/all_results_ranked.json`

탐색 공간:

```text
Layers: L16-L19
Directions: A, B, C, D
Alpha: 1.0
Configs: 71
  Phase 1: single layer × direction
  Phase 2: multi-layer same direction
  Phase 3: multi-layer mixed direction
```

### 10.1 Single layer × direction

KC suppression:

| Layer | A | B | C | D |
|---|---:|---:|---:|---:|
| L16 | 0.007 | 0.000 | 0.003 | 0.000 |
| L17 | 0.009 | 0.009 | 0.015 | 0.000 |
| L18 | 0.001 | 0.173 | 0.045 | 0.000 |
| L19 | 0.016 | 0.160 | 0.104 | 0.000 |

Clean accuracy:

| Layer | A | B | C | D |
|---|---:|---:|---:|---:|
| L16 | 0.916 | 0.921 | 0.765 | 0.949 |
| L17 | 0.747 | 0.882 | 0.843 | 0.964 |
| L18 | 0.723 | 0.860 | 0.837 | 0.955 |
| L19 | 0.937 | 0.854 | 0.893 | 0.946 |

결론:

- B@L18/L19가 단일 개입으로 가장 강하다.
- D는 clean은 보존하지만 KC suppression은 0이다.
- C@L19는 중간 효과가 있고, C@L16은 clean 손상이 크다.

### 10.2 Same direction multi-layer

| Direction | Best config | KC suppression | clean |
|---|---|---:|---:|
| A | L17:A + L19:A | 0.031 | 0.898 |
| B | L16:B + L17:B + L18:B | 0.205 | 0.854 |
| C | L16:C + L19:C | 0.129 | 0.714 |
| D | L16:D + L17:D | 0.000 | 0.958 |

해석:

- B만 여러 layer에서 제거해도 20.5%까지 오른다.
- 하지만 30% 목표에는 부족하다.

### 10.3 Mixed direction multi-layer

상위 config:

| Rank | Config | KC suppression | clean | supportive |
|---:|---|---:|---:|---:|
| 1 | L16:A + L17:C + L18:B + L19:B | 0.318 | 0.801 | 0.811 |
| 2 | L16:A + L17:C + L19:B | 0.315 | 0.813 | 0.822 |
| 3 | L17:C + L18:B + L19:B | 0.303 | 0.814 | 0.813 |
| 4 | L16:A + L17:C + L18:B | 0.295 | 0.810 | 0.816 |
| 5 | L17:C + L18:B | 0.291 | 0.817 | 0.813 |

핵심 시너지:

| Config | KC suppression |
|---|---:|
| L17:C only | 0.015 |
| L18:B only | 0.173 |
| L17:C + L18:B | 0.291 |

예상 단순합은 0.188인데 실제는 0.291이다. 즉 L17:C와 L18:B 사이에 시너지가 있다.

최종적으로 목표를 달성한 조합:

```text
KC suppression > 30%
clean accuracy > 80%
```

| Config | KC | clean | 장단점 |
|---|---:|---:|---|
| L16:A + L17:C + L18:B + L19:B | 0.318 | 0.801 | 최대 KC, clean 경계 |
| L16:A + L17:C + L19:B | 0.315 | 0.813 | 균형 최상 |
| L17:C + L18:B + L19:B | 0.303 | 0.814 | A 없이 목표 달성 |

---

## 11. 연구 전체 결론

### 11.1 Mechanistic story

현재까지의 실험을 하나로 묶으면 다음 스토리가 가장 자연스럽다.

```text
1. Conflict context는 초중반부터 모델의 answer preference를 바꾸기 시작한다.

2. IG/causal tracing 결과, 질문이나 option 자체보다 context evidence가 KC의 주된 원천이다.

3. Head ablation에서는 L17H3가 강하게 보였지만,
   이 head는 clean QA에도 필수적이어서 직접 ablation은 non-selective하다.

4. Exp1의 conflict-clean 방향(A)은 context-following 방향이다.
   supportive context와도 cosine이 매우 높아, 처음에는 KC-specific 분리에 실패했다.

5. conflict-supportive 비교(B)를 도입하면,
   후반 L18/L19에서 rank flip을 직접 만드는 강한 KC direction이 드러난다.

6. A는 log-prob IE와 double dissociation은 강하지만,
   초기 logit gap이 너무 커서 단일 A 제거만으로는 rank flip이 거의 안 된다.

7. B는 rank flip power가 강하지만 supportive도 일부 건드린다.

8. C는 단독으로 안정적이지 않지만 L17에서 B@L18/L19 효과를 증폭시키는 gate처럼 작동한다.

9. 따라서 KC를 선택적으로 줄이려면 단일 vector가 아니라
   layer별 역할이 다른 mixed intervention이 필요하다.
```

### 11.2 Direction별 역할

| Direction | 정의 | 역할 | 장점 | 한계 |
|---|---|---|---|---|
| A | SVD(conflict-clean) | context-following / stable KC-specific IE | supportive 손상 작음, dissociation 강함 | rank flip 약함 |
| B | SVD(conflict-supportive) | conflict-specific rank flip direction | L18/L19에서 강한 KC suppression | supportive 손상 큼 |
| C | cPCA | variance/gating direction | L17에서 B 효과와 시너지 | 단독 불안정, 일부 layer 역효과 |
| D | regression severity | severity encoding | severity 예측 가능 | causal suppression 없음 |

### 11.3 Layer별 역할

| Layer | 관찰된 역할 |
|---|---|
| L11-L13 | A IE가 이미 등장, 초기 KC representation 형성 |
| L14 | conflict-supportive 구조가 저차원화되는 전환점 |
| L16 | A direction의 stable double dissociation peak |
| L17 | C direction gate, L17H3 head도 강하지만 non-selective |
| L18 | B direction rank flip power 최고권 |
| L19 | B/A 모두 final answer rank에 직접적 영향 |

---

## 12. 다음 Step 제안

현재 가장 중요한 질문은 “grid search로 찾은 mixed intervention이 진짜 mechanistic story인지, 그리고 더 좋은 config를 찾을 수 있는지”이다.

### Step A. 최종 후보 3개를 더 정밀 검증

우선 다음 세 config를 고정 후보로 삼는 것이 좋다.

| 후보 | Config | KC | clean | 이유 |
|---|---|---:|---:|---|
| Balanced | L16:A + L17:C + L19:B | 0.315 | 0.813 | 3-layer, clean 여유 |
| Max KC | L16:A + L17:C + L18:B + L19:B | 0.318 | 0.801 | 최대 KC |
| Minimal | L17:C + L18:B | 0.291 | 0.817 | 2-layer, 메커니즘 단순 |

각 후보에 대해 다음을 추가 측정:

- type별 KC suppression
- severity bin별 suppression
- clean/supportive에서 어떤 option이 손상되는지
- flipped case와 non-flipped case의 차이
- generation 기반 answer 변화율

### Step B. Alpha sweep을 mixed config에 대해 수행

현재 grid는 alpha=1.0 고정이다. 다음 실험은 mixed config별 alpha sweep이 필요하다.

추천:

```text
alpha ∈ [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
configs:
  L17:C + L18:B
  L16:A + L17:C + L19:B
  L16:A + L17:C + L18:B + L19:B
```

목표:

- clean 80% 이상을 유지하면서 KC suppression을 35-40%까지 올릴 수 있는지 확인
- alpha>1에서 B/C가 clean을 얼마나 망가뜨리는지 확인

### Step C. Direction sign / projection target 재검토

현재 개입은 주로 direction 성분 제거이다.

```text
h' = h - alpha * (h @ d) * d
```

하지만 B/C는 단순 제거보다 다음이 더 좋을 수 있다.

```text
1. conflict projection을 supportive mean projection으로 이동
2. projection을 0이 아니라 clean/supportive target으로 clamp
3. case-wise projection excess만 제거
```

특히 B는 `conflict-supportive` 방향이므로 `0` 제거보다 supportive projection target이 더 자연스러울 수 있다.

### Step D. L17:C → L18:B 시너지 검증

현재 가장 흥미로운 mechanistic finding은 이 부분이다.

```text
L17:C only = 0.015
L18:B only = 0.173
L17:C + L18:B = 0.291
```

추가 실험:

- L17:C patch 후 L18:B projection이 실제로 줄어드는가?
- L17:C patch가 L18 residual의 B component를 어떻게 바꾸는가?
- L17:C를 넣은 경우와 넣지 않은 경우, L18:B의 case-wise IE가 어떻게 달라지는가?
- flipped case는 L17:C에 의해 B projection이 더 크게 변하는가?

이 결과가 강하면 논문 contribution이 된다.

### Step E. Cross-model validation

현재 mechanistic 실험은 주로 `gemma-2-2b`에 집중되어 있다. 다음 모델에서 최소한 축소 검증을 권한다.

우선순위:

| Model | 이유 |
|---|---|
| `google/gemma-2-9b` | 같은 Gemma2 계열, KC rate 100%, clean acc 높음 |
| `Qwen/Qwen2.5-7B` | 다른 family, KC rate 높음 |
| `Qwen/Qwen3-4B` | chat model, KC rate 높음 |

축소 검증 항목:

```text
1. Exp1 trajectory: A direction이 supportive와 겹치는가?
2. B direction: conflict-supportive SVD가 rank flip에 강한가?
3. Grid top config의 analogue가 있는가?
```

Gemma-3 4B/12B IT는 generation 기반 KC rate 평가부터 먼저 해야 한다.

### Step F. Evaluation metric 정리

현재 중요한 distinction:

```text
IE = log-prob improvement
KC suppression = rank flip
clean accuracy = clean prompt rank correctness
supportive accuracy = supportive prompt rank correctness
```

논문/발표에서는 이 distinction을 명확히 해야 한다.

특히:

- A는 IE 기준으로 강하고 selective하다.
- 그러나 rank flip 기준으로 약하다.
- B/mixed intervention은 rank flip 기준으로 강하다.

이 구분이 연구 내러티브의 핵심이다.

---

## 13. 한 문장 요약

```text
초기 pilot은 KC가 context evidence와 후반 attention/MLP 경로를 통해 발생함을 보였고,
vector 분석은 A(context-following), B(conflict-supportive), C(cPCA gate), D(severity probe)를 분리했다.
단일 A 제거는 매우 selective하지만 rank flip에는 약했고,
B와 C를 layer별로 조합한 mixed intervention이 처음으로 KC suppression 30%와 clean accuracy 80%를 동시에 달성했다.
다음 단계는 이 mixed intervention의 alpha/target/sign을 정밀화하고,
L17:C → L18/19:B 시너지의 mechanistic causal chain을 검증하는 것이다.
```

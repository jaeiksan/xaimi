# Override-Risk Validation: 실험 설계 및 결과 정리

작성일: 2026-08-13 (KST)  
상태: 실험 B와 실험 A 완료. harmful override 구분 가능성은 확인됐지만, beneficial override 보호는 실패한 negative result를 포함한다.

## 1. 연구 목표와 프레이밍

기존 연구는 Knowledge Conflict(KC), 즉 모델의 파라메트릭 지식과 inference-time context가 충돌할 때 모델이 context 쪽 답으로 바뀌는 현상을 다뤘다. 이 후속 검증은 이를 RAG 안전성 관점으로 재해석한다.

핵심 질문은 다음과 같다.

> 외부 context가 검증된 기존 지식을 잘못된 답으로 덮어쓸 위험(harmful override)을 residual-stream 신호로 감지하고, 위험할 때만 선택적으로 개입할 수 있는가?

중요한 안전성 요구는 context를 무조건 억제하는 것이 아니다. 모델의 지식이 오래됐을 때 최신의 정확한 context가 답을 올바르게 갱신하는 beneficial override는 보존해야 한다.

원래 검증하려던 주장은 다음과 같다.

> 위험 신호가 높을 때만 harmful context-induced override를 억제하고, 정확한 최신 context가 만드는 beneficial override에는 개입하지 않는다.

## 2. 기존 Phase 4에서 고정한 설정

이번 작업은 새 direction이나 intervention을 찾는 실험이 아니라, 기존 Phase 4에서 선택한 gate가 새로운 held-out 조건에서 일반화하는지 검증하는 실험이다. 따라서 새 데이터로 layer, direction, threshold, intervention strength를 재튜닝하지 않았다.

| 항목 | 고정값 |
|---|---|
| 모델 | `google/gemma-2-2b` (TransformerLens, bfloat16) |
| gate/개입 layer | L17, `blocks.17.hook_resid_post` |
| direction | Phase 4 train에서 추출한 L17 B direction |
| 복원 벡터 | `u_PK = mean(h_clean - h_conflict)` |
| alpha | 6.0 |
| gamma | 10.0 |
| tau | 19.256011962890625 |

L17 B direction은 기존 train에서 `h_conflict - h_supportive` 행렬의 첫 SVD right-singular vector로 얻었다. 모든 검증에서 다음 gate를 사용했다.

```text
p = h_last · normalize(d_B)
g = sigmoid(10.0 × (p - 19.256011962890625))

gated:   h_last' = h_last + g × 6.0 × u_PK
ungated: h_last' = h_last + 6.0 × u_PK
```

`gate_activation_rate`는 `p >= tau` 비율이며, `g >= 0.5` 비율과 동치다.

관련 기존 asset:

```text
phase4/vectors/direction_B_train.pt
phase4/vectors/u_PK_train.pt
phase4/vectors/train_proj_clean.pt
```

## 3. 평가 데이터와 조건

모든 최종 조건은 질문(question) 단위로 서로 겹치지 않게 구성했다. 조건 ①과 ② 내부에서는 동일 질문을 closed-book과 context view로 paired하게 평가한다.

| 조건 | 정의 | 최종 수 | 데이터 소스 |
|---|---|---:|---|
| ①a | context 없이 정답 | 300 | ConflictBank default evidence에서 screening |
| ①b | 정확한 supportive context에서도 정답 | 300 | ①a와 같은 질문의 paired view |
| ②a | context 없이 오답 | 105 | ConflictBank default evidence에서 screening |
| ②b | supportive context에서도 계속 오답 | 105 | ②a와 같은 질문의 paired view |
| ③ | M에서 최신 C로 올바르게 갱신되는 beneficial override | 300 | TempLAMA |
| ④ | 정답에서 conflict context의 오답으로 바뀌는 harmful override | 68 | 기존 Phase 4 dev |

### 3.1 조건 ③: TempLAMA factual beneficial override

ConflictBank temporal은 benchmark가 생성한 temporal conflict이며 replacement answer가 실제 최신 사실이라는 보장이 부족하다. 따라서 조건 ③에는 사용하지 않았다.

공식 TempLAMA test JSONL에서 historical answer를 M 후보, `most_recent_answer`를 C 후보로 사용했다. M과 C가 다른 3,232개 unique query를 만들고, C를 명시한 context를 제공했다. 다음 행동 필터를 모두 통과한 사례만 남겼다.

```text
closed-book full-choice prediction = M
newer-value context full-choice prediction = C
```

1,134개가 통과했고, relation 비율을 보존하는 seed 42 sampling으로 최종 300개 unique question을 고정했다.

### 3.2 조건 ①·②: ConflictBank safe context와 persistent error

ConflictBank `CB_qa`의 default evidence를 fact-aligned supportive context로 사용했다. 기존 Phase 0 harmful set 및 TempLAMA final set의 질문을 제외했다.

- 첫 5,000개 screening: 조건 ① 1,603개, 조건 ② 18개
- 추가 25,000개 screening: 조건 ① 8,963개, 조건 ② 87개
- 최종 pool: 조건 ① 10,566개, 조건 ② 105개
- 조건 ①은 seed 42로 300개를 뽑았고, 조건 ②는 희소하여 전수(105개)를 사용했다.

조건 ②가 희소한 이유는 정의가 엄격하기 때문이다. 모델은 closed-book과 정답 evidence가 있는 supportive context 양쪽에서 모두 틀려야 한다.

### 3.3 조건 ④: natural harmful override

조건 ④는 기존 Phase 4 `dev_200.json`에서 가져왔다. direction이 Phase 4 train에서 계산됐으므로 train을 사용하지 않았다. dev에는 동일 seed question의 conflict-type variant가 있어 질문 독립성을 위해 질문당 한 case만 deterministic하게 선택했고 최종 68개가 남았다.

최종 조건들 사이의 모든 pairwise question overlap은 0이며 조건 ①·②·③은 Phase 4 vector-training question과도 겹치지 않는다.

최종 manifest:

```text
data/processed/final_evaluation_manifest.json
```

## 4. 공통 평가 방식

기존 Phase 4는 주로 correct/replacement 두 option만 비교했다. 이번 검증에서는 조건 ②에 특정 replacement option이 없을 수 있으므로, 네 선택지의 letter-token log probability 중 최대값을 full-choice prediction으로 사용했다.

각 prompt view에서 다음을 저장했다.

- L17 마지막 position residual의 B-direction projection
- frozen sigmoid gate value
- gate activation 여부
- full-choice prediction 및 accuracy

## 5. 실험 B: harmful-override risk signal의 판별성

### 5.1 목적

B direction이 단순 context 존재, 일반적 오답, override 자체에 반응하는지와 harmful override에 더 특이적으로 반응하는지를 분리한다.

주 비교는 context 존재를 통제한 다음 비교다.

```text
조건 ④ harmful context vs 조건 ①b safe supportive context
```

보조 비교는 ④ vs ③, ②a vs ①a, ②b vs ①b다.

### 5.2 실행 규모와 통계

총 1,478개 prompt view를 수집했다.

```text
①: 300 closed-book + 300 context = 600
②: 105 closed-book + 105 context = 210
③: 300 closed-book + 300 context = 600
④: 68 harmful context = 68
합계 = 1,478
```

AUROC와 평균 projection 차이의 95% CI는 seed 42, 1,000회 bootstrap으로 계산했다.

### 5.3 조건별 결과

| 조건 | View | 평균 projection | 평균 gate | gate activation | 정확도 | n |
|---|---|---:|---:|---:|---:|---:|
| ① | closed-book | 2.655 | 0.088 | 8.7% | 100.0% | 300 |
| ① | safe context | 7.267 | 0.510 | 51.0% | 100.0% | 300 |
| ② | closed-book | 2.006 | 0.079 | 7.6% | 0.0% | 105 |
| ② | supportive context | 17.549 | 0.504 | 50.5% | 0.0% | 105 |
| ③ | closed-book | 7.456 | 0.297 | 29.7% | 0.0% | 300 |
| ③ | beneficial context | 20.781 | 0.665 | 66.3% | 100.0% | 300 |
| ④ | harmful context | 41.712 | 0.894 | 89.7% | 0.0% | 68 |

### 5.4 AUROC와 bootstrap 결과

| 비교 (positive vs negative) | AUROC | AUROC 95% CI | 평균 projection 차이 | 차이 95% CI |
|---|---:|---:|---:|---:|
| ④ harmful vs ①b safe context | 0.752 | [0.690, 0.812] | +34.445 | [+26.178, +41.788] |
| ④ harmful vs ③ beneficial context | 0.821 | [0.746, 0.888] | +20.932 | [+13.248, +27.574] |
| ②a error vs ①a correct | 0.487 | [0.425, 0.554] | -0.649 | [-3.226, +1.893] |
| ②b error vs ①b safe context | 0.549 | [0.497, 0.599] | +10.282 | [+4.240, +16.176] |

### 5.5 실험 B 해석

긍정적 증거:

- harmful condition ④는 safe context ①b보다 유의하게 높은 projection을 보였다.
- harmful condition ④는 factual beneficial override ③보다도 유의하게 높았다.
- closed-book 단순 오답 ②a는 closed-book correct ①a와 구분되지 않았다(AUROC 0.487, CI가 0.5 포함). 이 signal이 모델의 일반적 무지에 단순 반응한다고 보기는 어렵다.

제한:

- supportive context가 있는데도 계속 오답인 ②b는 safe context ①b보다 평균 projection이 +10.282 높았다. AUROC CI는 0.5를 약간 포함하지만, 평균 차이 bootstrap CI는 0을 포함하지 않는다.
- 따라서 B direction이 harmfulness에만 반응하는 완전히 순수한 risk signal이라고 주장할 수는 없다. context가 있는 실패 또는 context-induced uncertainty에도 약하게 반응할 가능성이 있다.

중간 결론:

> L17 B direction은 harmful override를 safe context 및 beneficial override보다 상당히 잘 구분한다. 그러나 harmfulness만을 완벽하게 나타내는 단일 신호라는 주장은 현재 지지되지 않는다.

## 6. 실험 A: beneficial override 보호 여부

### 6.1 목적과 arm

조건 ③의 factual beneficial override에서 frozen gate가 불필요하게 활성화해 최신 context에 의한 M → C 전환을 막는지 평가했다.

동일한 300개 사례에 대해 다음 세 arm을 paired하게 실행했다.

| Arm | 개입 |
|---|---|
| No intervention | 개입 없음 |
| Ungated | 항상 `h_last += 6.0 × u_PK` |
| B-gated | `h_last += gate × 6.0 × u_PK` |

평가 지표는 latest answer C를 full-choice prediction으로 고른 비율(`override_success_rate`)이다. no-intervention 대비 성공률 변화의 95% CI는 case-paired, 1,000회 bootstrap으로 계산했다.

### 6.2 결과

| Arm | Beneficial override 성공률 | No-intervention 대비 변화 | 95% CI |
|---|---:|---:|---:|
| No intervention | 100.0% | 0.0%p | [0.0, 0.0] |
| Ungated | 15.3% | -84.7%p | [-88.7, -80.3] |
| B-gated | 33.7% | -66.3%p | [-71.7, -61.0] |

Gated 방법은 ungated보다 beneficial override를 18.3%p 더 보존했지만, no-intervention과 비슷한 성공률을 유지해야 한다는 사전 성공 기준은 명확히 실패했다.

### 6.3 Gate activation subgroup

300개 중 gate가 활성화된 사례는 199개(66.3%)였다.

| B-gated subgroup | 사례 수 | Beneficial override 성공률 |
|---|---:|---:|
| gate 비활성 (`p < tau`) | 101 | 99.0% (100/101) |
| gate 활성 (`p >= tau`) | 199 | 0.5% (1/199) |

Gate가 꺼져 있을 때는 개입이 거의 발생하지 않아 C를 보존한다. 반면 gate가 켜지면 parametric restoration이 C를 거의 완전히 M 방향으로 되돌린다.

### 6.4 실험 A 결론

> Frozen L17 B-gated restoration은 harmful override만 선택적으로 억제하지 못한다. beneficial override의 66.3%에서 gate가 활성화되고, 활성 사례의 거의 전부에서 정확한 최신 context에 의한 answer update를 차단한다.

따라서 “beneficial override 보호”라는 원래 안전성 주장은 현 설정에서 성립하지 않는다. 이는 중요한 negative result다.

## 7. A와 B를 함께 해석한 결론

두 결과는 모순되지 않는다.

1. B direction은 harmful override에서 평균적으로 더 높고, ④는 ①b 및 ③과 구분 가능하다.
2. 하지만 beneficial override ③에서도 gate activation이 66.3%로 높다.
3. 현재의 단일 threshold `tau`와 parametric restoration action은 harmful/beneficial을 충분히 구분하는 의사결정을 만들지 못한다.

따라서 가장 정확한 현재 주장은 다음과 같다.

> Frozen L17 B-direction gate는 harmful context-induced override와 safe/beneficial context를 통계적으로 구분하지만, beneficial override에도 빈번히 활성화된다. 따라서 현재 single-signal, single-threshold restoration gate는 harmful override 억제와 beneficial update 보존을 동시에 달성하지 못한다.

## 8. 포스터/논문에서 가능한 주장과 불가능한 주장

### 주장 가능

- KC 기반 residual signal은 harmful context-induced answer change와 safe context를 구분하는 정보를 담고 있다.
- 이 signal은 closed-book 단순 error에는 반응하지 않았다.
- 기존 gated restoration은 ungated restoration보다 beneficial update를 더 보존한다.
- 하지만 frozen gate의 beneficial-update preservation은 불충분하며, 안전성 측면의 한계를 드러낸다.

### 주장하면 안 됨

- “이 gate가 harmful override만 선택적으로 막는다.”
- “이 gate가 beneficial override를 보호한다.”
- “B direction이 harmfulness에만 특이적이다.”

## 9. 한계

1. 조건 ② 표본 수는 105개로 상대적으로 작다.
2. held-out 검증은 Gemma-2-2B 한 모델에서만 실행했다.
3. L17 B direction과 `u_PK` restoration 한 가지 action만 검증했다.
4. 평가는 통제된 다지선다 letter-logit QA이며, 자유 생성 RAG 응답과는 다르다.
5. TempLAMA는 release 시점 Wikidata의 latest answer를 사용하며, 현재 웹 retrieval를 직접 평가한 것은 아니다.

## 10. 권장 다음 단계

현재 final test 역할을 하는 ③ 300개와 ④ 68개에 맞춰 다시 튜닝하면 안 된다. 개선을 시도한다면 별도 development set에서만 설계하고 현재 test에 한 번만 재평가해야 한다.

1. 결과 그림과 표 생성: 조건별 projection/gate distribution, 실험 A arm별 성공률과 CI, gate activation subgroup.
2. final 300개 밖의 TempLAMA behavioural-pass 사례를 development set으로 사용.
3. L10–L22 projection profile, answer logit gap, confidence/entropy 등을 결합한 calibrated risk classifier를 개발 set에서 학습.
4. harmful detection gate와 parametric restoration action을 분리해 평가.
5. 고정된 새 방법을 현재 final 조건 ①–④에 단 한 번 평가하고 harmful mitigation과 beneficial preservation을 함께 보고.

## 11. 주요 파일

### 코드

```text
scripts/02_build_condition3_templama.py
scripts/03_screen_condition3_templama.py
scripts/05_screen_conditions12_conflictbank.py
scripts/07_finalize_evaluation_sets.py
scripts/08_collect_experiment_b.py
scripts/09_analyze_experiment_b.py
scripts/10_run_experiment_a.py
scripts/11_analyze_experiment_a.py
```

### 로컬 데이터와 결과

```text
data/processed/final_evaluation_manifest.json
data/processed/final_cond1.jsonl
data/processed/final_cond2.jsonl
data/processed/final_cond3.jsonl
data/processed/final_cond4.jsonl

results/exp_B_per_case.jsonl
results/exp_B_specificity.csv
results/exp_B_auroc.csv
results/exp_A_per_case.jsonl
results/exp_A_beneficial_override.csv
```

실험 데이터와 raw result는 대용량·로컬 산출물 정책에 따라 GitHub에는 포함하지 않는다.

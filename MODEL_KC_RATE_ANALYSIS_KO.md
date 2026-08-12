# 모델별 ConflictBank KC 발생률 분석

## 실험 목적

현재 mechanistic experiment가 특정 모델 하나의 특이 사례가 아니라는 점을 보이기 위해, ConflictBank 1000개 query에서 여러 모델이 얼마나 자주 Knowledge Conflict(KC)를 보이는지 확인했다.

KC 판정 기준은 기존 `phase0_collect_kc.py`와 동일한 letter-logit exact match 방식이다.

- clean prompt에서 `correct_option`을 선택해야 함
- conflict context를 넣었을 때 `replace_option`을 선택하면 KC로 판정
- query-level KC는 misinformation / temporal / semantic 중 하나 이상에서 KC가 발생한 query

가장 중요한 지표는 `query_kc_rate_after_clean_correct_pct`이다. 이는 모델이 clean setting에서 정답을 알고 있던 query 중 conflict context로 인해 답이 바뀐 비율이다.

## 전체 결과 요약

| 모델 | Clean Acc. | 전체 Query KC | Clean-correct 조건부 KC | Trial-level KC | 상태 |
|---|---:|---:|---:|---:|---|
| Qwen/Qwen2.5-3B | 50.2% | 50.1% | 99.8% | 97.9% | 신뢰 가능 |
| Qwen/Qwen2.5-7B | 49.4% | 49.3% | 99.8% | 98.0% | 신뢰 가능 |
| Qwen/Qwen3-4B | 35.9% | 35.8% | 99.7% | 96.5% | 신뢰 가능 |
| Qwen/Qwen3-8B | 26.1% | 24.6% | 94.3% | 80.6% | 사용 가능, 약한 편 |
| google/gemma-2-2b | 37.4% | 37.3% | 99.7% | 99.0% | 신뢰 가능 |
| google/gemma-2-2b-it | 13.4% | 10.1% | 75.4% | 61.7% | 사용 가능, clean acc 낮음 |
| google/gemma-2-9b | 46.5% | 46.5% | 100.0% | 98.7% | 신뢰 가능 |
| google/gemma-3-1b-it | 29.2% | 28.9% | 99.0% | 95.7% | 신뢰 가능 |
| google/gemma-3-4b-it | 25.0% | 0.0% | 0.0% | 0.0% | 제외 권장: A collapse |
| google/gemma-3-12b-it | 25.0% | 0.0% | 0.0% | 0.0% | 제외 권장: A collapse |

`google/gemma-2-9b-it`는 실행 실패했다. 실행 모델명이 `google/gemma-2-9b_it`로 기록되어 있는데, 올바른 Hugging Face 모델명은 `google/gemma-2-9b-it`이다.

## 핵심 결론

대부분의 정상 실행 모델에서 KC는 매우 강하게 발생했다. 특히 clean에서 정답을 맞힌 query만 놓고 보면, Qwen2.5, Qwen3-4B, Gemma2 base, Gemma3-1B-it에서 조건부 KC 발생률이 95-100% 수준이다.

이는 ConflictBank의 conflict context가 단순히 일부 모델에만 영향을 주는 것이 아니라, 여러 계열의 LLM에서 모델의 clean answer를 replacement answer로 강하게 전환시킨다는 근거가 된다.

전체 query 기준 KC 발생률도 충분히 크다. 예를 들어 Qwen2.5-3B는 1000개 중 501개, Qwen2.5-7B는 493개, Gemma2-9B는 465개, Gemma2-2B는 373개 query에서 KC가 발생했다. 따라서 이후 mechanistic experiment에서 KC case를 분석하는 것이 충분히 정당화된다.

## 모델별 해석

### Gemma 2 Base

Gemma2 base 모델은 가장 안정적인 결과를 보였다.

- `gemma-2-2b`: clean accuracy 37.4%, 전체 query KC 37.3%, 조건부 KC 99.7%
- `gemma-2-9b`: clean accuracy 46.5%, 전체 query KC 46.5%, 조건부 KC 100.0%

크기가 커질수록 clean accuracy와 전체 query KC rate가 증가했다. 하지만 clean-correct 조건부 KC는 두 모델 모두 거의 100%로, conflict context가 매우 강하게 작동한다.

### Qwen 2.5

Qwen2.5 계열도 매우 강한 KC를 보였다.

- 3B: clean accuracy 50.2%, 전체 query KC 50.1%, 조건부 KC 99.8%
- 7B: clean accuracy 49.4%, 전체 query KC 49.3%, 조건부 KC 99.8%

두 모델 모두 clean에서 맞힌 거의 모든 query가 conflict context에서 뒤집혔다. 이 결과는 Gemma 계열 밖에서도 KC가 강하게 발생한다는 교차 검증 근거로 특히 유용하다.

### Qwen 3

Qwen3-4B는 안정적이다.

- Qwen3-4B: clean accuracy 35.9%, 전체 query KC 35.8%, 조건부 KC 99.7%

Qwen3-8B는 조건부 KC 94.3%로 여전히 높지만, clean prediction이 A에 67.5% 몰려 있어 Qwen3-4B보다 해석 시 주의가 필요하다. 그래도 246개 query에서 KC가 발생했으므로 보조 근거로는 사용할 수 있다.

### Gemma 3

Gemma3-1B-it는 신뢰 가능한 결과를 보였다.

- clean accuracy 29.2%
- 전체 query KC 28.9%
- 조건부 KC 99.0%

반면 Gemma3-4B-it와 Gemma3-12B-it는 clean prediction이 A 1000/1000으로 완전히 collapse했다. clean accuracy 25%와 KC 0%는 모델이 실제로 KC를 보이지 않는다는 의미라기보다, 현재 letter-logit 평가 방식이 해당 모델의 응답 형식과 맞지 않는다는 신호로 보는 것이 타당하다. 이 두 모델은 현재 표의 주요 결론에서 제외하는 것이 안전하다.

### Gemma 2 IT

Gemma2-2B-it는 clean accuracy가 13.4%로 낮고 조건부 KC도 75.4%로 다른 정상 모델보다 낮다. 그래도 clean-correct query 중 4분의 3이 conflict에 의해 뒤집혔으므로 KC 현상 자체의 보조 근거로는 사용할 수 있다.

## 발표/논문용 추천 정리

주요 표에는 다음 모델을 포함하는 것이 좋다.

- `google/gemma-2-2b`
- `google/gemma-2-9b`
- `Qwen/Qwen2.5-3B`
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen3-4B`
- `google/gemma-3-1b-it`

보조 결과로는 다음을 언급할 수 있다.

- `Qwen/Qwen3-8B`: KC는 충분히 발생하지만 prediction bias가 약간 큼
- `google/gemma-2-2b-it`: clean accuracy가 낮지만 조건부 KC는 75.4%

제외 또는 재평가 권장:

- `google/gemma-3-4b-it`
- `google/gemma-3-12b-it`

이 두 모델은 A collapse가 발생했으므로, 현재 결과를 KC 미발생 근거로 해석하면 안 된다. 필요하다면 generation-based answer parsing 또는 모델별 chat prompt를 더 강하게 통제한 평가로 별도 재실험해야 한다.

## 최종 결론

정상적으로 평가된 8개 모델 중 대부분에서 clean-correct 조건부 KC 발생률이 매우 높게 나타났다. 특히 Qwen2.5, Qwen3-4B, Gemma2 base, Gemma3-1B-it에서 조건부 KC가 약 95-100% 수준이다.

따라서 “ConflictBank에서 발생하는 knowledge conflict는 Gemma2-2B 하나의 특수한 현상이 아니라, 여러 모델 계열에서 반복적으로 관찰되는 현상”이라고 주장할 수 있다. 현재 mechanistic experiment를 Gemma2-2B에서 수행하는 것은, 먼저 현상이 다양한 모델에서 충분히 발생함을 확인한 뒤 대표 모델을 깊게 분석하는 설계로 정당화할 수 있다.

# Phase 3 개입 실험 분석

마지막 확인 시점: 2026-04-26

## 먼저 확인된 상태

현재 `phase3_intervention` 폴더에는 서로 다른 시점의 결과가 같이 있습니다.

- [intervention_results.json](/home/jaeik/xaimi/phase1/results/step6_intervention/intervention_results.json)
  - 수정 시각: `2026-04-25 23:36 KST`
  - 내용: `n_cases = 10`
  - 즉, 예전 소규모 실행 결과
- [phase2_checkpoint.json](/home/jaeik/xaimi/phase1/results/step6_intervention/phase2_checkpoint.json)
  - 수정 시각: `2026-04-26 20:43 KST`
  - 내용: `processed = 500`
  - 즉, 더 최근의 대규모 중간 실행 결과

따라서 지금 시점에서 Phase 3를 다시 해석할 때는 `intervention_results.json`이 아니라 `phase2_checkpoint.json`을 기준으로 보는 것이 맞습니다.

이 문서는 그 체크포인트 500개 결과를 `step6_intervention.py`의 집계 로직과 동일하게 다시 계산한 분석입니다.

요약 JSON:

- [intervention_results_checkpoint_summary.json](/home/jaeik/xaimi/phase1/results/step6_intervention/intervention_results_checkpoint_summary.json)

## 실험 목적

Phase 3의 목적은 앞선 Step 5에서 찾은 핵심 헤드와, Phase 1에서 얻은 방향벡터를 실제로 개입에 사용했을 때:

1. KC를 얼마나 줄일 수 있는지
2. clean QA 성능을 얼마나 유지하는지
3. 둘 사이의 trade-off가 어떤지

를 정량적으로 보는 것입니다.

## 실험 전략

코드:

- [step6_intervention.py](/home/jaeik/xaimi/phase1/scripts/step6_intervention.py)

### 전략 A

- `Layer 17, Head 3`에 대한 mean ablation
- Step 5에서 가장 강한 KC 기여 헤드로 나온 `L17H3`를 직접 약화시키는 방식

### 전략 B

정규화된 direction vector steering

- `B1`: Layer 17만 steering
- `B2`: Layer 18만 steering
- `B3`: Layer 17 + 18 동시 steering

alpha 후보:

- `0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3`

전략 B의 최적 조합은 무조건 KC 억제율만 크게 고르는 게 아니라,

- baseline clean accuracy의 `95% 이상`을 유지하는 후보를 우선 남기고
- 그 안에서 `tradeoff_score = KC억제율 - (100 - clean_accuracy)` 가 최대인 조합

을 선택합니다.

## 최신 집계 기준 결과

기준 데이터:

- 체크포인트 처리 수: `500`개
- baseline clean accuracy: `92.2%`
- baseline mean conflict diff: `0.595592`

### 1. 전략 A 결과

- KC 억제율: `57.1%`
- clean accuracy: `27.2%`
- tradeoff score: `-15.71`

type별:

- misinformation: KC 억제 `57.5%`, clean `27.1%`
- temporal: KC 억제 `57.5%`, clean `27.4%`
- semantic: KC 억제 `56.3%`, clean `27.1%`

### 해석

전략 A는 KC 억제 자체는 매우 강합니다.  
500개 중간 결과 기준으로 보면, `L17H3`를 직접 건드리는 것만으로 평균 KC 세기를 절반 이상 줄입니다.

하지만 부작용이 너무 큽니다.

- baseline clean accuracy가 `92.2%`였는데
- 전략 A에서는 `27.2%`까지 내려갑니다.

즉, `L17H3`는 KC에 기여하는 헤드이기도 하지만, 동시에 정상적인 정답 선택에도 깊게 관여하고 있다는 뜻입니다.

정리하면:

- 원인 타격력은 강함
- 선택성은 매우 낮음

입니다.

## 2. 전략 B 전체 패턴

전략 B는 전체적으로 굉장히 보수적인 결과를 보입니다.

대부분의 조합에서:

- clean accuracy는 `92%~95%` 수준으로 잘 유지됨
- KC 억제율은 `-0.03% ~ 0.17%` 수준으로 거의 0에 가까움

즉, 안전하지만 거의 효과가 없습니다.

### 각 조합 요약

#### B1: L17 only

- 가장 좋은 억제율: `0.17%` at `alpha=0.3`
- 하지만 clean accuracy는 `91.6%`
- tradeoff score는 `-8.23`

대체로:

- 약한 억제
- clean은 대체로 유지
- 그래도 “좋다”고 부를 만큼의 KC 감소는 없음

#### B2: L18 only

- 최고 tradeoff 후보는 아님
- 억제율은 대체로 `-0.03% ~ 0.11%`
- clean accuracy는 `92.8% ~ 94.0%`

이전 10개 결과에서는 `B2`가 최적으로 보였지만,
이번 500개 체크포인트 기준에서는 그 우위가 사라졌습니다.

#### B3: L17+18

- 최적 조합: `alpha=0.03`
- KC 억제율: `0.03%`
- clean accuracy: `94.6%`
- tradeoff score: `-5.37`

즉, 현재 체크포인트 기준에서 전략 B 최적은 `B3, alpha=0.03`입니다.

하지만 이 결과를 좋다고 보기는 어렵습니다.

왜냐하면:

- clean accuracy는 잘 유지했지만
- KC 억제율이 사실상 `0%`에 가깝기 때문입니다.

## 3. 전략 B 최적 조합 분석

현재 체크포인트 기준 최적 조합:

- 조건: `B3`
- 라벨: `L17+18`
- alpha: `0.03`
- KC 억제율: `0.03%`
- clean accuracy: `94.6%`
- tradeoff score: `-5.37`

type별:

- misinformation: KC 억제 `-0.01%`, clean `94.6%`
- temporal: KC 억제 `0.09%`, clean `94.6%`
- semantic: KC 억제 `0.00%`, clean `94.6%`

### 해석

이 조합의 장점은 딱 하나입니다.

- clean 성능을 거의 해치지 않음

하지만 핵심 목표였던 KC 억제는 사실상 실패입니다.

특히 misinformation에서는 억제율이 약간 음수라서,
평균적으로는 오히려 KC가 아주 미세하게 더 강해졌다고도 볼 수 있습니다.

그래서 이 조합은 “좋은 개입”이라기보다:

- 전체 후보 중 덜 나쁜 것
- 성능은 안전하지만 개입 효능은 거의 없는 것

으로 보는 게 맞습니다.

## 4. 10개 실행 결과와 500개 체크포인트 결과 비교

이전 `intervention_results.json` 10개 결과에서는:

- 전략 A: KC 억제 `40.9%`, clean `40.0%`
- 전략 B 최적: `B2, alpha=0.2`, KC 억제 `0.1%`, clean `90.0%`

이번 500개 체크포인트 기준에서는:

- 전략 A: KC 억제 `57.1%`, clean `27.2%`
- 전략 B 최적: `B3, alpha=0.03`, KC 억제 `0.03%`, clean `94.6%`

이 비교에서 보이는 변화는 다음과 같습니다.

### 전략 A

- KC 억제는 더 강해짐: `40.9% -> 57.1%`
- clean 성능은 더 나빠짐: `40.0% -> 27.2%`

즉, 표본이 커지면서 “강한 억제 + 큰 부작용” 패턴이 더 분명해졌습니다.

### 전략 B

- 10개 결과에서는 `B2`가 가장 나아 보였음
- 500개 결과에서는 `B3`가 tradeoff 기준 최적으로 바뀜
- 하지만 억제율 자체는 여전히 거의 0%

즉, 표본이 늘어나면서 전략 B의 본질이 더 분명해졌습니다.

- 안전함
- 그러나 실질적인 KC 억제는 거의 없음

## 5. 지금 시점의 결론

현재 Phase 3를 500개 체크포인트 기준으로 다시 해석하면, 결론은 꽤 명확합니다.

### 결론 1: Step 5에서 찾은 `L17H3`는 진짜 중요한 헤드다

전략 A가 `57.1%` 억제를 만들었다는 것은,
이 헤드가 KC 표현에 실질적으로 관여한다는 강한 증거입니다.

즉, Step 5의 메커니즘 가설은 Phase 3에서 인과적으로 재확인됐다고 볼 수 있습니다.

### 결론 2: 하지만 `L17H3`는 KC 전용 회로가 아니다

clean accuracy가 `27.2%`까지 무너졌다는 것은,
이 헤드가 단순히 “오답 유도 회로”만 담당하는 게 아니라,
정상적인 정답 생성에도 쓰이고 있다는 뜻입니다.

즉:

- 중요한 헤드이긴 하지만
- 선택적으로 제거 가능한 타깃은 아니다

입니다.

### 결론 3: 현재 steering 방식은 너무 약하다

전략 B는 clean 성능을 지키는 데는 성공했지만,
KC 억제율이 거의 `0%`라서 개입으로서는 의미가 약합니다.

이는 보통 다음 둘 중 하나를 시사합니다.

1. direction vector 자체가 너무 전역적이어서 KC 특이성이 약하다
2. 적용 위치(L17/L18 resid_post final position)가 개입 지점으로는 충분히 직접적이지 않다

## 6. 왜 이런 결과가 나왔는가

현재까지의 Step 1~5 결과와 묶어서 보면 자연스러운 해석은 이렇습니다.

1. KC는 late layer에서 강하게 굳어진다
2. 그 과정에 `L17H3` 같은 강한 헤드가 실제로 관여한다
3. 하지만 그 헤드는 정상 답변 회로와도 공유되어 있다
4. 그래서 직접 ablation은 KC도 깨지만 정상 능력도 같이 무너진다
5. 반대로 residual steering은 너무 보수적이라 회로를 충분히 흔들지 못한다

즉, 현재 개입 실패는 “메커니즘을 못 찾았다”기보다
“찾은 메커니즘이 너무 공용 회로라서 선택적으로 제어하기 어렵다”에 더 가깝습니다.

## 7. 다음 실험 제안

지금 결과를 바탕으로 다음 우선순위는 아래가 좋아 보입니다.

### 1. 전략 A를 더 국소화하기

지금은 `L17H3`를 final position에서 바로 mean ablation 합니다.
다음은 더 좁혀볼 만합니다.

- 특정 토큰 위치에서만 ablation
- 특정 케이스 그룹에서만 ablation
- head 전체가 아니라 일부 subspace만 제거

핵심은 “헤드 전체 제거”에서 “헤드 내부의 KC 관련 부분만 약화”로 가는 것입니다.

### 2. direction vector를 다시 정의하기

현재 vector는 layer 평균 neutral minus conflict residual입니다.
좀 더 선택적으로 바꾸려면:

- conflict type별 vector
- early / late transition group별 vector
- correct-vs-replace logit gradient 기반 vector

처럼 더 조건부인 방향을 써보는 편이 낫습니다.

### 3. 개입 위치를 바꾸기

현재 B 전략은 `resid_post`의 final position에만 들어갑니다.
하지만 앞선 결과를 보면:

- late layer가 중요하고
- context/final 경로가 모두 얽혀 있으므로

다음 위치도 볼 가치가 있습니다.

- `attn_out` 또는 `mlp_out`
- final position이 아니라 특정 context token position
- `L17H3`의 출력이 실제로 강한 토큰 위치

## 8. 한 줄 요약

2026년 4월 26일 기준 최신 Phase 3 상태는,
`intervention_results.json`의 10개 테스트가 아니라 `phase2_checkpoint.json`의 500개 중간 결과로 봐야 합니다.

그 기준에서 보면:

- `L17H3` ablation은 KC를 강하게 줄이지만 일반 성능을 크게 무너뜨리고
- `L17/L18` steering은 일반 성능은 지키지만 KC를 거의 줄이지 못합니다.

즉, 지금의 Phase 3는

- 메커니즘 확인에는 성공했지만
- 실용적인 억제 개입에는 아직 실패한 상태

라고 정리하는 것이 가장 정확합니다.

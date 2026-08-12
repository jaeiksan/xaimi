# Step 2 IG 1000 Case 결과 분석

## 실험 개요

Step 2는 Integrated Gradients(IG)를 사용해 KC를 유발하는 prompt token 후보를 탐색하는 단계다. 이번 실행은 `kc_cases_1000.json`의 1000개 KC case 전체에 대해 수행되었다.

- 모델: `google/gemma-2-2b`
- 방법: Manual Integrated Gradients, TransformerLens bfloat16 autograd
- 타겟: `log_prob(replace_option) - log_prob(correct_option)`
- baseline: conflict context가 없는 neutral prompt
- 성공 case: 1000개
- unstable / skipped case: 0개

결과 파일:

- `phase1/results/step2_ig/ig_results.json`
- `phase1/results/step2_ig/ig_stats.json`
- `phase1/results/step2_ig/group_attribution_by_type.png`
- `phase1/results/step2_ig/group_attribution_by_transition.png`
- `phase1/results/step2_ig/top_tokens_by_type.png`
- `phase1/results/step2_ig/severity_vs_attribution.png`

## 핵심 결과

전체 평균 attribution 비율은 다음과 같다.

| 그룹 | 평균 attribution |
|---|---:|
| context_evidence | 84.79% |
| format_tokens | 8.97% |
| context_claim | 6.24% |
| question | 0.00% |
| options | 0.00% |

가장 중요한 결론은 **KC를 만드는 attribution이 context evidence에 압도적으로 집중되어 있다**는 점이다. 전체 attribution의 약 85%가 evidence 부분에 할당되었고, claim은 약 6%, format token은 약 9%였다.

즉 모델의 answer flip은 conflict claim 한 문장 자체보다는, 그 뒤에 붙은 evidence-style context 전체에 의해 주로 지지되는 것으로 보인다.

## Conflict Type별 결과

| conflict type | context_claim | context_evidence | format_tokens |
|---|---:|---:|---:|
| misinformation | 5.28% | 85.50% | 9.22% |
| temporal | 8.19% | 83.06% | 8.74% |
| semantic | 5.25% | 85.80% | 8.95% |

세 conflict type 모두 evidence attribution이 83-86%로 매우 높다. type 간 패턴은 전반적으로 비슷하지만, temporal conflict에서 claim attribution이 상대적으로 높다.

해석:

- misinformation / semantic은 evidence가 거의 전적으로 answer flip을 지지한다.
- temporal은 claim 부분의 역할이 약간 커진다. 시간 변화, 대체 사실, 최신성 같은 signal이 claim 문장에 더 직접적으로 포함되기 때문일 가능성이 있다.

## Transition Group별 결과

| transition group | n | context_claim | context_evidence | format_tokens |
|---|---:|---:|---:|---:|
| early | 301 | 6.24% | 85.21% | 8.54% |
| mid | 646 | 6.24% | 84.60% | 9.16% |
| late | 16 | 7.75% | 81.98% | 10.26% |
| no_b | 37 | 5.53% | 85.87% | 8.60% |

early와 mid는 거의 동일한 구조를 보인다. late group은 claim과 format token 비율이 조금 더 높고 evidence 비율이 낮지만, n=16으로 표본이 매우 작기 때문에 강하게 해석하면 안 된다.

전체적으로 transition timing이 달라도 evidence dominance는 유지된다. 따라서 KC의 직접적 token-level driver는 transition layer의 빠르기보다 conflict evidence 자체에 더 안정적으로 묶여 있는 것으로 보인다.

## Severity와 Attribution의 관계

`context_claim attribution`과 `kc_severity`의 Pearson correlation:

```text
r = -0.0807, p = 0.0106
```

통계적으로는 p-value가 작지만, effect size는 매우 작다. 실질적으로는 claim attribution이 커진다고 KC severity가 강해진다고 보기 어렵다.

추가로 직접 계산한 correlation도 다음과 같다.

- context_claim vs severity: 약 -0.081
- context_evidence vs severity: 약 -0.005
- format_tokens vs severity: 약 +0.084

즉 KC severity는 단순히 특정 group attribution 비율 하나로 설명되지는 않는다. Evidence가 attribution의 대부분을 차지하지만, severity의 크기는 evidence attribution 비율보다는 evidence 내용, 모델 내부 상태, layer transition 패턴 등 다른 요인의 영향을 받을 가능성이 크다.

## Question / Options Attribution이 0인 이유

이번 결과에서 `question`, `options`, `replace_option_attr`, `correct_option_attr`가 모두 0으로 나온다. 이는 곧바로 버그라고 보기보다는 현재 IG baseline 설계의 직접적인 결과다.

현재 baseline은 conflict context가 없는 neutral prompt다.

```text
Conflict prompt = Context + Question + Options + Answer
Neutral prompt  =           Question + Options + Answer
```

코드에서는 neutral prompt embedding을 conflict prompt 길이에 맞춰 front-padding한다. 따라서 prompt의 뒤쪽에 있는 `Question + Options + Answer` 부분은 conflict prompt와 neutral prompt에서 동일하게 정렬된다. IG의 delta는 `conflict_emb - neutral_emb`이므로, 동일하게 정렬된 question/options 부분은 delta가 거의 0이 된다.

따라서 이 실험은 “전체 prompt에서 무엇이 중요한가”라기보다, **conflict context가 추가되면서 replace-vs-correct logit 차이를 만드는 token이 어디인가**를 측정한다. 이 관점에서는 question/options attribution이 0인 것이 자연스럽다.

다만 “선택지 자체의 영향”이나 “replace option token의 직접 attribution”을 보고 싶다면 다른 baseline이 필요하다. 예를 들어 zero embedding baseline, blank prompt baseline, 또는 option-masked baseline을 별도로 설계해야 한다.

## Top Token 결과 해석

`top_tokens_by_type`에는 type별로 attribution이 큰 token들이 저장되어 있다. 하지만 상위 token 다수가 count 1-3 수준의 희귀 token이다.

예:

- misinformation: `Bursa`, `spanned`, `citations`, `POLITICS`
- temporal: `opened`, `Stade`, `Wings`, `replaced`
- semantic: `shrouded`, `Pant`, `HOL`, `Mou`

따라서 top token 목록을 “일반적으로 중요한 단어”로 해석하기는 어렵다. 현재 aggregation은 token별 평균 attribution을 사용하므로, 한두 case에서 매우 큰 attribution을 받은 희귀 token이 상위에 올라올 수 있다.

논문/발표에서는 token 목록 자체보다 다음 결론을 강조하는 것이 안전하다.

- high-attribution token들은 대부분 context_evidence에 위치한다.
- top token의 의미론적 공통성보다는 위치/그룹 attribution이 더 안정적인 신호다.
- token-level 분석은 frequency threshold를 추가한 재분석이 필요하다.

## 이전 Step 2와 비교한 해석

이번 1000 case 결과는 표본이 커졌기 때문에 group-level 결론이 더 안정적이다. 특히 evidence attribution dominance가 모든 conflict type과 transition group에서 반복된다.

핵심적으로, Step 2는 다음 가설을 지지한다.

> Conflict context 중에서도 evidence segment가 replace answer 쪽 logit을 밀어 올리는 주된 token-level source다.

반면 claim 자체는 평균적으로 작은 비중을 차지한다. claim은 conflict direction을 명시할 수 있지만, 모델이 실제로 replacement answer로 넘어가는 과정에서는 evidence-style text가 더 넓고 강한 attribution mass를 제공하는 것으로 보인다.

## 한계 및 다음 분석 제안

1. 현재 IG는 context insertion effect를 측정한다.
   - question/options 영향은 baseline에서 제거되어 0으로 나온다.
   - 전체 prompt attribution으로 해석하면 안 된다.

2. top token 분석은 희귀 token에 민감하다.
   - count threshold를 적용한 top token 재집계가 필요하다.
   - 예: count >= 5 또는 count >= 10 token만 사용.

3. late transition group은 n=16으로 작다.
   - late group의 claim/format 증가 패턴은 참고용으로만 해석해야 한다.

4. severity와 group attribution의 상관은 매우 약하다.
   - severity를 설명하려면 layer-level causal tracing, path patching, head ablation 결과와 결합해야 한다.

## 최종 결론

1000개 case에 대한 Step 2 IG 결과는 KC 유발 신호가 conflict evidence에 강하게 집중되어 있음을 보여준다. 평균적으로 context_evidence가 전체 attribution의 약 85%를 차지하며, misinformation, temporal, semantic 모든 conflict type에서 이 패턴이 유지된다.

따라서 이후 단계의 causal tracing/path patching/head ablation에서는 단순히 claim token만 보는 것보다, evidence segment에서 형성된 representation이 어느 layer와 head를 통해 answer logit으로 전달되는지를 추적하는 것이 더 중요하다.

한 문장 요약:

> Step 2 IG over 1000 KC cases suggests that the model's conflict-induced answer flip is primarily driven by the evidence segment of the conflicting context, rather than by the question/options or the claim sentence alone.

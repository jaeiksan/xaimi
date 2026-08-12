# XAIMI 현재 실험 종합 분석

마지막 정리 시점: 2026-05-14 (Step 2 IG n=1000 결과 반영)

이 문서는 `/home/jaeik/xaimi`에서 `archived_previous_experiments/`를 제외한 현재 활성 워크스페이스만을 기준으로 작성했다. 따라서 아래 분석은 과거 A/B/C 실험 아카이브가 아니라, 현재 남아 있는 `phase0`, `step1~5`, `phase3_intervention` 파이프라인과 그 결과물에 대한 해석이다.

관련 코드와 결과:

- [phase0_collect_kc.py](/home/jaeik/xaimi/scripts/phase0_collect_kc.py)
- [step1_tuned_lens.py](/home/jaeik/xaimi/phase1/scripts/step1_tuned_lens.py)
- [step2_ig.py](/home/jaeik/xaimi/phase1/scripts/step2_ig.py)
- [step3_causal_tracing.py](/home/jaeik/xaimi/phase1/scripts/step3_causal_tracing.py)
- [step4_path_patching.py](/home/jaeik/xaimi/phase1/scripts/step4_path_patching.py)
- [step5_head_ablation.py](/home/jaeik/xaimi/phase1/scripts/step5_head_ablation.py)
- [step6_intervention.py](/home/jaeik/xaimi/phase1/scripts/step6_intervention.py)

## 1. 이 연구가 현재 무엇을 하려는가

이 프로젝트는 knowledge conflict를 단순한 실패 사례로 보지 않는다. 오히려 LLM 내부에서 서로 다른 두 지식 신호, 즉 파라메트릭 지식과 문맥 주입 지식이 어떤 시점과 어떤 회로에서 경쟁하고, 그 경쟁이 어떻게 최종 답 선택으로 이어지는지를 드러내는 창으로 본다.

현재 활성 파이프라인은 이 관점을 매우 일관되게 반영한다.

1. 먼저 모델이 실제로 conflict에 빠지는 사례만 모은다.
2. 그 다음 conflict preference가 어느 layer와 어느 position에서 나타나는지 localize한다.
3. 그 변화가 단순 상관이 아니라 실제 원인인지 causal하게 검증한다.
4. 그 뒤 late-stage mechanism을 attention, MLP, head 수준으로 더 세분화한다.
5. 마지막으로 activation steering이나 mean ablation 같은 최소개입 방식으로 완화 가능성을 시험한다.

즉, 이 저장소의 현재 실험들은 “문제를 고친다”보다 “문제가 어떤 내부 경쟁 구조로 발생하는지 설명한다”에 더 무게가 실려 있다. 그리고 지금 상태는 완성된 intervention 연구라기보다, 그런 연구가 가능한지를 점검하는 pilot에 가깝다.

## 2. 전체 실험 파이프라인

현재 워크스페이스는 `google/gemma-2-2b`를 대상으로 다음 순서로 구성되어 있다.

1. Phase 0: conflict 사례 수집
2. Step 1: layerwise transition localization
3. Step 2: token-level attribution
4. Step 3: causal tracing
5. Step 4: path patching으로 late-layer effect 분해
6. Step 5: head-level ablation
7. Phase 3: mechanism-based intervention

이 순서는 방법론적으로도 자연스럽다. Step 1이 “어디를 봐야 하는가”를 말해주고, Step 2가 “무엇이 conflict를 밀고 들어오는가”를 보여주며, Step 3이 “그 지점이 정말 원인인가”를 확인한다. 이후 Step 4와 Step 5는 intervention 후보를 고를 수 있을 만큼 회로를 더 좁히는 역할을 한다.

## 3. Phase 0: KC 사례 수집

파일:

- [collection_stats.json](/home/jaeik/xaimi/data/phase0/collection_stats.json)
- [kc_cases_1000.json](/home/jaeik/xaimi/data/phase0/kc_cases_1000.json)

### 왜 이 실험을 했는가

이후의 mechanistic analysis는 모두 “모델이 실제로 conflict를 일으킨 사례” 위에서만 의미가 있다. 따라서 가장 먼저 해야 할 일은 원 데이터셋 전체가 아니라, 해당 모델에서 clean prompt에는 정답을 고르고 conflict prompt에는 replacement answer로 뒤집히는 사례만 골라내는 것이다.

즉 이 단계는 데이터 전처리라기보다, “Gemma-2-2B가 실제로 두 지식 신호 사이에서 경쟁하는 장면”만 남기는 실험적 필터링 단계다.

### 어떻게 구현했는가

- `clean_prompt`와 `conflict_prompt`를 각각 구성한다.
- 마지막 토큰에서 `A/B/C/D` 각 선택지의 log-prob을 직접 비교한다.
- 공백 포함 토큰과 비포함 토큰 두 variant를 모두 검사해 더 높은 점수를 채택한다.
- clean에서 `correct_option`이 최고이고, conflict에서 `replace_option`이 최고인 경우만 KC 사례로 채택한다.
- conflict 조건에서 `logP(replace) - logP(correct)`를 `kc_severity`로 저장한다.

이 정의는 매우 operational하다. 따라서 여기서 만들어진 데이터는 “일반적인 지식 충돌 데이터셋”이 아니라, “Gemma-2-2B에서 conflict가 실제 발현된 사례 모음”이다.

### 결과는 어떠한가

- 총 처리 row: `900`
- 시도한 총 case: `1010`
- 최종 수집 KC case: `1000`
- 유형 균형:
  - misinformation `334`
  - temporal `333`
  - semantic `333`
- KC rate:
  - misinformation `97.9%`
  - temporal `99.7%`
  - semantic `99.4%`
- severity 평균: `3.1669`

### 의미는 무엇인가

이 결과는 두 가지를 뜻한다.

첫째, 현재 conflict prompt 설계가 매우 강하다. 거의 모든 케이스에서 문맥 주입만으로 clean 정답이 replace 정답으로 뒤집힌다.

둘째, 이후 모든 해석은 “원래 conflict가 약하게 일어나는 자연 분포”가 아니라 “이미 flip이 확인된 강한 conflict 사례”에 대한 해석이라는 점을 전제로 해야 한다. 다시 말해, 이 파이프라인은 prevalence estimation이 아니라 mechanism isolation에 적합하다.

## 4. Step 1: Tuned Lens 기반 transition localization

파일:

- [tuned_lens_stats.json](/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_stats.json)
- [preference_trajectory.png](/home/jaeik/xaimi/phase1/results/step1_tuned_lens/preference_trajectory.png)
- [transition_layer_distribution.png](/home/jaeik/xaimi/phase1/results/step1_tuned_lens/transition_layer_distribution.png)

### 왜 이 실험을 했는가

연구 질문의 첫 단계는 “conflict가 모델 내부 어디에서 처음 보이기 시작하는가”다. 단순히 최종 출력이 틀렸다는 사실만으로는, early ingestion 문제인지, mid-layer transformation 문제인지, late commitment 문제인지 구분할 수 없다.

그래서 이 단계는 knowledge conflict가 layer별로 언제 정답 쪽에서 conflict answer 쪽으로 넘어가는지, 그리고 그 전환이 prompt 내 어떤 위치에서 먼저 드러나는지 찾기 위한 localization 실험이다.

### 어떻게 구현했는가

- 각 레이어의 `resid_post` 위에 linear probe를 학습한다.
- 세 위치를 따로 본다.
  - `A`: 최종 답 예측 위치
  - `B`: `Answer:` 직전 위치
  - `C`: context 블록 끝 위치
- `A/B`는 neutral run residual을 기반으로 probe를 학습한 뒤 conflict run에 적용한다.
- `C`는 conflict run residual과 final log-prob을 따로 맞추는 방식으로 학습한다.
- probe가 예측한 preference가 정답 쪽에서 replace 쪽으로 넘어가는 첫 레이어를 transition layer로 정의한다.

이 설계는 단순한 logit lens보다 조금 더 직접적으로 “내부 표현이 어느 시점부터 conflict answer를 지지하느냐”를 보게 해준다.

### 결과는 어떠한가

전체 `1000`개 케이스 기준:

- `A`: 전 케이스 transition, 평균 `6.744`, 중앙값 `7`
- `B`: `963`개 transition, `37`개 no-transition, 평균 `10.405`, 중앙값 `11`
- `C`: 전 케이스 transition, 평균 `1.727`, 중앙값 `1`

유형별로도 패턴은 거의 동일하다.

### 의미는 무엇인가

이 실험은 현재 워크스페이스에서 가장 중요한 구조적 결과 중 하나다.

핵심은 conflict가 한 번에 생기지 않는다는 점이다.

1. context block 끝 위치 `C`에서는 거의 초기 layer에서 이미 conflict 방향 신호가 생긴다.
2. 하지만 실제 최종 답 예측 위치 `A`에서 preference가 뒤집히는 것은 평균 layer 7 부근이다.
3. `Answer:` 직전 위치 `B`는 더 늦게, 평균 layer 11 부근에서야 transition한다.

즉, 문맥 신호의 유입과 최종 답 선택의 확정은 다른 단계다. 이는 conflict를 “early contamination” 하나로 설명하기보다, “초기 유입 후 중후반에 재통합되고 최종 경로에서 굳어지는 경쟁 과정”으로 봐야 함을 시사한다.

특히 `B` 위치의 `37`개 no-transition 사례는 중요하다. 이들은 pre-answer representation이 명확히 뒤집히지 않았는데도 최종 출력은 conflict answer로 가는 경우다. 이는 decision commitment가 특정 위치 하나의 표상만으로 설명되지 않고, 다른 residual path나 late head의 기여를 통해 형성될 수 있음을 암시한다.

## 5. Step 2: Integrated Gradients로 입력 토큰 기여 분석

파일:

- [ig_stats.json](/home/jaeik/xaimi/phase1/results/step2_ig/ig_stats.json)
- [group_attribution_by_type.png](/home/jaeik/xaimi/phase1/results/step2_ig/group_attribution_by_type.png)
- [group_attribution_by_transition.png](/home/jaeik/xaimi/phase1/results/step2_ig/group_attribution_by_transition.png)

### 왜 이 실험을 했는가

Step 1이 “어디서 뒤집히는가”를 봤다면, Step 2는 “무슨 입력이 그 뒤집힘을 밀어 넣는가”를 보는 단계다. 연구 프레이밍상 중요 포인트는 conflict를 유발하는 요소가 단순히 옵션 문구인지, 질문 자체인지, 아니면 문맥 내부의 특정 evidence span인지 가르는 것이다.

### 어떻게 구현했는가

- TransformerLens 기반 수동 Integrated Gradients를 사용한다.
- baseline은 neutral prompt embedding이며, conflict prompt 길이에 맞춰 padding한다.
- target score는 `logP(replace_option) - logP(correct_option)`이다.
- 입력 토큰을 5개 그룹으로 묶는다.
  - `context_claim`
  - `context_evidence`
  - `question`
  - `options`
  - `format_tokens`
- Step 1 transition group과도 연결해 attribution 차이를 볼 수 있게 한다.

### 결과는 어떠한가

전체 `1000`개 케이스 기준 그룹별 attribution 평균 (양수 = replace_option 지지, KC 유발 방향):

| 그룹 | 평균 attribution | std | 양수 비율 |
|---|---|---|---|
| `context_evidence` | `+0.848` | `0.038` | `100%` |
| `format_tokens` | `+0.090` | `0.028` | `100%` |
| `context_claim` | `+0.062` | `0.027` | `100%` |
| `question` | `0.000` | `0.000` | `0%` |
| `options` | `0.000` | `0.000` | `0%` |

conflict_type별로 나눠도 패턴은 구조적으로 동일하지만, temporal만 소폭 다르다.

| | `context_claim` | `context_evidence` |
|---|---|---|
| misinformation (n=334) | `+0.053` | `+0.855` |
| temporal (n=333) | `+0.082` | `+0.831` |
| semantic (n=333) | `+0.053` | `+0.858` |

temporal은 context_claim 비중이 다른 유형보다 약간 높고 evidence가 살짝 낮다. 이는 시간적 주장 자체(“As of 2019, …”)가 evidence 서술만큼이나 직접적인 KC 유발 역할을 하는 구조의 특성으로 해석된다.

transition_group별(early/mid/late/no_b)로는 유의미한 차이가 없다. 어느 그룹이든 context_evidence가 약 85%를 차지한다.

kc_severity와 context_evidence attribution 사이의 상관은 Pearson r = `-0.005` (p = `0.886`)로 실질적으로 0이다. 즉 severity가 높아도 어느 토큰이 KC를 유발하는지 패턴은 바뀌지 않는다.

**top_tokens 1위 분포 주의사항**: `<bos>` 토큰이 `936`건에서 top-1 토큰으로 나타난다. 이는 zero-embedding baseline을 사용하는 IG의 알려진 아티팩트로, BOS는 항상 큰 gradient를 받기 때문이다. 실질적 KC 유발 토큰으로 해석해서는 안 된다. BOS를 제외하면 context_evidence 그룹 내 `'\n\n'`, `' is'`, `','` 등이 상위에 나타난다.

`options_attribution`은 replace_option과 correct_option 모두 `0.000`이며, replace > correct 비율도 `0.0%`다. 즉 선택지 문구 자체는 KC에 전혀 기여하지 않는다.

### 의미는 무엇인가

n=1000 전체 케이스에서도 소규모 pilot과 동일한 구조가 안정적으로 확인됐다.

**핵심 결론**: KC는 파라메트릭 지식 토큰(question, options)이 억압되는 방식이 아니라, context 내 evidence 서술이 적극적으로 replace_option을 driving하는 방식으로 발생한다. context_evidence의 attribution이 평균 84.8%이고, 전체 1000케이스에서 예외 없이 양수라는 점이 이를 강하게 뒷받침한다.

이 결과는 Phase 3 intervention 결과와도 정합적이다. Strategy A에서 L17H3를 죽이면 KC suppression 57.5%가 달성되지만 clean accuracy가 25.9%로 붕괴한다. 이는 KC를 유발하는 경로가 clean answer circuitry와 구조적으로 얽혀 있어, evidence-driven input signal을 선택적으로만 억제하기 어렵다는 사실과 일치한다.

또한 transition_group별 차이가 없다는 점은, conflict 유발의 입력 측 메커니즘이 “문맥이 언제 flip을 유발하는가”와 독립적임을 의미한다. 즉 KC의 토큰 기여 구조는 transition timing에 따라 다른 패턴을 갖지 않는 단일 메커니즘에 가깝다.

이후 실험에서 evidence-to-answer 경로를 더 직접 겨냥하는 path patching이나 evidence span 내 token-level patching이 자연스러운 다음 단계다.

## 6. Step 3: Causal Tracing으로 원인 위치 검증

파일:

- [ct_stats.json](/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ct_stats.json)
- [ie_heatmap_overall.png](/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ie_heatmap_overall.png)
- [ie_layer_profile.png](/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ie_layer_profile.png)

### 왜 이 실험을 했는가

Step 1의 transition은 상관적 localization이다. 하지만 어떤 레이어에서 representation이 flip된다고 해서, 그 지점이 정말 원인이라고 단정할 수는 없다. 따라서 다음 단계는 특정 layer와 segment를 neutral 상태로 복원했을 때 정답 확률이 실제로 회복되는지를 보는 causal test여야 한다.

이 실험은 바로 그 점을 겨냥한다. 즉 “conflict가 보이는 위치”가 아니라 “correct behavior를 되살리는 데 실제로 지렛대가 되는 위치”를 찾는 것이다.

### 어떻게 구현했는가

- neutral prompt run에서 각 레이어 residual을 캐싱한다.
- conflict prompt에서는 context embedding에 가우시안 노이즈를 넣어 corrupted run을 만든다.
- 각 `layer × segment` 조합에 대해 neutral residual slice를 복원한다.
- `IE(layer, segment) = P_restored(correct) - P_corrupted(correct)`를 계산한다.
- 세그먼트는 다음 다섯 개다.
  - `context_first_sent`
  - `context_rest`
  - `question`
  - `options`
  - `final`

이 방식은 Rome-style activation patching의 segment-level 변형으로 볼 수 있다.

### 결과는 어떠한가

전체 `1000`개 케이스 기준:

- corrupted correct-answer probability 평균: `0.2383`
- `final` segment의 late layer에서 가장 큰 positive IE가 나타난다.
- 최고 peak layer는 주로 `17~22` 사이에 몰린다.
- max IE 상위 layer:
  - layer `18`: `0.1363`
  - layer `20`: `0.1343`
  - layer `17`: `0.1331`
  - layer `21`: `0.1311`
  - layer `19`: `0.1304`

segment 평균을 보면:

- `context_first_sent`: 음수
- `context_rest`: 음수
- `question`: 거의 0
- `options`: 약한 음수
- `final`: 뚜렷한 양수

### 의미는 무엇인가

이 결과는 연구의 핵심 주장과 잘 맞는다.

Step 1에서 context signal은 매우 이르게 들어왔지만, Step 3에서는 정답을 복원하는 실질적 causal leverage가 late layers의 final segment에 집중된다. 즉 conflict는 early ingestion에서 시작될 수 있지만, 최종적 승패는 late decision pathway에서 결정된다.

이 점은 연구적으로 중요하다. 단순히 “문맥이 모델을 오염시킨다”는 수준에서 멈추지 않고, “오염된 신호가 후반부 어느 경로에서 의사결정으로 고정되는가”라는 더 정밀한 설명이 가능해지기 때문이다.

또한 context segment를 early layer에서 복원하는 것이 평균적으로 유익하지 않다는 점은, conflict가 한번 내부 회로로 흡수된 뒤에는 raw context 자체보다 late residual state가 더 직접적인 제어점이 된다는 뜻으로 해석할 수 있다.

## 7. Step 4: Path patching으로 attention과 MLP 분해

파일:

- [pp_stats.json](/home/jaeik/xaimi/phase1/results/step4_path_patching/pp_stats.json)
- [attn_vs_mlp_by_layer.png](/home/jaeik/xaimi/phase1/results/step4_path_patching/attn_vs_mlp_by_layer.png)
- [ie_explanation_ratio.png](/home/jaeik/xaimi/phase1/results/step4_path_patching/ie_explanation_ratio.png)

### 왜 이 실험을 했는가

Step 3에서 late residual이 중요하다는 사실은 알았지만, residual 전체는 너무 넓은 설명 단위다. intervention 가능한 수준의 메커니즘으로 가려면 late causal effect가 attention output인지, MLP output인지, 아니면 둘의 상호작용인지 더 쪼개야 한다.

### 어떻게 구현했는가

- Step 3에서 중요한 late window를 따라 layers `13~25`만 본다.
- final token 위치에서만 patching한다.
- 각 레이어마다 세 가지를 비교한다.
  - `hook_attn_out`
  - `hook_mlp_out`
  - `hook_resid_post`
- 이를 통해 `IE_attn`, `IE_mlp`, `IE_full`을 얻고,
  `(IE_attn + IE_mlp) / IE_full` 설명률도 계산한다.

### 결과는 어떠한가

dominant component:

- attention 우세: `13, 14, 18, 25`
- MLP 우세: `15, 16, 17, 19, 20, 21, 22, 23, 24`

하지만 평균 설명률은 약 `12.4%`에 불과하다.

레이어별 설명률도 다수 레이어에서 낮거나 음수이며, `attn_out + mlp_out`의 단순 합이 `resid_post` 전체 효과를 잘 복원하지 못한다.

### 의미는 무엇인가

이 실험은 긍정적 발견과 부정적 발견을 동시에 준다.

긍정적으로는, late layer의 많은 구간에서 MLP가 attention보다 더 큰 역할을 하는 경향이 보인다. 특히 layer `17`, `19~24`에서 그렇다. 이는 conflict resolution 혹은 conflict consolidation이 단순 retrieval head 하나의 문제가 아니라, 후반부 MLP를 포함한 더 넓은 변환 과정과 연결돼 있음을 시사한다.

부정적으로는, 현재의 component decomposition이 전체 residual 효과를 거의 설명하지 못한다. 이는 다음 중 하나를 시사한다.

- causal effect가 token-to-token 상호작용까지 포함하는 더 분산된 회로에 퍼져 있다.
- single-component patching으로는 회로의 결합 효과를 회수할 수 없다.
- final token 한 점만 보는 설계가 지나치게 좁다.

즉, Step 4는 “late effect가 순수 attention 하나로 설명되지는 않는다”는 제약 결과로서 가치가 크다.

## 8. Step 5: Head-level ablation으로 강한 attention head 찾기

파일:

- [head_ablation_results.json](/home/jaeik/xaimi/phase1/results/step5_head_ablation/head_ablation_results.json)
- [head_suppression_heatmap.png](/home/jaeik/xaimi/phase1/results/step5_head_ablation/head_suppression_heatmap.png)
- [top_heads_profile.png](/home/jaeik/xaimi/phase1/results/step5_head_ablation/top_heads_profile.png)

### 왜 이 실험을 했는가

Step 4가 residual 전체 효과를 단순 분해하기 어렵다고 보여준 상황에서도, intervention 후보를 찾으려면 여전히 “상대적으로 국소적이고 지렛대가 큰 요소”를 찾아야 한다. Step 5는 late-layer attention 중에서도 어떤 head가 conflict를 특히 강하게 떠받치는지를 찾기 위한 단계다.

### 어떻게 구현했는가

- layers `13~25`, heads `0~7`을 대상으로 한다.
- `blocks.{layer}.attn.hook_z`에 mean ablation을 적용한다.
- baseline diff는 `P(replace) - P(correct)`이다.
- ablation 후 같은 diff를 다시 계산한다.
- `suppression = baseline_diff - ablated_diff`로 정의한다.
- 1단계에서 `300`개로 screening하고, 2단계에서 top head를 `1000`개 전체로 검증한다.

positive suppression은 해당 head를 없앴을 때 conflict가 약해진다는 뜻이므로, 곧 그 head가 conflict에 기여한다는 의미다.

### 결과는 어떠한가

stage 1 screening 최고치는 매우 뚜렷하다.

- `L17H3`: `0.3332`

전체 검증 top heads:

1. `L17H3`: `0.3387`
2. `L18H6`: `0.1017`
3. `L23H5`: `0.0410`
4. `L17H4`: `0.0271`
5. `L17H6`: `0.0268`

`L17H3`는 세 conflict type 모두에서 안정적으로 강하다.

### 의미는 무엇인가

이 결과는 현재 워크스페이스에서 intervention으로 가장 직접 이어지는 발견이다.

- conflict support는 전체 attention head에 균등하게 퍼져 있지 않다.
- 특히 `L17H3`가 압도적으로 강한 single-point leverage를 가진다.
- `L18H6`도 보이지만 크기 차이가 상당하다.

이는 late residual effect가 완전히 diffuse한 것만은 아니며, 적어도 일부는 고레버리지 attention subcircuit에 기대고 있음을 보여준다. 즉, “완전히 퍼져 있어서 아무 것도 못 건드린다”는 상황은 아니다. 대신 “건드릴 수는 있지만, 그 회로가 clean 기능과 얼마나 공유되는가”가 다음 문제로 남는다.

## 9. Phase 3: Mechanism-based intervention

파일:

- [intervention_results_checkpoint_summary.json](/home/jaeik/xaimi/phase1/results/step6_intervention/intervention_results_checkpoint_summary.json)
- [tradeoff_curve.png](/home/jaeik/xaimi/phase1/results/step6_intervention/tradeoff_curve.png)
- [suppression_by_type.png](/home/jaeik/xaimi/phase1/results/step6_intervention/suppression_by_type.png)

### 왜 이 실험을 했는가

앞선 Step 1~5는 localization과 explanation에 해당한다. 하지만 연구적으로는 “그렇게 찾은 메커니즘이 실제로 개입 가능한가”를 한 번은 확인해야 한다. 그래서 이 단계는 explanation을 intervention으로 연결하는 pilot test다.

다만 사용자의 연구 방향상, 이 단계의 목적은 완성된 해법 제시보다도 “현재까지 찾은 메커니즘이 진짜 causal handle이 되는지”를 확인하는 데 있다.

### 어떻게 구현했는가

전략 A:

- `L17H3`에 대해 mean ablation

전략 B:

- neutral residual 평균과 conflict residual 평균의 차이로 direction vector를 만든다.
- 이를 정규화한 뒤 `resid_post`에 steering으로 주입한다.
- 세 조건을 시험한다.
  - `B1`: layer 17
  - `B2`: layer 18
  - `B3`: layer 17 + 18
- alpha는 `0.01`부터 `0.3`까지 탐색한다.

현재 최신 해석 기준은 10개짜리 오래된 `intervention_results.json`이 아니라, `500`개 처리된 [intervention_results_checkpoint_summary.json](/home/jaeik/xaimi/phase1/results/step6_intervention/intervention_results_checkpoint_summary.json)이다.

### 결과는 어떠한가

baseline:

- clean accuracy: `92.2%`
- mean baseline diff: `0.595592`

전략 A:

- KC suppression: `57.1%`
- clean accuracy: `27.2%`

전략 B 최적:

- 조건: `B3`
- alpha: `0.03`
- KC suppression: `0.03%`
- clean accuracy: `94.6%`

전략 B 전반 패턴:

- clean accuracy는 대체로 `92~95%`로 유지
- KC suppression은 거의 `0%` 근처

### 의미는 무엇인가

이 결과는 매우 중요하다. 왜냐하면 단순히 intervention이 “잘 됐다/안 됐다”가 아니라, 메커니즘의 성격 자체를 말해주기 때문이다.

전략 A는 KC를 강하게 줄인다. 따라서 `L17H3`는 진짜 원인적 중요도를 가진다. Step 5에서 찾은 head가 단순 correlational artifact는 아니라는 뜻이다.

하지만 clean accuracy가 `27.2%`까지 무너진다. 즉 이 head는 conflict 전용 회로가 아니라 정상적 정답 선택에도 깊이 관여하는 공용 회로일 가능성이 높다.

반면 steering은 clean 성능을 거의 해치지 않지만 KC도 거의 줄이지 못한다. 이는 현재 steering vector가 너무 전역적이거나, 적용 위치가 충분히 직접적이지 않거나, 또는 conflict-specific subspace를 제대로 겨냥하지 못하고 있음을 의미한다.

결국 Phase 3는 다음 결론을 준다.

- 메커니즘은 찾았다.
- 그 메커니즘은 causal하다.
- 하지만 현재 수준의 최소개입은 아직 선택성이 부족하다.

즉 intervention 실패는 “메커니즘을 몰라서”가 아니라 “찾은 메커니즘이 clean 기능과 너무 얽혀 있어서 조심스럽게 분리해내기 어렵다”에 가깝다.

## 10. 현재 결과들을 하나의 메커니즘 이야기로 묶으면

아카이브를 제외한 현재 실험들만 봐도, 스토리는 꽤 일관적이다.

1. conflict evidence는 매우 강하게 모델 내부로 들어온다. 입력 측에서 KC를 거의 전적으로(84.8%) 유발하는 것은 context 내 evidence 서술 토큰이며, question이나 options 문구는 기여가 없다 (n=1000 확인).
2. 그 신호는 context 끝 위치에서 극초기 layer부터 감지된다.
3. 그러나 최종 답 선택의 실제 causal bottleneck은 late layer `17~22` 부근에 있다.
4. 이 late-stage effect는 단순 attention 하나로 다 설명되지 않으며, MLP와 residual-level 통합이 함께 중요하다.
5. 그 안에서도 `L17H3` 같은 일부 head는 conflict를 크게 지지하는 고레버리지 요소다.
6. 하지만 그 요소는 clean answer circuitry와도 공유된다.

따라서 현재 가장 설득력 있는 해석은 다음과 같다.

knowledge conflict는 “초기 문맥 오염” 하나로 끝나는 현상이 아니라, 초기에 유입된 evidence-like signal이 중후반 레이어에서 기존 파라메트릭 지식과 경쟁하고, late decision pathway에서 특정 회로에 의해 안정화되는 현상이다.

이 프레임은 사용자가 제시한 연구 철학과도 정확히 맞닿아 있다. conflict는 제거해야 할 노이즈가 아니라, 모델 내부 지식 경쟁이 어디서 어떻게 발생하는지 보여주는 관찰창이다.

## 11. 현재 연구의 강점

### 1. 실험 설계의 단계성이 좋다

현재 파이프라인은 dataset construction, localization, attribution, causal validation, decomposition, intervention이 자연스럽게 이어진다. 이는 pilot 연구로서 매우 강한 구조다.

### 2. localization과 causal validation이 분리되어 있다

많은 초기 mechanistic 실험은 “어디서 보였다”와 “거기가 원인이다”를 혼동하는데, 여기서는 Step 1과 Step 3이 분리되어 있어 설명력이 훨씬 높다.

### 3. intervention이 explanation에 직접 연결된다

Step 5에서 고른 `L17H3`를 실제로 Phase 3에서 건드려 보는 구조 덕분에, 발견이 단순 서술에 그치지 않고 causal handle인지 확인할 수 있다.

## 12. 현재 연구의 한계

### 1. Step 2의 question/options attribution이 정확히 0이다

n=1000 결과에서 question과 options 그룹 attribution이 모두 0.000이다. 이것이 진짜 기여가 없는 것인지, 아니면 group_attribution 정규화 방식의 부작용인지 검토가 필요하다. 또한 `<bos>` 토큰이 top-1으로 936건 등장하는 것은 zero-embedding baseline의 아티팩트이므로, baseline 설계를 text-level neutral prompt로 바꾸면 해석이 더 깔끔해질 수 있다.

### 2. Step 4의 분해력은 낮다

`attn_out + mlp_out`가 `resid_post` 효과를 거의 설명하지 못하므로, “late effect의 세부 회로”를 말하기에는 아직 거칠다.

### 3. 현재 intervention은 선택적이지 않다

강한 억제는 가능하지만 clean 성능 붕괴가 크고, 안전한 개입은 효과가 거의 없다. 즉 “메커니즘 발견”과 “실용적 제어” 사이에 아직 간극이 있다.

### 4. 단일 모델, 단일 operational definition에 묶여 있다

현재 결과는 Gemma-2-2B와 현재 prompt format에 최적화된 분석이다. 일반화 가능성은 아직 별도 검증이 필요하다.

## 13. 앞으로 가능한 방법론

사용자의 연구 방향이 intervention 이전의 localization과 explanation 정밀화에 더 가깝다는 점을 고려하면, 다음 방법론들이 특히 잘 맞는다.

### 1. position-specific causal tracing

현재는 segment 단위 복원이 중심이다. 다음 단계에서는 동일 segment 내에서도 어떤 토큰 위치가 late layer causal bottleneck과 실제로 연결되는지 finer-grained하게 볼 필요가 있다.

예를 들어:

- evidence token별 patching
- final token뿐 아니라 특정 context token에서의 residual patching
- `L17H3`가 강하게 attend하는 source position만 선택 패치

이렇게 하면 “어느 문장이 conflict를 일으키는가”에서 더 나아가 “어느 토큰 경로가 그 신호를 late answer circuit로 전달하는가”까지 볼 수 있다.

### 2. transition subgroup별 메커니즘 비교

Step 1에서 이미 `early`, `mid`, `late`, `no-transition(B)` 그룹이 나온다. 이 그룹은 매우 좋은 자연 실험 단위다.

가능한 분석:

- 그룹별 IG 비교
- 그룹별 causal tracing profile 비교
- 그룹별 top head 비교
- 그룹별 steering vector 분리 학습

이렇게 하면 “knowledge conflict가 항상 같은 회로로 일어나느냐”와 “서로 다른 transition profile이 서로 다른 메커니즘을 반영하느냐”를 볼 수 있다.

### 3. conflict-type specific mechanism 분석

현재 수치상 misinformation, temporal, semantic은 상당히 비슷하지만, 그것이 곧 메커니즘 동일성을 뜻하지는 않는다.

추천되는 실험:

- type별 direction vector 분리
- type별 top head 재선정
- type별 causal bottleneck layer 비교
- type별 claim/evidence attribution 세분화

이렇게 하면 “semantic conflict는 개념 재정렬 문제이고 temporal conflict는 recency override 문제” 같은 더 이론적인 구분으로 이어질 수 있다.

### 4. subspace-level intervention

현재 mean ablation은 head 전체를 죽이므로 너무 거칠다. 다음은 더 국소적이어야 한다.

가능한 방향:

- `L17H3` 출력의 PCA 또는 SAE-like subspace 분해
- conflict vs clean을 가장 잘 구분하는 head-internal direction만 제거
- gradient-based direction만 부분 subtract
- token-conditioned head output masking

이 방향은 “head 전체 제거”에서 “head 내부의 KC 관련 성분만 약화”로 가는 방법론이다.

### 5. path-specific intervention

현재 steering은 `resid_post` final position에 들어간다. 하지만 Step 3와 Step 5를 보면 진짜 중요한 것은 특정 head와 late path다.

더 나은 개입은 예를 들어:

- `attn.hook_z` 수준 steering
- 특정 source token에 대한 attention contribution만 억제
- `L17H3 -> L18/L19 downstream` 경로만 패치
- final position이 아니라 evidence token representation을 조정

이런 식의 path-specific intervention이 더 선택적일 가능성이 높다.

### 6. representation similarity와 trajectory 분석

transition layer만 보지 말고, clean/conflict residual trajectory의 cosine similarity나 logit difference trajectory를 layer별로 추적하면 “언제부터 conflict run이 clean manifold를 이탈하는가”를 볼 수 있다.

이건 특히 Step 1과 Step 3 사이를 잇는 좋은 중간 설명층이 될 수 있다.

## 14. 이 연구가 가능한가에 대한 판단

현재 코드와 결과만 보면, 이 연구는 충분히 가능하다. 그리고 단순 가능 수준이 아니라, 이미 유의미한 mechanistic signal을 잡아내기 시작한 상태다.

그 근거는 세 가지다.

1. transition localization이 재현성 있게 나온다.
2. causal tracing이 late bottleneck을 명확히 보여준다.
3. head ablation이 개별 causal handle을 실제로 제시한다.

즉, 지금 프로젝트는 “conflict를 해석할 수 있을까?” 단계는 이미 지나서, “어느 수준까지 정밀하게 해석하고 얼마나 선택적으로 개입할 수 있을까?” 단계로 들어와 있다.

다만 아직은 pilot답게, explanation은 꽤 유망하지만 intervention은 거칠다. 따라서 다음 연구의 중심은 새로운 억제 기법을 많이 늘리기보다, 현재 찾은 late mechanism을 더 세분화해 clean 기능과 conflict 기능이 어디서 갈라지는지 찾는 데 두는 편이 훨씬 설득력 있다.

## 15. 추천되는 다음 연구 우선순위

### 1순위

Step 1 subgroup를 기준으로 Step 2~5를 다시 나누어 돌려라.

이건 지금 있는 자산을 가장 효율적으로 활용하는 방법이다. transition timing이 다른 사례가 같은 메커니즘인지 아닌지만 밝혀도 논문의 설명력이 크게 올라간다.

### 2순위

Step 2 IG를 token-level로 세분화하라.

n=1000 결과로 context_evidence가 KC의 거의 전부를 유발한다는 사실은 확정됐다. 다음 단계는 evidence span 내에서 어떤 specific token이 어떤 late-layer path로 전달되는지 finer-grained하게 볼 필요가 있다. 구체적으로는 evidence span 내 token-level patching과, `L17H3`가 강하게 attend하는 source position 분석을 결합하는 것이 자연스럽다.

### 3순위

`L17H3` 전체 ablation 대신 subspace-level suppression을 시도하라.

현재 Phase 3 실패의 본질은 선택성 부족이다. 그러므로 새 head를 더 찾기보다, 이미 찾은 head 안에서 clean과 conflict 성분을 분리하는 쪽이 낫다.

### 4순위

late layer final segment뿐 아니라 evidence-to-final path를 직접 겨냥하는 path patching을 설계하라.

지금까지 결과를 종합하면 conflict의 핵심은 단순 final state가 아니라 evidence signal이 late answer circuit으로 넘어오는 과정일 가능성이 높다.

## 16. 최종 정리

현재 `archived_previous_experiments/`를 제외한 활성 워크스페이스는, knowledge conflict를 failure case가 아니라 내부 지식 경쟁 메커니즘의 창으로 보려는 연구 방향과 매우 잘 맞는 구조를 갖고 있다.

가장 강한 결론은 다음과 같다.

- conflict를 유발하는 입력 측 신호는 context_evidence 토큰에 거의 전적으로 집중된다 (attribution 84.8%, n=1000, conflict type·transition group 무관하게 안정).
- 그 신호는 초기 layer에서 context 끝 위치에 먼저 감지된다.
- 그러나 최종 답 선택을 뒤집는 실제 causal bottleneck은 late layers `17~22` 부근에 있다.
- 그 과정에는 `L17H3` 같은 고레버리지 head가 실제로 관여한다.
- 하지만 그 회로는 clean 정답 생성과도 공유되어 있어, 현재 개입은 강하면 부작용이 크고 안전하면 효과가 거의 없다.

따라서 이 연구의 다음 단계는 “더 강한 intervention”을 찾는 것이 아니라, “late conflict circuit 내부에서 clean 기능과 conflict 기능이 어디서 분기하는지”를 더 정밀하게 localize하고 explain하는 것이다. 바로 그 지점이 이 프로젝트가 가장 잘할 수 있는 다음 연구 주제다.

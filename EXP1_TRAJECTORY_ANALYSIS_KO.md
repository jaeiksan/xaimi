# Exp1 KC Trajectory 결과 분석

## 실험 목적

`exp1_trajectory.py`는 KC case에서 residual stream에 나타나는 `conflict - clean` 차이를 layer별로 모아 SVD를 수행하고, 각 layer의 1st singular vector를 KC direction으로 정의한다.

비교 조건은 세 가지다.

- `conflict`: conflict context가 포함된 prompt
- `clean`: context 없이 question/options만 있는 prompt
- `support`: correct answer를 지지하는 synthetic context가 붙은 prompt

핵심 비교:

```text
D_kc  = residual(conflict) - residual(clean)
D_sup = residual(support)  - residual(clean)
```

따라서 이 실험은 “conflict context가 clean 대비 residual stream에 어떤 주방향 변화를 만드는가”를 보는 실험이다.

## 생성된 결과 파일

```text
phase2/results/exp1_trajectory/trajectory_stats.pt
phase2/results/exp1_trajectory/trajectory_stats_summary.json
phase2/results/exp1_trajectory/trajectory_plot.png
phase2/results/exp1_trajectory/direction_overlap_plot.png
```

`trajectory_stats.pt`에는 exp2/exp3에서 사용할 `kc_dir` tensor가 포함되어 있다.

## Layer별 주요 수치

| layer | KC strength | Support strength | KC/Support | KC-Support overlap | Direction continuity |
|---:|---:|---:|---:|---:|---:|
| 6 | 1191.08 | 1189.64 | 1.001 | 1.000 | - |
| 7 | 1308.30 | 1307.82 | 1.000 | 1.000 | 0.715 |
| 8 | 1379.31 | 1379.14 | 1.000 | 1.000 | 0.758 |
| 9 | 1579.17 | 1581.30 | 0.999 | 1.000 | 0.768 |
| 10 | 1650.05 | 1651.04 | 0.999 | 1.000 | 0.760 |
| 11 | 1736.99 | 1738.96 | 0.999 | 1.000 | 0.784 |
| 12 | 1694.42 | 1699.51 | 0.997 | 0.999 | 0.788 |
| 13 | 1878.14 | 1893.34 | 0.992 | 0.998 | 0.797 |
| 14 | 2105.70 | 2056.91 | 1.024 | 0.974 | 0.729 |
| 15 | 2106.22 | 2107.62 | 0.999 | 0.984 | 0.740 |
| 16 | 2428.27 | 2372.74 | 1.023 | 0.977 | 0.800 |
| 17 | 2845.81 | 2512.29 | 1.133 | 0.914 | 0.735 |
| 18 | 2947.62 | 2563.69 | 1.150 | 0.892 | 0.890 |
| 19 | 3050.84 | 2669.17 | 1.143 | 0.881 | 0.897 |

## 핵심 관찰 1: KC strength는 깊은 layer로 갈수록 커진다

KC strength는 layer 6에서 약 1191이고, layer 19에서 약 3051까지 증가한다. 중간에 layer 12에서 약간 낮아지는 구간은 있지만 전체 추세는 명확하게 후반부로 갈수록 커진다.

상위 KC strength layer:

```text
layer 19: 3050.84
layer 18: 2947.62
layer 17: 2845.81
layer 16: 2428.27
layer 15: 2106.22
```

해석:

KC 관련 residual difference는 초중반 layer에서 생기기 시작하지만, 후반 layer로 갈수록 더 큰 주성분 구조로 정렬된다. 즉 answer prediction에 가까운 후반부에서 conflict-induced representation이 더 강하게 응축되는 것으로 볼 수 있다.

## 핵심 관찰 2: KC direction은 supportive direction과 거의 완전히 겹친다

초중반 layer 6-13에서 `kc_sup_overlap`은 거의 1.0이다.

```text
layer 6-13: overlap ≈ 0.998-1.000
```

후반 layer로 갈수록 overlap이 조금 낮아지지만, 가장 낮은 layer 19도 0.881이다.

낮은 overlap 순서:

```text
layer 19: 0.881
layer 18: 0.892
layer 17: 0.914
layer 14: 0.974
layer 16: 0.977
```

중요한 해석:

현재 결과는 “KC만의 독립적인 direction이 supportive direction과 분리되어 있다”는 증거는 아니다. 오히려 conflict context와 supportive context가 clean 대비 residual stream을 매우 비슷한 방향으로 이동시킨다는 결과다.

따라서 `kc_dir`이라는 이름은 유지해도 되지만, 논문/발표에서는 더 조심스럽게:

```text
conflict-clean dominant direction
```

또는

```text
context-induced answer-support direction
```

처럼 표현하는 것이 안전하다.

## 핵심 관찰 3: 후반 layer에서 KC가 supportive보다 상대적으로 강해진다

KC/Support strength ratio는 layer 17-19에서 뚜렷하게 1보다 커진다.

```text
layer 17: 1.133
layer 18: 1.150
layer 19: 1.143
```

초중반에서는 KC와 supportive strength가 거의 동일하지만, 후반부에서는 conflict-clean 차이가 supportive-clean 차이보다 더 커진다.

해석:

direction 자체는 supportive와 상당히 겹치지만, 후반 layer에서는 conflict condition이 그 방향을 더 강하게 사용한다. 즉 KC-specificity가 “방향의 독립성”보다는 “같은 방향 성분의 강도 차이”로 나타나는 구조일 수 있다.

## 핵심 관찰 4: direction continuity는 대체로 유지되지만 몇 번 꺾인다

인접 layer 간 KC direction continuity는 대부분 0.72-0.90 범위다. 완전히 불연속적이지는 않지만, 몇 layer에서 방향 변화가 상대적으로 크다.

낮은 continuity:

```text
layer 7:  0.715
layer 14: 0.729
layer 17: 0.735
layer 15: 0.740
```

해석:

Layer 14-17 부근은 KC direction이 후반부 강한 representation으로 넘어가는 전환 구간일 가능성이 있다. 특히 layer 17은 기존 pilot에서 L17H3가 중요했던 지점과도 맞물리므로 후속 exp2/exp3에서 주목할 가치가 있다.

## exp2/exp3에 주는 의미

Exp1만으로는 direction이 causal하다고 말할 수 없다. Exp1은 어디까지나 residual difference의 주성분 구조를 찾는 단계다.

다만 후속 실험 후보는 명확하다.

1. 후반 강도 peak:
   - layer 17, 18, 19

2. direction 변화 지점:
   - layer 14, 17

3. mediation 후보:
   - layer 16-17 주변
   - L17H3와 연결해서 보기 좋음

Exp2에서는 layer별 `kc_dir` 성분만 제거했을 때 correct probability가 회복되는지 봐야 한다. Exp3에서는 mid-layer direction patch가 L17H3의 evidence attention과 output projection을 바꾸는지 확인해야 한다.

## 주의점

### 1. `supportive_prompt`가 synthetic하다

support condition은 실제 evidence가 아니라:

```text
The answer is {correct_answer}.
```

형태의 synthetic context다. 따라서 `KC vs supportive overlap`은 “잘 구성된 support evidence와 conflict evidence의 차이”가 아니라, “conflict context insertion과 minimal correct-answer support insertion의 차이”로 해석해야 한다.

### 2. overlap이 너무 높다

`kc_sup_overlap`이 전 layer에서 0.88 이상이다. 이는 KC direction이 supportive direction과 분리되어 있지 않다는 뜻이다. 그러므로 “KC-specific layer”라는 표현은 현재 결과에 맞지 않는다.

### 3. Singular value 절대값은 case 수에 의존한다

SVD singular value는 sample 수가 늘면 커질 수 있다. 따라서 이전 pilot 결과와 비교할 때 절대값보다 layer별 상대 패턴을 봐야 한다.

## 최종 결론

Exp1 결과는 KC가 residual stream에서 layer가 깊어질수록 강한 주성분 방향으로 정렬된다는 것을 보여준다. 특히 layer 17-19에서 KC strength가 가장 크고, KC/support strength ratio도 1보다 커진다.

하지만 KC direction과 supportive direction의 cosine overlap이 매우 높기 때문에, 현재 결과만으로 KC만의 독립적인 방향이 발견되었다고 말하기는 어렵다. 더 정확한 결론은 다음과 같다.

> Conflict context는 clean prompt 대비 residual stream을 강한 context-induced answer-support direction으로 이동시키며, 이 효과는 후반 layer에서 특히 커진다. 이 direction은 supportive context direction과 상당히 공유되지만, layer 17-19에서는 conflict condition이 더 강한 magnitude를 보인다.

따라서 exp2/exp3의 핵심 질문은 “이 shared-but-strong conflict direction이 실제로 answer flip에 causal한가, 그리고 L17H3를 통해 매개되는가?”가 되어야 한다.

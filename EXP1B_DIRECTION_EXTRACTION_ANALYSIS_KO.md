# Exp1b Direction Extraction 결과 분석

## 1. 이 실험이 무엇을 한 것인가

Exp1에서는 `conflict - clean` residual의 SVD 1번 방향을 KC direction으로 잡았고, 그 방향이 supportive context direction과 전 layer에서 매우 크게 겹쳤다. 즉 Exp1의 결론은 “KC direction을 뽑았다고 생각했지만, 사실상 context-following 방향과 거의 같다”에 가까웠다.

Exp1b는 이 문제를 해결하려고 baseline을 `clean`이 아니라 `supportive`로 바꿨다. 핵심 비교는 다음이다.

```text
conflict residual - supportive residual
```

따라서 Exp1b는 “context가 있다는 공통 효과”를 제거하고, conflict에서만 더 나타나는 residual 성분이 있는지 찾는 실험이다. 이를 위해 네 가지 방법을 썼다.

- Method A: `conflict - supportive`의 평균 방향
- Method B: `conflict - supportive` 행렬의 SVD 1번 방향
- Method C: conflict에서는 분산이 크고 supportive에서는 분산이 작은 cPCA 방향
- Method D: KC severity 또는 excess severity를 예측하는 ridge regression 방향

결과 저장 위치:

```text
/home/jaeik/xaimi/phase2/results/exp1b_direction_extraction
```

주요 산출물:

- `exp1b_summary.json`
- `exp1b_report.md`
- `direction_to_use.pt`
- `method_A_mean_diff.pt`
- `method_B_svd_conf_sup.pt`
- `method_C_cpca.pt`
- `method_D_regression.pt`
- `stability_stats.json`
- `type_alignment_stats.json`
- `convergence_stats.json`

## 2. 자동 결론

자동 결론은 다음이다.

```text
conclusion: case2_weak_direction
best_dir_key: D_reg_excess
best_layers: [18, 15, 19]
n_cases: 1000
```

즉 스크립트 기준으로는 “KC-specific structured direction이 강하게 수렴한 것은 아니지만, KC excess severity와 상관된 약한 방향은 있다”는 결론이다.

Exp2/3에 넘기도록 저장된 `direction_to_use.pt`는 `D_reg_excess` 방향을 사용한다. patch target은 0이다.

```text
h_patched = h_conf - (proj_conf - 0) * patch_dir
```

## 3. 핵심 결과 요약

### 3.1 A/B는 conflict-supportive excess 방향을 꽤 일관되게 찾았다

Method A의 평균 방향은 모든 layer에서 양의 정렬을 보였다. `positive_alignment_ratio`는 전 layer에서 0.73 이상이고, 중간 layer에서는 0.80 이상까지 올라간다.

특히 Method A와 Method B는 대부분 layer에서 매우 잘 맞는다.

```text
Layer 8  A-B overlap = 0.976
Layer 9  A-B overlap = 0.976
Layer 14 A-B overlap = 0.963
Layer 15 A-B overlap = 0.952
Layer 17 A-B overlap = 0.979
Layer 19 A-B overlap = 0.943
```

이는 `conflict - supportive`에 평균적으로 같은 방향의 excess가 존재한다는 뜻이다.

### 3.2 SVD energy는 후반 layer에서 강하다

Method B의 `explained_ratio`는 layer 14 이후 크게 올라간다.

```text
L14: 0.4419
L15: 0.3740
L16: 0.2946
L17: 0.4520
L18: 0.4103
L19: 0.4060
```

stability에서도 같은 패턴이 나온다. 후반 layer의 effective rank가 크게 낮아진다.

```text
L14 effective_rank = 20.63
L17 effective_rank = 10.12
L18 effective_rank = 8.60
L19 effective_rank = 9.39
```

즉 후반 layer에서는 `conflict - supportive` 차이가 비교적 저차원 구조로 모인다.

### 3.3 Exp1 direction과는 꽤 분리된다

Exp1의 `conflict - clean` direction과 Method B direction의 overlap은 중간 layer에서 매우 낮다.

```text
L8  overlap_exp1 = 0.0460
L9  overlap_exp1 = 0.0414
L10 overlap_exp1 = 0.0051
L11 overlap_exp1 = 0.0005
L12 overlap_exp1 = 0.0557
L13 overlap_exp1 = 0.0737
L15 overlap_exp1 = 0.0797
```

후반 layer에서는 overlap이 조금 올라간다.

```text
L17 overlap_exp1 = 0.4846
L18 overlap_exp1 = 0.5221
L19 overlap_exp1 = 0.5092
```

그래도 Exp1에서 KC/support overlap이 0.88 이상이었던 것과 비교하면 훨씬 분리된 방향이다. 따라서 Exp1b는 Exp1보다 KC-specific 후보를 더 잘 분리한 것은 맞다.

### 3.4 하지만 전체 방법 수렴도는 낮다

자동 결론이 case1이 아니라 case2가 된 핵심 이유는 `mean_convergence`가 0.5를 넘는 layer가 없기 때문이다.

```text
최고 mean_convergence:
L9  = 0.3649
L8  = 0.3625
L12 = 0.3530
L10 = 0.3458
L14 = 0.3383
```

이는 A/B/C/D가 하나의 동일한 KC-specific direction을 가리킨다고 보기 어렵다는 뜻이다.

특히 중요한 점은 Method D의 regression excess direction이 A/B/C와 거의 겹치지 않는다는 것이다.

```text
L14 D_excess vs A = 0.004, vs B = 0.012
L15 D_excess vs A = 0.002, vs B = 0.013
L17 D_excess vs A = 0.003, vs B = 0.004
L18 D_excess vs A = 0.008, vs B = 0.000
L19 D_excess vs A = 0.009, vs B = 0.001
```

반면 D 내부에서는 `kc_severity` direction과 `excess` direction이 서로 높게 겹친다.

```text
D_reg_kc vs D_reg_excess ≈ 0.86-0.92
```

따라서 결과는 “하나의 KC-specific direction”이라기보다 다음처럼 보는 것이 더 정확하다.

```text
1. A/B/C: conflict와 supportive를 분리하는 구조적 excess direction
2. D: KC severity를 예측하는 별도의 약한 방향
```

## 4. Regression 결과는 의미가 있다

Method D의 held-out Pearson r은 전 layer에서 양수이고, 특히 후반 layer에서 높다.

```text
excess target:
L18 r = 0.524
L15 r = 0.522
L19 r = 0.521
L14 r = 0.494
L13 r = 0.484
L17 r = 0.482
```

이는 residual 안에 KC severity와 상관된 정보가 있다는 뜻이다. 하지만 그 방향이 A/B/C와 거의 직교에 가까우므로, 이 direction이 “conflict-supportive 차이의 주방향”은 아니다.

해석하면 다음과 같다.

```text
KC 발생 여부 또는 conflict-supportive 차이의 큰 구조는 A/B/C가 잡는다.
KC가 얼마나 강한지는 D가 잡지만, 그 신호는 별도의 작은 축에 있다.
```

## 5. Conflict type별 결과

type-specific mean direction은 매우 잘 정렬된다.

전 layer에서 misinformation, temporal, semantic 간 alignment가 대부분 0.90 이상이고, 후반 layer에서는 거의 0.99에 가깝다.

```text
L17:
misinformation vs temporal = 0.9964
misinformation vs semantic = 0.9957
temporal vs semantic = 0.9942

L19:
misinformation vs temporal = 0.9957
misinformation vs semantic = 0.9966
temporal vs semantic = 0.9951
```

따라서 `conflict - supportive` excess 방향은 conflict type별로 크게 다르지 않다. 이 점은 논문/발표에서 좋은 근거가 된다.

## 6. Exp2/3로 이어갈 때의 주의점

현재 `direction_to_use.pt`는 `D_reg_excess`를 선택한다. 이는 스크립트의 case2 결론과 일관된다.

다만 해석상 주의가 필요하다.

### Exp2

Exp2에서 `D_reg_excess`를 patch하면 “KC severity와 상관된 약한 방향을 제거했을 때 답변이 바뀌는가”를 검증하게 된다.

만약 효과가 작게 나오더라도 이상한 결과는 아니다. D 방향은 A/B/C가 찾은 큰 conflict-supportive excess direction과 거의 겹치지 않기 때문이다.

따라서 Exp2는 두 가지 patch를 비교하면 더 설득력 있다.

```text
1. D_reg_excess patch: severity-linked weak direction의 causal test
2. B_svd 또는 A_mean_diff patch: conflict-supportive structural excess의 causal test
```

### Exp3

best_layers가 `[18, 15, 19]`인데, 기존 Exp3의 관심 mediator가 `L17H3`라면 L18, L19는 L17H3의 원인이 될 수 없다. L18/L19는 L17H3보다 뒤에 있기 때문이다.

따라서 L17H3 mediation을 보려면 pre-L17 layer를 우선해야 한다.

추천 후보:

```text
L15: D_excess r = 0.522, explained_ratio = 0.3740
L14: D_excess r = 0.494, explained_ratio = 0.4419
L13: D_excess r = 0.484, explained_ratio = 0.1971
L16: D_excess r = 0.463, explained_ratio = 0.2946
```

특히 L15는 자동 best_layers에도 들어가므로 Exp3의 1순위로 적절하다.

## 7. 결론

Exp1b는 Exp1보다 확실히 더 좋은 분리 실험이다. `clean`이 아니라 `supportive`를 baseline으로 둔 덕분에, 단순 context-following 방향과 구분되는 `conflict - supportive` excess 구조를 잡았다.

하지만 네 방법이 하나의 방향으로 수렴하지는 않았다. 가장 큰 구조적 차이는 A/B/C가 잡고, KC severity를 예측하는 방향은 D가 따로 잡는다. 그래서 현재 결과는 `case2_weak_direction`으로 보는 것이 맞다.

논문식 결론으로는 다음 표현이 가장 안전하다.

```text
Exp1b identifies a robust conflict-supportive excess structure, but not a single fully convergent KC-specific direction. KC severity is decodable from residual states, yet the severity-predictive direction is nearly orthogonal to the dominant conflict-supportive SVD direction.
```

한국어로는 다음처럼 정리할 수 있다.

```text
KC는 단일하고 강한 독립 방향으로 나타난다기보다, conflict-supportive 차이를 만드는 큰 구조적 excess와 KC 강도를 예측하는 약한 별도 신호로 분리되어 나타난다.
```

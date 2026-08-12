# Pilot 후속 실험 exp1/exp2/exp3 코드 파악

## 전체 구조

새로 추가된 세 실험은 기존 pilot에서 관찰한 KC 현상을 더 구조적으로 검증하기 위한 후속 실험 체인이다.

전체 흐름은 다음과 같다.

```text
phase2/scripts/exp1_trajectory.py
  ↓ KC direction 추출
phase2/results/exp1_trajectory/trajectory_stats.pt
  ↓
exp2_direction_patching.py
  ↓ KC direction만 제거했을 때 causal effect 측정
results/exp2_direction_patching/comparison_with_step3.json
  ↓
exp3_mediation.py
  ↓ mid-layer KC direction이 L17H3 behavior를 매개하는지 테스트
```

공통 유틸리티는 `scripts/exp_utils.py`에 있다.

공통 설정:

- 모델: `google/gemma-2-2b`
- 데이터: `data/phase0/kc_cases_1000.json`
- 기본 case 수: `N_CASES = 200`
- 분석 layer: 6-19
- mediation 대상 head: L17H3

주의: 기본값은 1000개가 아니라 200개다. 전체 1000개를 쓰려면 `exp_utils.py`의 `N_CASES`를 바꾸거나 CLI 옵션을 추가해야 한다.

## exp1_trajectory.py

### 목적

KC가 residual stream 안에서 어떤 direction으로 나타나는지 layer별로 찾는 실험이다. 각 layer에서 `conflict - clean` residual 차이를 모아 SVD를 수행하고, 첫 번째 right singular vector를 해당 layer의 KC direction으로 정의한다.

### 비교 조건

각 case에 대해 세 prompt를 사용한다.

1. conflict prompt
   - 기존 conflict context 포함
2. clean prompt
   - context 없이 question/options만 포함
3. supportive prompt
   - correct answer를 지지하는 synthetic context 추가
   - 현재 구현은 `The answer is {correct_answer}.`를 context로 붙임

### 핵심 계산

각 layer `l`에서:

```text
D_kc  = h_conflict[l] - h_clean[l]
D_sup = h_support[l]  - h_clean[l]
```

그리고 각각 SVD:

```text
kc_dir[l]  = first right singular vector of D_kc
sup_dir[l] = first right singular vector of D_sup
```

저장되는 주요 지표:

- `kc_strength`: KC direction의 1st singular value
- `sup_strength`: supportive direction의 1st singular value
- `kc_sup_overlap`: `|cos(kc_dir, sup_dir)|`
- `direction_continuity`: 인접 layer 간 KC direction cosine
- `clean_mean_proj`: clean residual의 KC direction 평균 projection
- `kc_dir`: exp2/exp3에서 사용할 direction vector

### 현재 생성된 결과

이미 다음 파일이 있다.

- `phase2/results/exp1_trajectory/trajectory_stats.pt`
- `phase2/results/exp1_trajectory/trajectory_stats_summary.json`
- `trajectory_plot.png`
- `direction_overlap_plot.png`

현재 summary를 보면 KC strength는 layer가 깊어질수록 커지고, 가장 강한 층은 layer 19다. `kc_sup_overlap`은 0.62-0.86 범위로, 코드 주석의 “overlap < 0.3이면 KC-specific” 기준을 만족하는 layer는 없다. 즉 현재 결과만 보면 KC direction이 supportive direction과 완전히 분리된 특수 방향이라기보다는, support/context insertion 방향과 상당히 공유되는 성분이 있다.

## exp2_direction_patching.py

### 목적

exp1에서 찾은 KC direction이 실제로 causal한지 검증한다.

아이디어는 conflict residual에서 KC direction 성분만 clean 평균 projection으로 되돌리고, 나머지 residual 성분은 그대로 둔 뒤 정답 확률이 얼마나 회복되는지 보는 것이다.

### 개입 방식

각 layer와 position마다:

```text
h_patched = h_conflict - (proj_conflict - clean_mean_proj) * kc_dir
```

즉 residual 전체를 clean으로 바꾸는 것이 아니라, `kc_dir` 방향 성분만 clean 평균 수준으로 맞춘다.

### 측정 위치

두 position을 본다.

- `final`: Answer prediction 위치
- `pre_answer`: `\nAnswer:` 직전 token 위치

### 효과 지표

```text
IE_dir[layer][pos] =
  mean(P(correct | patched) - P(correct | conflict))
```

양수면 KC direction 성분을 제거했을 때 correct option 확률이 올라간다는 뜻이므로, 해당 direction이 KC를 causal하게 밀고 있었다고 해석한다.

### 출력

- `results/exp2_direction_patching/IE_dir.pt`
- `IE_dir_heatmap.png`
- `IE_dir_vs_step3.png`
- `comparison_with_step3.json`

`comparison_with_step3.json`은 exp3에서 patch target layer를 고를 때 사용된다.

### 기존 pilot과의 관계

기존 Step3 causal tracing은 chunk/span 전체를 clean activation으로 restore하는 방식이었다. exp2는 그보다 더 세밀하게, residual 전체가 아니라 KC direction 하나만 patch한다. 따라서 “정말 direction-level feature가 causal한가?”를 묻는 후속 검증이다.

## exp3_mediation.py

### 목적

exp2에서 causal하다고 의심되는 mid-layer KC direction이, pilot에서 중요하게 나온 L17H3의 behavior를 매개하는지 확인한다.

핵심 질문:

```text
Layer 6-16의 KC direction을 patch하면
L17H3의 evidence attention / KC projection / final logit behavior가 바뀌는가?
```

### Patch target

기본 patch layers:

```text
[6, 8, 10, 12, 14, 16]
```

여기에 exp2의 top IE layers가 있으면 추가한다. 단, L17H3보다 뒤의 layer는 제외한다.

### 측정값

각 patch layer에 대해 세 값을 측정한다.

1. `delta_logit_diff`

```text
logit_diff_before - logit_diff_after
```

여기서 logit diff는 `logP(replace) - logP(correct)`다. 양수면 patch 후 KC 경향이 줄었다는 뜻이다.

2. `delta_ev_attn`

```text
L17H3 evidence attention before - after
```

양수면 patch 후 L17H3가 evidence span을 덜 본다는 뜻이다.

3. `delta_kc_proj`

```text
L17H3 hook_z output의 KC direction projection before - after
```

양수면 patch 후 L17H3 output에서 KC 성분이 줄었다는 뜻이다.

### d_model → d_head 변환

L17H3의 `hook_z`는 `d_head` 공간이므로, exp1의 `kc_dir`은 다음 방식으로 변환된다.

```text
kc_dir_head = W_O[17, 3] @ kc_dir_dmodel
```

이 방향에 L17H3 z output을 projection한다.

### 출력

- `phase2/results/exp3_mediation/mediation_results.pt`
- `mediation_summary.json`
- `mediation_plot.png`

### 기존 pilot과의 관계

Step5 head ablation에서 L17H3가 중요했다면, exp3는 “mid-layer KC direction이 L17H3로 전달되어 L17H3의 evidence attention/output을 바꾸는가?”를 테스트한다. 즉 head ablation의 결과를 direction-level mechanism으로 연결하려는 실험이다.

## 중요한 주의점

### 1. 현재 기본 case 수는 200개

`exp_utils.py`에 `N_CASES = 200`으로 되어 있다. 파일 이름이나 주석과 달리 기본 실행은 1000개 전체가 아니다.

### 2. exp1의 supportive prompt는 synthetic하다

supportive condition은 실제 ConflictBank supportive evidence가 아니라:

```text
The answer is {correct_answer}.
```

를 붙이는 방식이다. 따라서 `kc_sup_overlap`은 “conflict evidence vs realistic supportive evidence”라기보다 “conflict context insertion vs synthetic correct-answer support” 비교로 해석해야 한다.

### 3. exp1 현재 결과에서는 KC-specific overlap이 낮지 않다

현재 `kc_sup_overlap`은 모든 layer에서 0.62 이상이다. 코드의 기준인 `< 0.3`에 해당하는 KC-specific layer는 없다. 따라서 exp2/exp3에서 “KC-specific direction”이라는 표현은 조심해야 한다. 더 정확히는 “conflict-clean dominant direction”이다.

### 4. exp2/exp3는 exp1 결과에 강하게 의존한다

`trajectory_stats.pt`가 없으면 exp2/exp3는 실행되지 않는다. 또 exp1을 다른 case 수나 다른 prompt 구성으로 다시 돌리면 exp2/exp3 결과도 다시 계산해야 한다.

### 5. exp3의 mediation 해석은 세 지표가 같이 움직일 때 강하다

`delta_logit_diff`만 양수인 경우와, `delta_ev_attn`, `delta_kc_proj`까지 같이 양수인 경우의 해석 강도가 다르다. mediation 주장은 세 지표가 같은 patch layer에서 함께 커질 때 가장 설득력 있다.

## 한 줄 요약

세 실험은 pilot에서 관찰한 KC 현상을 다음 방식으로 확장한다.

```text
exp1: KC가 residual stream의 어떤 direction으로 나타나는가?
exp2: 그 direction 성분만 제거하면 correct answer가 회복되는가?
exp3: 그 direction이 L17H3의 evidence attention/output을 매개하는가?
```

즉 기존 pilot이 “어느 layer/head가 중요해 보이는가”를 보였다면, 이 후속 실험들은 “KC를 나타내는 residual direction이 있고, 그 direction이 causal하며, L17H3로 전달되는가”를 검증하려는 설계다.

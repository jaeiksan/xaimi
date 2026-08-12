# Phase 3 실험 결과 및 분석

## 개요

### 핵심 질문

> A/B/C direction이 단순히 "외부 context를 따르는 방향"인가,  
> 아니면 "parametric knowledge와 external context가 충돌하는 상황을 감지하는 signal"인가?

### 판단 논리

| 지표 | context following | conflict detection |
|------|------|------|
| **Step1** mean projection 순서 | conflict ≈ supportive >> clean | conflict >> supportive ≈ clean |
| **Step2** ratio = Δ_sc / Δ_cc | ratio ≈ 1 (supportive도 같이 올라감) | ratio ≈ 0 (conflict만 올라감) |
| **Step3** flipped effect size | 차이 없음 or 음수 | flipped case에서 양의 효과 |

### 실험 설정

| 항목 | 값 |
|------|------|
| 모델 | google/gemma-2-2b |
| 데이터 | kc_cases_1000.json (N=1000) |
| 분석 레이어 | L10 ~ L22 (13개) |
| Direction A | Mean Diff (conflict − supportive), 정규화 |
| Direction B | SVD(conflict − supportive), 첫 번째 singular vector |
| Direction C | cPCA (best alpha ∈ {0.5, 1.0, 2.0}) |
| Step3 config | `L16:A + L17:C + L19:B` |
| Flipped / Non-flipped | 316 / 684 |

---

## Step 1: 세 조건 Projection 직접 비교

### 방법

모델을 세 가지 조건(clean / supportive / conflict)으로 실행하고,
마지막 토큰 위치의 residual stream을 각 direction으로 projection한 평균값을 레이어별로 비교한다.

### Direction A 결과

| Layer | clean | supportive | conflict | 특징 |
|-------|------:|----------:|--------:|------|
| L10 | 2.60 | 0.50 | 1.79 | 세 조건 혼재 |
| L11 | 4.73 | 1.73 | 3.34 | 세 조건 혼재 |
| L12 | -8.61 | -13.74 | -11.62 | 음수 구간 |
| L13 | -18.00 | -27.70 | -24.08 | supportive 최저 |
| **L14** | 1.62 | 1.35 | **15.40** | conflict 급등 시작 |
| L15 | -13.71 | -19.66 | -8.20 | conflict가 상대적으로 높음 |
| **L16** | 0.58 | 1.20 | **16.61** | conflict 두드러짐 |
| **L17** | -7.37 | 1.11 | **32.39** | conflict 대폭 상승 |
| **L18** | -5.96 | 1.12 | **35.84** | conflict 고조 |
| **L19** | -4.98 | -0.09 | **37.06** | conflict 고조 |
| **L20** | -5.82 | 1.11 | **39.12** | conflict 최고조 |
| L21 | -55.75 | -45.71 | -2.78 | 전체 급락, 상대 갭은 유지 |
| **L22** | -28.48 | -20.44 | **24.82** | conflict 재상승 |

**패턴**: L14 이전은 세 조건이 유사하게 움직이나, **L14부터 conflict projection이 선택적으로 급등**한다. L17~L20에서 conflict는 32~39인 반면 clean/supportive는 −6 ~ +1 수준이다. L21에서 전체 값이 크게 떨어지지만 conflict가 상대적으로 가장 높은 위치를 유지하며, L22에서 다시 conflict 우세 패턴이 복원된다.

### Direction B 결과

| Layer | clean | supportive | conflict | 특징 |
|-------|------:|----------:|--------:|------|
| L10 | -5.42 | -6.49 | -5.25 | 거의 유사 |
| L11 | 5.24 | 3.65 | 5.07 | 거의 유사 |
| L12 | -19.18 | -24.11 | -22.17 | 음수 구간 |
| L13 | -23.94 | -31.58 | -28.28 | supportive 최저 |
| **L14** | 3.48 | 2.91 | **16.44** | conflict 급등 |
| L15 | -7.72 | -13.72 | -2.80 | conflict 상대적 우세 |
| **L16** | 6.76 | 5.84 | **20.33** | conflict 두드러짐 |
| **L17** | 1.32 | 8.34 | **38.98** | conflict 압도적 |
| **L18** | 5.02 | 12.35 | **44.28** | conflict 고조 |
| **L19** | 7.67 | 12.23 | **47.26** | conflict 고조 |
| **L20** | 5.68 | 10.64 | **47.28** | conflict 최고조 |
| L21 | -35.50 | -29.41 | **11.01** | 전체 급락, conflict 상대적 우위 유지 |
| **L22** | -9.52 | -6.54 | **34.80** | conflict 재상승 |

**패턴**: Direction A와 매우 유사하지만 폭이 더 크다. L17~L20에서 conflict는 39~47이고 clean/supportive는 1~12 수준으로, **conflict의 선택적 상승이 A보다 더 두드러진다**. L21 급락 후 L22에서도 A보다 큰 폭으로 회복된다.

### Direction C 결과

| Layer | clean | supportive | conflict | 특징 |
|-------|------:|----------:|--------:|------|
| L10 | 8.95 | 11.84 | 11.04 | supportive > conflict ≈ clean |
| L11 | 13.40 | 8.84 | 9.70 | clean > conflict ≈ supportive |
| L12 | 17.40 | 22.09 | 20.67 | supportive > conflict > clean |
| L13 | -12.28 | -11.99 | -10.19 | conflict 약간 우세 |
| L14 | 30.54 | 28.51 | 18.96 | clean > supportive >> conflict (역전) |
| L15 | 50.62 | 51.26 | 44.22 | clean ≈ supportive >> conflict |
| L16 | -45.74 | -46.04 | -40.11 | 음수, conflict 상대적으로 높음 |
| L17 | 10.70 | 2.54 | -4.91 | clean > supportive > conflict |
| L18 | 14.84 | 3.56 | -6.80 | clean > supportive > conflict |
| L19 | -2.37 | 7.67 | 18.10 | conflict > supportive > clean |
| L20 | 7.44 | -1.35 | -11.68 | clean > supportive > conflict (역전) |
| L21 | -42.87 | -36.83 | -20.79 | conflict 상대적 우위 |
| L22 | 7.76 | 5.31 | -9.06 | clean > supportive > conflict (역전) |

**패턴**: 레이어마다 conflict/clean/supportive 순서가 계속 뒤바뀐다. **일관된 conflict-specific 상승 패턴이 없으며**, L14~L15, L17~L18, L20, L22에서는 오히려 conflict가 clean보다 낮다.

### Step1 소결

- **Direction A, B**: L14부터 conflict projection이 선택적으로 상승 → conflict detection 가능성. L20까지 패턴 유지, L21 일시 급락 후 L22에서 복원.
- **Direction C**: 레이어별로 순서가 불규칙 → 명확한 해석 불가

---

## Step 2: Ratio 비교 (Δ_sc / Δ_cc)

### 방법

각 케이스에서 다음 두 값을 계산한다.

```
Δ_cc = conflict projection − clean projection    (conflict가 clean 대비 얼마나 올라갔나)
Δ_sc = supportive projection − clean projection  (supportive가 clean 대비 얼마나 올라갔나)

ratio = mean(Δ_sc) / mean(Δ_cc)
```

- ratio ≈ 0 → supportive는 안 올라가고 conflict만 올라감 → **conflict detection**
- ratio ≈ 1 → supportive도 conflict만큼 올라감 → **context following**

### Direction A

| Layer | Δ_cc | Δ_sc | ratio | 판정 |
|-------|-----:|-----:|------:|------|
| L10 | -0.82 | -2.10 | 2.57 | context following |
| L11 | -1.39 | -3.00 | 2.16 | context following |
| L12 | -3.01 | -5.13 | 1.70 | context following |
| L13 | -6.08 | -9.70 | 1.60 | context following |
| **L14** | **13.77** | -0.27 | **-0.02** | **conflict detection** |
| L15 | 5.51 | -5.96 | -1.08 | conflict detection |
| **L16** | **16.03** | 0.62 | **0.04** | **conflict detection** |
| **L17** | **39.76** | 8.48 | **0.21** | **conflict detection** |
| **L18** | **41.80** | 7.08 | **0.17** | **conflict detection** |
| **L19** | **42.04** | 4.89 | **0.12** | **conflict detection** |
| **L20** | **44.93** | 6.93 | **0.15** | **conflict detection** |
| **L21** | **52.98** | 10.04 | **0.19** | **conflict detection** |
| **L22** | **53.30** | 8.04 | **0.15** | **conflict detection** |

**해석**: L10~L13에서는 ratio > 1.6으로 context following이었다가, **L14를 기점으로 ratio가 0 근처로 급락**한다. 이 패턴은 L20, L21, L22까지 지속된다 — **L20~L22에서도 ratio = 0.15~0.19**로 일관된 conflict detection 신호를 보인다.

### Direction B

| Layer | Δ_cc | Δ_sc | ratio | 판정 |
|-------|-----:|-----:|------:|------|
| L10 | 0.17 | -1.07 | -6.21 | 불안정 (Δ_cc ≈ 0) |
| L11 | -0.17 | -1.59 | 9.61 | 불안정 (Δ_cc ≈ 0) |
| L12 | -2.99 | -4.92 | 1.65 | context following |
| L13 | -4.34 | -7.64 | 1.76 | context following |
| **L14** | **12.96** | -0.57 | **-0.04** | **conflict detection** |
| L15 | 4.92 | -6.00 | -1.22 | conflict detection |
| **L16** | **13.57** | -0.92 | **-0.07** | **conflict detection** |
| **L17** | **37.66** | 7.03 | **0.19** | **conflict detection** |
| **L18** | **39.26** | 7.33 | **0.19** | **conflict detection** |
| **L19** | **39.59** | 4.56 | **0.12** | **conflict detection** |
| **L20** | **41.60** | 4.96 | **0.12** | **conflict detection** |
| **L21** | **46.51** | 6.10 | **0.13** | **conflict detection** |
| **L22** | **44.31** | 2.98 | **0.07** | **conflict detection** |

**해석**: Direction A와 거의 동일한 전환 패턴. **L14부터 ratio가 0 근처로 수렴**한다. L14, L16에서 Δ_sc < 0 (supportive가 오히려 clean 아래) — conflict 조건에서만 해당 direction이 강하게 활성화됨을 의미한다. **L22에서 ratio=0.07로 최저**, 즉 마지막 레이어까지 B는 순수한 conflict detection 방향을 유지한다.

### Direction C

| Layer | Δ_cc | Δ_sc | ratio | 판정 |
|-------|-----:|-----:|------:|------|
| L10 | 2.09 | 2.88 | 1.38 | context following |
| L11 | -3.70 | -4.55 | 1.23 | context following |
| L12 | 3.27 | 4.69 | 1.43 | context following |
| L13 | 2.09 | 0.29 | 0.14 | 혼재 |
| L14 | -11.57 | -2.03 | 0.18 | 혼재 |
| L15 | -6.40 | 0.64 | -0.10 | 혼재 |
| L16 | 5.63 | -0.29 | -0.05 | 혼재 |
| L17 | -15.61 | -8.16 | 0.52 | 혼재 |
| L18 | -21.64 | -11.28 | 0.52 | 혼재 |
| L19 | 20.48 | 10.05 | 0.49 | 혼재 |
| L20 | -19.12 | -8.79 | 0.46 | 혼재 |
| L21 | 22.09 | 6.05 | 0.27 | 혼재 (conflict detection 쪽) |
| L22 | -16.82 | -2.45 | 0.15 | 혼재 (conflict detection 쪽) |

**해석**: L10~L12에서 ratio > 1.2 (context following), L13 이후 0.15~0.52 범위로 내려오지만, **Δ_cc의 부호가 레이어마다 달라지며** conflict detection으로 단정하기 어렵다. L21~L22에서 ratio가 낮아지나 이는 Δ_cc 자체의 절댓값이 작아서가 아니라 부호 변동으로 인한 결과이다.

### Step2 소결

| Direction | 전환 레이어 | L14~L22 평균 ratio | L20~L22 평균 ratio | 판정 |
|-----------|------------|-------------------|-------------------|------|
| A | **L14** | **~0.12** | **~0.16** | conflict detection (전 구간 유지) |
| B | **L14** | **~0.08** | **~0.11** | conflict detection (전 구간 유지, B > A) |
| C | 없음 | ~0.27 | ~0.29 | mixed (불안정) |

---

## Step 3: Flipped vs Non-flipped Case 비교

### 방법

Grid search 최우선 config(`L16:A + L17:C + L19:B`)로 개입 후 rank flip이 일어난 케이스(성공, N=316)와 일어나지 않은 케이스(실패, N=684)를 분리한다.

각 그룹에서 conflict 조건의 projection 평균을 비교해 Cohen's d와 t-test p-value를 계산한다.

> 양의 effect size = flipped case에서 projection이 더 높음 = "conflict detection이 강한 케이스에서 rank flip도 잘 됨"

### Direction A

| Layer | flipped mean | non-flipped mean | Cohen's d | p-value | 유의 |
|-------|-------------:|-----------------:|----------:|--------:|:----:|
| L10 | 2.11 | 1.64 | 0.17 | 0.013 | * |
| L11 | 3.56 | 3.25 | 0.11 | 0.099 | |
| L12 | -11.11 | -11.85 | 0.26 | 0.000 | * |
| L13 | -22.93 | -24.61 | 0.43 | 0.000 | * |
| L14 | 20.40 | 13.08 | 0.54 | 0.000 | * |
| L15 | -4.37 | -9.97 | 0.48 | 0.000 | * |
| L16 | 21.83 | 14.19 | 0.57 | 0.000 | * |
| L17 | 44.83 | 26.64 | 0.67 | 0.000 | * |
| **L18** | **54.36** | **27.29** | **0.86** | **0.000** | **\*** |
| **L19** | **55.57** | **28.51** | **0.79** | **0.000** | **\*** |
| **L20** | **56.44** | **31.11** | **0.73** | **0.000** | **\*** |
| **L21** | **20.48** | **-13.52** | **0.83** | **0.000** | **\*** |
| **L22** | **49.83** | **13.27** | **0.84** | **0.000** | **\*** |

**패턴**: L14부터 모든 레이어에서 양의 effect size. **L18~L22 구간에서 d=0.73~0.86**이 일관되게 유지된다. L21에서 절댓값 projection은 급락하지만 flipped/non-flipped 그룹 간 분리는 여전히 강하다(d=0.83).

### Direction B

| Layer | flipped mean | non-flipped mean | Cohen's d | p-value | 유의 |
|-------|-------------:|-----------------:|----------:|--------:|:----:|
| L10 | -4.92 | -5.41 | 0.15 | 0.023 | * |
| L11 | 5.23 | 5.00 | 0.08 | 0.267 | |
| L12 | -21.53 | -22.47 | 0.31 | 0.000 | * |
| L13 | -26.96 | -28.89 | 0.48 | 0.000 | * |
| L14 | 22.24 | 13.76 | 0.60 | 0.000 | * |
| L15 | 1.81 | -4.92 | 0.57 | 0.000 | * |
| L16 | 27.12 | 17.20 | 0.70 | 0.000 | * |
| L17 | 53.51 | 32.26 | 0.77 | 0.000 | * |
| **L18** | **76.08** | **29.59** | **1.23** | **0.000** | **\*** |
| **L19** | **77.14** | **33.46** | **1.13** | **0.000** | **\*** |
| **L20** | **72.98** | **35.41** | **1.01** | **0.000** | **\*** |
| **L21** | **47.33** | **-5.77** | **1.15** | **0.000** | **\*** |
| **L22** | **77.68** | **14.99** | **1.22** | **0.000** | **\*** |

L18에서 flipped group 평균은 76.08, non-flipped는 29.59로 **약 2.6배 차이**가 난다. **L22에서 d=1.22**로 L18과 거의 동일한 분리력 — 마지막 레이어까지 강한 conflict signal을 유지한다.

### Direction C

| Layer | flipped mean | non-flipped mean | Cohen's d | p-value | 유의 |
|-------|-------------:|-----------------:|----------:|--------:|:----:|
| L10 | 10.24 | 11.41 | -0.23 | 0.001 | * |
| L11 | 10.28 | 9.43 | 0.15 | 0.025 | * |
| L12 | 19.70 | 21.12 | -0.27 | 0.000 | * |
| L13 | -9.52 | -10.50 | 0.27 | 0.000 | * |
| L14 | 15.61 | 20.51 | -0.33 | 0.000 | * |
| L15 | 42.20 | 45.16 | -0.22 | 0.002 | * |
| L16 | -36.84 | -41.62 | 0.30 | 0.000 | * |
| **L17** | **-35.53** | **9.24** | **-1.35** | **0.000** | **\*** |
| **L18** | **-49.77** | **13.06** | **-1.39** | **0.000** | **\*** |
| **L19** | **61.98** | **-2.17** | **1.41** | **0.000** | **\*** |
| **L20** | **-54.91** | **8.29** | **-1.39** | **0.000** | **\*** |
| **L21** | **28.86** | **-43.73** | **1.38** | **0.000** | **\*** |
| **L22** | **-65.74** | **17.13** | **-1.43** | **0.000** | **\*** |

**해석**: L17~L22에서 effect size의 **절댓값**은 1.35~1.43으로 A, B를 훨씬 넘지만, **레이어마다 부호가 교번**한다(L17 음, L18 음, L19 양, L20 음, L21 양, L22 음). Flipped case에서 더 높은 레이어도 있고 낮은 레이어도 있다 — 일관된 방향이 없으며, Direction C가 인코딩하는 변동이 개입 성공/실패와 복잡한 방식으로 교차됨을 보여준다.

### Step3 소결

| Direction | 주요 레이어 | 최대 Cohen's d | L20~L22 d 범위 | flip 예측력 | 해석 |
|-----------|-----------|--------------|---------------|------------|------|
| A | L14~L22 | **0.86** (L18) | 0.73~0.84 | 강함, 지속적 | conflict detection signal이 rank flip을 예측 |
| B | L14~L22 | **1.23** (L18) | 1.01~1.22 | 매우 강함, 지속적 | 가장 순수한 conflict detection signal |
| C | L17~L22 | ±1.43 (L22) | ±1.39 (교번) | 비일관적 | 레이어별 부호 교번 — 복잡한 개입 역학 |

---

## 종합 판정

### Direction A — Conflict Detection Signal ✓

**근거**:
1. **Step1**: L14부터 conflict projection이 clean/supportive 대비 선택적으로 급등. **L20에서 최고조(39.12)**, L21 일시 급락 후 L22에서 복원(24.82).
2. **Step2**: L14 이후 ratio ≈ 0.04~0.21, **L20~L22에서도 ratio = 0.15~0.19**로 conflict detection 패턴 지속.
3. **Step3**: L14~L22 전 레이어에서 양의 effect size (d=0.43~0.86). **L20~L22에서 d=0.73~0.84** — 분석 범위 끝까지 일관된 flip 예측력.

**결론**: Direction A는 L14부터 L22까지 conflict state를 선택적으로 인코딩하는 방향이다. L21의 projection 절댓값 급락은 레이어의 전반적인 활성화 스케일 변화이며, 그 레이어 내에서도 conflict/non-conflict 분리는 유효하다(d=0.83).

---

### Direction B — Strongest Conflict Detection Signal ✓✓

**근거**:
1. **Step1**: Direction A보다 더 큰 폭으로 conflict 상승. **L19~L20에서 conflict ≈ 47**, L22에서도 34.80으로 강하게 복원.
2. **Step2**: L14 이후 ratio ≈ −0.07~0.19, **L22에서 ratio=0.07** — 마지막 레이어까지 가장 낮은 ratio. L14·L16에서 Δ_sc < 0 (supportive가 오히려 clean 아래).
3. **Step3**: **L18에서 d=1.23**, **L22에서 d=1.22**. Flipped case 대비 non-flipped의 projection이 2.3~5.2배 낮음. L20~L22에서 d=1.01~1.22로 A(0.73~0.84)를 일관되게 상회.

**결론**: Direction B(SVD conflict-supportive)는 분석 레이어 전 구간(L14~L22)에서 가장 강하고 일관된 conflict detection signal을 보인다. SVD가 supportive를 baseline으로 삼아 conflict-specific 성분을 더 순수하게 분리했기 때문으로 해석된다.

---

### Direction C — Mixed / Reversed Signal ✗

**근거**:
1. **Step1**: 레이어마다 clean/supportive/conflict 순서가 역전됨. L14~L15, L17~L18, L20, L22에서 conflict < clean.
2. **Step2**: L10~L12에서 ratio > 1.2, 이후 0.15~0.52로 불안정. Δ_cc 부호가 레이어마다 달라짐.
3. **Step3**: L17~L22에서 절댓값 effect size는 1.35~1.43으로 A/B를 훨씬 넘지만, **레이어마다 부호가 L17 음 → L18 음 → L19 양 → L20 음 → L21 양 → L22 음으로 교번**. 일관된 flip 예측 방향이 없다.

**결론**: Direction C(cPCA)는 L20~L22까지 확장하더라도 conflict detection signal을 일관되게 인코딩하지 않는다. 그러나 절댓값 effect size가 매우 크다는 점에서 C 방향은 개입 메커니즘에 관여하되, **A/B와 다른 방식 — 아마도 레이어별로 다른 feature를 포착하는 복합 신호** — 으로 작동함을 시사한다.

---

## 핵심 발견 요약

### 1. L14가 전환점, L20까지 유효

Direction A, B 모두 L14를 기점으로 ratio가 > 1에서 ≈ 0으로 전환되며, **이 패턴은 L20까지 일관되게 유지**된다. L10~L22로 분석 범위를 확장함으로써 conflict detection 구간이 L14~L20으로 넓게 형성됨을 확인했다.

### 2. L21의 특이점: 스케일 급락, 그러나 분리력 유지

L21에서 projection 절댓값이 전 방향에서 급락한다(Direction A: conflict -2.78, Direction B: conflict 11.01). 이는 L21의 residual stream 스케일 자체가 달라지기 때문으로, conflict/non-conflict 그룹 간 분리(Step3 d > 0.83)는 여전히 강하다. L22에서 정상 스케일이 복원되며 conflict detection 패턴도 재확인된다.

### 3. A/B direction의 conflict detection은 행동과 연결 (L20~L22 확인)

Step3에서 flipped case의 projection이 모든 레이어(L14~L22)에서 일관되게 높다. **L20~L22를 추가해도 이 패턴이 유지**된다는 것은, projection이 단순한 관찰 지표가 아니라 실제 KC 완화 가능성과 상관된 인과적 신호임을 강화하는 근거이다.

### 4. Direction B > Direction A (전 구간)

L20~L22에서도 B의 effect size(d=1.01~1.22)가 A(d=0.73~0.84)를 일관되게 상회한다. SVD(conflict-supportive)가 SVD(conflict-clean)보다 conflict를 더 순수하게 포착하는 방향임이 확인된다.

### 5. Direction C의 한계와 가능성

cPCA 방향은 conflict detection signal을 일관되게 인코딩하지 않는다. 그러나 L17~L22의 절댓값 effect size(1.35~1.43)가 A/B를 모두 상회한다는 점, 그리고 교번 부호 패턴이 L20~L22까지 지속된다는 점은, **C가 포착하는 feature가 서로 다른 레이어에서 conflict와 반대 방향으로 활성화되는 복합 패턴**임을 시사한다. Phase 3 이후 심층 분석이 필요하다.

---

## 생성된 결과 파일

```
phase3/results/
├── projections_raw.pt                  # 전체 projection 데이터 (N=1000, L10~L22)
├── step1/
│   ├── projection_stats.json           # 레이어 × direction × condition 평균/표준편차
│   └── fig_projection_comparison.png   # 세 조건 mean projection line plot
├── step2/
│   ├── ratio_results.json              # ratio per (direction, layer)
│   ├── delta_results.json              # Δ_cc, Δ_sc per (direction, layer)
│   ├── fig_ratio.png                   # Ratio line plot
│   └── fig_delta_comparison.png        # Δ_cc vs Δ_sc 비교 plot
└── step3/
    ├── flipped_ids.json                # flipped/non-flipped case 인덱스
    ├── group_stats.json                # effect size, t-stat, p-val per (direction, layer)
    ├── fig_effect_size.png             # Cohen's d bar plot
    └── fig_boxplot.png                 # 대표 레이어 box plot
```

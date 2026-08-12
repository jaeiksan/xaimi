# Phase3 실험 재설계 지시문

## 사전 작업

### 기존 Phase3 폴더 초기화

```bash
rm -rf phase3/results/
rm -rf phase3/scripts/
mkdir -p phase3/results/step1
mkdir -p phase3/results/step2
mkdir -p phase3/results/step3
mkdir -p phase3/scripts
```

---

## 공통 설정

```python
MODEL_NAME    = "google/gemma-2-2b"
DATA_PATH     = "data/phase0/kc_cases_1000.json"
RESULTS_DIR   = "phase3/results"
DEVICE        = "cuda"

# 분석 대상 layer
TARGET_LAYERS = list(range(10, 23))  # L10 ~ L22

# 분석 대상 direction (기존 exp1b에서 추출된 것 재사용)
# 로드 경로:
#   A: results/exp1b_direction_extraction/direction_A.pt  (SVD conflict-clean)
#   B: results/exp1b_direction_extraction/direction_B.pt  (SVD conflict-supportive)
#   C: results/exp1b_direction_extraction/direction_C.pt  (cPCA)
# 각 direction shape: [n_layers, d_model] — layer별 unit vector
DIRECTIONS = ["A", "B", "C"]

# 세 조건
CONDITIONS = ["clean", "supportive", "conflict"]
```

---

## 데이터 구조

`data/phase0/kc_cases_1000.json` 각 케이스는 다음 필드를 가진다:

```json
{
  "case_id": 0,
  "clean_prompt": "...",
  "conflict_prompt": "...",
  "supportive_prompt": "...",
  "correct_answer": "A",
  "conflict_answer": "B"
}
```

---

## Step0: Projection 추출 (공통 전처리)

**스크립트**: `phase3/scripts/phase3_step0_extract_projections.py`

**목적**: 세 조건의 forward pass를 돌려 마지막 토큰 위치의 residual stream을 추출하고, 각 direction으로 projection 값을 계산해 저장한다. Step1/2/3 모두 이 결과를 재사용한다.

**구현 사항**:

```python
import torch
import json
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(MODEL_NAME)
model.eval()

# direction 로드
directions = {}
for d in ["A", "B", "C"]:
    directions[d] = torch.load(f"results/exp1b_direction_extraction/direction_{d}.pt")
    # shape: [n_layers, d_model], 각 layer별 unit vector

# 결과 저장 구조
# projections[direction][condition][layer] = Tensor[N]  (N=1000)
projections = {d: {c: {l: [] for l in TARGET_LAYERS} for c in CONDITIONS} for d in DIRECTIONS}

for case in kc_cases:  # 1000개
    for condition in CONDITIONS:
        prompt = case[f"{condition}_prompt"]

        with torch.no_grad():
            _, cache = model.run_with_cache(prompt)

        for layer in TARGET_LAYERS:
            # 마지막 토큰 위치의 residual stream
            residual = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  # [d_model]

            for d in DIRECTIONS:
                direction_vec = directions[d][layer]  # [d_model], unit vector
                proj = (residual @ direction_vec).item()
                projections[d][condition][layer].append(proj)

# Tensor로 변환 후 저장
for d in DIRECTIONS:
    for c in CONDITIONS:
        for l in TARGET_LAYERS:
            projections[d][c][l] = torch.tensor(projections[d][c][l])

torch.save(projections, "phase3/results/projections_raw.pt")
print("Saved: phase3/results/projections_raw.pt")
```

**저장**:
```
phase3/results/projections_raw.pt
  dict[direction: str][condition: str][layer: int] -> Tensor[1000]
```

---

## Step1: 세 조건 Projection 직접 비교

**스크립트**: `phase3/scripts/phase3_step1_comparison.py`

**목적**: 각 direction × layer에서 clean / supportive / conflict projection 값의 평균을 직접 비교한다. "conflict가 clean과 supportive보다 유독 큰가"를 눈으로 확인한다.

**구현 사항**:

```python
projections = torch.load("phase3/results/projections_raw.pt")

# 각 (direction, layer, condition)의 평균값 계산
stats = {}
for d in DIRECTIONS:
    stats[d] = {}
    for layer in TARGET_LAYERS:
        stats[d][layer] = {}
        for c in CONDITIONS:
            vals = projections[d][c][layer]
            stats[d][layer][c] = {
                "mean": vals.mean().item(),
                "std":  vals.std().item(),
            }

# 저장
import json
with open("phase3/results/step1/projection_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
```

**시각화**: `phase3/scripts/phase3_step1_plot.py`

그림 1: Direction별 세 조건 평균 projection line plot

```
subplot 3개 (A, B, C)
x축: layer (L10 ~ L22)
y축: mean projection value
선 3개: clean (회색) / supportive (초록) / conflict (빨강)
```

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, d in enumerate(["A", "B", "C"]):
    ax = axes[i]
    for c, color, label in [
        ("clean", "gray", "clean"),
        ("supportive", "green", "supportive"),
        ("conflict", "red", "conflict"),
    ]:
        means = [stats[d][l][c]["mean"] for l in TARGET_LAYERS]
        ax.plot(TARGET_LAYERS, means, color=color, label=label, marker="o", markersize=4)
    ax.set_title(f"Direction {d}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean projection")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

plt.tight_layout()
plt.savefig("phase3/results/step1/fig_projection_comparison.png", dpi=150)
```

**저장**:
```
phase3/results/step1/projection_stats.json
phase3/results/step1/fig_projection_comparison.png
```

---

## Step2: Ratio 비교 (Δ_cc vs Δ_sc)

**스크립트**: `phase3/scripts/phase3_step2_ratio.py`

**목적**: 각 케이스에서 다음 두 값을 계산한다.

```
Δ_cc = conflict projection - clean projection
Δ_sc = supportive projection - clean projection
```

그리고 ratio = mean(Δ_sc) / mean(Δ_cc) 를 layer별로 계산한다.

**판단 기준**:
```
ratio ≈ 0  → supportive가 clean에서 거의 안 올라감
             conflict만 올라가는 것 → conflict detection

ratio ≈ 1  → supportive도 conflict만큼 올라감
             context가 있으면 다 올라가는 것 → context following
```

**구현 사항**:

```python
projections = torch.load("phase3/results/projections_raw.pt")

ratio_results = {}
delta_results = {}

for d in DIRECTIONS:
    ratio_results[d] = {}
    delta_results[d] = {}
    for layer in TARGET_LAYERS:
        proj_clean     = projections[d]["clean"][layer]       # [N]
        proj_supportive = projections[d]["supportive"][layer]  # [N]
        proj_conflict  = projections[d]["conflict"][layer]    # [N]

        delta_cc = proj_conflict - proj_clean       # [N]
        delta_sc = proj_supportive - proj_clean     # [N]

        mean_cc = delta_cc.mean().item()
        mean_sc = delta_sc.mean().item()

        # ratio 계산 (분모가 0에 가까우면 nan 처리)
        if abs(mean_cc) < 1e-6:
            ratio = float("nan")
        else:
            ratio = mean_sc / mean_cc

        ratio_results[d][layer] = ratio
        delta_results[d][layer] = {
            "mean_delta_cc": mean_cc,
            "mean_delta_sc": mean_sc,
            "std_delta_cc": delta_cc.std().item(),
            "std_delta_sc": delta_sc.std().item(),
        }

# 저장
import json
with open("phase3/results/step2/ratio_results.json", "w") as f:
    json.dump(ratio_results, f, indent=2)
with open("phase3/results/step2/delta_results.json", "w") as f:
    json.dump(delta_results, f, indent=2)
```

**시각화**: `phase3/scripts/phase3_step2_plot.py`

그림 1: Ratio line plot (direction별)

```
subplot 3개 (A, B, C)
x축: layer (L10 ~ L22)
y축: ratio (Δ_sc / Δ_cc)
수평선: ratio=0 (conflict detection 기준), ratio=1 (context following 기준)
```

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, d in enumerate(["A", "B", "C"]):
    ax = axes[i]
    ratios = [ratio_results[d][l] for l in TARGET_LAYERS]
    ax.plot(TARGET_LAYERS, ratios, color="steelblue", marker="o", markersize=4)
    ax.axhline(0, color="green", linewidth=1, linestyle="--", label="conflict detection (ratio=0)")
    ax.axhline(1, color="red",   linewidth=1, linestyle="--", label="context following (ratio=1)")
    ax.set_title(f"Direction {d} — Δ_sc / Δ_cc")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 3)

plt.tight_layout()
plt.savefig("phase3/results/step2/fig_ratio.png", dpi=150)
```

그림 2: Δ_cc와 Δ_sc를 같은 축에 나란히 (Step1과 Step2 통합 시각화)

```
subplot 3개 (A, B, C)
x축: layer (L10 ~ L22)
y축: mean delta value
선 2개: Δ_cc (빨강, conflict-clean), Δ_sc (초록, supportive-clean)
ratio 값을 각 layer에 숫자로 표시
```

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, d in enumerate(["A", "B", "C"]):
    ax = axes[i]
    dcc = [delta_results[d][l]["mean_delta_cc"] for l in TARGET_LAYERS]
    dsc = [delta_results[d][l]["mean_delta_sc"] for l in TARGET_LAYERS]
    ratios = [ratio_results[d][l] for l in TARGET_LAYERS]

    ax.plot(TARGET_LAYERS, dcc, color="red",   marker="o", markersize=4, label="Δ_cc (conflict−clean)")
    ax.plot(TARGET_LAYERS, dsc, color="green", marker="s", markersize=4, label="Δ_sc (supportive−clean)")

    for j, (l, r) in enumerate(zip(TARGET_LAYERS, ratios)):
        if not (r != r):  # nan 제외
            ax.annotate(f"r={r:.2f}", (l, dsc[j]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color="gray")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Direction {d}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Δ projection")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("phase3/results/step2/fig_delta_comparison.png", dpi=150)
```

**저장**:
```
phase3/results/step2/ratio_results.json
phase3/results/step2/delta_results.json
phase3/results/step2/fig_ratio.png
phase3/results/step2/fig_delta_comparison.png
```

---

## Step3: Flipped vs Non-flipped Case 비교

**스크립트**: `phase3/scripts/phase3_step3_flipped.py`

**목적**: Phase2 grid search에서 개입 후 rank flip이 일어난 케이스(성공)와 안 일어난 케이스(실패)를 분리해서, conflict projection이 강한 케이스에서 실제로 rank flip도 더 잘 일어나는지 확인한다.

이를 통해 A/B/C projection이 단순한 관찰 신호가 아니라 실제 모델 동작과 연결된 신호임을 검증한다.

**flipped case 로드**:

```python
import json

# 기존 grid search 최우선 config 결과 로드
# config: L16:A + L17:C + L19:B (balanced config)
grid_results = json.load(open("results/exp_grid/all_results_ranked.json"))
best_config_key = "L16:A+L17:C+L19:B"  # 실제 key 이름 확인 필요

case_results = grid_results[best_config_key]  # list of dict, 케이스별 결과

flipped_ids     = [i for i, r in enumerate(case_results) if r["rank_flipped"] == True]
non_flipped_ids = [i for i, r in enumerate(case_results) if r["rank_flipped"] == False]

print(f"Flipped: {len(flipped_ids)}, Non-flipped: {len(non_flipped_ids)}")

# flipped case 수가 50 미만이면 fallback config 사용
# fallback: "L17:C+L18:B+L19:B"
```

**구현 사항**:

```python
from scipy import stats as scipy_stats

projections = torch.load("phase3/results/projections_raw.pt")

group_stats = {}

for d in DIRECTIONS:
    group_stats[d] = {}
    for layer in TARGET_LAYERS:
        # conflict 조건의 projection만 사용
        proj_conflict = projections[d]["conflict"][layer]  # [N]

        flipped_proj     = proj_conflict[flipped_ids]      # [n_flipped]
        non_flipped_proj = proj_conflict[non_flipped_ids]  # [n_non_flipped]

        # t-test
        t_stat, p_val = scipy_stats.ttest_ind(
            flipped_proj.numpy(),
            non_flipped_proj.numpy()
        )

        # effect size (Cohen's d)
        pooled_std = proj_conflict.std().item()
        effect_size = (flipped_proj.mean().item() - non_flipped_proj.mean().item()) / (pooled_std + 1e-8)

        group_stats[d][layer] = {
            "flipped_mean":     flipped_proj.mean().item(),
            "non_flipped_mean": non_flipped_proj.mean().item(),
            "effect_size":      effect_size,
            "t_stat":           t_stat,
            "p_val":            p_val,
        }

# 저장
with open("phase3/results/step3/group_stats.json", "w") as f:
    json.dump(group_stats, f, indent=2)

with open("phase3/results/step3/flipped_ids.json", "w") as f:
    json.dump({"flipped": flipped_ids, "non_flipped": non_flipped_ids}, f)
```

**시각화**: `phase3/scripts/phase3_step3_plot.py`

그림 1: Effect size line plot (direction별)

```
subplot 3개 (A, B, C)
x축: layer (L10 ~ L22)
y축: Cohen's d (effect size)
수평선: effect_size=0
양수 → flipped case에서 projection이 더 큼 (conflict detection signal이 강한 케이스가 flip도 잘 됨)
음수 → non-flipped case에서 더 큼
```

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, d in enumerate(["A", "B", "C"]):
    ax = axes[i]
    effects = [group_stats[d][l]["effect_size"] for l in TARGET_LAYERS]
    colors  = ["steelblue" if e >= 0 else "salmon" for e in effects]
    ax.bar(TARGET_LAYERS, effects, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Direction {d} — Effect size (flipped vs non-flipped)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d")

plt.tight_layout()
plt.savefig("phase3/results/step3/fig_effect_size.png", dpi=150)
```

그림 2: Flipped vs Non-flipped projection 분포 비교 (box plot, direction별 대표 layer)

```python
# 대표 layer: L15, L18 (기존 실험에서 핵심으로 나온 layer)
rep_layers = [15, 18]

fig, axes = plt.subplots(len(DIRECTIONS), len(rep_layers), figsize=(10, 12))
for i, d in enumerate(DIRECTIONS):
    for j, layer in enumerate(rep_layers):
        ax = axes[i][j]
        proj = projections[d]["conflict"][layer]
        data = [
            proj[flipped_ids].numpy(),
            proj[non_flipped_ids].numpy()
        ]
        ax.boxplot(data, labels=["flipped", "non-flipped"])
        ax.set_title(f"Direction {d} — L{layer}")
        ax.set_ylabel("Projection (conflict)")

plt.tight_layout()
plt.savefig("phase3/results/step3/fig_boxplot.png", dpi=150)
```

**저장**:
```
phase3/results/step3/group_stats.json
phase3/results/step3/flipped_ids.json
phase3/results/step3/fig_effect_size.png
phase3/results/step3/fig_boxplot.png
```

---

## 실행 순서

```bash
# 1. Projection 추출 (Step1/2/3 공통 전처리)
python phase3/scripts/phase3_step0_extract_projections.py

# 2. Step1: 세 조건 직접 비교
python phase3/scripts/phase3_step1_comparison.py
python phase3/scripts/phase3_step1_plot.py

# 3. Step2: Ratio 비교
python phase3/scripts/phase3_step2_ratio.py
python phase3/scripts/phase3_step2_plot.py

# 4. Step3: Flipped vs Non-flipped
python phase3/scripts/phase3_step3_flipped.py
python phase3/scripts/phase3_step3_plot.py
```

---

## 주의사항

- Direction 벡터는 기존 `results/exp1b_direction_extraction/`에서 로드. 새로 추출하지 않는다.
- **마지막 토큰 위치의 residual stream만 사용한다** (`cache[...][0, -1, :]`).
- `hook_resid_post`를 사용한다 (attention + MLP 모두 반영된 후).
- Step3에서 grid search 결과 json의 실제 key 이름을 먼저 확인하고 코드를 맞춘다.
- Flipped case 수가 50 미만이면 fallback config(`L17:C+L18:B+L19:B`)로 교체한다.
- 모든 forward pass는 `torch.no_grad()` 환경에서 수행한다.
- KC type별 분리 분석은 하지 않는다.

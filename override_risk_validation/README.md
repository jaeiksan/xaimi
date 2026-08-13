# Override-Risk Validation

이 디렉터리는 기존 Knowledge Conflict 연구를 **harmful override 위험 감지·완화** 관점에서
검증하는 독립 후속 실험을 담는다.

기존 `phase1/`~`phase4/` 구조를 바꾸거나 대체하지 않는다. 기존 Phase 4에서 확정한
direction, gate, intervention layer, hyperparameter를 그대로 재사용하고, 새 데이터는
held-out 평가용으로만 사용한다.

## 디렉터리 구조

```text
data/raw/        원본 데이터셋
data/processed/  조건별 후보·screening·최종 평가 데이터
scripts/         데이터 구성, screening, 병합, 평가 코드
results/         표와 그림 등 로컬 산출물
```

기존 파이프라인과 고정 설정은 `notes_existing_pipeline.md`에, 작업 이력과 실행 상태는
`PROGRESS.md`에 기록한다.

## 현재 진행 순서

1. TempLAMA에서 factual beneficial override 후보를 만들고 model behaviour로 screening한다.
2. ConflictBank default evidence에서 paired safe-context 조건 ①과 persistent-error 조건 ②를 만든다.
3. 조건 ①·②·③·④의 projection/gate 분포와 AUROC를 측정하는 실험 B를 실행한다.
4. 고정 gate 설정으로 no intervention / ungated / gated를 비교하는 실험 A를 실행한다.

## 조건 ③: TempLAMA beneficial override

후보 데이터 생성:

```bash
cd ~/xaimi/override_risk_validation
python scripts/02_build_condition3_templama.py
```

GPU 환경에서 behavioural screen 실행:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n apa python scripts/03_screen_condition3_templama.py
```

screening은 다음을 모두 만족하는 사례만 남긴다.

```text
closed-book 예측 = 과거 답 M
새로운 사실 C를 포함한 context 예측 = C
```

현재 이 절차는 완료됐으며, relation 비율을 보존한 seed 42 기반 최종 300개가
`data/processed/cond3_beneficial_override_300.jsonl`에 있다.

## 대규모 실행 원칙

코드 변경과 소규모 pilot은 대화형으로 실행한다. 수천~수만 건의 screening 및 전체 실험은
사용자가 직접 실행한다. 이렇게 하면 에이전트 실행 시간과 토큰 사용을 줄이고, GPU 두 장을
효율적으로 활용할 수 있다.

## 조건 ①·②: ConflictBank 두 GPU 분할 screening

`cond12_conflictbank_candidates_more.jsonl`의 25,000개 후보를 서로 겹치지 않게 12,500개씩
나눈다. 아래 두 명령을 **서로 다른 터미널**에서 실행한다.

```bash
cd ~/xaimi/override_risk_validation

# 터미널 1 — GPU 0
CUDA_VISIBLE_DEVICES=0 conda run -n apa python scripts/05_screen_conditions12_conflictbank.py \
  --input data/processed/cond12_conflictbank_candidates_more.jsonl \
  --start-index 0 --limit 12500 \
  --screened-output data/processed/cond12_screened_shard0.jsonl \
  --cond1-output data/processed/cond1_shard0.jsonl \
  --cond2-output data/processed/cond2_shard0.jsonl

# 터미널 2 — GPU 1
CUDA_VISIBLE_DEVICES=1 conda run -n apa python scripts/05_screen_conditions12_conflictbank.py \
  --input data/processed/cond12_conflictbank_candidates_more.jsonl \
  --start-index 12500 --limit 12500 \
  --screened-output data/processed/cond12_screened_shard1.jsonl \
  --cond1-output data/processed/cond1_shard1.jsonl \
  --cond2-output data/processed/cond2_shard1.jsonl
```

두 작업이 모두 끝난 뒤, 병합과 중복 검증을 실행한다.

```bash
conda run -n apa python scripts/06_merge_condition12_shards.py \
  data/processed/cond12_screened_shard0.jsonl \
  data/processed/cond12_screened_shard1.jsonl
```

병합 스크립트는 shard 사이에 중복된 ID 또는 질문이 있으면 실패한다. 정상 완료 시 다음 파일을 만든다.

```text
data/processed/cond12_conflictbank_screened_merged.jsonl
data/processed/cond1_safe_paired_merged.jsonl
data/processed/cond2_persistent_error_paired_merged.jsonl
```

## 실험 B: harmful-override risk 신호 특이성

조건별 최종 고정 세트에서 개입 없이 L17 B-direction projection, frozen gate value,
gate activation, full-choice accuracy를 수집한다. 이 실행은 약 1,478개 prompt view를
처리하므로 GPU 0에서 직접 실행한다.

```bash
cd ~/xaimi/override_risk_validation

CUDA_VISIBLE_DEVICES=0 conda run -n apa python scripts/08_collect_experiment_b.py \
  --output results/exp_B_per_case.jsonl
```

중간에 중단됐다면 같은 명령에 `--resume`을 붙여 기존 checkpoint 다음부터 재개한다.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n apa python scripts/08_collect_experiment_b.py \
  --output results/exp_B_per_case.jsonl --resume
```

수집이 완료된 뒤 CPU에서 summary·AUROC·bootstrap 95% CI를 계산한다.

```bash
conda run -n apa python scripts/09_analyze_experiment_b.py \
  --input results/exp_B_per_case.jsonl
```

결과 파일:

```text
results/exp_B_per_case.jsonl
results/exp_B_specificity.csv
results/exp_B_auroc.csv
```

## 실험 A: beneficial override 보호

조건 ③의 300개 사례에 기존 Phase 4의 고정 L17 설정을 적용한다. 비교 arm은 개입 없음,
무조건 parametric restoration, B-gated parametric restoration이다.

```bash
cd ~/xaimi/override_risk_validation

CUDA_VISIBLE_DEVICES=0 conda run -n apa python scripts/10_run_experiment_a.py \
  --output results/exp_A_per_case.jsonl
```

중단됐다면 다음 명령으로 완료된 세 intervention arm 다음 사례부터 재개한다.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n apa python scripts/10_run_experiment_a.py \
  --output results/exp_A_per_case.jsonl --resume
```

완료 후 paired bootstrap 95% CI를 계산한다.

```bash
conda run -n apa python scripts/11_analyze_experiment_a.py \
  --input results/exp_A_per_case.jsonl
```

결과 파일:

```text
results/exp_A_per_case.jsonl
results/exp_A_beneficial_override.csv
```

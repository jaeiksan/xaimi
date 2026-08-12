# Phase 4 Exp3: Intervention 방식 비교 실험 지시문

## 개요

이 지시문을 읽기 전에 반드시 아래 파일들을 먼저 확인해라.

```
phase4/scripts/eval_utils.py       # hook factory, 지표 계산 함수
phase4/scripts/setup_vectors.py    # 벡터 구조 확인
phase4/scripts/exp0_clean_patching.py  # 기존 실험 패턴 확인
phase4/scripts/exp3_gated_restoration.py  # 기존 Exp3 패턴 확인
scripts/exp_utils.py               # build_*_prompt, load_model, collect_residuals
phase4/data/train_800.json         # 데이터 필드명 확인
phase4/vectors/                    # 사전계산된 벡터 확인
```

기존 코드 스타일, hook 방식, 데이터 필드명을 모두 파악한 후 코드를 작성해라.

---

## 실험 목적

Phase 3에서 B direction이 conflict detection signal임을 확인했다.
Phase 4 Exp3에서는 이 B signal을 gate로 고정하고,
**어떤 방식의 개입(action)이 모델의 parametric knowledge 선택을 효과적으로 유도하는가**를 비교한다.

```
Gate (고정):   B direction projection
               → "지금 conflict 상황인가?" 판단

Action (비교): 5가지 방법
               → "어떻게 개입하면 parametric을 선택하는가?"
```

gate를 고정하고 action만 바꾸기 때문에,
5가지 방법의 차이가 순수하게 개입 방식의 효과 차이가 된다.

---

## 데이터 및 벡터

```python
# 사전계산된 벡터 (setup_vectors.py에서 생성됨)
# 모두 train 800개에서만 계산됨

dir_B    = phase4/vectors/direction_B_train.pt   # {layer: unit_tensor[d_model]}
u_PK     = phase4/vectors/u_PK_train.pt          # {layer: tensor[d_model]}
proj_clean = phase4/vectors/train_proj_clean.pt  # {"B": {layer: tensor[n_train]}}
baseline = phase4/vectors/baseline_logprobs.pt   # logprob + dev_ids
residuals = phase4/vectors/residuals.pt          # {cond: {layer: tensor[N, d_model]}}

# 평가 데이터
dev_cases = phase4/data/dev_200.json
```

---

## 공통 설정

```python
CANDIDATE_LAYERS = list(range(14, 23))  # L14 ~ L22
ALPHAS  = [0.5, 1.0, 2.0, 4.0]
GAMMAS  = [1.0, 2.0, 5.0, 10.0]
K_VALS  = [0.0, 0.5, 1.0, 1.5, 2.0]   # tau = mean + k * std

# Gate: B direction (고정)
# hyperparameter는 dev set 기준으로 선택
# test에서는 한 번만 평가
```

### Gate 계산 (모든 방법 공통)

```python
# B projection 기반 gate
gate = sigmoid(gamma * (h_last @ d_B_l - tau_B_l))

# tau 계산 (train clean projections에서)
tau_B_l = mean_train_clean_proj_B_l + k * std_train_clean_proj_B_l
```

---

## 스크립트 구조

```
phase4/scripts/
├── exp3_intervention_comparison.py   # 메인 실험 스크립트
├── intervention_methods.py           # 5가지 action hook factory
└── exp3_plot.py                      # 시각화
```

---

## intervention_methods.py

5가지 action 방법의 hook factory를 모두 이 파일에 구현한다.
모든 hook은 `blocks.{layer}.hook_resid_post` 또는 별도 hook point를 사용한다.
gate는 외부에서 계산해서 넘기거나 hook 안에서 계산한다.

### 공통 Gate Hook (residual stream에서 gate 계산)

```python
def make_gate_reader_hook(gate_dir: torch.Tensor, tau: float, gamma: float,
                          gate_store: list):
    """
    hook_name: blocks.{layer}.hook_resid_post
    마지막 position의 B projection으로 gate 값을 계산해서 gate_store[0]에 저장.
    residual stream은 수정하지 않음.
    """
    nd = F.normalize(gate_dir, dim=0)
    def hook_fn(value, hook):
        v = value.float()
        proj = v[:, -1, :] @ nd.to(v.device)  # [batch]
        gate_store[0] = torch.sigmoid(gamma * (proj - tau)).detach()
        return value  # 수정 없이 반환
    return hook_fn
```

---

### 방법 1: B-gated + u_PK

```
hook_name: blocks.{layer}.hook_resid_post
개입: h' = h + gate * alpha * u_PK_l
설명: conflict-clean 평균 차이 방향으로 이동 (clean 상태로 복원)
```

```python
def make_upk_hook(gate_store: list, u_PK: torch.Tensor, alpha: float):
    """
    gate_store[0]: 사전에 계산된 gate 값 [batch]
    u_PK: [d_model]
    """
    def hook_fn(value, hook):
        v  = value.float()
        up = u_PK.to(v.device)
        gate = gate_store[0].to(v.device) if gate_store[0] is not None else 1.0
        v[:, -1, :] = v[:, -1, :] + gate.unsqueeze(-1) * alpha * up
        return v.to(value.dtype)
    return hook_fn
```

---

### 방법 2: B-gated + Contrastive Direction

```
hook_name: blocks.{layer}.hook_resid_post
개입: h' = h + gate * alpha * d_contrast_l
설명: train에서 정답 케이스와 오답 케이스의 평균 residual 차이 방향으로 이동
      d_contrast = normalize(mean(h | correct) - mean(h | wrong))
      correct: conflict prompt에서 정답을 맞힌 케이스
      wrong:   conflict prompt에서 오답을 낸 케이스 (KC 발생)
```

#### Contrastive Direction 사전계산 (setup_vectors.py에 추가)

```python
def compute_contrastive_direction(residuals, baseline, layers, train_ids):
    """
    train 케이스에서 conflict condition 기준으로
    정답 케이스와 오답 케이스를 나누어 평균 차이를 구한다.

    correct_ids: conflict에서 lp_correct > lp_conflict인 train 케이스
    wrong_ids:   conflict에서 lp_correct < lp_conflict인 train 케이스 (KC 발생)
    """
    lp_cor  = baseline["lp_correct"]["conflict"][train_ids]
    lp_conf = baseline["lp_conflict"]["conflict"][train_ids]

    correct_mask = lp_cor > lp_conf  # [n_train], bool
    wrong_mask   = ~correct_mask

    d_contrast = {}
    for l in layers:
        h = residuals["conflict"][l][train_ids].float()
        h_correct = h[correct_mask]   # [n_correct, d_model]
        h_wrong   = h[wrong_mask]     # [n_wrong, d_model]

        if len(h_correct) == 0 or len(h_wrong) == 0:
            d_contrast[l] = torch.zeros(h.shape[-1])
            continue

        diff = h_correct.mean(0) - h_wrong.mean(0)  # [d_model]
        d_contrast[l] = F.normalize(diff, dim=0)

    return d_contrast

# 저장: phase4/vectors/direction_contrast_train.pt
```

#### Hook 구현

```python
def make_contrast_hook(gate_store: list, d_contrast: torch.Tensor, alpha: float):
    def hook_fn(value, hook):
        v  = value.float()
        dc = d_contrast.to(v.device)
        gate = gate_store[0].to(v.device) if gate_store[0] is not None else 1.0
        v[:, -1, :] = v[:, -1, :] + gate.unsqueeze(-1) * alpha * dc
        return v.to(value.dtype)
    return hook_fn
```

---

### 방법 3: B-gated + Attention Score Re-weighting

```
hook_name: blocks.{layer}.attn.hook_attn_scores
개입: conflict token position에 대한 attention score를 gate 비례로 감소
설명: conflict context token을 덜 참조하도록 유도
     완전히 끄는 게 아니라 부드럽게 줄임 (soft masking)
```

#### Conflict Token Position 파악

```python
def get_conflict_context_positions(model, prompt: str, ctx: str) -> list:
    """
    prompt에서 conflict context가 차지하는 token position을 반환한다.
    ctx: conflict_context 문자열
    """
    full_ids = model.to_tokens(prompt, prepend_bos=True)[0]
    ctx_ids  = model.to_tokens(ctx, prepend_bos=False)[0]

    # conflict context의 시작 position 탐색
    n_ctx = len(ctx_ids)
    for i in range(len(full_ids) - n_ctx + 1):
        if (full_ids[i:i+n_ctx] == ctx_ids).all():
            return list(range(i, i + n_ctx))
    return []  # fallback: 못 찾으면 빈 리스트
```

#### Hook 구현

```python
def make_attn_reweight_hook(gate_store: list, conflict_positions: list,
                             suppress_value: float):
    """
    hook_name: blocks.{layer}.attn.hook_attn_scores
    shape: [batch, n_heads, seq_len, seq_len]

    conflict_positions: conflict context token의 position 인덱스 리스트
    suppress_value: attention score에서 뺄 값 (gate 비례로 조절)

    h'_attn_score[:, :, :, conflict_positions] -= gate * suppress_value
    """
    def hook_fn(value, hook):
        v = value.float()
        if not conflict_positions:
            return value
        gate = gate_store[0]
        if gate is None:
            return value
        g = gate.to(v.device)  # [batch]
        for pos in conflict_positions:
            if pos < v.shape[-1]:
                v[:, :, :, pos] -= g.unsqueeze(-1).unsqueeze(-1) * suppress_value
        return v.to(value.dtype)
    return hook_fn
```

#### Grid

```python
SUPPRESS_VALUES = [1.0, 2.0, 5.0, 10.0]  # attention score 감소량
```

---

### 방법 4: B-gated + Attention Pattern Redirect

```
hook_name: blocks.{layer}.attn.hook_attn_scores
개입: conflict token attention을 줄이고, question token으로 재분배
설명: 단순히 줄이는 것이 아니라 attention을 다른 곳으로 redirect
```

#### Question Token Position 파악

```python
def get_question_positions(model, prompt: str, question: str) -> list:
    """
    prompt에서 question 문자열이 차지하는 token position을 반환한다.
    """
    full_ids = model.to_tokens(prompt, prepend_bos=True)[0]
    q_ids    = model.to_tokens(question, prepend_bos=False)[0]

    n_q = len(q_ids)
    for i in range(len(full_ids) - n_q + 1):
        if (full_ids[i:i+n_q] == q_ids).all():
            return list(range(i, i + n_q))
    return []
```

#### Hook 구현

```python
def make_attn_redirect_hook(gate_store: list,
                             conflict_positions: list,
                             question_positions: list,
                             suppress_value: float,
                             boost_value: float):
    """
    hook_name: blocks.{layer}.attn.hook_attn_scores
    shape: [batch, n_heads, seq_len, seq_len]

    conflict_positions → attention score 감소
    question_positions → attention score 증가

    conflict attention을 줄이고 question으로 재분배
    """
    def hook_fn(value, hook):
        v = value.float()
        gate = gate_store[0]
        if gate is None or not conflict_positions:
            return value
        g = gate.to(v.device)  # [batch]

        # conflict context suppress
        for pos in conflict_positions:
            if pos < v.shape[-1]:
                v[:, :, :, pos] -= g.unsqueeze(-1).unsqueeze(-1) * suppress_value

        # question redirect
        for pos in question_positions:
            if pos < v.shape[-1]:
                v[:, :, :, pos] += g.unsqueeze(-1).unsqueeze(-1) * boost_value

        return v.to(value.dtype)
    return hook_fn
```

#### Grid

```python
SUPPRESS_VALUES = [1.0, 2.0, 5.0]
BOOST_VALUES    = [0.5, 1.0, 2.0]
```

---

### 방법 5: B-gated + MLP Neuron Suppression

```
hook_name: blocks.{layer}.mlp.hook_post
개입: B direction과 aligned된 MLP neuron activation을 gate 비례로 감소
설명: residual stream 전체가 아니라 MLP의 특정 neuron subset에만 개입
     conflict signal을 증폭하는 neuron을 선택적으로 억제
```

#### Conflict-aligned Neuron 선택

```python
def get_conflict_neurons(model, residuals, dir_B, layer, train_ids, top_k=50):
    """
    MLP post-activation에서 B direction과 가장 강하게 aligned된 neuron을 선택.
    train conflict cases 기준.

    방법:
      W_out: [d_mlp, d_model]  (MLP output projection)
      neuron_b_alignment[i] = |W_out[i] @ d_B|
      → B direction에 가장 많이 기여하는 neuron top-k 선택
    """
    W_out = model.blocks[layer].mlp.W_out.float()  # [d_mlp, d_model]
    d_B   = dir_B[layer].float()
    alignment = (W_out @ d_B).abs()  # [d_mlp]
    top_neuron_ids = alignment.topk(top_k).indices.tolist()
    return top_neuron_ids

# 저장: phase4/vectors/conflict_neurons.pt
# {layer: list[int]}  (top-k neuron indices)
```

#### Hook 구현

```python
def make_mlp_suppression_hook(gate_store: list, neuron_ids: list,
                               beta: float):
    """
    hook_name: blocks.{layer}.mlp.hook_post
    shape: [batch, seq_len, d_mlp]

    conflict-aligned neuron의 activation을 gate 비례로 감소
    h_mlp[:, -1, neuron_ids] -= gate * beta * h_mlp[:, -1, neuron_ids].clamp(min=0)
    """
    def hook_fn(value, hook):
        v = value.float()
        gate = gate_store[0]
        if gate is None:
            return value
        g = gate.to(v.device)  # [batch]

        # positive activation만 억제 (음수는 건드리지 않음)
        acts = v[:, -1, :][:, neuron_ids]  # [batch, k]
        v[:, -1, :][:, neuron_ids] = acts - g.unsqueeze(-1) * beta * acts.clamp(min=0)

        return v.to(value.dtype)
    return hook_fn
```

#### Grid

```python
TOP_K_NEURONS = [10, 20, 50, 100]
BETA_VALUES   = [0.5, 1.0, 2.0]
```

---

## exp3_intervention_comparison.py

### 전체 구조

```python
"""
Phase 4 Exp3: Intervention 방식 비교

B direction을 gate로 고정하고 5가지 action 방법을 비교한다.

Baseline:
  1. no_intervention
  2. ungated_upk          (gate 없이 항상 u_PK 주입)
  3. shuffled_B_gated_upk (gate distribution 유지, case-specific 정보 제거)

B-gated 방법 5가지:
  4. B_gated_upk          (Track 1: Residual - u_PK)
  5. B_gated_contrast     (Track 1: Residual - contrastive direction)
  6. B_gated_attn_rw      (Track 2: Attention re-weighting)
  7. B_gated_attn_rd      (Track 3: Attention redirect)
  8. B_gated_mlp          (Track 3: MLP neuron suppression)

핵심 지표: selective_gain = conflict_gain - mean(clean_drop, supportive_drop)

실행:
  cd /home/jaeik/xaimi
  CUDA_VISIBLE_DEVICES=0 python -u phase4/scripts/exp3_intervention_comparison.py
"""
```

### Forward Pass 구조

방법마다 사용하는 hook point가 다르기 때문에 아래 원칙을 따른다.

**Residual 방법 (1, 2):**
```python
# gate를 같은 layer hook_resid_post에서 계산하고 action 적용
gate_store = [None]
hooks = [
    (f"blocks.{layer}.hook_resid_post", make_gate_reader_hook(d_B, tau, gamma, gate_store)),
    # gate_reader가 먼저 gate_store를 채움
    # 같은 hook_name이면 등록 순서대로 실행됨
    (f"blocks.{layer}.hook_resid_post", make_upk_hook(gate_store, u_PK, alpha)),
]
```

**Attention 방법 (3, 4):**
```python
# gate는 hook_resid_pre에서 계산 (attention 계산 전)
# action은 hook_attn_scores에서 적용
gate_store = [None]
hooks = [
    (f"blocks.{layer}.hook_resid_pre", make_gate_reader_hook(d_B, tau, gamma, gate_store)),
    (f"blocks.{layer}.attn.hook_attn_scores", make_attn_reweight_hook(gate_store, conflict_pos, suppress_value)),
]
```

**MLP 방법 (5):**
```python
# gate는 hook_resid_post에서 계산
# action은 hook_post (MLP activation 후)에서 적용
gate_store = [None]
hooks = [
    (f"blocks.{layer}.hook_resid_pre", make_gate_reader_hook(d_B, tau, gamma, gate_store)),
    (f"blocks.{layer}.mlp.hook_post", make_mlp_suppression_hook(gate_store, neuron_ids, beta)),
]
```

### 평가 루프 구조

```python
def evaluate_method(model, cases, layer, hooks_fn, conditions=("conflict","clean","supportive")):
    """
    hooks_fn(case) → List[(hook_name, hook_fn)]
    케이스마다 다른 hook이 필요한 경우(attention position 등) hooks_fn이 case를 받아서 생성
    """
    N = len(cases)
    result = {c: {"lp_correct": torch.zeros(N), "lp_conflict": torch.zeros(N)}
              for c in conditions}

    for j, case in enumerate(cases):
        cor_id = model.tokenizer.encode(f" {case['correct_option']}",  add_special_tokens=False)[-1]
        rep_id = model.tokenizer.encode(f" {case['replace_option']}",  add_special_tokens=False)[-1]
        hooks  = hooks_fn(case)

        for cond, prompt in [
            ("conflict",   build_conflict_prompt(case["conflict_context"], case["question"], case["options"])),
            ("clean",      build_clean_prompt(case["question"], case["options"])),
            ("supportive", build_supportive_prompt(case["question"], case["options"],
                                                   case["correct_answer"], case["conflict_context"],
                                                   case["conflict_answer"])),
        ]:
            with torch.no_grad(), model.hooks(fwd_hooks=hooks):
                ids    = model.to_tokens(prompt, prepend_bos=True)
                logits = model(ids)[0, -1, :].float().cpu()
            result[cond]["lp_correct"][j]  = logits[cor_id].item()
            result[cond]["lp_conflict"][j] = logits[rep_id].item()

    return result
```

### Shuffled Gate Baseline

```python
def compute_shuffled_gate_result(model, cases, layer, d_B, tau, gamma, u_PK, alpha):
    """
    B-gate 값을 case 간 shuffle해서 사용.
    gate distribution은 유지하면서 case-specific 정보만 제거.
    """
    # 1. conflict condition에서 gate 값 사전 계산
    gate_vals = torch.zeros(len(cases))
    for j, case in enumerate(cases):
        prompt = build_conflict_prompt(case["conflict_context"], case["question"], case["options"])
        ids = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                ids, names_filter=lambda n: n == f"blocks.{layer}.hook_resid_post"
            )
        h    = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].float()
        nd   = F.normalize(d_B[layer], dim=0).to(h.device)
        proj = float(h @ nd)
        gate_vals[j] = float(torch.sigmoid(torch.tensor(gamma * (proj - tau))))

    # 2. Shuffle
    perm    = torch.randperm(len(gate_vals))
    shuffled = gate_vals[perm]

    # 3. Shuffled gate로 평가
    # shuffled gate는 case j에 대해 shuffled[j]를 사용
    # make_fixed_gate_hook: gate 값을 고정해서 주입하는 hook
    ...
```

### Grid 탐색 및 Best Config 선택

```python
# dev set 기준으로 각 방법의 best config 선택
# 선택 기준: selective_gain 최대

best_configs = {}

for method_name in METHODS:
    best = None
    for layer in CANDIDATE_LAYERS:
        for alpha in ALPHAS:
            for gamma in GAMMAS:
                for k in K_VALS:
                    # 방법별 추가 grid (suppress_value, beta 등)
                    ...
                    result = evaluate_method(...)
                    m = compute_metrics(baseline, result)
                    if best is None or m["selective_gain"] > best["selective_gain"]:
                        best = {**m, "layer": layer, "alpha": alpha, ...}
    best_configs[method_name] = best
```

---

## 평가 지표 (compute_metrics 확장)

eval_utils.py의 `compute_metrics`를 그대로 사용하되,
Track별 추가 지표를 별도 함수로 구현한다.

```python
def compute_attention_shift(model, cases, layer, hooks_fn) -> float:
    """
    Track 2용: conflict token position으로의 attention weight 평균 변화량
    before: no hook
    after:  attention hook 적용
    """
    ...

def compute_mlp_suppression_rate(model, cases, layer, hooks_fn) -> float:
    """
    Track 3용: 억제된 neuron의 activation 평균 변화량
    """
    ...
```

---

## 저장 구조

```
phase4/results/exp3_comparison/
├── all_results.json              # 전체 grid 결과
├── best_configs.json             # 방법별 best config (dev 기준)
├── gate_stats.json               # gate activation 통계
└── figures/
    ├── fig_selective_gain.png    # 8가지 방법 selective gain 비교
    ├── fig_accuracy_comparison.png  # conflict/clean/supportive accuracy
    ├── fig_logit_margin.png      # logit margin delta 비교
    ├── fig_gate_activation.png   # gate activation rate (conflict vs clean)
    └── fig_layer_sweep.png       # layer별 selective gain (방법별)
```

---

## 결과 비교 테이블 구조

```
Method                   | Conflict↑ | Clean↓ | Sup↓ | Selective↑ | Margin↑
-------------------------|-----------|--------|------|------------|--------
no_intervention          |           |        |      |            |
ungated_upk              |           |        |      |            |
shuffled_B_gated_upk     |           |        |      |            |
B_gated_upk              |           |        |      |            |
B_gated_contrast         |           |        |      |            |
B_gated_attn_rw          |           |        |      |            |
B_gated_attn_rd          |           |        |      |            |
B_gated_mlp              |           |        |      |            |
```

---

## 실행 순서

```bash
# 0. contrastive direction 및 conflict neuron 사전계산 (setup_vectors.py에 추가)
python phase4/scripts/setup_vectors.py  # 이미 실행했으면 추가 부분만

# 1. 메인 실험
CUDA_VISIBLE_DEVICES=0 python -u phase4/scripts/exp3_intervention_comparison.py

# 2. 시각화
python phase4/scripts/exp3_plot.py
```

---

## 주의사항

- **모든 hyperparameter는 dev set에서만 선택한다.** test에서는 한 번만 평가한다.
- **Direction B는 unit vector로 정규화되어 있어야 한다.** setup_vectors.py에서 이미 처리됨.
- **Attention hook position 탐색 실패 시 (get_conflict_context_positions 반환 빈 리스트)** 해당 케이스는 no_intervention으로 처리한다.
- **같은 hook_name에 여러 hook 등록 시** TransformerLens는 등록 순서대로 실행한다. gate_reader가 반드시 action hook보다 먼저 등록되어야 한다.
- **Attention 방법에서 gate는 hook_resid_pre에서 계산한다.** hook_resid_post는 attention + MLP 이후라서 attention hook보다 늦다.
- **MLP hook_post는 MLP activation function 적용 후 값이다.** hook_pre는 적용 전.
- **Shuffled gate는 conflict condition에서 계산한 gate 값을 shuffle한다.** clean/supportive에서 계산하지 않는다.
- **KC type별 분리 분석은 하지 않는다.**
- **L21은 포함하되 결과 해석 시 scale 급락 특성을 언급한다.**

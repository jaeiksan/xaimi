# Knowledge Conflict (KC) 연구 — Claude Code 실험 가이드

## 연구 개요

이 연구는 LLM의 **Knowledge Conflict(KC)** 현상을 XAI와 Mechanistic Interpretability(MI) 두 관점에서 분석한다.

**KC 정의**: 모델의 파라메트릭 지식(학습 당시 알고 있던 것)과 추론 시 입력되는 외부 문맥이 충돌하여, 모델이 어느 한쪽으로 쏠린 출력을 생성하는 현상.

---

## 방법론 전체 구조

각 방법이 어떤 역할을 담당하는지 먼저 명확히 한다.

```
역할                방법                      수준
──────────────────────────────────────────────────────
XAI 위치 파악       Attention Attribution     레이어 × Head
XAI 방향 결정       IG (Integrated Gradients) 토큰
MI  인과 검증       Activation Patching       레이어
MI  방향 검증       Probing                   레이어
MI  정밀 분석       Gemma Scope SAE           feature
지식전환 구간 탐색  Logit Lens                레이어
수렴 정량화         TKOR                      레이어
```

**핵심 원칙**:
- `Attention Attribution` → `XAI_topK` 구성 (어느 레이어×Head가 중요한가)
- `Activation Patching` → `MI_topK` 구성 (어느 레이어가 인과적인가)
- `TKOR(K) = |XAI_topK ∩ MI_topK| / K` → 수렴 정량화
- `IG` → 위치 파악에 쓰지 않음. **방향 결정(d_IG)에만 사용**
  - `s_ctx = Σ IG(문맥 토큰)`, `s_param = Σ IG(질문 토큰)`
  - `s_ctx > s_param` → `d = -1` (파라메트릭 복원)
  - `s_param > s_ctx` → `d = +1` (문맥 강화)

---

## 핵심 가설

**가설 1 (수렴)**:
> KC 케이스에서 Attention Attribution이 중요하다고 본 레이어×Head 구간과 Activation Patching이 인과적이라고 본 레이어 구간은 충돌 유형에 따라 다른 정도로 수렴하며, 국소화된 충돌(Misinformation)일수록 TKOR이 높을 것이다.

**가설 2 (전환 레이어)**:
> KC가 발생하는 케이스에서 파라메트릭 지식 → 외부 문맥으로의 전환은 Logit Lens로 추적했을 때 특정 레이어 구간에 집중되어 나타날 것이다.

**가설 3 (방향 합의)**:
> IG(d_IG)와 Probing(d_probe)이 개입 방향에서 합의한 케이스에서의 KC 완화율이, 불일치 케이스보다 유의미하게 높을 것이다.

---

## 환경 설정

```bash
pip install transformer_lens transformers torch datasets
pip install inseq          # IG attribution 계산용 (실험 E 전용)
pip install sae-lens       # Gemma Scope SAE 사용 시
pip install numpy matplotlib seaborn scipy
```

**메인 모델**: `google/gemma-2-9b`
**교차 검증 모델**: `meta-llama/Llama-3.1-8B`
**데이터셋**: ConflictBank (NeurIPS 2024), QACC (Findings of NAACL 2025)

---

## KC 케이스 선별 기준

```python
# 반드시 아래 두 조건을 모두 만족하는 케이스만 사용한다.
# 조건 1: 충돌 문맥 없이 질문 시 → 모델이 정답 출력
# 조건 2: 충돌 문맥 주고 질문 시 → 모델이 오답 출력

KC_CASE_EXAMPLE = {
    "id": "kc_001",
    "conflict_type": "misinformation",   # misinformation / temporal / semantic
    "parametric_answer": "팀 쿡",        # 모델이 학습 당시 알고 있던 정답
    "conflict_context": "최근 보고서에 따르면 애플의 CEO는 홍길동이다.",
    "question": "애플의 CEO는 누구인가?",
    "conflict_answer": "홍길동",         # KC 발생 시 모델의 오답 출력
}
```

충돌 유형 3가지:
- `misinformation`: 허위 정보가 문맥에 주입된 경우
- `temporal`: 시간적으로 오래된 정보 vs 최신 정보 충돌
- `semantic`: 의미적으로 유사하지만 다른 정보 충돌

---

## Toy 실험 A: Attention Attribution — XAI 위치 파악

### 목적
KC 케이스에서 어느 레이어×Head 조합이 출력 결정에 크게 기여하는지 측정한다.
이 결과가 `XAI_topK` ranked list가 된다.

### 코드 구조

```python
from transformer_lens import HookedTransformer
import torch
import numpy as np
from typing import List, Dict, Tuple


def load_model(model_name: str = "google/gemma-2-9b") -> HookedTransformer:
    """TransformerLens로 모델을 로드한다."""
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    return model


def compute_attention_attribution(
    model: HookedTransformer,
    context: str,
    question: str,
    conflict_answer: str,
    parametric_answer: str,
) -> Dict:
    """
    Attention Attribution으로 레이어×Head별 기여도를 측정한다.

    방법: gradient × attention activation 기반 attribution
    타겟: conflict_answer 토큰 로짓 - parametric_answer 토큰 로짓

    Returns:
        {
            "layer_head_scores": np.ndarray,  # shape: (n_layers, n_heads)
            "layer_scores": np.ndarray,       # shape: (n_layers,) head 합산
            "top_k_layers": List[int],        # 상위 K개 레이어 (XAI_topK)
            "K": int,
        }
    """
    conflict_prompt = f"{context}\n{question}"
    tokens = model.to_tokens(conflict_prompt)

    attention_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))

    for layer in range(model.cfg.n_layers):
        attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        # gradient × attention 기반 attribution 계산 구현
        pass

    layer_scores = attention_scores.sum(axis=1)

    K = max(1, int(model.cfg.n_layers * 0.25))
    top_k_layers = np.argsort(layer_scores)[-K:][::-1].tolist()

    return {
        "layer_head_scores": attention_scores,
        "layer_scores": layer_scores,
        "top_k_layers": top_k_layers,
        "K": K,
    }


def visualize_attention_heatmap(
    layer_head_scores: np.ndarray,
    title: str,
    save_path: str
):
    """레이어(y축) × Head(x축) attribution score 히트맵 시각화."""
    pass
```

### 출력 파일
- `results/exp_a_attention_heatmap_{conflict_type}.png`
- `results/exp_a_top_k_layers.json`: XAI_topK ranked list

---

## Toy 실험 B: Logit Lens — 지식 전환 레이어 구간 식별

### 목적
KC 케이스에서 레이어별로 예측 토큰이 "파라메트릭 답" → "문맥 답"으로 전환되는
레이어 구간을 식별한다. 이 구간이 Activation Patching 탐색 범위를 좁혀준다.

### 코드 구조

```python
def compute_logit_lens(
    model: HookedTransformer,
    prompt: str,
    target_tokens: Dict[str, str],
    # {"parametric": "팀 쿡", "context": "홍길동"}
) -> Dict:
    """
    각 레이어의 hidden state를 vocab으로 투영하여
    레이어별 예측 토큰 확률을 계산한다.

    Returns:
        {
            "layers": List[int],
            "parametric_probs": List[float],
            "context_probs": List[float],
            "transition_layer": int,  # 문맥 답이 파라메트릭 답을 처음 넘는 레이어
        }
    """
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)

    parametric_probs, context_probs = [], []

    for layer_idx in range(model.cfg.n_layers):
        resid = cache[f"blocks.{layer_idx}.hook_resid_post"]
        layer_logits = model.unembed(model.ln_final(resid))
        probs = torch.softmax(layer_logits[0, -1], dim=-1)

        parametric_id = model.to_single_token(target_tokens["parametric"])
        context_id = model.to_single_token(target_tokens["context"])

        parametric_probs.append(probs[parametric_id].item())
        context_probs.append(probs[context_id].item())

    transition_layer = next(
        (i for i in range(len(parametric_probs))
         if context_probs[i] > parametric_probs[i]),
        None
    )

    return {
        "layers": list(range(model.cfg.n_layers)),
        "parametric_probs": parametric_probs,
        "context_probs": context_probs,
        "transition_layer": transition_layer,
    }


def plot_logit_lens_trajectory(result: Dict, case_info: Dict, save_path: str):
    """
    레이어(x축) × 확률(y축) 그래프.
    파라메트릭 답: 파란 실선 / 문맥 답: 빨간 실선
    전환 레이어: 수직 점선
    """
    pass
```

### 출력 파일
- `results/exp_b_logit_lens_{case_id}.png`
- `results/exp_b_transition_histogram.png`
- `results/exp_b_stats.json`

---

## Toy 실험 C: Activation Patching — MI 인과 검증

### 목적
레이어별로 hidden state를 clean run 값으로 교체했을 때 출력 변화를 측정한다.
AIE가 높은 레이어 = 인과적으로 중요한 레이어 → `MI_topK` ranked list.

### 코드 구조

```python
def compute_activation_patching(
    model: HookedTransformer,
    conflict_prompt: str,
    clean_prompt: str,
    parametric_answer: str,
    conflict_answer: str,
) -> Dict:
    """
    레이어별 AIE(Average Indirect Effect) 측정.

    AIE_l = (패치 후 파라메트릭 로짓 - 문맥 로짓)
            - (베이스라인 파라메트릭 로짓 - 문맥 로짓)

    값이 클수록 해당 레이어가 KC에 인과적으로 중요.

    Returns:
        {
            "layer_aie": np.ndarray,   # shape: (n_layers,)
            "top_k_layers": List[int], # MI_topK
            "K": int,
        }
    """
    conflict_tokens = model.to_tokens(conflict_prompt)
    clean_tokens = model.to_tokens(clean_prompt)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        conflict_logits = model(conflict_tokens)

    parametric_id = model.to_single_token(parametric_answer)
    conflict_id = model.to_single_token(conflict_answer)

    baseline_diff = (
        conflict_logits[0, -1, parametric_id]
        - conflict_logits[0, -1, conflict_id]
    ).item()

    layer_aie = np.zeros(model.cfg.n_layers)

    for layer_idx in range(model.cfg.n_layers):
        def patch_hook(value, hook):
            value[:] = clean_cache[f"blocks.{layer_idx}.hook_resid_post"]
            return value

        with model.hooks(fwd_hooks=[(
            f"blocks.{layer_idx}.hook_resid_post", patch_hook
        )]):
            patched_logits = model(conflict_tokens)

        patched_diff = (
            patched_logits[0, -1, parametric_id]
            - patched_logits[0, -1, conflict_id]
        ).item()

        layer_aie[layer_idx] = patched_diff - baseline_diff

    K = max(1, int(model.cfg.n_layers * 0.25))
    top_k_layers = np.argsort(layer_aie)[-K:][::-1].tolist()

    return {
        "layer_aie": layer_aie,
        "top_k_layers": top_k_layers,
        "K": K,
    }
```

### 출력 파일
- `results/exp_c_aie_barplot_{conflict_type}.png`
- `results/exp_c_top_k_layers.json`: MI_topK ranked list

---

## Toy 실험 D: TKOR 수렴 정량화

### 목적
실험 A(XAI_topK)와 실험 C(MI_topK)를 합쳐 수렴 정도를 정량화한다.

```python
def compute_tkor(xai_top_k: List[int], mi_top_k: List[int], K: int) -> float:
    """
    TKOR(K) = |XAI_topK ∩ MI_topK| / K

    완전수렴: TKOR >= 0.6
    부분수렴: 0.3 <= TKOR < 0.6
    비수렴:   TKOR < 0.3
    """
    return len(set(xai_top_k) & set(mi_top_k)) / K


def compute_spearman(xai_scores: np.ndarray, mi_aie: np.ndarray):
    from scipy.stats import spearmanr
    return spearmanr(xai_scores, mi_aie)
```

### 기대 결과

| 충돌 유형 | 예상 TKOR | 수렴 등급 |
|-----------|-----------|-----------|
| Misinformation | 0.65~0.80 | 완전수렴 |
| Temporal | 0.50~0.65 | 완전수렴 |
| Semantic | 0.30~0.50 | 부분수렴 |

---

## Toy 실험 E: IG Attribution — 개입 방향 결정 (d_IG)

### 목적
IG는 **위치 파악이 아닌 개입 방향 결정에만 사용**한다.

```python
import inseq

def compute_ig_direction(
    model_name: str,
    context: str,
    question: str,
    conflict_answer: str,
    n_steps: int = 50,
) -> Dict:
    """
    s_ctx   = Σ IG(문맥 토큰들)
    s_param = Σ IG(질문/파라메트릭 토큰들)

    d_ig = -1  if s_ctx > s_param  (문맥이 주도 → 파라메트릭 복원)
    d_ig = +1  if s_param > s_ctx  (파라메트릭 고집 → 문맥 강화)
    """
    model = inseq.load_model(model_name, "integrated_gradients")
    prompt = f"{context}\n{question}"

    attribution_result = model.attribute(
        input_texts=prompt,
        generated_texts=conflict_answer,
        n_steps=n_steps,
    )
    # s_ctx, s_param 계산 후 d_ig 반환
    pass
```

### 주의
- inseq와 TransformerLens를 같은 스크립트에서 로드하지 않는다.
- 실험 A~D는 TransformerLens만 사용, 실험 E는 inseq 전용 스크립트로 분리.

---

## 실험 실행 순서

```bash
# 0. KC 케이스 선별
python scripts/prepare_data.py \
    --dataset conflictbank \
    --n_samples 20 \
    --output data/kc_cases.json

# 1. Logit Lens (빠름, 탐색 범위 설정용)
python scripts/exp_b_logit_lens.py \
    --model google/gemma-2-9b \
    --data data/kc_cases.json \
    --output results/exp_b/

# 2. Attention Attribution (XAI_topK)
python scripts/exp_a_attention_attr.py \
    --model google/gemma-2-9b \
    --data data/kc_cases.json \
    --output results/exp_a/

# 3. Activation Patching (MI_topK)
python scripts/exp_c_activation_patching.py \
    --model google/gemma-2-9b \
    --data data/kc_cases.json \
    --output results/exp_c/

# 4. TKOR 수렴 정량화
python scripts/exp_d_tkor.py \
    --xai results/exp_a/ \
    --mi results/exp_c/ \
    --output results/exp_d/

# 5. IG 방향 결정 (별도 환경)
python scripts/exp_e_ig_direction.py \
    --model google/gemma-2-9b \
    --data data/kc_cases.json \
    --output results/exp_e/
```

---

## 주의사항

1. **GPU 메모리**: Gemma-2-9B는 약 18GB VRAM 필요. 부족하면 `google/gemma-2-2b`로 먼저 테스트.

2. **inseq-TransformerLens 충돌**: 실험 A~D는 TransformerLens만, 실험 E는 inseq 전용으로 분리.

3. **Activation Patching 토큰 길이**: conflict_prompt와 clean_prompt의 토큰 길이가 다를 수 있다. 패치 시 position 정렬 주의.

4. **KC 케이스 선별 엄격히**: "문맥 없을 때 정답 + 문맥 있을 때 오답" 두 조건 모두 실제로 모델에 넣어서 확인한다.

5. **중간 결과 캐싱 필수**: Activation Patching은 레이어 수만큼 forward pass 반복. JSON으로 저장해둔다.

---

## 전체 연구 맥락

```
Phase 1 (toy 실험 — 지금):
  실험 A: Attention Attribution → XAI_topK
  실험 B: Logit Lens            → 전환 레이어 탐색
  실험 C: Activation Patching   → MI_topK
  실험 D: TKOR                  → XAI-MI 수렴 정량화
  실험 E: IG                    → d_IG (방향 결정)

Phase 2 (본 실험):
  Probing                       → d_probe (억압 방향)
  d_IG == d_probe 합의 확인
  Gemma Scope SAE               → feature 수준 억압 분석

Phase 3 (개입):
  레이어 개입: h'_l = h_l + α·d·Δh_l
  Feature 개입: SAE feature 직접 조작
  두 방법 KC 완화율 + 일반 성능 저하 비교

Phase 4 (검증):
  TKOR 수준별 개입 효과 비교
  LLaMA 3.1 8B 교차 검증
```

---

## 참고문헌

- [1] Longpre et al. (EMNLP 2021). Entity-Based Knowledge Conflicts in QA.
- [2] Xu et al. (EMNLP 2024). Knowledge Conflicts for LLMs: A Survey.
- [3] Su et al. (NeurIPS 2024). ConflictBank.
- [4] Jin et al. (ACL 2024). Cutting Off the Head Ends the Conflict.
- [5] Chen et al. (ICML 2025 Spotlight). Taming Knowledge Conflicts in LMs.
- [6] Xie et al. (ICLR 2024). Adaptive Chameleon or Stubborn Sloth.
- [7] Liu et al. (NAACL 2025). Open Domain QA with Conflicting Contexts.
- [8] Elhage et al. (2021). A Mathematical Framework for Transformer Circuits.
- [9] Zou et al. (2023). Representation Engineering. arXiv:2310.01405.
- [10] Lieberum et al. (2024). Gemma Scope. arXiv:2408.05147.

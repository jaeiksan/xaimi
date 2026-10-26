# 실험 지시서: Override-Risk Gating의 특이성 검증 및 Beneficial Override 보호

## 배경 (연구 맥락 요약)

이 프로젝트는 RAG(Retrieval-Augmented Generation) 상황에서, 검증된 parametric 정답이
오염되거나 잘못된 context로 인해 잘못된 답으로 전환(override)되는 위험을 residual stream의
방향성 신호(direction)로 답 생성 전에 예측하고, 그 위험도에 비례하여 선택적으로 개입하는
방어 메커니즘을 개발 중이다. 이미 존재하는 실험 자산:

- Residual stream에서 conflict-specific direction을 추출하는 파이프라인
- Direction의 causal mediation을 검증하는 activation patching / causal tracing 코드
- Direction의 projection 값을 게이트로 사용해 선택적으로 개입하는 gating 메커니즘
  (naive knowledge conflict 조건에서 conflict 완화 약 31.8%, 일반 QA 성능 80%대 유지를 확인함)

**지금까지 검증되지 않은 것**이 이번 실험 A, B의 목적이다:
1. 이 신호가 정말로 "override" 자체에 특이적인가, 아니면 단순히 "context 유무" 또는
   "오답 여부"에 반응하는 것뿐인가 (confound 가능성) — **실험 B**
2. 이 신호/게이트가 "beneficial override" (정확한 최신 context가 모델의 outdated된
   parametric 지식을 올바르게 고쳐주는 경우)까지 실수로 막아버리지는 않는가 — **실험 A**

**A와 B를 합쳤을 때 포스터가 노리는 최종 메시지**는 다음 한 문장이다. 모든 실험 설계는
이 문장을 뒷받침하도록 되어 있다 (실험 A는 이 문장의 "개입하지 않는다" 부분, 실험 B는
"위험 신호가 높을 때만" 부분의 근거가 됨):

> 위험 신호가 높을 때만 context-induced harmful override를 억제한다.
> 정확한 최신 context가 유익하게 지식을 갱신하는 경우에는 개입하지 않는다.

## 중요: 용어/프레이밍 매핑 (기존 코드 ↔ 이번 실험)

**기존 코드베이스는 "knowledge conflict" 프레이밍으로 작성되어 있고, "RAG"나
"override"라는 단어를 코드/변수명/주석에서 전혀 찾을 수 없을 가능성이 높다.** 이건
코드가 실험과 무관하다는 뜻이 아니라, **같은 실험을 다른 이름으로 불렀을 뿐**이라는
뜻이다. 코딩 에이전트는 이 매핑을 반드시 먼저 이해하고, 기존 코드를 "override 관련
코드가 없다"고 판단해서 새로 처음부터 작성하는 실수를 하지 않아야 한다.

| 기존 코드에서 쓰는 용어 (knowledge conflict 프레이밍) | 이번 실험에서의 의미 (override-risk 프레이밍) |
|---|---|
| "clean context" 또는 evidence `E_M` (parametric과 일치하는 context) | 대략 조건 ① (clean correct)에 해당. 단, 모델이 실제로 맞혔는지는 별도 확인 필요 |
| "conflict context" 또는 evidence `E_C` (parametric과 불일치하는 context) | **주의: 이 하나의 카테고리 안에 조건 ③(beneficial override)과 ④(harmful override)가 뒤섞여 있을 가능성이 매우 높음.** 기존 코드는 "context가 parametric과 다른가"만 보고 "그 차이가 옳은 방향인가 틀린 방향인가"는 구분하지 않았을 것임 |
| 답이 `M`(memorised/parametric)에서 `C`(context) 답으로 바뀜 | 이게 바로 "override"가 일어난 것. 기존 코드에 override라는 단어가 없어도, "M에서 C로 답이 바뀌는 현상"을 다루고 있었다면 이미 override를 관찰하고 있었던 것임 |
| "conflict 완화" (mitigation), "게이트가 M을 지키도록 개입" | 지금 말하는 "무조건적 parametric 강제 개입"과 유사하거나 동일한 코드일 가능성이 높음. 다만 기존 실험에서는 이게 항상 "좋은 개입"이라고 가정했을 수 있는데, beneficial override 조건에서는 이 개입이 오히려 나쁜 개입이 됨 |
| RAG라는 단어 자체가 코드에 없음 | 문제 없음 — "context를 주고 답을 생성하는 QA 세팅" 자체가 이미 RAG의 최소 형태이므로, RAG를 위한 별도 파이프라인을 새로 만들 필요가 없음 |

**코딩 에이전트가 반드시 확인해야 할 것**: 기존에 쓰던 "conflict" 데이터셋(evidence
`E_C`)을 열어서, 각 인스턴스가 실제로 조건 ③(정답 방향으로 override, beneficial)인지
조건 ④(오답 방향으로 override, harmful)인지 구분되어 있는지 확인한다. 구분이 안
되어 있다면(즉 그냥 "conflict냐 아니냐"만 라벨링되어 있다면), 이것 자체가 지금
검증하려는 confound 문제이며, 이번 실험 B가 정확히 이 문제를 겨냥한 것이라는 점을
README에 명시한다.

### ⚠️ 절대 하지 말 것: 기존 "supportive" 데이터를 조건 ③으로 재사용하는 것

기존 코드에 "supportive" prompt(또는 유사한 이름의 context)가 있다면, 이건 **conflict
문맥의 오답 텍스트를 정답 텍스트로 치환해서 만든 합성(synthetic) context**일 가능성이
높다. 이 경우 근본적으로 **모델이 원래부터 정답을 알고 있던 사례**를 기반으로 만든
것이므로, "outdated parametric 지식이 context로 인해 올바르게 고쳐지는" beneficial
override와는 다른 것이다 (supportive 조건에서는 애초에 override가 일어날 필요가 없음 —
모델이 처음부터 맞았기 때문).

**따라서 조건 ③(beneficial override)은 반드시 별도로 새로 만들어야 하며, 기존
supportive 데이터를 그대로 갖다 쓰면 안 된다.** 아래 실험 A의 데이터 섹션에 있는
절차(closed-book에서는 outdated 답 M, context 포함 시 최신 정답 C로 실제로 바뀌는
인스턴스만 선택)를 반드시 따른다.

## 코드 작성 전에 반드시 먼저 할 일

**절대 새 파이프라인을 처음부터 작성하지 말 것.** 먼저 아래 순서로 기존 코드베이스를
탐색하고, 재사용 가능한 부분을 파악한 뒤, 그 위에 최소한만 추가한다.

1. 저장소 전체 구조를 훑어본다 (`find`, `tree`, 또는 디렉토리 목록으로).
2. 다음 4가지 기능을 담당하는 기존 스크립트/함수를 찾는다:
   - (a) LLM에 (question, context) 쌍을 넣어 forward pass를 돌리고 특정 layer의
     residual stream activation(hidden state)을 추출/저장하는 코드
   - (b) 저장된 activation으로부터 conflict direction(또는 A/B direction)을
     계산하는 코드 (예: 평균 차이, SVD 등)
   - (c) 임의의 입력에 대해 해당 direction으로의 projection(스칼라 신호값)을
     계산하는 코드
   - (d) projection 값을 게이트로 써서 특정 layer의 activation에 개입(steering /
     activation addition)하고, 이후 생성된 답을 얻는 코드
3. 각 스크립트의 입출력 형식(함수 시그니처, 데이터 포맷, 저장 경로 규칙)을 정리해서
   `notes_existing_pipeline.md` 파일로 새 실험 디렉토리에 남긴다. (다음 사람이 실험을
   재현할 때 참고하도록.)
3-1. **위 "용어/프레이밍 매핑" 표를 참고하여**, 기존에 쓰던 conflict 데이터셋을 직접
   열어보고 다음을 확인한다: (i) 각 인스턴스에 override 방향(정답→오답인지, 오답→정답인지)을
   판단할 수 있는 정보가 있는가 (예: 정답 라벨과 parametric 답을 비교하면 계산 가능한가),
   (ii) 없다면 어떤 필드(질문, evidence, M, C)를 조합해서 이번 실험에서 새로 계산해야
   하는가를 `notes_existing_pipeline.md`에 기록한다.
4. 기존 코드가 어떤 모델(예: Llama, Mistral 등)과 어떤 layer 번호를 쓰는지 확인하고,
   이번 실험도 **동일한 모델/layer**를 그대로 사용한다 (비교 가능성을 위해 절대 바꾸지 않음).
5. 위 탐색이 끝나기 전까지는 실험 A, B의 코드를 작성하지 않는다.

## 새 디렉토리 구조 제안

```
experiments/
  override_specificity_and_beneficial/
    README.md                       # 이 실험의 목적, 실행 방법, 결과 요약 (최종 업데이트)
    notes_existing_pipeline.md      # 위 2단계에서 정리한 기존 코드 사용법
    data/
      raw/                          # 원본 다운로드 (ConflictBank 등)
      processed/
        cond1a_closedbook_correct.jsonl
        cond1b_supportive_context_correct.jsonl
        cond2_clean_context_error.jsonl
        cond3_beneficial_override.jsonl
        cond4_natural_harmful_override.jsonl
    scripts/
      01_build_condition2_clean_error.py
      02_build_condition3_beneficial.py
      03_collect_activations.py       # 가능하면 기존 (a) 스크립트를 얇게 감싸서 재사용
      04_compute_projection_and_gate.py  # 가능하면 기존 (c),(d) 재사용
      05_run_experiment_A.py
      06_run_experiment_B.py
      07_make_tables_and_plots.py
    results/
      exp_A_beneficial_override.csv
      exp_B_specificity.csv
      figures/
```

## 실험 A: Beneficial Override 보호 검증

**질문**: 모델이 outdated된 parametric 지식을 갖고 있고, context가 정확한 최신 정보를
줘서 정답으로 override되는 상황(beneficial override)에서, 지금의 direction/gate가
불필요하게 개입해서 이 정당한 override를 막아버리는가?

**데이터 (조건 ③: beneficial override) — 반드시 새로 구성, 기존 supportive 데이터 재사용 금지**:

1. ConflictBank의 "temporal conflict" 서브셋을 원본에서 가져온다
   (https://github.com/zhaochen0110/conflictbank).
2. **"temporal"이라는 카테고리 이름만 믿지 않는다.** 원본 데이터의 정답 라벨과 시간
   메타데이터(예: 각 claim이 언제 사실이었는지, ground-truth answer가 무엇인지)를
   직접 확인하여, **context 쪽 답(C)이 실제로 최신 기준 정답이고 parametric 쪽 답(M)이
   실제로 outdated된 오답인지를 인스턴스 단위로 검증**한다. temporal conflict로
   분류되어 있어도 C가 틀렸거나 애매한 경우가 섞여 있을 수 있으므로, 이 검증을 거치지
   않은 인스턴스는 조건 ③에서 제외한다.
3. 검증을 통과한 인스턴스 중에서도, 모델이 실제로 그렇게 행동하는 경우만 최종 선택한다:
   - closed-book(질문 단독)으로 물었을 때 모델이 outdated 답 M을 낸다
   - 같은 질문에 context를 주면 모델이 최신 정답 C로 답을 바꾼다
   - (즉 "정답 라벨상 beneficial"인 것과 "이 모델이 실제로 override되는 사례"를 모두
     만족해야 조건 ③에 포함됨)
- 최소 200~300개 인스턴스면 포스터용 예비 결과로 충분하다. 시간 여유가 있으면 늘린다.

**중요 — 하이퍼파라미터는 절대 새로 튜닝하지 않는다**: 기존에 이미 확정되어 있는
direction(예: B direction), 개입 layer, `alpha`(개입 강도), `gamma`(게이트 sigmoid
기울기), `tau`(게이트 threshold) 값을 그대로 고정해서 사용한다. 이번에 새로 만드는
조건 ②③ 데이터는 **완전히 held-out 평가용**으로만 쓴다 (여기에 맞춰 하이퍼파라미터를
다시 고르면 안 됨). 이렇게 해야 "이 게이트가 원래 튜닝했던 naive conflict 조건뿐
아니라, override-risk라는 새 프레이밍/새 데이터에서도 일반화한다"는 주장이 성립한다.

**절차**:
1. 각 인스턴스에 대해 (질문 단독), (질문+context) 두 가지로 forward pass를 돌려
   각각의 답과 마지막 토큰의 residual state를 얻는다.
2. 기존 direction(고정값)으로 projection(위험도 신호)을 계산한다.
3. 게이트 개입을 켠 상태로 답을 생성한다. 이때도 기존에 확정된 `alpha/gamma/tau`를
   그대로 사용한다.
4. 아래 3가지를 비교한다:
   - **개입 없음**: override 성공률 (즉 context 덕분에 정답으로 바뀐 비율) — 이게 기준선
   - **무조건 parametric 강제 개입** (ungated, 항상 parametric 쪽으로 steering) — 이 baseline은
     beneficial override를 대부분 막아버릴 것으로 예상됨 (이게 "무조건 개입의 한계"를 보여주는 대조군)
   - **risk-gated 개입 (제안 방법)**: beneficial override 성공률이 "개입 없음"과 비슷하게
     유지되는지 확인 (즉 게이트가 여기서는 거의 발동하지 않아야 함)

**출력**: `results/exp_A_beneficial_override.csv` — 컬럼: `condition, intervention_type,
override_success_rate, mean_projection, mean_gate_value, gate_activation_rate, n_samples`

(주의: `projection`은 direction 내적값 그 자체이고, `gate_value`는 이걸 sigmoid 등에
통과시킨 후의 값이다. 이 둘을 하나의 컬럼으로 뭉치면 해석이 흐려지므로 반드시 분리해서
기록한다. 게이트가 `sigmoid(gamma * (projection - tau))` 형태라면, `gate_activation_rate`는
다음과 같이 정의한다: `projection >= (고정된) tau`인 인스턴스의 비율. 이는 정의상
`gate_value >= 0.5`인 것과 정확히 동치이다.)

**통계 검정**: 같은 인스턴스에 대해 개입 전/후(또는 개입 방식 간)를 비교하는 것이므로
**paired bootstrap 또는 McNemar test**로 유의성을 검정한다 (단순 비율 차이만 보고하지
않음). 부트스트랩을 쓸 경우 최소 1,000회 resampling, 95% CI를 같이 보고한다.

**성공 기준**: risk-gated 개입의 override_success_rate가 개입 없음과 통계적으로 유의하게
다르지 않고, ungated(무조건 개입)보다 훨씬 높아야 한다. 만약 risk-gated도 beneficial override를
많이 막는다면, 이는 negative result이며 그대로 보고한다 (포스터의 discussion/limitation에
넣을 수 있는 정직한 결과임).

## 실험 B: Harmful-Override Risk 신호의 판별성 검증 (Confound 배제)

**정확한 질문 (주의: "override 특이성"이 아니라 "harmful-override risk 특이성"임)**:
신호가 "override가 일어났다는 사실 자체"에 반응하는 것인지, 아니면 그중에서도
**"해로운 방향의 override(harmful)"에만 특이적으로** 반응하는 것인지를 구분해야 한다.
이 둘은 다른 주장이다 — 신호가 ③(beneficial)과 ④(harmful)를 구분하지 못하고 둘 다에서
비슷하게 높게 나온다면, "신호가 override 발생 여부는 감지하지만 해로움까지는
판별하지 못한다"는 것이며, 이는 실패가 아니라 **연구적으로 중요한 negative result**로
그대로 보고해야 한다 (게이트 설계를 재고해야 한다는 신호이기 때문).

**데이터 (조건 ①a·①b·②③④ 비교) — ①을 같은 질문 안의 두 view로 나눈다**:
- **①a Closed-book correct**와 **①b Supportive-context correct**는 **반드시 같은
  질문(question)에 대한 두 가지 prompt view**로 구성한다 — ①a는 그 질문을 context 없이
  물은 것, ①b는 같은 질문에 정답과 일치하는 context를 붙여서 물은 것. 이렇게 해야
  "context 유무"만 바뀌고 질문의 난이도·주제는 그대로 유지되어, 신호 차이가 순수하게
  context 존재 효과임을 알 수 있다. **①a/①b를 서로 다른 질문 집합으로 나누면 안 된다**
  (질문 난이도/주제 차이가 섞여서 원인 분리가 안 됨).
  - ①a/①b는 이렇게 하나의 "paired safe-context 그룹"을 이루고, 이 그룹 전체가
    ②·③·④와는 질문 단위로 중복되지 않게 분리된다 (즉 분리는 그룹 대 그룹으로 하고,
    그룹 내부의 ①a/①b끼리는 같은 질문을 공유함).
  - **①b가 실험 B의 주(main) 비교 기준선이다.** ④(harmful override)와 비교할 때는
    반드시 이 ①b를 써야 한다 — ①b도 ④와 마찬가지로 "context가 존재하는" 상황이므로,
    ①b vs ④ 비교에서 신호값 차이가 나온다면 그건 순수하게 "context 존재 여부"나
    "prompt 길이/형식" 때문이 아니라 "그 context가 안전한지 위험한지"에 신호가
    반응한다는 훨씬 강한 증거가 된다.
- **② Clean-context error**: **①a/①b와 마찬가지로, 같은 질문 집합에 대해 closed-book
  view와 supportive-context view 둘 다를 평가한다.** 이 그룹에 포함되는 질문은
  "질문 단독(closed-book)에서도 오답이고, supportive context를 줘도 여전히 오답인
  경우"로 고정한다 (context가 있든 없든 모델이 아예 모르는 질문). 이렇게 하면 아래
  두 비교를 각각 남길 수 있다:
  - `② closed-book error` vs `①a closed-book correct`
  - `② supportive-context error` vs `①b supportive-context correct`

  이 두 비교를 같은 prompt 형식 안에서 각각 보면, "단순 오답 효과"와 "context 존재
  효과"를 서로 분리해서 확인할 수 있다.
- **③ Beneficial override**: 실험 A에서 새로 만든 데이터 재사용 (기존 supportive
  데이터가 아님 — 위 경고 참고)
- **④ Natural harmful override**: 기존에 이미 쓰고 있던 naive conflict 데이터 그대로 재사용
  (6월 실험에서 conflict 완화 31.8% 낼 때 쓴 데이터셋)

**데이터 중복 금지 (그룹 단위)**: "①(a+b) 그룹", "②(closed-book+supportive) 그룹",
③, ④는 서로 질문 단위로 중복되지 않게 구성한다. 단, 각 그룹 **내부**에서는 (①a-①b
쌍, ②의 closed-book-supportive 쌍) 같은 질문을 공유하는 것이 맞다. seed를 고정해서
어떤 질문이 어느 그룹으로 갔는지 기록을 남긴다.

**평가 함수에 대한 주의**: 기존 평가 코드는 주로 `correct_option` 대 `replace_option`의
2지선다 logit 비교로 되어 있을 가능성이 높다. 조건 ②에서는 `replace_option`이라는
개념 자체가 의미가 없을 수 있으므로(원래 정답을 대체할 특정 오답이 지정되어 있지 않을
수 있음), **네 개(또는 원래 선택지 수만큼)의 전체 선택지 중 정답을 골랐는지를 계산하는
공통 평가 함수를 새로 추가**해서 모든 조건(①②③④)에 동일하게 적용한다. 기존
2지선다 평가와 새 다지선다 평가 결과를 둘 다 남겨서, 두 평가 방식 간 결과가 크게
다르지 않은지도 확인한다.

**절차**:
1. ①a, ①b, ②(closed-book/supportive 두 view), ③, ④ 각 조건에서 direction projection
   값의 분포를 계산한다 (gate_value도 별도로 계산해서 둘 다 남긴다).
2. **주 비교는 ①b vs ④**로 한다. 보조로 다음 비교들을 같이 기록한다:
   - `①a vs ①b` (같은 질문에서 context 존재 자체의 영향)
   - `② closed-book error vs ①a closed-book correct` (closed-book 상황에서의 단순
     오답 효과)
   - `② supportive-context error vs ①b supportive-context correct` (context가 있는
     상황에서의 단순 오답 효과 — 이게 ①b vs ④ 비교와 같은 prompt 형식 안에서 비교되므로,
     "context가 있다"는 조건을 통제한 채 "오답이냐 harmful override냐"만 다른 두 비교를
     나란히 볼 수 있게 됨)
3. 다음이 성립하는지 확인한다:
   - ④(harmful override) > ①b (신호가 "안전한 context가 있는 상황"보다 harmful
     override 상황에서 유의하게 높아야 함 — 이게 이번 실험의 핵심 비교)
   - ②의 두 view가 각각 ①a, ①b와 신호값이 비슷해야 함 (오답 자체가 아니라 override에
     반응한다는 증거이며, closed-book/supportive 두 상황 모두에서 이게 성립해야 더 강함)
   - ③(beneficial override, override는 일어났지만 해로운 게 아님)과 ④의 신호값을 비교 —
     만약 ③도 ④만큼 신호가 높다면, 신호는 "override 발생 여부"에만 반응하고 "해로운지"는
     구분하지 못한다는 뜻 (이러면 실험 A의 게이트 성공 여부와 직접 연결되는 중요한 발견)
4. AUROC으로 "④ vs ①b"를 구분하는 성능을 주 지표로 계산하고, "④ vs ①a·②·③ 전체"를
   구분하는 성능을 보조 지표로 같이 계산한다 (SpARE 논문의 probing 방식과 동일한
   방법을 재사용).

**출력**: `results/exp_B_specificity.csv` — 컬럼: `condition, mean_projection,
std_projection, mean_gate_value, gate_activation_rate, n_samples`, 그리고 AUROC 값은
별도 파일(`results/exp_B_auroc.csv`)에 `comparison_pair, auroc, auroc_ci_lower,
auroc_ci_upper`로 기록.

**통계 검정**: AUROC은 반드시 **95% bootstrap CI**와 같이 보고한다 (예: 1,000회
resampling). 조건 간 projection 평균 차이(예: ④ vs ①b)도 마찬가지로 bootstrap CI를
계산해서 0을 포함하지 않는지로 유의성을 판단한다. 단순 t-test 하나만으로 결론
내리지 않는다.

**결과 해석 기준 (성공/실패가 아니라 두 가지 가능한 결과로 이해할 것)**:
- **②가 ①a와 신호값이 비슷함**: 이건 반드시 확인되어야 하는 전제조건이다. 여기서
  벗어나면(② 신호가 ①a보다 유의하게 높으면) "단순 오답"에도 신호가 반응하는 것이므로,
  override 프레이밍 자체가 흔들리는 심각한 문제다.
  - **결과 (a) — 이상적**: ④ 신호값이 ①b보다(즉 "안전하게 context가 있는 상황"보다)
    유의하게 높고, ③은 ①b와 비슷함. → 신호가 harmful override에 특이적이라는, 가장
    강한 주장이 성립. 이 경우 "context 존재 자체"가 원인이 아니라는 것까지 ①b 비교로
    확실히 배제된 것임.
  - **결과 (b) — 중요한 negative result**: ④와 ③ 둘 다 ①b보다 높고, ④와 ③ 사이에는
    유의한 차이가 없음. → 신호는 "override가 일어났다"는 것은 감지하지만 "그게 해로운
    방향인지"는 구분하지 못한다는 뜻. 이 경우 포스터의 메시지를 "신호=override 감지,
    게이트=위험도 판별"이 아니라 "신호=override 감지, 그 자체로는 harmful/beneficial을
    구분 못 하며 향후 이 구분을 위한 추가 feature가 필요하다"는 방향으로 조정해야 한다.
    (이 경우 실험 A의 결과와 반드시 같이 해석한다: A에서 beneficial override가
    보호됐다면 게이트의 다른 부분(threshold 설계, 다른 정보 결합 등)이 이미 이 구분을
    암묵적으로 하고 있다는 뜻이 되어 오히려 흥미로운 후속 질문이 됨)
  - **참고**: ①a와 ①b 사이에 신호값 차이가 크게 난다면(즉 context가 있다는 사실
    자체만으로 신호가 움직인다면), 이는 별도로 보고해야 하는 관찰이다 — 이 경우 ④
    vs ①b 비교의 해석에 이 효과를 함께 고려해야 한다.

## 실험 C (A, B를 반드시 먼저 끝낸 뒤 진행하는 보조 확장): 신호 단위 비교

**A, B가 먼저다.** C는 A, B의 결과가 나온 뒤에 시간이 남을 때만 진행한다. C의 목적은
"direction projection이 단순한 대안 신호(logit gap, entropy)보다 실제로 더 유용한가"를
보여주는 것이며, 이건 A/B의 핵심 주장(harmful override만 선택적으로 막는다)을 보강하는
역할이지 그 자체로 독립적인 메인 결과는 아니다.

실험 A, B에서 이미 수집한 activation을 **재사용**해서 (추가 forward pass 없이), 다음을
추가로 계산해 비교표를 만든다:
- Linear probe (logistic regression, layer별 hidden state로 학습): AUROC
- Logit gap (parametric 후보 답과 contextual 후보 답의 logit 차이): AUROC
- Entropy (생성 답의 토큰 분포 entropy): AUROC
- 기존 direction projection: AUROC (실험 B에서 이미 계산됨)

비교는 실험 B와 동일한 기준(①a vs ②의 구분 가능성, 주 비교인 ①b vs ④의 구분 가능성,
③ vs ④의 구분 가능성)으로 각 신호 단위에 대해 반복한다.

## 최종 체크리스트

- [ ] 기존 파이프라인 탐색 및 `notes_existing_pipeline.md` 작성 완료
- [ ] 기존 "conflict"/"supportive" 데이터가 beneficial/harmful을 구분하는지 확인 완료
- [ ] 기존 direction/layer/alpha/gamma/tau 값을 확인하고 고정 (재튜닝 금지) 완료
- [ ] ConflictBank temporal 서브셋의 정답·시간 라벨을 직접 검증해서 조건 ③ 구성
      (기존 supportive 데이터 재사용 아님)
- [ ] 조건 ①을 같은 질문에 대한 ①a(closed-book)/①b(supportive-context) paired view로 구성
- [ ] 조건 ②도 같은 질문 집합에 대해 closed-book/supportive-context 두 view로 구성
      (양쪽 다 오답인 질문만 선택)
- [ ] "①그룹", "②그룹", ③, ④가 서로 질문 단위로 중복되지 않도록 분리 (그룹 내부의
      paired view끼리는 같은 질문을 공유, seed 고정, 기록 남김)
- [ ] 4지선다(또는 전체 선택지) 정답 판별용 공통 평가 함수 작성
- [ ] 실험 A 실행 및 `exp_A_beneficial_override.csv` 생성 (held-out 평가로,
      `mean_projection`/`mean_gate_value`/`gate_activation_rate` 분리 기록)
- [ ] 실험 A: paired bootstrap 또는 McNemar test로 개입 전/후 유의성 검정
- [ ] 실험 B 실행 및 `exp_B_specificity.csv`, `exp_B_auroc.csv` 생성
      (주 비교는 ①b vs ④), 결과 (a)/(b) 중 어느 쪽인지 판정
- [ ] 실험 B: AUROC 95% bootstrap CI, 조건 간 projection 차이 bootstrap CI 보고
- [ ] (선택, A/B 이후) 실험 C 실행
- [ ] 포스터용 그림 2개 이상 생성 (예: 조건별 projection 분포 boxplot, 개입 방식별
      override_success_rate bar chart, 신뢰구간 포함)
- [ ] README.md에 실행 커맨드와 결과 요약 3~5줄 정리, 최종 메시지 문장이 실제
      결과로 뒷받침되는지 확인

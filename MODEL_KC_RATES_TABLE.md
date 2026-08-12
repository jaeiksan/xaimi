# 모델별 Knowledge Conflict 발생률 정리

기준: `results/kc_rates_*` 폴더의 `run_summary.json` 집계. `Query KC / clean-correct`는 clean prompt에서 정답을 맞힌 query 중 conflict context에서 replace option으로 뒤집힌 비율이다.

| Model | N | Clean acc | Query KC / all | Query KC / clean-correct | Trial KC / clean-correct | Misinformation | Temporal | Semantic | Mode | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `google/gemma-2-2b` | 1000 | 37.4% | 37.3% | 99.7% | 99.0% | 98.1% | 99.5% | 99.5% | plain |  |
| `google/gemma-2-2b-it` | 1000 | 13.4% | 10.1% | 75.4% | 61.7% | 59.7% | 59.7% | 65.7% | chat | clean acc 낮음 |
| `google/gemma-2-9b` | 1000 | 46.5% | 46.5% | 100.0% | 98.7% | 98.5% | 97.8% | 99.8% | plain |  |
| `google/gemma-2-9b-it` | 1000 | 19.9% | 16.2% | 81.4% | 67.8% | 67.8% | 61.8% | 73.9% | chat |  |
| `google/gemma-3-1b-it` | 1000 | 29.2% | 28.9% | 99.0% | 95.7% | 95.2% | 95.9% | 95.9% | chat |  |
| `google/gemma-3-4b-it` | 1000 | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | chat | A-collapse 의심 |
| `google/gemma-3-12b-it` | 1000 | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | chat | A-collapse 의심 |
| `Qwen/Qwen2.5-3B` | 1000 | 50.2% | 50.1% | 99.8% | 97.9% | 95.6% | 98.2% | 99.8% | plain |  |
| `Qwen/Qwen2.5-7B` | 1000 | 49.4% | 49.3% | 99.8% | 98.0% | 96.2% | 98.4% | 99.6% | plain |  |
| `Qwen/Qwen3-4B` | 1000 | 35.9% | 35.8% | 99.7% | 96.5% | 93.9% | 96.7% | 98.9% | chat |  |
| `Qwen/Qwen3-8B` | 1000 | 26.1% | 24.6% | 94.3% | 80.6% | 78.5% | 76.2% | 87.0% | chat | A 쏠림 약간 있음 |

## 해석 메모

- 발표/논문에서 “KC가 여러 모델에서 충분히 발생한다”는 근거로는 `Query KC / clean-correct`가 가장 적절하다. 모델이 원래 알고 맞힌 문제만 대상으로 context 때문에 답이 바뀌었는지를 보기 때문이다.
- `Query KC / all`은 clean accuracy까지 반영한 보수적 지표다. clean accuracy가 낮은 모델은 이 값이 낮아진다.
- Gemma 3 4B/12B IT는 clean prediction이 A로 붕괴한 평가 실패 가능성이 커서, “KC가 없다”는 결론으로 쓰면 안 된다.
- 전체적으로 정상 평가된 base/Qwen 계열에서는 clean-correct 조건부 KC가 대체로 94-100% 수준으로 매우 높다.

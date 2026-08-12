#!/usr/bin/env python3
"""
Step 1: Tuned Lens — KC에서 레이어별 answer preference 추적

각 레이어 linear probe를 학습하여 conflict run에서 transition layer를 탐지한다.

측정 위치:
  A — final token (모델이 직접 predict하는 위치, "Answer:" 마지막 토큰)
  B — \nAnswer: 직전 마지막 토큰 (선택지 다 읽은 직후)
  C — 문맥 블록 끝 토큰 (충돌 문맥 처리 직후, conflict prompt 전용)

Probe 학습 전략:
  pos A/B → neutral run residuals + neutral final lp4로 학습 → conflict에 적용
  pos C   → conflict run residuals + conflict final lp4로 학습 (별도 80/20 split)

실행:
  CUDA_VISIBLE_DEVICES=1 conda run -n apa python -u phase1/scripts/step1_tuned_lens.py
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer

# ── 경로 / 상수 ────────────────────────────────────────────────────────────────
KC_CASES_PATH = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
OUT_DIR       = Path("/home/jaeik/xaimi/phase1/results/step1_tuned_lens")
MODEL_NAME    = "google/gemma-2-2b"
SEED          = 42
MAX_SEQ_LEN   = 2048
TRAIN_RATIO   = 0.8
PROBE_EPOCHS  = 30
PROBE_LR      = 1e-3
BATCH_SIZE    = 64

LETTER_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate("ABCD")}
LETTER_VARIANTS: Dict[str, List[int]] = {}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 토큰 설정 ─────────────────────────────────────────────────────────────────
def setup_letter_variants(tokenizer) -> None:
    global LETTER_VARIANTS
    for letter in "ABCD":
        ids: List[int] = []
        for text in (f" {letter}", letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.append(toks[-1] if len(toks) > 1 else toks[0])
        LETTER_VARIANTS[letter] = list(dict.fromkeys(ids))
    print(f"[setup] LETTER_VARIANTS={LETTER_VARIANTS}")


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────
def build_clean_prompt(question: str, options: Dict[str, str]) -> str:
    lines = [f"Question: {question}"]
    for l in "ABCD":
        lines.append(f"{l}. {options[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_conflict_prompt(context: str, question: str, options: Dict[str, str]) -> str:
    lines = ["Context:", context, "", f"Question: {question}"]
    for l in "ABCD":
        lines.append(f"{l}. {options[l]}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── 토큰 위치 탐지 ────────────────────────────────────────────────────────────
def find_pos_B(model: HookedTransformer, prompt: str) -> int:
    """\nAnswer: 직전 마지막 토큰 인덱스."""
    idx = prompt.rfind("\nAnswer:")
    if idx == -1:
        return model.to_tokens(prompt, prepend_bos=True).shape[1] - 1
    prefix = prompt[:idx]
    return model.to_tokens(prefix, prepend_bos=True).shape[1] - 1


def find_pos_C(model: HookedTransformer, conflict_prompt: str, context: str) -> int:
    """문맥 블록 끝 토큰 인덱스 (Context:\n{context} prefix 마지막 토큰)."""
    prefix   = f"Context:\n{context}"
    n        = model.to_tokens(prefix, prepend_bos=True).shape[1]
    full_len = model.to_tokens(conflict_prompt, prepend_bos=True).shape[1]
    return min(n - 1, min(full_len, MAX_SEQ_LEN) - 1)


# ── Residual 수집 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_residuals(
    model: HookedTransformer,
    prompt: str,
    token_positions: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        resids: float16 [len(token_positions), n_layers, d_model]
        lp4:    float32 [4] — A/B/C/D log_probs from final output logits
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > MAX_SEQ_LEN:
        tokens = tokens[:, :MAX_SEQ_LEN]
    seq_len  = tokens.shape[1]
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model

    logits, cache = model.run_with_cache(
        tokens,
        names_filter=lambda n: n.endswith("hook_resid_post"),
        return_type="logits",
    )

    resids = torch.zeros(len(token_positions), n_layers, d_model, dtype=torch.float16)
    for li in range(n_layers):
        rp = cache[f"blocks.{li}.hook_resid_post"][0]  # [seq_len, d_model]
        for pi, pos in enumerate(token_positions):
            resids[pi, li] = rp[min(pos, seq_len - 1)].half()

    lp_all = torch.log_softmax(logits[0, -1].float(), dim=-1)
    lp4    = torch.tensor([
        max(lp_all[tid].item() for tid in LETTER_VARIANTS[l])
        for l in "ABCD"
    ])
    return resids, lp4


# ── Linear Probe ──────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, d_in: int = 2304, d_out: int = 4):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_one_probe(
    X: np.ndarray,  # [n_cases, d_model]
    y: np.ndarray,  # [n_cases, 4]
    device: str,
) -> Tuple[LinearProbe, float]:
    n       = X.shape[0]
    n_train = int(n * TRAIN_RATIO)
    idx     = list(range(n))
    random.shuffle(idx)

    Xt = torch.tensor(X[idx[:n_train]], dtype=torch.float32).to(device)
    Xv = torch.tensor(X[idx[n_train:]], dtype=torch.float32).to(device)
    yt = torch.tensor(y[idx[:n_train]], dtype=torch.float32).to(device)
    yv = torch.tensor(y[idx[n_train:]], dtype=torch.float32).to(device)

    probe   = LinearProbe(X.shape[1], 4).to(device)
    opt     = optim.Adam(probe.parameters(), lr=PROBE_LR)
    loss_fn = nn.MSELoss()

    for _ in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, BATCH_SIZE):
            bi   = perm[start:start + BATCH_SIZE]
            loss = loss_fn(probe(Xt[bi]), yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        val_mse = loss_fn(probe(Xv), yv).item()
    return probe, val_mse


def train_probes_for_position(
    all_resids: np.ndarray,  # [n_cases, n_layers, d_model]
    all_lp4:   np.ndarray,   # [n_cases, 4]
    device:    str,
    label:     str,
) -> Tuple[List[LinearProbe], List[float]]:
    n_layers = all_resids.shape[1]
    probes:   List[LinearProbe] = []
    val_mses: List[float]       = []
    for li in range(n_layers):
        probe, mse = train_one_probe(all_resids[:, li, :], all_lp4, device)
        probes.append(probe)
        val_mses.append(mse)
    print(f"[probes] pos={label} trained, mean_val_mse={np.mean(val_mses):.4f}")
    return probes, val_mses


# ── 추론 (벡터화) ─────────────────────────────────────────────────────────────
def apply_probes_vectorized(
    probes:     List[LinearProbe],  # n_layers
    all_resids: np.ndarray,          # [n_cases, n_layers, d_model]
    device:     str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        preds:   int32  [n_cases, n_layers] — argmax option index (0=A,…,3=D)
        logits:  float32[n_cases, n_layers, 4]
    """
    n_cases  = all_resids.shape[0]
    n_layers = len(probes)
    preds    = np.zeros((n_cases, n_layers), dtype=np.int32)
    logits   = np.zeros((n_cases, n_layers, 4), dtype=np.float32)

    for li, probe in enumerate(probes):
        X = torch.tensor(all_resids[:, li, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = probe(X).cpu().numpy()  # [n_cases, 4]
        logits[:, li, :] = out
        preds[:, li]     = out.argmax(axis=1)

    return preds, logits


def find_transition_layer(
    preds:        np.ndarray,  # [n_layers] int32
    target_idx:   int,
) -> Optional[int]:
    hits = np.where(preds == target_idx)[0]
    return int(hits[0]) if len(hits) > 0 else None


# ── 시각화 ────────────────────────────────────────────────────────────────────
_CTYPE_COLORS = {
    "misinformation": "#e74c3c",
    "temporal":       "#3498db",
    "semantic":       "#2ecc71",
}


def plot_transition_distribution(results: List[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, pos in zip(axes, "ABC"):
        for ctype, color in _CTYPE_COLORS.items():
            layers = [
                r["positions"][pos]["transition_layer"]
                for r in results
                if r["conflict_type"] == ctype
                   and r["positions"][pos]["transition_layer"] is not None
            ]
            if layers:
                ax.hist(layers, bins=range(0, 28), alpha=0.65, label=ctype,
                        color=color, edgecolor="white")
        ax.set_title(f"Position {pos}", fontsize=13)
        ax.set_xlabel("Transition Layer")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.5, 26.5)
    fig.suptitle("KC Transition Layer Distribution (Tuned Lens)", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


def plot_preference_trajectory(
    results:  List[Dict],
    n_layers: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, pos in zip(axes, "ABC"):
        for ctype, color in _CTYPE_COLORS.items():
            cases = [r for r in results if r["conflict_type"] == ctype]
            if not cases:
                continue
            mean_diffs = []
            for li in range(n_layers):
                diffs = []
                for r in cases:
                    traj = r["positions"][pos]["logit_trajectory"]
                    if li < len(traj):
                        ci = LETTER_TO_IDX[r["replace_option"]]
                        pi = LETTER_TO_IDX[r["correct_option"]]
                        diffs.append(traj[li][ci] - traj[li][pi])
                mean_diffs.append(float(np.mean(diffs)) if diffs else 0.0)
            ax.plot(range(n_layers), mean_diffs, color=color, label=ctype, linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Position {pos}", fontsize=13)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δlogit (conflict − correct)")
        ax.legend(fontsize=9)
    fig.suptitle("KC Preference Trajectory by Layer (Tuned Lens)", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    print("[1/8] KC 케이스 로드...")
    with KC_CASES_PATH.open() as f:
        cases = json.load(f)
    N = len(cases)
    print(f"      {N} cases")

    # 2. 모델 로드
    print("[2/8] TransformerLens 모델 로드...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device="cuda", dtype=torch.float16,
    )
    model.eval()
    setup_letter_variants(model.tokenizer)
    dev      = str(next(model.parameters()).device)
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    print(f"      d_model={d_model}, n_layers={n_layers}, device={dev}")

    # 3. Neutral run residuals 수집 (pos A, B)
    print(f"[3/8] Neutral run residuals 수집 ({N} cases)...")
    neutral_A   = np.zeros((N, n_layers, d_model), dtype=np.float16)
    neutral_B   = np.zeros((N, n_layers, d_model), dtype=np.float16)
    neutral_lp4 = np.zeros((N, 4), dtype=np.float32)

    t_ph = time.time()
    for i, case in enumerate(cases):
        prompt   = build_clean_prompt(case["question"], case["options"])
        pos_a    = model.to_tokens(prompt, prepend_bos=True).shape[1] - 1
        pos_b    = find_pos_B(model, prompt)
        uni_pos  = sorted(set([pos_a, pos_b]))
        resids, lp4 = collect_residuals(model, prompt, uni_pos)
        neutral_A[i]   = resids[uni_pos.index(pos_a)].numpy()
        neutral_B[i]   = resids[uni_pos.index(pos_b)].numpy()
        neutral_lp4[i] = lp4.numpy()
        if (i + 1) % 100 == 0:
            el = time.time() - t_ph
            eta = (N - i - 1) / ((i + 1) / el)
            print(f"      [{i+1}/{N}] {el:.0f}s elapsed, ETA={eta:.0f}s")
    print(f"      neutral 수집 완료 ({time.time()-t_ph:.0f}s)")

    # 4. Probe 학습 (pos A, B)
    print("[4/8] Probes 학습 (pos A, B)...")
    probes_A, vmse_A = train_probes_for_position(neutral_A, neutral_lp4, dev, "A")
    probes_B, vmse_B = train_probes_for_position(neutral_B, neutral_lp4, dev, "B")

    # 5. Conflict run residuals 수집 (pos A, B, C)
    print(f"[5/8] Conflict run residuals 수집 ({N} cases)...")
    conflict_A   = np.zeros((N, n_layers, d_model), dtype=np.float16)
    conflict_B   = np.zeros((N, n_layers, d_model), dtype=np.float16)
    conflict_C   = np.zeros((N, n_layers, d_model), dtype=np.float16)
    conflict_lp4 = np.zeros((N, 4), dtype=np.float32)

    t_ph = time.time()
    for i, case in enumerate(cases):
        ctx    = case["conflict_context"]
        prompt = build_conflict_prompt(ctx, case["question"], case["options"])
        pos_a  = model.to_tokens(prompt, prepend_bos=True).shape[1] - 1
        pos_b  = find_pos_B(model, prompt)
        pos_c  = find_pos_C(model, prompt, ctx)
        uni_pos = sorted(set([pos_a, pos_b, pos_c]))
        resids, lp4 = collect_residuals(model, prompt, uni_pos)
        conflict_A[i]   = resids[uni_pos.index(pos_a)].numpy()
        conflict_B[i]   = resids[uni_pos.index(pos_b)].numpy()
        conflict_C[i]   = resids[uni_pos.index(pos_c)].numpy()
        conflict_lp4[i] = lp4.numpy()
        if (i + 1) % 100 == 0:
            el = time.time() - t_ph
            eta = (N - i - 1) / ((i + 1) / el)
            print(f"      [{i+1}/{N}] {el:.0f}s elapsed, ETA={eta:.0f}s")
    print(f"      conflict 수집 완료 ({time.time()-t_ph:.0f}s)")

    # 6. Probe 학습 (pos C, conflict data)
    print("[6/8] Probes 학습 (pos C, conflict data)...")
    probes_C, vmse_C = train_probes_for_position(conflict_C, conflict_lp4, dev, "C")

    # 7. Apply probes → predictions (vectorized)
    print("[7/8] Probes 적용 (벡터화)...")
    preds_A, logits_A = apply_probes_vectorized(probes_A, conflict_A, dev)
    preds_B, logits_B = apply_probes_vectorized(probes_B, conflict_B, dev)
    preds_C, logits_C = apply_probes_vectorized(probes_C, conflict_C, dev)

    # 8. 결과 조합 및 저장
    print("[8/8] 결과 분석 및 저장...")
    results = []
    for i, case in enumerate(cases):
        rep_idx = LETTER_TO_IDX[case["replace_option"]]
        cor_idx = LETTER_TO_IDX[case["correct_option"]]
        results.append({
            "id":             case["id"],
            "conflict_type":  case["conflict_type"],
            "correct_option": case["correct_option"],
            "replace_option": case["replace_option"],
            "kc_severity":    case.get("kc_severity", 0.0),
            "positions": {
                pos: {
                    "transition_layer":   find_transition_layer(preds[i], rep_idx),
                    "predicted_sequence": [
                        "ABCD"[preds[i, li]] for li in range(n_layers)
                    ],
                    "logit_trajectory": logits[i].tolist(),
                }
                for pos, preds, logits in [
                    ("A", preds_A, logits_A),
                    ("B", preds_B, logits_B),
                    ("C", preds_C, logits_C),
                ]
            },
        })

    # 통계
    def pos_stats(pos: str) -> Dict:
        layers = [
            r["positions"][pos]["transition_layer"]
            for r in results
            if r["positions"][pos]["transition_layer"] is not None
        ]
        return {
            "n_with_transition":    len(layers),
            "n_without_transition": N - len(layers),
            "mean_layer":   float(np.mean(layers)) if layers else None,
            "median_layer": float(np.median(layers)) if layers else None,
            "std_layer":    float(np.std(layers)) if layers else None,
        }

    def ctype_pos_stats(ctype: str, pos: str) -> Dict:
        layers = [
            r["positions"][pos]["transition_layer"]
            for r in results
            if r["conflict_type"] == ctype
               and r["positions"][pos]["transition_layer"] is not None
        ]
        n_type = sum(1 for r in results if r["conflict_type"] == ctype)
        return {
            "n_with_transition": len(layers),
            "n_total":           n_type,
            "mean_layer":   float(np.mean(layers)) if layers else None,
            "median_layer": float(np.median(layers)) if layers else None,
        }

    stats = {
        "n_cases": N,
        "n_layers": n_layers,
        "by_position": {pos: pos_stats(pos) for pos in "ABC"},
        "by_type": {
            ctype: {pos: ctype_pos_stats(ctype, pos) for pos in "ABC"}
            for ctype in ("misinformation", "temporal", "semantic")
        },
        "probe_val_mse": {
            "A": {"mean": float(np.mean(vmse_A)), "per_layer": [float(v) for v in vmse_A]},
            "B": {"mean": float(np.mean(vmse_B)), "per_layer": [float(v) for v in vmse_B]},
            "C": {"mean": float(np.mean(vmse_C)), "per_layer": [float(v) for v in vmse_C]},
        },
    }

    # slim results (logit_trajectory 제외)
    results_slim = []
    for r in results:
        slim = {k: v for k, v in r.items() if k != "positions"}
        slim["positions"] = {
            pos: {
                "transition_layer":   r["positions"][pos]["transition_layer"],
                "predicted_sequence": r["positions"][pos]["predicted_sequence"],
            }
            for pos in "ABC"
        }
        results_slim.append(slim)

    with (OUT_DIR / "tuned_lens_results.json").open("w") as f:
        json.dump(results_slim, f, ensure_ascii=False, indent=2)
    with (OUT_DIR / "tuned_lens_stats.json").open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[save] {OUT_DIR}/tuned_lens_results.json, tuned_lens_stats.json")

    # 시각화
    plot_transition_distribution(results, OUT_DIR / "transition_layer_distribution.png")
    plot_preference_trajectory(results, n_layers, OUT_DIR / "preference_trajectory.png")

    # 최종 요약
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Step 1 완료 (총 {elapsed/60:.1f}분)")
    print(f"{'='*60}")
    for pos in "ABC":
        s = stats["by_position"][pos]
        pct = 100 * s["n_with_transition"] / N
        ml  = f"{s['mean_layer']:.1f}" if s["mean_layer"] is not None else "—"
        md  = f"{s['median_layer']:.1f}" if s["median_layer"] is not None else "—"
        print(f"  pos={pos}: transition={s['n_with_transition']}/{N} ({pct:.1f}%),  "
              f"mean_layer={ml},  median={md}")
    print(f"{'='*60}")
    for ctype in ("misinformation", "temporal", "semantic"):
        for pos in "ABC":
            s   = stats["by_type"][ctype][pos]
            pct = 100 * s["n_with_transition"] / max(s["n_total"], 1)
            ml  = f"{s['mean_layer']:.1f}" if s["mean_layer"] is not None else "—"
            print(f"  {ctype:15} pos={pos}: {s['n_with_transition']}/{s['n_total']} ({pct:.0f}%)  "
                  f"mean={ml}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

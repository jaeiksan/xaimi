#!/usr/bin/env python3
"""
Step 1 개별 케이스 분석 — Tuned Lens 결과 심층 분석

분석 1: Position A 전환 레이어 분포 히스토그램
분석 2: 이른/늦은 전환 그룹 극단 케이스 비교
분석 3: Position B no-transition 37개 케이스 분석
분석 4: severity vs 전환 레이어 상관관계

실행:
  conda run -n apa python -u phase1/scripts/step1_case_analysis.py
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

# ── 경로 ─────────────────────────────────────────────────────────────────────
RESULTS_PATH  = Path("/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_results.json")
KC_PATH       = Path("/home/jaeik/xaimi/data/phase0/kc_cases_1000.json")
OUT_DIR       = Path("/home/jaeik/xaimi/phase1/results/step1_tuned_lens/case_analysis")

EARLY_THRESH  = 3   # pos A transition_layer <= 이 값이면 이른 전환
LATE_THRESH   = 15  # pos A transition_layer >= 이 값이면 늦은 전환
N_LAYERS      = 26

CTYPE_COLORS  = {
    "misinformation": "#e74c3c",
    "temporal":       "#3498db",
    "semantic":       "#2ecc71",
}
CTYPES        = ("misinformation", "temporal", "semantic")


# ── 데이터 로드 및 병합 ───────────────────────────────────────────────────────
def load_data() -> List[Dict]:
    with RESULTS_PATH.open() as f:
        results = json.load(f)
    with KC_PATH.open() as f:
        kc_cases = json.load(f)

    kc_by_id = {c["id"]: c for c in kc_cases}
    merged = []
    for r in results:
        kc = kc_by_id.get(r["id"], {})
        merged.append({
            **r,
            "question":        kc.get("question", ""),
            "options":         kc.get("options", {}),
            "conflict_context": kc.get("conflict_context", ""),
            "correct_answer":  kc.get("correct_answer", ""),
            "conflict_answer": kc.get("conflict_answer", ""),
        })
    return merged


# ── 분석 1: 전환 레이어 분포 히스토그램 ──────────────────────────────────────
def analysis_1_distribution(cases: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("분석 1: Position A 전환 레이어 분포")
    print("=" * 60)

    # 레이어별 카운트 수집 (conflict type 구분)
    counts_by_type: Dict[str, List[int]] = {ct: [] for ct in CTYPES}
    for c in cases:
        tl = c["positions"]["A"]["transition_layer"]
        if tl is not None:
            counts_by_type[c["conflict_type"]].append(tl)

    # 콘솔 출력: 레이어별 케이스 수 + 누적 비율
    layer_total = [0] * N_LAYERS
    for layers in counts_by_type.values():
        for l in layers:
            layer_total[l] += 1

    total_with_trans = sum(layer_total)
    cumulative = 0
    for li in range(N_LAYERS):
        n = layer_total[li]
        pct = 100 * n / max(total_with_trans, 1)
        cumulative += pct
        bar = "█" * min(n // 3, 40)
        print(f"  레이어 {li:2d}: {n:3d}개 ({pct:4.1f}%) | 누적 {cumulative:5.1f}% {bar}")

    # 히스토그램 플롯
    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(N_LAYERS)
    for ct in CTYPES:
        hist = np.zeros(N_LAYERS, dtype=int)
        for l in counts_by_type[ct]:
            hist[l] += 1
        ax.bar(range(N_LAYERS), hist, bottom=bottoms,
               color=CTYPE_COLORS[ct], label=ct, alpha=0.85, edgecolor="white")
        bottoms += hist

    ax.set_xlabel("Transition Layer (Position A)", fontsize=12)
    ax.set_ylabel("Case Count", fontsize=12)
    ax.set_title("KC Transition Layer Distribution — Position A (Final Token)", fontsize=14)
    ax.set_xticks(range(N_LAYERS))
    ax.legend(fontsize=10)
    ax.axvline(x=np.mean([c["positions"]["A"]["transition_layer"]
                           for c in cases
                           if c["positions"]["A"]["transition_layer"] is not None]),
               color="black", linestyle="--", linewidth=1.5, label="mean")
    plt.tight_layout()
    out = OUT_DIR / "transition_distribution_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [저장] {out}")


# ── 분석 2: 극단 케이스 비교 ─────────────────────────────────────────────────
def analysis_2_extreme(cases: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print(f"분석 2: 극단 케이스 비교 (이른 ≤{EARLY_THRESH} vs 늦은 ≥{LATE_THRESH})")
    print("=" * 60)

    early = [c for c in cases
             if c["positions"]["A"]["transition_layer"] is not None
             and c["positions"]["A"]["transition_layer"] <= EARLY_THRESH]
    late  = [c for c in cases
             if c["positions"]["A"]["transition_layer"] is not None
             and c["positions"]["A"]["transition_layer"] >= LATE_THRESH]

    def group_stats(grp: List[Dict], label: str) -> Dict:
        severities = [c["kc_severity"] for c in grp]
        ctx_lens   = [len(c["conflict_context"]) for c in grp]
        ctype_dist = {ct: sum(1 for c in grp if c["conflict_type"] == ct) for ct in CTYPES}
        trans_lays = [c["positions"]["A"]["transition_layer"] for c in grp]

        print(f"\n  [{label}] n={len(grp)}")
        print(f"    평균 severity : {np.mean(severities):.3f}  (std={np.std(severities):.3f})")
        print(f"    평균 ctx 길이 : {np.mean(ctx_lens):.0f} chars")
        print(f"    conflict type : {ctype_dist}")
        print(f"    transition lay: mean={np.mean(trans_lays):.1f}, "
              f"range=[{min(trans_lays)}, {max(trans_lays)}]")

        reps = grp[:3]
        print(f"\n    대표 케이스 3개:")
        for i, c in enumerate(reps, 1):
            ctx_preview = c["conflict_context"][:120].replace("\n", " ")
            print(f"      [{i}] {c['id']} | layer={c['positions']['A']['transition_layer']} | "
                  f"sev={c['kc_severity']:.2f}")
            print(f"          Q: {c['question'][:80]}")
            print(f"          ctx: {ctx_preview}...")

        return {
            "label":        label,
            "n":            len(grp),
            "mean_severity": float(np.mean(severities)),
            "std_severity":  float(np.std(severities)),
            "mean_ctx_len":  float(np.mean(ctx_lens)),
            "ctype_dist":    ctype_dist,
            "mean_transition_layer": float(np.mean(trans_lays)),
            "representative_cases": [
                {
                    "id":               c["id"],
                    "conflict_type":    c["conflict_type"],
                    "transition_layer": c["positions"]["A"]["transition_layer"],
                    "kc_severity":      c["kc_severity"],
                    "question":         c["question"],
                    "conflict_context_preview": c["conflict_context"][:300],
                }
                for c in reps
            ],
        }

    early_stats = group_stats(early, f"이른 전환 (layer ≤{EARLY_THRESH})")
    late_stats  = group_stats(late,  f"늦은 전환 (layer ≥{LATE_THRESH})")

    print(f"\n  === 핵심 차이 ===")
    sev_diff = early_stats["mean_severity"] - late_stats["mean_severity"]
    len_diff = early_stats["mean_ctx_len"] - late_stats["mean_ctx_len"]
    print(f"    severity 차이 (이른 - 늦은): {sev_diff:+.3f}")
    print(f"    ctx 길이 차이  (이른 - 늦은): {len_diff:+.0f} chars")

    out = OUT_DIR / "extreme_cases_comparison.json"
    with out.open("w") as f:
        json.dump({"early": early_stats, "late": late_stats}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n  [저장] {out}")


# ── 분석 3: no-transition 케이스 분석 ────────────────────────────────────────
def analysis_3_no_transition(cases: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("분석 3: Position B no-transition 케이스 분석")
    print("=" * 60)

    no_trans = [c for c in cases if c["positions"]["B"]["transition_layer"] is None]
    yes_trans = [c for c in cases if c["positions"]["B"]["transition_layer"] is not None]

    print(f"  no-transition: {len(no_trans)}개 / yes-transition: {len(yes_trans)}개")

    def sev_stats(grp: List[Dict]) -> Tuple[float, float]:
        sevs = [c["kc_severity"] for c in grp]
        return float(np.mean(sevs)), float(np.median(sevs))

    sev_no_m, sev_no_med   = sev_stats(no_trans)
    sev_yes_m, sev_yes_med = sev_stats(yes_trans)

    print(f"\n  severity 비교:")
    print(f"    no-transition  : mean={sev_no_m:.3f}, median={sev_no_med:.3f}")
    print(f"    yes-transition : mean={sev_yes_m:.3f}, median={sev_yes_med:.3f}")

    ctype_dist = {ct: sum(1 for c in no_trans if c["conflict_type"] == ct) for ct in CTYPES}
    ctx_lens   = [len(c["conflict_context"]) for c in no_trans]
    print(f"\n  conflict type 분포: {ctype_dist}")
    print(f"  ctx 길이: mean={np.mean(ctx_lens):.0f}, "
          f"median={np.median(ctx_lens):.0f}, "
          f"min={min(ctx_lens)}, max={max(ctx_lens)}")

    print(f"\n  대표 케이스 5개 (pos_B no-transition):")
    for i, c in enumerate(no_trans[:5], 1):
        ctx_preview = c["conflict_context"][:150].replace("\n", " ")
        pos_a_tl = c["positions"]["A"]["transition_layer"]
        print(f"    [{i}] {c['id']} | sev={c['kc_severity']:.2f} | "
              f"posA_layer={pos_a_tl} | type={c['conflict_type']}")
        print(f"         Q: {c['question'][:80]}")
        print(f"         ctx: {ctx_preview}...")

    out = OUT_DIR / "no_transition_cases.json"
    with out.open("w") as f:
        json.dump({
            "n_no_transition":  len(no_trans),
            "n_yes_transition": len(yes_trans),
            "no_transition": {
                "mean_severity":   sev_no_m,
                "median_severity": sev_no_med,
                "ctype_dist":      ctype_dist,
                "mean_ctx_len":    float(np.mean(ctx_lens)),
                "median_ctx_len":  float(np.median(ctx_lens)),
                "cases": [
                    {
                        "id":               c["id"],
                        "conflict_type":    c["conflict_type"],
                        "kc_severity":      c["kc_severity"],
                        "pos_A_transition": c["positions"]["A"]["transition_layer"],
                        "pos_C_transition": c["positions"]["C"]["transition_layer"],
                        "ctx_len":          len(c["conflict_context"]),
                        "question":         c["question"],
                        "conflict_context_preview": c["conflict_context"][:400],
                    }
                    for c in no_trans
                ],
            },
            "yes_transition": {
                "mean_severity":   sev_yes_m,
                "median_severity": sev_yes_med,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  [저장] {out}")


# ── 분석 4: severity vs transition layer 상관관계 ────────────────────────────
def analysis_4_correlation(cases: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("분석 4: severity vs transition layer 상관관계 (Position A)")
    print("=" * 60)

    all_tl  = np.array([c["positions"]["A"]["transition_layer"]
                         for c in cases
                         if c["positions"]["A"]["transition_layer"] is not None],
                        dtype=float)
    all_sev = np.array([c["kc_severity"]
                         for c in cases
                         if c["positions"]["A"]["transition_layer"] is not None])

    r_all, p_all = scipy_stats.pearsonr(all_tl, all_sev)
    print(f"\n  전체 상관계수: r = {r_all:+.4f}  (p = {p_all:.4f})")

    # 그룹별 상관계수
    fig, ax = plt.subplots(figsize=(9, 6))
    jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(all_tl))

    offset = 0
    for ct in CTYPES:
        grp = [(c["positions"]["A"]["transition_layer"], c["kc_severity"])
               for c in cases
               if c["conflict_type"] == ct
               and c["positions"]["A"]["transition_layer"] is not None]
        tl_g  = np.array([g[0] for g in grp], dtype=float)
        sev_g = np.array([g[1] for g in grp])
        r_g, p_g = scipy_stats.pearsonr(tl_g, sev_g)
        print(f"  {ct:15}: r = {r_g:+.4f}  (p = {p_g:.4f}, n={len(grp)})")

        ax.scatter(tl_g + jitter[offset:offset + len(tl_g)], sev_g,
                   color=CTYPE_COLORS[ct], alpha=0.45, s=18, label=ct)
        # 회귀선
        m, b = np.polyfit(tl_g, sev_g, 1)
        xs = np.linspace(tl_g.min(), tl_g.max(), 80)
        ax.plot(xs, m * xs + b, color=CTYPE_COLORS[ct], linewidth=1.8, alpha=0.8)
        offset += len(tl_g)

    # 전체 회귀선
    m_all, b_all = np.polyfit(all_tl, all_sev, 1)
    xs_all = np.linspace(all_tl.min(), all_tl.max(), 100)
    ax.plot(xs_all, m_all * xs_all + b_all, color="black", linewidth=2.0,
            linestyle="--", label=f"all  r={r_all:+.3f}")

    ax.set_xlabel("Transition Layer (Position A)", fontsize=12)
    ax.set_ylabel("KC Severity  (conflict_lp − correct_lp)", fontsize=12)
    ax.set_title("Severity vs Transition Layer — Pearson Correlation", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / "severity_vs_transition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [저장] {out}")

    return r_all, p_all


# ── 최종 해석 요약 ────────────────────────────────────────────────────────────
def print_summary(cases: List[Dict], r_all: float, p_all: float) -> None:
    print("\n" + "=" * 60)
    print("최종 해석 요약")
    print("=" * 60)

    # 분포 모양 분석
    all_tl = [c["positions"]["A"]["transition_layer"]
              for c in cases
              if c["positions"]["A"]["transition_layer"] is not None]
    q25, q50, q75 = np.percentile(all_tl, [25, 50, 75])
    iqr = q75 - q25
    early_cnt = sum(1 for t in all_tl if t <= EARLY_THRESH)
    late_cnt  = sum(1 for t in all_tl if t >= LATE_THRESH)
    mid_cnt   = len(all_tl) - early_cnt - late_cnt

    print(f"\n  [분포 모양]")
    print(f"    Q25={q25:.0f} / Q50={q50:.0f} / Q75={q75:.0f} / IQR={iqr:.0f}")
    print(f"    이른(≤{EARLY_THRESH}): {early_cnt}개 ({100*early_cnt/len(all_tl):.1f}%)")
    print(f"    중간({EARLY_THRESH+1}~{LATE_THRESH-1}): {mid_cnt}개 ({100*mid_cnt/len(all_tl):.1f}%)")
    print(f"    늦은(≥{LATE_THRESH}): {late_cnt}개 ({100*late_cnt/len(all_tl):.1f}%)")
    if q75 - q25 < 8:
        print("    → 분포가 집중되어 있음 (KC 전환이 특정 레이어 구간에 국소화)")
    else:
        print("    → 분포가 넓게 퍼져 있음 (KC 전환 레이어가 케이스마다 다양)")

    # 이른/늦은 그룹 핵심 차이
    early_sev = np.mean([c["kc_severity"] for c in cases
                         if c["positions"]["A"]["transition_layer"] is not None
                         and c["positions"]["A"]["transition_layer"] <= EARLY_THRESH])
    late_sev  = np.mean([c["kc_severity"] for c in cases
                         if c["positions"]["A"]["transition_layer"] is not None
                         and c["positions"]["A"]["transition_layer"] >= LATE_THRESH])
    print(f"\n  [이른 vs 늦은 전환 핵심 차이]")
    print(f"    severity: 이른={early_sev:.3f}, 늦은={late_sev:.3f}")
    if early_sev > late_sev:
        print("    → 이른 전환 케이스의 severity가 더 높음 "
              "(충돌이 강할수록 초기 레이어에서 preference 전환)")
    else:
        print("    → 늦은 전환 케이스의 severity가 더 높음 "
              "(충돌이 강할수록 더 많은 레이어 처리 후 전환)")

    # no-transition 특징
    no_trans_sev = np.mean([c["kc_severity"] for c in cases
                            if c["positions"]["B"]["transition_layer"] is None])
    yes_trans_sev = np.mean([c["kc_severity"] for c in cases
                             if c["positions"]["B"]["transition_layer"] is not None])
    print(f"\n  [no-transition 특징 (pos B)]")
    print(f"    평균 severity: no-trans={no_trans_sev:.3f}, yes-trans={yes_trans_sev:.3f}")
    if no_trans_sev < yes_trans_sev:
        print("    → no-transition 케이스는 severity가 낮음 "
              "(약한 KC는 pos B에서 preference 전환이 불완전)")
    else:
        print("    → no-transition 케이스도 severity는 유사/높음 "
              "(pos B에서 미전환이 severity와 무관할 수 있음)")

    # severity 상관관계
    print(f"\n  [severity ↔ transition layer 관계]")
    print(f"    Pearson r = {r_all:+.4f}  (p = {p_all:.4f})")
    if abs(r_all) < 0.1:
        strength = "거의 없음"
    elif abs(r_all) < 0.3:
        strength = "약함"
    elif abs(r_all) < 0.5:
        strength = "중간"
    else:
        strength = "강함"
    direction = "음의 상관 (severity 높을수록 이른 레이어에서 전환)" if r_all < 0 \
        else "양의 상관 (severity 높을수록 늦은 레이어에서 전환)"
    sig = "유의미함 (p<0.05)" if p_all < 0.05 else "유의미하지 않음 (p≥0.05)"
    print(f"    방향: {direction}")
    print(f"    강도: {strength}")
    print(f"    통계: {sig}")

    print("\n" + "=" * 60)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] 데이터 로드 및 병합...")
    cases = load_data()
    print(f"       {len(cases)} cases loaded")

    analysis_1_distribution(cases)
    analysis_2_extreme(cases)
    analysis_3_no_transition(cases)
    r_all, p_all = analysis_4_correlation(cases)
    print_summary(cases, r_all, p_all)

    print(f"\n[완료] 결과 저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()

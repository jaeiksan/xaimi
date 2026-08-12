#!/usr/bin/env python3
"""
Phase 4 Exp3 v2 시각화
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "phase4/results/exp3_comparison_v2"
FIG_DIR = OUT_DIR / "figures"
VEC_DIR = ROOT / "phase4/vectors"
DATA_DIR = ROOT / "phase4/data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "phase4/scripts"))
from eval_utils import CANDIDATE_LAYERS

METHOD_ORDER = [
    "no_intervention",
    "ungated_upk",
    "shuffled_B_gated_upk",
    "B_gated_upk",
    "B_gated_contrast",
    "B_gated_attn_rw_sentence",
    "B_gated_attn_rd_sentence",
    "B_gated_attn_dynamic_rw",
    "B_gated_attn_dynamic_rd",
    "B_gated_attn_multilayer",
    "B_gated_mlp_weight",
    "B_gated_mlp_activation",
]

ACCURACY_METHOD_ORDER = [
    "B_gated_upk",
    "B_gated_contrast",
    "B_gated_attn_dynamic_rw",
    "B_gated_attn_rd_sentence",
    "B_gated_mlp_activation",
]

METHOD_LABELS = {
    "no_intervention": "No\nInterv",
    "ungated_upk": "Ungated\nu_PK",
    "shuffled_B_gated_upk": "Shuffled\nGate",
    "B_gated_upk": "B-gated\nResidual",
    "B_gated_contrast": "B-gated\nContrast",
    "B_gated_attn_rw_sentence": "Attention\nReweight",
    "B_gated_attn_rd_sentence": "Attention\nRedirect",
    "B_gated_attn_dynamic_rw": "Attention\nReweight",
    "B_gated_attn_dynamic_rd": "DynamicRD",
    "B_gated_attn_multilayer": "MultiLayer",
    "B_gated_mlp_weight": "MLP\nWeight",
    "B_gated_mlp_activation": "MLP\nAct",
}

METHOD_COLORS = {
    "no_intervention": "#aaaaaa",
    "ungated_upk": "#4e79a7",
    "shuffled_B_gated_upk": "#f28e2b",
    "B_gated_upk": "#e15759",
    "B_gated_contrast": "#76b7b2",
    "B_gated_attn_rw_sentence": "#59a14f",
    "B_gated_attn_rd_sentence": "#3a7d2a",
    "B_gated_attn_dynamic_rw": "#8cd17d",
    "B_gated_attn_dynamic_rd": "#b6992d",
    "B_gated_attn_multilayer": "#edc948",
    "B_gated_mlp_weight": "#af7aa1",
    "B_gated_mlp_activation": "#ff9da7",
}


def load_results():
    best_path = OUT_DIR / "best_configs_7plus2.json"
    if not best_path.exists():
        best_path = OUT_DIR / "best_configs.json"
    with open(best_path) as f:
        best = json.load(f)
    with open(OUT_DIR / "all_results.json") as f:
        all_r = json.load(f)
    return best, all_r


def get_val(entry, key, default=0.0):
    return entry.get(key, default)


def load_conflict_signal_strength():
    residuals = torch.load(VEC_DIR / "residuals.pt", map_location="cpu", weights_only=False)
    dir_B = torch.load(VEC_DIR / "direction_B_train.pt", map_location="cpu", weights_only=False)
    train_cases = json.loads((DATA_DIR / "train_800.json").read_text())
    n_train = len(train_cases)
    train_ids = list(range(n_train))

    signal = {}
    for layer in sorted(dir_B.keys()):
        d = dir_B[layer].float()
        d = d / d.norm()
        h_conf = residuals["conflict"][layer][train_ids].float()
        h_sup = residuals["supportive"][layer][train_ids].float()
        proj_conf = h_conf @ d
        proj_sup = h_sup @ d
        signal[layer] = float(proj_conf.mean() - proj_sup.mean())
    return signal


def conflict_clean_gain(entry):
    return get_val(entry, "conflict_acc_gain") + get_val(entry, "clean_acc_gain")


def plot_conflict_clean_gain(best):
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    vals = [conflict_clean_gain(best.get(m, {})) for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    bars = ax.bar(labels, vals, color=colors, alpha=0.9, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    for idx in (2.5, 4.5, 9.5):
        ax.axvline(idx, color="gray", lw=1, linestyle="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:+.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Conflict-Clean Gain\n(conflict_acc_gain + clean_acc_gain)")
    ax.set_title("Exp3 v2: Conflict-Clean Gain by Method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_conflict_clean_gain.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_selective_gain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(best):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = [METHOD_LABELS[m] for m in ACCURACY_METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in ACCURACY_METHOD_ORDER]
    baseline_entry = best.get("no_intervention", {})

    for ax, key_after, key_before, title in [
        (axes[0], "conflict_acc_after", "conflict_acc_before", "Conflict Accuracy"),
        (axes[1], "clean_acc_after", "clean_acc_before", "Clean Accuracy"),
    ]:
        vals = [get_val(best.get(m, {}), key_after, get_val(best.get(m, {}), key_before)) for m in ACCURACY_METHOD_ORDER]
        bars = ax.bar(labels, vals, color=colors, alpha=0.9, edgecolor="white")
        ax.axhline(get_val(baseline_entry, key_before), color="#aaaaaa", lw=1.5, linestyle="--", alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=7)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.015,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_logit_margin(best):
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    vals = [get_val(best.get(m, {}), "conflict_margin_delta") for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    bars = ax.bar(labels, vals, color=colors, alpha=0.9, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:+.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Conflict Margin Delta")
    ax.set_title("Exp3 v2: Conflict Logit Margin Change")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_logit_margin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_summary(best):
    methods = [
        "B_gated_attn_rw_sentence",
        "B_gated_attn_rd_sentence",
        "B_gated_attn_dynamic_rw",
    ]
    labels = [
        "Attention\nReweight\n(Static)",
        "Attention\nRedirect\n(Static)",
        "Attention\nReweight\n(Dynamic)",
    ]
    colors = [METHOD_COLORS[m] for m in methods]

    conflict_acc = [get_val(best.get(m, {}), "conflict_acc_after") for m in methods]
    conflict_margin = [get_val(best.get(m, {}), "conflict_margin_delta") for m in methods]
    clean_acc = [get_val(best.get(m, {}), "clean_acc_after") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    bars = axes[0].bar(labels, conflict_acc, color=colors, alpha=0.9, edgecolor="white")
    axes[0].set_ylim(0, 0.12)
    axes[0].set_title("Conflict Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, conflict_acc):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.004,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    x = np.arange(len(methods))
    bars = axes[1].bar(x, conflict_margin, color=colors, alpha=0.9, edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_title("Conflict Margin Delta")
    axes[1].set_ylabel("Delta")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, v, ca in zip(bars, conflict_margin, clean_acc):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.04,
            f"{v:+.2f}\n(clean {ca:.3f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Attention Methods: Score Movement vs Answer Flips", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_attention_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_vs_residual_mlp(best):
    methods = [
        "B_gated_upk",
        "B_gated_attn_dynamic_rw",
        "B_gated_mlp_activation",
    ]
    labels = [
        "B-gated\nResidual",
        "Attention\nReweight",
        "MLP\nActivation",
    ]
    colors = [METHOD_COLORS[m] for m in methods]
    conflict_margin = [get_val(best.get(m, {}), "conflict_margin_delta") for m in methods]
    conflict_acc = [get_val(best.get(m, {}), "conflict_acc_after") for m in methods]
    clean_acc = [get_val(best.get(m, {}), "clean_acc_after") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    bars = axes[0].bar(labels, conflict_margin, color=colors, alpha=0.9, edgecolor="white")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Conflict Logit Margin Delta")
    axes[0].set_ylabel("Delta")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, conflict_margin):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.08,
            f"{v:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bars = axes[1].bar(labels, conflict_acc, color=colors, alpha=0.9, edgecolor="white")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_title("Conflict Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", labelsize=8)
    for bar, v, ca in zip(bars, conflict_acc, clean_acc):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.3f}\n(clean {ca:.3f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Dynamic Attention vs Residual / MLP", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_attention_vs_residual_mlp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gate_comparison(best):
    labels = ["Ungated", "B-gated"]
    conflict_vals = [
        get_val(best.get("ungated_upk", {}), "conflict_acc_after"),
        get_val(best.get("B_gated_upk", {}), "conflict_acc_after"),
    ]
    clean_vals = [
        get_val(best.get("ungated_upk", {}), "clean_acc_after"),
        get_val(best.get("B_gated_upk", {}), "clean_acc_after"),
    ]
    colors = ["#4e79a7", "#e15759"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8))

    for ax, vals, title in [
        (axes[0], conflict_vals, "Conflict"),
        (axes[1], clean_vals, "Clean"),
    ]:
        bars = ax.bar(labels, vals, color=colors, alpha=0.9, edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gate_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_layer_profile(all_r):
    layers = list(range(14, 23))
    records = [r for r in all_r if r.get("method") == "B_gated_upk" and "layer" in r]
    signal = load_conflict_signal_strength()
    best_by_layer = []
    for layer in layers:
        layer_recs = [r for r in records if r.get("layer") == layer]
        if not layer_recs:
            best_by_layer.append(None)
            continue
        best = max(
            layer_recs,
            key=lambda r: get_val(r, "conflict_acc_gain") + get_val(r, "clean_acc_gain"),
        )
        best_by_layer.append(best)

    conflict = [get_val(r, "conflict_acc_after") if r else np.nan for r in best_by_layer]
    signal_vals = [signal.get(layer, np.nan) for layer in layers]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax2 = ax.twinx()
    ax.plot(layers, conflict, marker="o", color=METHOD_COLORS["B_gated_upk"], label="Accuracy")
    ax2.plot(layers, signal_vals, marker="^", color="#2f4b7c", label="Conflict signal strength")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy", color=METHOD_COLORS["B_gated_upk"])
    ax2.set_ylabel("Conflict signal strength", color="#2f4b7c")
    ax.set_ylim(0.4, 0.75)
    ax.grid(alpha=0.3)
    ax.set_title("Accuracy vs Signal Strength by Layer")
    ax.tick_params(axis="y", colors=METHOD_COLORS["B_gated_upk"])
    ax2.tick_params(axis="y", colors="#2f4b7c")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.12),
        fontsize=8,
    )
    for layer in [14, 17, 20]:
        idx = layers.index(layer)
        if layer == 20:
            ax.text(layer, conflict[idx] - 0.02, f"{conflict[idx]:.3f}", ha="center", va="top", fontsize=8)
        else:
            ax.text(layer, conflict[idx] + 0.015, f"{conflict[idx]:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.text(layer, signal_vals[idx] + 0.2, f"{signal_vals[idx]:.1f}", ha="center", va="bottom", fontsize=8, color="#2f4b7c")

    fig.suptitle("Accuracy and Signal Strength by Layer", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_residual_layer_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_sweep(all_r):
    methods = [m for m in METHOD_ORDER if m.startswith("B_gated_") and "multilayer" not in m]
    n_cols = 4
    n_rows = (len(methods) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes_flat = np.array(axes).reshape(-1)

    for ax_i, method in enumerate(methods):
        ax = axes_flat[ax_i]
        records = [r for r in all_r if r.get("method") == method and "layer" in r]
        gains = []
        layers = sorted({r["layer"] for r in records}) or CANDIDATE_LAYERS
        for layer in layers:
            layer_recs = [r for r in records if r.get("layer") == layer]
            gain = max((conflict_clean_gain(r) for r in layer_recs), default=0.0)
            gains.append(gain)
        xpos = np.arange(len(layers))
        ax.bar(xpos, gains, color=METHOD_COLORS[method], alpha=0.9)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=7)
        ax.set_title(METHOD_LABELS[method].replace("\n", " "), fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(methods), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_layer_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attn_dynamic_vs_static(all_r):
    fig, ax = plt.subplots(figsize=(8, 5))
    attn_methods = [
        ("B_gated_attn_rw_sentence", "Static Sentence"),
        ("B_gated_attn_dynamic_rw", "Dynamic Top-k"),
        ("B_gated_attn_multilayer", "Multilayer"),
    ]
    layers = list(range(10, 23))

    for method, label in attn_methods:
        gains = []
        for layer in layers:
            if method == "B_gated_attn_multilayer":
                recs = [
                    r for r in all_r
                    if r.get("method") == method and layer in r.get("layer_list", [])
                ]
            else:
                recs = [r for r in all_r if r.get("method") == method and r.get("layer") == layer]
            gains.append(max((conflict_clean_gain(r) for r in recs), default=np.nan))
        ax.plot(layers, gains, marker="o", label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Conflict-Clean Gain")
    ax.set_title("Attention: Static vs Dynamic vs Multilayer")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_attn_dynamic_vs_static.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_weight_vs_activation(all_r):
    fig, ax = plt.subplots(figsize=(8, 5))
    top_ks = [10, 20, 50, 100]
    for method, label in [
        ("B_gated_mlp_weight", "Weight-based"),
        ("B_gated_mlp_activation", "Activation-based"),
    ]:
        gains = []
        for top_k in top_ks:
            recs = [r for r in all_r if r.get("method") == method and r.get("top_k_neurons") == top_k]
            gains.append(max((conflict_clean_gain(r) for r in recs), default=np.nan))
        ax.plot(top_ks, gains, marker="o", label=label)

    ax.set_xlabel("Top-k neurons")
    ax.set_ylabel("Conflict-Clean Gain")
    ax.set_title("MLP: Weight vs Activation Selection")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mlp_weight_vs_activation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    best, all_r = load_results()
    plot_conflict_clean_gain(best)
    plot_accuracy_comparison(best)
    plot_logit_margin(best)
    plot_attention_summary(best)
    plot_attention_vs_residual_mlp(best)
    plot_gate_comparison(best)
    plot_residual_layer_profile(all_r)
    plot_layer_sweep(all_r)
    plot_attn_dynamic_vs_static(all_r)
    plot_mlp_weight_vs_activation(all_r)
    print(f"All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

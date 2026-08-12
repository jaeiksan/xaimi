from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def normalize(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def main() -> None:
    base = Path("/home/jaeik/xaimi/results")
    assets = base / "report_assets"
    assets.mkdir(exist_ok=True)

    with (base / "exp_a_500/exp_a_top_k_layers.json").open() as f:
        exp_a = json.load(f)
    with (base / "exp_b_500/exp_b_stats.json").open() as f:
        exp_b = json.load(f)
    with (base / "exp_c_500/exp_c_top_k_layers.json").open() as f:
        exp_c = json.load(f)

    layers = list(range(exp_a["n_layers"]))

    # Experiment C overall mean AIE.
    mean_aie = exp_c["overall"]["mean_layer_aie"]
    plt.figure(figsize=(8, 4.5))
    plt.plot(layers, mean_aie, marker="o", linewidth=2, color="#b23a48")
    plt.axvspan(20, 25, color="#f2c9cf", alpha=0.35)
    plt.title("Experiment C: Mean AIE by Layer (500 cases)")
    plt.xlabel("Layer")
    plt.ylabel("Mean AIE")
    plt.xticks(layers)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(assets / "exp_c_mean_layer_aie_overall.png", dpi=180)
    plt.close()

    # A/B/C comparison on normalized scales.
    a_vec = exp_a["overall"]["mean_layer_scores"]
    c_vec = exp_c["overall"]["mean_layer_aie"]
    b_hist = [0] * exp_a["n_layers"]
    for layer in exp_b["transition_layers"]:
        if layer is None:
            continue
        idx = int(layer)
        if 0 <= idx < len(b_hist):
            b_hist[idx] += 1

    plt.figure(figsize=(9, 5))
    plt.plot(
        layers,
        normalize(a_vec),
        marker="o",
        linewidth=2,
        label="A: attention attribution",
        color="#355070",
    )
    plt.plot(
        layers,
        normalize(b_hist),
        marker="s",
        linewidth=2,
        label="B: transition frequency",
        color="#6d597a",
    )
    plt.plot(
        layers,
        normalize(c_vec),
        marker="^",
        linewidth=2,
        label="C: activation patching AIE",
        color="#b56576",
    )
    plt.axvspan(0, 2, color="#d8e2dc", alpha=0.5)
    plt.axvspan(5, 7, color="#cfe1f2", alpha=0.4)
    plt.axvspan(20, 25, color="#f2d0d7", alpha=0.35)
    plt.title("A/B/C Layer Pattern Comparison (normalized, 500 cases)")
    plt.xlabel("Layer")
    plt.ylabel("Normalized score / frequency")
    plt.xticks(layers)
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(assets / "abc_layer_pattern_comparison.png", dpi=180)
    plt.close()

    print(assets / "exp_c_mean_layer_aie_overall.png")
    print(assets / "abc_layer_pattern_comparison.png")


if __name__ == "__main__":
    main()

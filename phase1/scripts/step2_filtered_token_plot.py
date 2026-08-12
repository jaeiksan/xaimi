#!/usr/bin/env python3
"""
Make a cleaner token-level attribution plot from Step 2 IG outputs.

This script aggregates per-case `top_tokens` from ig_results.json, filters out
special tokens / punctuation / common stopwords / obviously broken tokens, and
renders a presentation-friendly bar chart for a single conflict type.

Default target: all saved cases
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_PATH = Path("/home/jaeik/xaimi/phase1/results/step2_ig/ig_results.json")
OUT_DIR = Path("/home/jaeik/xaimi/phase1/results/step2_ig")

TYPE_COLORS = {
    "all": "#34495e",
    "misinformation": "#e74c3c",
    "temporal": "#3498db",
    "semantic": "#2ecc71",
}

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with",
    "from", "is", "was", "are", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "he", "she", "they", "them", "his",
    "her", "their", "and", "or", "but", "as", "into", "over", "under",
    "after", "before", "during", "through", "between", "among", "about",
    "around", "because", "while", "where", "when", "which", "who", "whom",
    "what", "how", "why", "can", "could", "would", "should", "may", "might",
    "will", "shall", "do", "does", "did", "done", "have", "has", "had",
    "i", "we", "you", "our", "your", "my", "me", "us",
    "context", "question", "answer", "references", "reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        default="all",
        choices=["all", "misinformation", "temporal", "semantic"],
        help="Conflict type to visualize. Use 'all' to aggregate every saved case.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=12,
        help="Number of filtered tokens to plot.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Keep only tokens that appear at least this many times.",
    )
    return parser.parse_args()


def normalize_token(token: str) -> str:
    token = token.replace("\n", " ").strip()
    token = re.sub(r"\s+", " ", token)
    return token


def keep_token(token: str) -> bool:
    tok = normalize_token(token)
    if not tok:
        return False
    if tok.startswith("<") and tok.endswith(">"):
        return False
    if tok.lower() in STOPWORDS:
        return False
    if re.fullmatch(r"[\W_]+", tok):
        return False
    if re.fullmatch(r"\d+", tok):
        return False
    # Keep mostly readable alphabetic tokens and short title-like tokens.
    if not re.search(r"[A-Za-z]", tok):
        return False
    # Exclude obviously broken tokenizer fragments with non-ascii chars.
    if any(ord(ch) > 127 for ch in tok):
        return False
    if len(tok) <= 2:
        return False
    return True


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = json.loads(RESULTS_PATH.read_text())
    subset = (
        results
        if args.type == "all"
        else [row for row in results if row["conflict_type"] == args.type]
    )
    if not subset:
        raise SystemExit(f"No rows found for type={args.type}")

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    groups: dict[str, str] = {}

    for row in subset:
        for item in row.get("top_tokens", []):
            tok = normalize_token(item["token"])
            if not keep_token(tok):
                continue
            sums[tok] += float(item["attribution"])
            counts[tok] += 1
            groups[tok] = item.get("group", "unknown")

    ranked = sorted(
        (
            {
                "token": tok,
                "total_attr": sums[tok],
                "count": counts[tok],
                "mean_attr": sums[tok] / counts[tok],
                "group": groups[tok],
            }
            for tok in sums
            if counts[tok] >= args.min_count
        ),
        key=lambda x: (x["total_attr"], x["mean_attr"]),
        reverse=True,
    )[:args.top_n]

    if not ranked:
        raise SystemExit("No tokens survived filtering.")

    tokens = [r["token"] for r in ranked][::-1]
    values = [r["total_attr"] for r in ranked][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(tokens, values, color=TYPE_COLORS[args.type], alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_title("Top Token Attributions", fontsize=13)
    ax.set_xlabel("Aggregated attribution over saved top-token lists")
    ax.set_ylabel("Token")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out_png = OUT_DIR / f"filtered_top_tokens_{args.type}.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    out_json = OUT_DIR / f"filtered_top_tokens_{args.type}.json"
    out_json.write_text(json.dumps(ranked, indent=2))

    print(f"saved plot: {out_png}")
    print(f"saved data: {out_json}")


if __name__ == "__main__":
    main()

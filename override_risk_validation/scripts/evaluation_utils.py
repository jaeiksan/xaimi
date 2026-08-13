"""Local evaluation helpers for the override-risk validation experiments.

These wrappers deliberately reuse XAIMI's established letter-token handling,
but add full-choice prediction rather than comparing only correct/replacement
options.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from exp_utils import get_letter_lp  # noqa: E402


@torch.no_grad()
def option_logprobs(model, prompt: str, options: Mapping[str, str]) -> dict[str, float]:
    """Return each offered option's best letter-token log probability."""
    ids = model.to_tokens(prompt, prepend_bos=True)
    logits = model(ids)[0, -1].float()
    return option_logprobs_from_logits(logits, options)


def option_logprobs_from_logits(logits: torch.Tensor, options: Mapping[str, str]) -> dict[str, float]:
    """Score offered choice letters from a final-position logits vector."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return {letter: get_letter_lp(logprobs, letter) for letter in options}


def predict_option(option_scores: Mapping[str, float]) -> str:
    """Return the highest-scoring offered choice; ties resolve alphabetically."""
    return max(sorted(option_scores), key=lambda letter: option_scores[letter])


def is_correct(option_scores: Mapping[str, float], correct_option: str) -> bool:
    return predict_option(option_scores) == correct_option

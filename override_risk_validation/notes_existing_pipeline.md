# Existing pipeline reused by override-risk validation

This document fixes the existing Phase 4 pipeline before any new validation data are evaluated. New condition data are evaluation-only: none of the values below may be retuned on them.

## Fixed model and Phase 4 assets

- Model: `google/gemma-2-2b`, loaded in bfloat16 through TransformerLens.
- Candidate residual layers used to derive vectors: L10–L22.
- Direction/gate: Phase 4 **B direction**, the first right-singular vector of `h_conflict - h_supportive`, derived from Phase 4 training cases only.
- Restoration vector: `u_PK = mean(h_clean - h_conflict)` from the same training cases.
- Stored assets: `phase4/vectors/direction_B_train.pt`, `phase4/vectors/u_PK_train.pt`, and `phase4/vectors/train_proj_clean.pt`.

## Frozen intervention configuration

The latest full-dev Phase 4 V2 result is used as the fixed reference configuration (not the earlier 30-case sweep):

```text
method = B_gated_upk
layer  = 17
alpha  = 6.0
gamma  = 10.0
k      = 1.5
tau    = 19.256011962890625
```

Source: `phase4/results/exp3_comparison_v2/best_configs.json`.

The gated hook applies at `blocks.17.hook_resid_post`:

```text
projection = h_last · normalize(direction_B[17])
gate       = sigmoid(10.0 * (projection - 19.256011962890625))
h_last'    = h_last + gate * 6.0 * u_PK[17]
```

The ungated baseline applies the same `u_PK[17]` and `alpha=6.0` at the same hook point, omitting the gate. `gate_activation_rate` is the fraction with `projection >= tau` (equivalently, `gate >= 0.5`).

## Existing reusable code

| Need | Existing implementation | Input/output |
|---|---|---|
| Model + prompt construction | `scripts/exp_utils.py` | Multiple-choice question, options, and optional context → prompt/token logits |
| Residual collection | `collect_residuals` in `scripts/exp_utils.py` | model, prompt, layers, token position → `[layers, d_model]` residuals and final logits |
| Full Phase 4 vector derivation | `phase4/scripts/setup_vectors.py` | Phase 4 train split → A/B directions, `u_PK`, clean projections |
| Threshold calculation | `compute_tau` in `phase4/scripts/eval_utils.py` | train-clean projections and k → tau (already frozen above) |
| Gated/ungated hooks | `make_gate_reader_resid_post`, `make_upk_hook` in `phase4/scripts/intervention_methods.py` | B direction, tau, gamma, u_PK, alpha → hooks |
| Existing two-answer metrics | `evaluate_cases` / `compute_metrics` in `phase4/scripts/eval_utils.py` | correct vs replacement-option logits |

## Existing data and its limits

`data/phase0/kc_cases_1000.json` contains 1,000 cases, all marked `passes_kc_filter=true`, with `clean_answer` equal to `correct_option` and `conflict_answer_model` equal to `replace_option`. It therefore supplies condition ④ (natural harmful override): a model-known correct answer changes to the conflicting replacement answer. It has 334 misinformation, 333 temporal, and 333 semantic entries.

Its fields are `question`, `options`, `correct_option`, `replace_option`, `correct_answer`, `conflict_answer`, and `conflict_context`. This derived file does not itself establish whether a temporal context answer is a verified latest ground truth. It must not be used as condition ③ without checking the original ConflictBank source and model behaviour.

## Evaluation additions required

Existing Phase 4 accuracy compares only `correct_option` against `replace_option`. The validation wrapper must additionally compute full-choice accuracy: the logit-maximising option among all supplied choices is compared to the ground-truth option. Save both metrics where a replacement option exists.

For B, each safe/error group is evaluated with paired closed-book and supportive-context prompt views. Groups ①, ②, ③, and ④ must use disjoint question IDs; the two views within ① or ② intentionally share an ID.

## Runtime environment

The existing Conda environment `/home/jaeik/miniconda3/envs/apa` contains the
required packages: Python 3.10.19, `torch 2.9.1+cu128`, and
`transformer_lens` (installed). This is the environment to use for all
model-dependent validation commands; do not install a second package stack.

At setup time, `nvidia-smi` could not communicate with an NVIDIA driver, so the
CUDA-dependent forward-pass, residual, and intervention stages cannot run in
this shell until GPU access is restored.

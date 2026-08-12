# XAIMI: Knowledge Conflict Mechanistic Analysis

This repository contains the active analysis code and research notes for studying
**Knowledge Conflict (KC)** in language models: cases where a model's parametric
knowledge conflicts with external context supplied at inference time.

The current work uses XAI and mechanistic-interpretability methods to locate KC
signals, extract candidate directions, and test selective interventions.

## Repository scope

Only source code and lightweight documentation are versioned here. Datasets,
generated plots, logs, model/vector checkpoints, and experimental results stay
local and are intentionally ignored. See `RESEARCH_EXPERIMENT_SUMMARY_KO.md` for
the research narrative and current findings.

## Layout

```text
scripts/          Shared utilities, Phase 0 collection, and cross-phase helpers
phase1/scripts/   Initial diagnosis: tuned lens, IG, causal tracing, patching
phase2/scripts/   Direction extraction and causal-intervention experiments
phase3/scripts/   Conflict-signal comparison and analysis
phase4/scripts/   Multi-vector / multi-layer intervention experiments
```

`archived_previous_experiments/` remains on the local machine but is not part of
the repository.

## Environment

The experiments use Python with PyTorch and TransformerLens. The original
working environment included:

```bash
pip install transformer_lens transformers torch datasets
pip install inseq sae-lens numpy matplotlib seaborn scipy
```

Model access may require a Hugging Face token configured outside the repository.

## Research summary

The project investigates whether KC can be diagnosed and reduced through
layer-wise residual-stream analysis. Current evidence indicates that KC is not
fully captured by one KC-specific vector; effective suppression requires a
combination of directions across later layers. Details and experiment-specific
results are documented in `RESEARCH_EXPERIMENT_SUMMARY_KO.md`.

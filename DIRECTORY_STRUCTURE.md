# xaimi Directory Structure

Last checked: 2026-06-18

This document records the current project layout after moving previous experiment
artifacts into `archived_previous_experiments/`.

## Current Active Layout

```text
xaimi/
├── AGENTS.md
├── CLAUDE.md
├── DIRECTORY_STRUCTURE.md
├── data/
│   └── phase0/
│       ├── collection_progress.json
│       ├── collection_stats.json
│       └── kc_cases_1000.json
├── phase1/
│   ├── scripts/
│   │   ├── step1_tuned_lens.py
│   │   ├── step1_case_analysis.py
│   │   ├── step2_ig.py
│   │   ├── step2_filtered_token_plot.py
│   │   ├── step3_causal_tracing.py
│   │   ├── step3_redraw_heatmap.py
│   │   ├── step3_sanity_check.py
│   │   ├── step4_path_patching.py
│   │   ├── step4_analysis.py
│   │   ├── step5_head_ablation.py
│   │   ├── step6_intervention.py
│   │   └── step6_redraw_plots.py
│   ├── results/
│   │   ├── step1_tuned_lens/
│   │   ├── step2_ig/
│   │   ├── step3_causal_tracing/
│   │   ├── step4_path_patching/
│   │   ├── step5_head_ablation/
│   │   └── step6_intervention/
│   └── STEP6_INTERVENTION_ANALYSIS_KO.md
├── phase2/
│   ├── scripts/
│   │   ├── exp1_trajectory.py
│   │   ├── exp1b_candidates.py
│   │   ├── exp1b_direction_extraction.py
│   │   ├── exp2_causal_patching.py
│   │   ├── exp3_dissociation_mediation.py
│   │   ├── exp3_mediation.py
│   │   ├── exp4_selective_intervention.py
│   │   ├── exp_grid.py
│   │   ├── exp_ie_dissociation.py
│   │   ├── exp_logit_severity.py
│   │   └── exp_supp.py
│   └── results/
│       ├── exp1_trajectory/
│       ├── exp1b_candidates/
│       ├── exp1b_direction_extraction/
│       ├── exp2_causal_patching/
│       ├── exp3/
│       ├── exp4/
│       ├── exp_grid/
│       ├── exp_ie_dissociation/
│       ├── exp_logit_severity/
│       └── exp_supp/
├── phase3/
│   ├── scripts/
│   │   ├── phase3_utils.py
│   │   ├── phase3_step1_projection_dist.py
│   │   ├── phase3_step1_plot.py
│   │   ├── phase3_step2_paired_comparison.py
│   │   ├── phase3_step2_plot.py
│   │   ├── phase3_step3_flipped_analysis.py
│   │   └── phase3_summary.py
│   └── results/
│       ├── directions/
│       ├── step1/
│       ├── step2/
│       └── step3/
├── scripts/
│   ├── make_report_assets.py
│   ├── phase0_collect_kc.py
│   ├── prepare_data.py
│   ├── exp2_direction_patching.py
│   ├── exp1b_candidate_C_analysis.py
│   ├── exp1b_cohens_d.py
│   ├── exp_utils.py
│   ├── check_kc_rates_multi_model.py
│   ├── check_kc_query_rates_phase0_style.py
│   └── redraw_figures_larger_fonts.py
└── archived_previous_experiments/
    ├── data/
    ├── results/
    └── scripts/
```

## Active Files

### `data/phase0/`

These are the currently retained Phase 0 collection outputs.

- `collection_progress.json`
- `collection_stats.json`
- `kc_cases_1000.json`

### `phase1/`

Core step-based pipeline artifacts live here:

- `scripts/step1_*` through `scripts/step6_*`
- `results/step1_tuned_lens/` through `results/step6_intervention/`
- `STEP6_INTERVENTION_ANALYSIS_KO.md`

### `phase2/`

Follow-up experiment artifacts live here:

- `scripts/exp1_*`, `scripts/exp1b_*`, `scripts/exp2_*`, `scripts/exp3_*`, `scripts/exp4_*`
- `scripts/exp_ie_dissociation.py`
- `scripts/exp_grid.py`
- `scripts/exp_logit_severity.py`
- `scripts/exp_supp.py`
- matching `results/exp*/` directories

### `scripts/`

These are the scripts still kept at the root because they are shared utilities,
Phase 0 collection code, or cross-phase helpers.

- `make_report_assets.py`
- `phase0_collect_kc.py`
- `prepare_data.py`
- `exp2_direction_patching.py`
- `exp1b_candidate_C_analysis.py`
- `exp1b_cohens_d.py`
- `exp_utils.py`
- `check_kc_rates_multi_model.py`
- `check_kc_query_rates_phase0_style.py`
- `redraw_figures_larger_fonts.py`

### `phase3/`

Conflict-detection analysis artifacts live here:

- `scripts/phase3_step1_*`
- `scripts/phase3_step2_*`
- `scripts/phase3_step3_flipped_analysis.py`
- `scripts/phase3_summary.py`
- `results/step1/`, `results/step2/`, `results/step3/`

## Archived Previous Experiments

Previous A/B/C experiment code, old datasets, and generated results are stored in:

```text
xaimi/archived_previous_experiments/
├── data/
│   ├── kc_cases.json
│   ├── kc_cases.sample.json
│   ├── kc_cases.verified.json
│   ├── kc_cases_100.json
│   └── kc_cases_500.json
├── scripts/
│   ├── __pycache__/
│   ├── exp_a_attention_attr.py
│   ├── exp_b_logit_lens.py
│   ├── exp_c_activation_patching.py
│   ├── inspect_model_components.py
│   └── merge_exp_c_results.py
└── results/
    ├── ab_experiment_summary.md
    ├── ab_result_analysis_100cases.md
    ├── ab_result_analysis_30cases.md
    ├── abc_detailed_interpretation_report_500cases.md
    ├── abc_examples_gallery_500cases.md
    ├── abc_integrated_report_500cases.md
    ├── abc_methodology_and_execution.md
    ├── exp_a_500/
    ├── exp_b_500/
    ├── exp_c_500/
    ├── exp_c_500_shard0/
    ├── exp_c_500_shard1/
    ├── model_inspect_basic/
    ├── model_inspect_prompt/
    ├── report_assets/
    └── research_framework_and_progress.md
```

## Size Summary

- `xaimi/`: about 116M
- `xaimi/data/`: about 2.9M
- `xaimi/phase1/`: step-based pipeline outputs and scripts
- `xaimi/phase2/`: follow-up experiment outputs and scripts
- `xaimi/scripts/`: shared and utility scripts
- `xaimi/archived_previous_experiments/`: about 113M

## Notes

- `data/kc_cases_100.json` is no longer in the active `data/` directory.
  It has been moved to `archived_previous_experiments/data/kc_cases_100.json`.
- The previous `results/` directory has been moved wholesale to
  `archived_previous_experiments/results/`.
- The active `data/` directory now contains only the `phase0/` folder.

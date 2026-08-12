# Current Experiment Analysis

Last checked: 2026-04-26

Scope of this note:

- Includes only the current `xaimi/` workspace
- Excludes `archived_previous_experiments/`
- Summarizes what experiments were run, how they were implemented, and what results they produced

Relevant active code:

- [phase0_collect_kc.py](/home/jaeik/xaimi/scripts/phase0_collect_kc.py)
- [prepare_data.py](/home/jaeik/xaimi/scripts/prepare_data.py)
- [step1_tuned_lens.py](/home/jaeik/xaimi/phase1/scripts/step1_tuned_lens.py)
- [step1_case_analysis.py](/home/jaeik/xaimi/phase1/scripts/step1_case_analysis.py)
- [step2_ig.py](/home/jaeik/xaimi/phase1/scripts/step2_ig.py)
- [step3_causal_tracing.py](/home/jaeik/xaimi/phase1/scripts/step3_causal_tracing.py)
- [step3_sanity_check.py](/home/jaeik/xaimi/phase1/scripts/step3_sanity_check.py)
- [step4_path_patching.py](/home/jaeik/xaimi/phase1/scripts/step4_path_patching.py)
- [step4_analysis.py](/home/jaeik/xaimi/phase1/scripts/step4_analysis.py)
- [step5_head_ablation.py](/home/jaeik/xaimi/phase1/scripts/step5_head_ablation.py)
- [step6_intervention.py](/home/jaeik/xaimi/phase1/scripts/step6_intervention.py)

## 1. Overall pipeline

The current project is a staged KC analysis pipeline built around `google/gemma-2-2b`.

1. Phase 0: collect model-verified KC cases from ConflictBank-style multiple-choice data
2. Step 1: find where answer preference flips across layers
3. Step 2: identify which input tokens drive the KC effect
4. Step 3: locate causally important layer/segment sites with causal tracing
5. Step 4: decompose late-layer causal effects into attention vs MLP components
6. Step 5: find specific attention heads that strongly support KC
7. Phase 3 intervention: use those mechanism findings to suppress KC

The active dataset is [kc_cases_1000.json](/home/jaeik/xaimi/data/phase0/kc_cases_1000.json), and almost every later step uses it directly.

## 2. Phase 0: KC case collection

Files:

- [phase0_collect_kc.py](/home/jaeik/xaimi/scripts/phase0_collect_kc.py)
- [collection_stats.json](/home/jaeik/xaimi/data/phase0/collection_stats.json)
- [collection_progress.json](/home/jaeik/xaimi/data/phase0/collection_progress.json)

### What it does

This is the dataset construction stage. A case is counted as KC only if:

- the clean prompt makes the model choose the correct option
- the conflict prompt makes the model choose the replacement option

The script uses a 4-choice exact-match setup with answer letters `A/B/C/D`.

### How it is implemented

- Builds `clean_prompt` and `conflict_prompt`
- Computes the final-token log-probability of each answer letter
- Uses the best letter-token variant score for each of `A/B/C/D`
- Keeps a case only when clean predicts `correct_option` and conflict predicts `replace_option`
- Stores `kc_severity = logP(replace) - logP(correct)` under conflict

### Result

The saved Phase 0 stats are:

- processed rows: `900`
- KC cases collected: `1000`
- type balance:
  - misinformation `334`
  - temporal `333`
  - semantic `333`
- KC rates:
  - misinformation `97.9%`
  - temporal `99.7%`
  - semantic `99.4%`
- severity mean: `3.1669`

### Interpretation

This is an aggressively filtered KC dataset. The collection criterion is operational and model-specific, so later results should be read as “mechanisms of Gemma-2-2B on these curated KC cases,” not as a universal KC distribution.

## 3. Step 1: Tuned Lens transition analysis

Files:

- [step1_tuned_lens.py](/home/jaeik/xaimi/phase1/scripts/step1_tuned_lens.py)
- [tuned_lens_stats.json](/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_stats.json)
- [tuned_lens_results.json](/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_results.json)
- [step1_case_analysis.py](/home/jaeik/xaimi/phase1/scripts/step1_case_analysis.py)

### What it asks

At which layer does the model’s internal preference switch toward the conflict answer?

### How it is implemented

The script trains a linear probe for each layer on `resid_post`.

It tracks three token positions:

- `A`: final answer position
- `B`: right before `Answer:`
- `C`: end of context block

Training strategy:

- positions `A/B`: train probes on neutral-run residuals and neutral final answer log-probs, then apply to conflict runs
- position `C`: train separately on conflict-run residuals and conflict final log-probs

Transition layer is defined as the first layer where the predicted preference crosses from correct-side to conflict-side.

### Result

From [tuned_lens_stats.json](/home/jaeik/xaimi/phase1/results/step1_tuned_lens/tuned_lens_stats.json):

- `n_cases = 1000`
- `n_layers = 26`

By position:

- `A`: all `1000` cases transition, mean layer `6.744`, median `7`
- `B`: `963` transition, `37` do not, mean layer `10.405`, median `11`
- `C`: all `1000` cases transition, mean layer `1.727`, median `1`

Position B subgroup counts:

- early: `180`
- mid: `543`
- late: `240`
- no-transition: `37`

### Interpretation

This is the first strong structural finding.

- Context processing itself flips preference very early at position `C`
- The final answer position flips later, around layer `7`
- The pre-answer representation at position `B` flips even later on average, around layer `11`

That suggests a staggered process rather than a one-shot switch:

1. context signal enters very early
2. answer preference at the final decision path flips in mid layers
3. some pre-answer traces remain resistant until later layers

The `37` no-transition cases at `B` are especially interesting because they imply the model can still output the conflict answer even when the pre-answer representation at that location never clearly flips.

## 4. Step 2: Integrated Gradients token attribution

Files:

- [step2_ig.py](/home/jaeik/xaimi/phase1/scripts/step2_ig.py)
- [ig_stats.json](/home/jaeik/xaimi/phase1/results/step2_ig/ig_stats.json)
- [ig_results.json](/home/jaeik/xaimi/phase1/results/step2_ig/ig_results.json)

### What it asks

Which input tokens are actually pushing the model toward the conflict answer?

### How it is implemented

- Manual IG with TransformerLens
- Baseline: neutral prompt embedding, padded to conflict prompt length
- Target score: `logP(replace_option) - logP(correct_option)`
- 5 token groups:
  - `context_claim`
  - `context_evidence`
  - `question`
  - `options`
  - `format_tokens`

The script also reads Step 1 results so it can compare IG patterns across transition groups.

### Result

The currently saved run in [ig_stats.json](/home/jaeik/xaimi/phase1/results/step2_ig/ig_stats.json) is a small run with:

- `n = 10`
- unstable cases: `0`

Overall mean attribution by group:

- `context_evidence`: `0.8518`
- `format_tokens`: `0.0917`
- `context_claim`: `0.0564`
- `question`: `0.0`
- `options`: `0.0`

The same pattern holds across all three conflict types in the saved file.

### Interpretation

On the saved run, the conflict effect is overwhelmingly driven by evidence-style context tokens, not by the question or options.

Two important cautions:

1. the current result file only has `10` cases, so this is suggestive rather than final
2. the top-token list is noisy and includes tokenizer artifacts, which means the group-level result is more trustworthy than individual token rankings

Still, the broad message is useful: the conflict answer seems to be carried mainly by the injected context body, especially the evidence span.

## 5. Step 3: Causal tracing

Files:

- [step3_causal_tracing.py](/home/jaeik/xaimi/phase1/scripts/step3_causal_tracing.py)
- [step3_sanity_check.py](/home/jaeik/xaimi/phase1/scripts/step3_sanity_check.py)
- [ct_stats.json](/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ct_stats.json)
- [ct_results.json](/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ct_results.json)

### What it asks

Which layer-and-segment activations causally matter for restoring the correct answer?

### How it is implemented

Rome-style segment-level causal tracing:

1. run a neutral prompt and cache residuals
2. run a corrupted conflict prompt with Gaussian noise injected into context embeddings
3. restore one `layer × segment` slice from the neutral cache
4. measure indirect effect:

`IE(layer, segment) = P_restored(correct) - P_corrupted(correct)`

Segments:

- `context_first_sent`
- `context_rest`
- `question`
- `options`
- `final`

### Result

From [ct_stats.json](/home/jaeik/xaimi/phase1/results/step3_causal_tracing/ct_stats.json):

- `n = 1000`
- corrupted correct-answer probability mean: `0.2383`

Top layers per segment:

- `context_first_sent`: layer `20`
- `context_rest`: layer `22`
- `question`: layer `4`
- `options`: layer `22`
- `final`: layer `18`

Highest peak layers overall by max IE:

- layer `18`: `0.1363`
- layer `20`: `0.1343`
- layer `17`: `0.1331`
- layer `21`: `0.1311`
- layer `19`: `0.1304`

Segment means across layers:

- `context_first_sent`: `-0.0625`
- `context_rest`: `-0.1093`
- `question`: `0.0013`
- `options`: `-0.0178`
- `final`: `0.0448`

### Interpretation

This is the clearest causal result in the current pipeline.

- Late layers `17-22` are the main causal bottleneck for restoring correct behavior
- The largest positive restoration appears on the `final` segment in late layers
- Earlier restoration of raw context segments is not consistently helpful on average

In other words, the model may ingest the conflict context early, but the most leverage for correcting the final answer sits much later, near the final decision pathway.

That lines up well with Step 1: early entry of context, late causal commitment.

## 6. Step 4: Path patching, attention vs MLP

Files:

- [step4_path_patching.py](/home/jaeik/xaimi/phase1/scripts/step4_path_patching.py)
- [step4_analysis.py](/home/jaeik/xaimi/phase1/scripts/step4_analysis.py)
- [pp_stats.json](/home/jaeik/xaimi/phase1/results/step4_path_patching/pp_stats.json)
- [pp_results.json](/home/jaeik/xaimi/phase1/results/step4_path_patching/pp_results.json)

### What it asks

Inside the late causal window, is KC mainly maintained by attention outputs or MLP outputs?

### How it is implemented

The script focuses on layers `13-25` and only on the final token.

For each layer it compares:

- `hook_attn_out` patching
- `hook_mlp_out` patching
- full `hook_resid_post` patching

So it can estimate:

- `IE_attn(layer)`
- `IE_mlp(layer)`
- `IE_full(layer)`

### Result

From [pp_stats.json](/home/jaeik/xaimi/phase1/results/step4_path_patching/pp_stats.json):

Dominant component by layer:

- attention-dominant: `13, 14, 18, 25`
- MLP-dominant: `15, 16, 17, 19, 20, 21, 22, 23, 24`

Overall mean explanation ratios `(IE_attn + IE_mlp) / IE_full * 100`:

- layer 13: `-25.7%`
- layer 14: `24.7%`
- layer 15: `22.4%`
- layer 16: `54.5%`
- layer 17: `32.7%`
- layer 18: `9.4%`
- layer 19: `-0.1%`
- layer 20: `-0.8%`
- layer 21: `22.8%`
- layer 22: `3.2%`
- layer 23: `11.7%`
- layer 24: `7.9%`
- layer 25: `-1.6%`

Average explanation ratio across these layers: about `12.4%`.

### Interpretation

This step gives a nuanced answer.

- MLP dominates most late layers
- But attention still has strong local importance at a few specific layers, especially 18
- The low explanation ratio means `attn_out + mlp_out` captures only a modest fraction of full residual restoration

So the full causal effect is not cleanly reducible to just “attention vs MLP output” at the final token. A lot of the effect seems distributed, entangled, or dependent on interactions not isolated by this decomposition.

That makes Step 4 more of a negative/constraint result:

- it supports a strong role for late MLP processing
- but it also says the final residual effect is not fully explained by these two component swaps alone

## 7. Step 5: head-level ablation

Files:

- [step5_head_ablation.py](/home/jaeik/xaimi/phase1/scripts/step5_head_ablation.py)
- [head_ablation_results.json](/home/jaeik/xaimi/phase1/results/step5_head_ablation/head_ablation_results.json)
- [mean_values.pt](/home/jaeik/xaimi/phase1/results/step5_head_ablation/mean_values.pt)

### What it asks

Which individual attention heads most strongly support KC?

### How it is implemented

Mean ablation on `blocks.{layer}.attn.hook_z` for layers `13-25`.

Definition:

- baseline diff = `P(replace) - P(correct)`
- ablated diff = same quantity after replacing one head’s `hook_z` with its mean value
- suppression = baseline diff - ablated diff

Positive suppression means ablating that head weakens KC.

The pipeline has two stages:

1. screen on `300` cases
2. validate selected top heads on all `1000` cases

### Result

The strongest head in the current project is very clear.

Stage 1 screening:

- layer `17`, head `3`: suppression `0.3332`

Stage 2 full validation top heads:

1. `L17H3`: suppression mean `0.3387`
2. `L18H6`: suppression mean `0.1017`
3. `L23H5`: suppression mean `0.0410`
4. `L17H4`: suppression mean `0.0271`
5. `L17H6`: suppression mean `0.0268`

The strongest head, `L17H3`, is stable across conflict types:

- misinformation: `0.3429`
- temporal: `0.3320`
- semantic: `0.3414`

### Interpretation

This is the most actionable result in the current workspace.

- KC support is not uniformly spread across all heads
- one head, `L17H3`, stands out dramatically
- `L18H6` is the only other head with clearly notable effect, but still far smaller

This strongly suggests that late-layer attention contains at least one compact, high-leverage KC-supporting component.

It also explains why the intervention script centers on layer 17 head 3.

## 8. Phase 3 intervention

Files:

- [step6_intervention.py](/home/jaeik/xaimi/phase1/scripts/step6_intervention.py)
- [intervention_results.json](/home/jaeik/xaimi/phase1/results/step6_intervention/intervention_results.json)
- [direction_vectors.pt](/home/jaeik/xaimi/phase1/results/step6_intervention/direction_vectors.pt)

### What it asks

Can the identified mechanism actually suppress KC at inference time?

### How it is implemented

Two intervention families:

- Strategy A: mean ablation on `L17H3`
- Strategy B: residual steering using direction vectors at layers `17` and/or `18`

Direction vectors are computed from average neutral residual minus average conflict residual, then normalized.

Tested steering conditions:

- `B1`: layer 17 only
- `B2`: layer 18 only
- `B3`: layers 17 and 18 together

Tested alphas:

- `0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3`

### Result

The currently saved result file is a small run with `n_cases = 10`.

Baseline:

- KC suppression: `0.0%`
- clean accuracy: `90.0%`

Strategy A:

- KC suppression: `40.9%`
- clean accuracy: `40.0%`
- tradeoff score: `-19.06`

Best steering result:

- condition: `B2`
- layer setup: `L18only`
- alpha: `0.2`
- KC suppression: `0.1%`
- clean accuracy: `90.0%`
- tradeoff score: `-9.91`

### Interpretation

The intervention result is mixed, but still informative.

- Mean ablation of `L17H3` clearly suppresses KC
- But it also badly damages clean accuracy
- Steering is much safer, but basically ineffective in the saved run

So the current state is:

- the mechanism seems real enough to break
- but the intervention is not yet selective enough

This is a classic mechanistic-interpretability outcome: strong causal leverage exists, but using it without collateral damage is hard.

## 9. Cross-step synthesis

Across the active workspace, the current story is surprisingly coherent.

### Main pattern

1. Phase 0 builds a dataset where conflict context reliably flips answers
2. Step 1 shows context influence enters early, but final answer commitment shifts later
3. Step 2 suggests the context evidence span is the main input driver
4. Step 3 shows the strongest causal restoration lives in late layers `17-22`
5. Step 4 shows those late effects are mostly more MLP-shaped than attention-shaped, but not fully decomposed by simple component swaps
6. Step 5 identifies one especially strong late attention head, `L17H3`
7. Phase 3 shows ablating that head reduces KC, but with major capability loss

### Working mechanistic picture

The current active project seems to support this interpretation:

- conflict evidence enters early
- preference starts shifting during the middle of the network
- late layers consolidate that shift into a final decision
- most late causal mass is distributed at the residual level
- one or two attention heads, especially `L17H3`, act as strong amplifiers or stabilizers of the wrong answer

### What looks strongest

The most convincing results in the current workspace are:

1. Step 1 transition structure
2. Step 3 late-layer causal bottleneck
3. Step 5 identification of `L17H3`

These three fit together well and point to a stable late-stage KC mechanism.

### What still looks tentative

The weaker parts are:

1. Step 2, because the saved run only has `10` cases
2. Phase 3 intervention, because the saved run only has `10` cases and the tradeoff is poor
3. Step 4’s decomposition, because its low explanation ratio implies the chosen component split is not capturing the whole story

## 10. Bottom line

Excluding `archived_previous_experiments/`, the current `xaimi` project is no longer a generic exploratory notebook set. It is a fairly specific mechanism pipeline:

- collect KC cases
- locate transition timing
- localize causal restoration
- isolate head-level contributors
- test intervention

The project’s strongest present conclusion is:

Late layers around `17-22`, especially attention head `L17H3`, are high-leverage sites for KC expression in this Gemma-2-2B setup, but naive suppression damages clean behavior too much to be a satisfactory fix.

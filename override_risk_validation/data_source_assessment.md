# Data-source assessment for conditions ①–④

## What the existing project supplies

`data/phase0/kc_cases_1000.json` is directly usable for condition ④ (natural
harmful override). Every retained case satisfies the project's original
behavioural filter: the closed-book answer is `correct_option`, while the
conflict prompt answer is `replace_option`.

It cannot supply condition ② because all retained cases are selected for the
opposite behaviour (closed-book correct followed by a harmful context-driven
change).

## ConflictBank temporal data: not sufficient for condition ③ by itself

The official `CB_qa` dataset exposes fields including
`temporal_conflict_time_span`, `temporal_conflict_claim`, and
`temporal_conflict_evidence`, alongside the default fact and evidence.
However, its temporal examples are benchmark-generated conflict scenarios. The
time span records the simulated claim interval; it does not establish that the
replacement answer is a real, currently true answer.

For example, the dataset viewer includes future-dated and historically
impossible temporal claims. Consequently, selecting rows merely because they
are `temporal_conflict` would not meet this study's stated requirement that C
is a verified latest ground truth and M is a real outdated answer.

## Required source decision

Condition ③ needs a temporally grounded source with an explicit, externally
verifiable current answer and a prior answer. Possible approaches are:

1. Curate a small provenance-backed set from authoritative time-stamped sources
   (recommended for a 200–300 case poster study, but requires a source and
   annotation policy).
2. Adopt an established temporal QA benchmark only after verifying that its
   labels encode the target date and the prior/current answer relationship.
3. Use ConflictBank only for ①/②/④ and label its temporal examples as
   *synthetic beneficial-like controls*, not factual beneficial overrides.

No condition-③ selection or model evaluation should be presented as beneficial
override validation until one of these approaches is explicitly chosen.

## Environment status

The current shell has the `datasets` package but does not have `torch` or
`transformer_lens`. It therefore cannot run the existing Gemma-2-2B Phase 4
forward-pass and hook pipeline yet. The source data can be inspected without
this environment; condition construction requiring model behaviour and all
activation/intervention experiments require the original CUDA environment (or
an approved equivalent) first.

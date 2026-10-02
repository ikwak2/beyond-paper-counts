# PRI Construct Audit v0.4: post-collection runbook

## Scope

This runbook executes only the locked Analysis A in
`facct_locked_completion_protocol_v0.4.md`. It does not collect data, call an
API, add an outcome, or change the research question. The local exact-year
matcher in script 42 is a legacy smoke diagnostic and is never a substitute
for the final private OR-batch plus/minus-one-year linkage from script 44.

## External inputs that must finish first

1. Broad-AI full compact cache:
   `data/interim/openalex_pri_v04/private/broad_ai_full_v041/`
2. Broad validation:
   `outputs/facct_locked_completion_v04/pri_qc/broad_ai_validation_full.json`
   with `status == PASS`.
3. Frozen top-roster batch matches:
   `data/interim/openalex_pri_v04/private/top_batch_match_results_v041.jsonl`
   with exactly 26,872 distinct `paper_key` rows.
4. Top-match validation:
   `outputs/facct_locked_completion_v04/pri_qc/top_batch_match_validation.json`
   with `status == PASS`.

The collection/matching processes are external prerequisites. The downstream
driver never invokes scripts 41 or 44 and makes zero network calls.

## Read-only readiness check

From the project root:

```bash
python3 scripts/46_run_pri_construct_audit_after_collection_v04.py
```

Expected before both branches finish: `WAIT_FOR_EXTERNAL_DATA`. This is not an
error and produces no PRI estimate.

Required readiness gates:

- all required inputs exist;
- broad validation is `PASS` and every 2018--2024 year is complete;
- frozen roster has 26,872 rows and 26,872 unique papers;
- batch linkage has 26,872 rows and no duplicate `paper_key`;
- batch validation is `PASS`;
- every venue-year linkage rate is at least 80%;
- maximum minus minimum venue-year linkage rate is at most 10 percentage points.

## Exact downstream command after READY

```bash
python3 scripts/46_run_pri_construct_audit_after_collection_v04.py --execute
```

The driver calls script 43 only after every preflight gate passes. Outputs are
written under:

`outputs/facct_locked_completion_v04/pri_audit_full/`

Primary validation files:

- `tables/pri_full_driver_preflight.json`
- `tables/pri_construct_decision.json`
- `tables/pri_full_driver_validation.json`
- `tables/pri_full_driver_sha256.csv`

The final driver passes only when:

- script 43 exits zero;
- the frozen country-panel population aligns one-to-one with the 26,872 roster;
- broad cache completion and final batch-linkage gates pass;
- work-level and topic tables are nonempty;
- the locked full decision is exactly one of
  `DIRECTIONALLY_ROBUST`,
  `DIRECTIONALLY_ROBUST_CONSTRUCT_SENSITIVE`, or
  `PRI_CONSTRUCT_SENSITIVE_WARNING`.

Any adverse or warning classification is retained. It changes interpretation,
not the research question.

## Fixed construct rules

- Ordering is evaluated only over mutually exclusive China, India, Core
  Anglophone, and Other non-core groups.
- The legacy table label `Non-core Anglophone` actually denotes all non-core
  countries. It remains only for reproduction of previously inspected output.
- Topic-standardized primary specifications keep the frozen all-top
  first-listed numerator. The linked top-paper topic distribution supplies the
  within-year standardization weights.
- A matched-compatible numerator is retained only as an explicitly labeled
  selection-sensitive diagnostic.
- Conservative and sensitivity country mappings and every valid adverse
  denominator are reported side by side.

## Privacy

Record-level batch linkage, titles, names, and paper identifiers remain under
the private interim directory. Public tables contain only aggregates; public
country linkage diagnostics obey the established k=20 rule. SHA-256 manifests
classify private/internal inputs separately from public aggregate/QC outputs.

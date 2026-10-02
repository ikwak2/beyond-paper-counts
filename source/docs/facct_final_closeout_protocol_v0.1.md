# FAccT scientific closeout protocol v0.1

## Scope lock

This closeout performs package QA, terminology and claim-source auditing,
reformats already-frozen missingness and exact-index+2 sensitivity results,
constructs a Conference Inclusion Claim Card, verifies related-work evidence,
performs figure accessibility checks, creates an anonymous reproducibility
package, and records reviewer risks. It adds no conference, outcome, subgroup,
causal framework, or longitudinal horizon.

Existing frozen artifacts are read-only. All new files are written under
`outputs/facct_final_closeout/` except the two closeout protocol documents and
the new closeout implementation script.

## Scientific definitions that remain fixed

- Family A: ICML and NeurIPS accepted first-listed institutional-country
  representation, 2018--2024, relative to the frozen OpenAlex production
  baselines.
- Family B: ICML and NeurIPS 2018--2022 first-listed entrant cohorts, no
  observed accepted appearance in the preceding five publication cycles in
  the named Top-4 record universe, recent accepted-history coauthor
  configuration, and exact index+2 accepted appearances.
- Topic linkage gate: 80%.
- Country-by-pathway minimum cell gate: k=20.
- Primary Family B analytic sample and model are unchanged.
- Bootstrap resampling count and seed are inherited from frozen artifacts.

## Frozen-state reconciliation

The full cohort contains 184 entrants with `index_venue == Both`; the
prespecified team-size-restricted analytic sample contains 181. The requested
Both-exclusion sensitivity therefore removes 181 from the analytic sample and
does not conflict with the cohort provenance. An earlier completed
functional-form sensitivity used a natural cubic spline with df=4. The
closeout executes the requested df=3 sensitivity as a new robustness check
without changing the frozen primary model, and preserves the prior df=4
artifact as provenance rather than silently relabeling it.

## Reporting restrictions

- No publication-ready Abstract, Introduction, Discussion, or rebuttal prose.
- Institutional country is not nationality or language background.
- Accepted-paper records do not identify submission access or review equity.
- Exact index+2 accepted appearance is not career survival.
- Recent accepted-history coauthor status is not mentorship or a causal
  treatment.
- Failed gates remain visible and are not relaxed.


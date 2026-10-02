# FAccT Persistence Technical Specification v0.3.1

This document operationalizes the new RQ3 in
`docs/facct_access_persistence_plan_v0.3.md`. It is frozen before return
outcomes are computed.

## Author-cohort construction

- Eligible index papers are accepted ICML/NeurIPS papers in 2018--2022 whose
  first-listed author is `top4_newcomer_5y == True`.
- Primary identity requires `identity_source == dblp_pid` and a nonblank
  `author_key`.
- An author contributes only the earliest eligible index year.
- All eligible ICML/NeurIPS papers led by that author in the index year are
  aggregated into one author-cohort record.
- `n_index_papers` is the number of those papers.
- `entry_team_size_mean` is the arithmetic mean number of listed authors over
  those papers.
- `index_venue` is `ICML`, `NeurIPS`, or `Both` when the author has qualifying
  papers at both venues in the index year.
- `entry_with_experienced_top4_coauthor` is one if any qualifying index paper
  includes a non-first author with `top4_newcomer_5y == False` in that cycle.

## Country aggregation

- Each qualifying index paper is merged to its paper-time first-author country
  record.
- For each conservative/sensitivity mapping, all observed codes across the
  author's index-cycle papers are pooled.
- The author is assigned to China, India, Core Anglophone, or Other non-core
  only when every pooled code maps to the same comparison group.
- No observed code is `Unresolved`; codes spanning comparison groups are
  `Mixed-group`.
- The primary persistence model does not condition on country availability.

## Return outcomes

- Primary observation ceiling is 2024; 2025 is not used.
- A return is an accepted paper in AAAI, ICLR, ICML, or NeurIPS in index year
  +1 or +2 carrying the same stable DBLP author key.
- `any_top4_return_2cycle` counts return at any author position.
- `first_author_top4_return_2cycle` requires `author_order == 1`.
- Same-cycle papers do not count as returns.
- Cycle 1 is always in the risk set. Cycle 2 is included only if no cycle-1
  event occurred for the corresponding outcome.

## Models and standardization

Primary person-cycle model:

`event ~ experienced_entry_network + C(followup_cycle) + C(index_year) + C(index_venue) + log1p(entry_team_size_mean)`

- Binomial GLM with complementary-log-log link.
- Author-clustered sandwich covariance.
- Separate fits for any-author and first-author return.
- Adjusted two-cycle cumulative probabilities are standardized over the full
  eligible cohort distribution.
- Confidence intervals use 2,000 coefficient draws from the fitted cluster-
  robust multivariate normal approximation with fixed seed 202603.

Secondary country model:

`event ~ experienced_entry_network * C(comparison_group) + C(followup_cycle) + C(index_year) + C(index_venue) + log1p(entry_team_size_mean)`

- Fitted separately under conservative and sensitivity country mappings.
- Only China and Other non-core form the inferential comparison.
- India and Core Anglophone are reported descriptively unless all required
  exposure-by-outcome cells contain at least 20 unique authors.

## Smoke-test PASS gates

- Exactly five index cohorts: 2018--2022.
- Stable-PID coverage at least 90% in every venue-cohort.
- Both experienced and all-new entry-network categories occur in every cohort.
- Both event and non-event outcomes occur in each primary exposure category.
- Primary models converge with finite coefficients and covariance.
- Aggregate output contains no author names, PIDs, author keys, titles, or
  profile identifiers.

Failure of a country interaction or a null retention difference is an
empirical result, not a smoke-test failure.

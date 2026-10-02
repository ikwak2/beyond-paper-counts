# OpenAlex missing-country tipping-point protocol v0.1

## Status and provenance

This document freezes the reporting contract for the final closeout rerun. It
does not claim to be an outcome-blind preregistration: an earlier frozen
implementation and aggregate result already exist in
`facct_final_two_robustness_protocol_v0.1.md` and
`outputs/facct_final_robustness_v01/tipping/`. The closeout rerun must preserve
the scientific inputs and geometry while replacing overbroad legacy wording.

## Fixed scientific question

For each frozen institutional-country group, what combinations of country
composition among OpenAlex works without an observed country would make the
2018--2024 PRI endpoint difference equal zero?

This is a tipping-point analysis, not an imputation estimator and not a
corrected PRI.

## Fixed inputs and estimand

- Years: 2018 and 2024.
- Groups: China, Core Anglophone, India, Other non-core.
- Country mapping: frozen conservative mapping.
- Main top-venue numerator counting: fractional.
- Denominator: frozen role-matched first-position fractional strict broad-AI
  denominator.
- Let `q_2018` and `q_2024` be the group-specific fractional credit among
  works whose country is unobserved in the corresponding year.
- Evaluate the complete unit square `[0,1] x [0,1]`.
- The break-even set satisfies `PRI_2024(q_2024) - PRI_2018(q_2018) = 0`.

## Outputs and classifications

For every group, report the observed endpoint direction, the minimum and
maximum endpoint difference over the unit square, whether its sign is
identified, the feasible break-even boundary, and the closest point on that
boundary to the observed-country composition reference.

- Sign remains unchanged over the entire unit square:
  `SIGN_IDENTIFIED_UNDER_DENOMINATOR_MISSINGNESS_MODEL`.
- A sign reversal is feasible:
  `MISSINGNESS_DIRECTION_NOT_IDENTIFIED`.
- Required inputs or geometry are unavailable: `UNINFORMATIVE`.

The only permitted umbrella wording is **under the prespecified denominator
missing-country model**. The analysis does not address missingness in the
top-venue numerator, topic composition, OpenAlex classification error, or the
causal origin of representation changes.

## Required files

- `coverage_tipping_points.csv`
- `coverage_tipping_surface.csv`
- `coverage_tipping_summary.json`
- `coverage_tipping_report.md`
- `figure_coverage_tipping_point.pdf`
- `figure_coverage_tipping_point.svg`

## No-adaptation rule

Results may not be used to change the primary PRI specification, country
groups, endpoint years, coverage definition, or the interpretation threshold.


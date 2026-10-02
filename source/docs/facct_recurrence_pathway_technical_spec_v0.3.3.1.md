# FAccT v0.3.3.1 Recurrence-Pathway Technical Specification

- Raw outcome shares use all 6,704 entrant proxies.
- Adjusted exposure contrasts use common support `entry_team_size_mean >= 2`,
  because a solo index paper cannot contain an experienced coauthor.
- The primary Top-4 return-universe model uses 1,000 entrant bootstrap
  resamples with seed 202605. Covariate standardization uses each bootstrap
  sample and paired exposure scenarios, so probability-difference intervals
  preserve covariance.
- Mandatory return-universe sensitivities use the identical common-support
  multinomial model and report point estimates: no AAAI, ICML/NeurIPS only,
  and the strict temporal proxy dropping +1 AAAI/ICLR recurrence.
- If fewer than 950 of 1,000 primary bootstrap fits succeed, the interval gate
  fails.
- Public cells require k >= 20. No name, PID, author key, title, or paper ID is
  written to output.

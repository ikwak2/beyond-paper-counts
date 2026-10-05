# Public analysis inputs

## entrant_model_inputs.csv

One row per observed entrant (12,094 rows). No author names, source IDs, original entrant IDs, titles or paper linkages. Blank distance cells represent missing measurements. Original row order is retained for seeded resampling.

- index_year, index_venue: entry year and venue; Both is retained.
- n_index_papers, entry_team_size_mean: entry-paper count and mean team size.
- entry_with_experienced_top4_coauthor: original recent-history coauthor indicator; legacy variable name is preserved.
- history_start_year, history_end_year: prior-history window.
- exact_plus2_top4, exact_plus2_exclude_aaai, exact_plus2_icml_neurips: exact +2 pathway categories, with missing/noneligible states retained. Categories used in the models: no_recurrence, coauthor_continuity_only, at_least_one_no_index_coauthor.
- n_unique_index_titles, n_unique_primary_prior_titles, n_unique_alltype_prior_titles: deduplicated title counts.
- specter2_title_distance_primary, specter2_title_distance_alltypes_sensitivity: frozen semantic distances.
- tfidf_word_title_distance_sensitivity, tfidf_char_title_distance_sensitivity: frozen lexical distances.
- primary_distance_observed: original measurement-observation flag.

The primary distance analysis uses 9,639 observations; D1 uses 9,521; the primary exact +2 model uses 5,122. These are filtered subsets of the exported cohort.

## figure3_calibration_coordinates.csv

249 observed pilot points. `distance_observed` is true for these rows. `specter2_title_portfolio_distance` is the x-coordinate; `permuted_other_portfolio_median_distance` is the y-coordinate; `own_portfolio_closer_than_permuted_median` is the calibration color flag. These frozen coordinates reproduce the scatterplot; the original permutation experiment is not rerun.

## Terms

New numeric exports above: CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/). Code: MIT. Upstream sources retain their own terms.

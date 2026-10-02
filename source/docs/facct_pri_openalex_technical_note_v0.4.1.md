# PRI OpenAlex work-level technical note v0.4.1

## Fixed scope

- Top-venue roster: 26,872 accepted ICML and NeurIPS papers, 2018--2024,
  from the frozen DBLP accepted-paper table.
- Broad-AI denominator: OpenAlex works whose primary-topic subfield is
  Artificial Intelligence (`1702`) and whose type is `article` or
  `conference-paper`, 2018--2024.
- This collection supports only the locked PRI construct audit. It does not
  introduce an additional outcome.

## Why OpenAlex source filtering is not used for the numerator

The OpenAlex source records returned for “International Conference on Machine
Learning” (`S4306419644`) and “Neural Information Processing Systems”
(`S4306420609`) contained only 1,866 and 4,131 works respectively at the
2026-08-13 preflight. These totals are far below the frozen 26,872-paper
roster. Source filtering would therefore silently change the target
population.

Each frozen paper is instead searched by title within a plus/minus one-year window around its roster year. Exact-year agreement is retained as an explicit score.
The ten returned candidates are all scored. The first API result is never
silently accepted. Automatic acceptance requires strong normalized-title
agreement, author agreement, and separation from the second candidate.
Competing, weak, or unmatched cases remain `ambiguous`, `review`, or
`rejected`. Candidate summaries and the score margin are retained for audit.

## Work-level broad-AI denominator

The preflight count is 864,215 works, or approximately 8,643 cursor requests
at 100 works per request. Full raw OpenAlex responses would require roughly
7 GB based on a one-page measurement. The pipeline therefore transforms every
page immediately into a compact, append-only cache and records an atomic page
manifest with SHA-256, next cursor, page count, and cumulative count.

The compact record retains:

- OpenAlex work ID, year, type, and primary topic;
- country set across all observed authorships;
- country set for the first-position authorship;
- paper-country full and fractional credit;
- first-author full and fractional credit;
- authorship-fractional country credit using all-author and covered-author
  denominators;
- missing-country and truncated-authorship diagnostics.

Names and raw affiliation strings are not retained in the broad-AI cache.

## Supported denominator constructs

The cache supports side-by-side comparison of:

1. hybrid PRI: top-venue first-listed country share divided by broad-AI
   all-authorship/paper-country participation share;
2. role-matched PRI: top-venue first-listed country share divided by broad-AI
   first-position country share;
3. full versus fractional multi-country assignment; and
4. topic-standardized versions using the same work-level primary-topic cells.

No denominator is designated as ground truth. Dependence of direction or
magnitude on these constructs is itself the measurement-audit result.

## Mandatory QC before inference

- report match disposition for every venue-year cell;
- report candidate ambiguity and score-margin distributions;
- audit rather than silently accept weak matches;
- report topic, any-country, and first-author-country coverage by year;
- report duplicate IDs, truncated authorships, API filters, retrieval dates,
  page counts, cache hashes, and software version;
- compare OpenAlex counts with the already frozen grouped-count metadata;
- do not describe unmatched papers or missing affiliations as belonging to a
  country;
- do not interpret OpenAlex topic or author-position metadata as error-free.

## Scripts and branches

- `scripts/40_match_top_venues_openalex_v04.py`
- `scripts/41_collect_openalex_broad_ai_v04.py`
- compact cache: `data/interim/openalex_pri_v04/`
- QC: `outputs/facct_locked_completion_v04/pri_qc/`

Smoke and full branches are separate. Both are resumable and rate-limited.

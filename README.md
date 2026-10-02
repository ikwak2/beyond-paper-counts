# Beyond paper counts

Research code and frozen aggregate results for **Measuring participation in AI
conferences: production-adjusted country representation, researcher entry, and
subsequent reappearance in ICML and NeurIPS**.

This package follows the supplied Korean manuscript, *AI 학회 참여의 측정:
ICML · NeurIPS의 생산량 대비 국가별 대표성, 연구자 진입과 후속 재등장*,
including its three-panel Figure 2 and restricted missingness analysis (Table 3c,
supplementary CSV S10b). The attached PDF's filename is a template filename;
the manuscript text and its frozen numerical tables define this release.

The S10b critical thresholds now use the manuscript's **closed-form equation
with 0/1 allocation boundaries**. Across all eight group/counting combinations,
the largest change from the original Brent solver is 7.92 × 10⁻¹². Reported
rounding and conclusions are unchanged. See the
[calculation comparison](docs/CLOSED_FORM_CHECK_KO.md).

## Quick start: offline reproduction

Use Python 3.11. No API key, network collection, GPU, or individual records are
needed after installing dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s tests -v
```

Outputs go to `outputs/`; frozen inputs remain unchanged. To verify the packaged
files and manuscript arithmetic only, use `python reproduce.py --check-only`.

The full command recalculates the **12 annual API-aggregate PRI specifications**,
checks their endpoints against the 44-specification dashboard, reconstructs
restricted missingness bounds from frozen S10, and regenerates Figures 1, 2, 4,
Figure 3B's distance trajectory, and supplementary Figure S3. Generated numeric
tables are checked against the distributed reference tables. Figure 4's layout is
refreshed while its estimates and intervals are unchanged.

The full manuscript Figure 3, including the pilot calibration scatterplot, is
preserved in [`results/figures/fig3.pdf`](results/figures/fig3.pdf). Recreating its
individual scatter points requires the pilot records; those records are not
distributed. The original figure-generation code is included in `source/`.

## What corresponds to the paper?

| Manuscript analysis | Scope | Code / results |
| --- | --- | --- |
| RQ1: country representation | 26,872 accepted papers, 2018–2024; country-credit shares | [`scripts/build_pri.py`](scripts/build_pri.py), [`source/scripts/43_calculate_pri_construct_audit_v04.py`](source/scripts/43_calculate_pri_construct_audit_v04.py), supplementary S9–S11 |
| RQ2: title-portfolio distance | 12,094 entrants → 9,639 with prior-title distance | [`source/scripts/81_build_epj_distance_full_cohort.py`](source/scripts/81_build_epj_distance_full_cohort.py) through [`84_analyze_epj_distance_full.py`](source/scripts/84_analyze_epj_distance_full.py) |
| D1: coauthor composition | 9,521 entrants with distance and mean team size ≥ 2 | `84_analyze_epj_distance_full.py`; legacy `rq2_*` aggregate files |
| RQ3: exact +2 reappearance | 2018–2022 cohorts: 6,704 → 5,207 with distance → 5,122 in the primary model | `81_build_epj_distance_full_cohort.py`, `84_analyze_epj_distance_full.py`; `rq3_*` aggregates |
| Restricted denominator missingness | Table 3c / supplementary S10b | [`scripts/build_restricted_bounds.py`](scripts/build_restricted_bounds.py) |
| Supplementary analyses | External ETO production shares, country-distance summaries, cumulative follow-up | [`source/scripts/90_build_reviewer_additions.py`](source/scripts/90_build_reviewer_additions.py) |

Legacy source filenames call the distance trajectory `rq1_*` and D1 `rq2_*`.
These filenames are preserved for provenance; they do not replace the current
manuscript numbering. The five original main-table CSVs map to **six current
main tables** through
[`results/main_table_source_mapping.csv`](results/main_table_source_mapping.csv).

## Reproducibility boundaries

| Material | Included | Reproduction supported |
| --- | --- | --- |
| 21 annual country-level API responses: 3 definitions × 2018–2024 | Yes | Country counts and 12 annual PRI specifications |
| Accepted-paper country credits, aggregated by year and mapping rule | Yes | Full/fractional numerator arithmetic without paper identifiers |
| 5 original main-table CSVs and 18 supplementary CSVs | Yes; S10b thresholds refreshed using the closed form, other estimates unchanged | Numerical verification and aggregate plots |
| Analysis code, fixed configs and protocols | Yes | Inspection; source-data refits where inputs are available |
| 863,752-record OpenAlex work-level intermediate | **Not preserved** | Its 32 specifications and role-matched endpoints survive as frozen aggregates only |
| Individual author–paper links, histories, embeddings and pilot rows | Not distributed | Full model refits require separately obtained source inputs |

The primary OpenAlex query-target total is **864,215**. The separate work-level
analysis used **863,752** records. Neither count is the country-credit denominator
used for country shares. Re-querying OpenAlex is a new collection and is not an
exact reconstruction of the 11 August 2026 snapshot.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for data access, full-refit
instructions, and remaining provenance limitations.

## Files

- `source/`: original analysis scripts, protocols and configurations; original bytes retained.
- `scripts/`: portable offline reproduction and validation.
- `data/`: country aggregates, preserved API responses and aggregate model results.
- `results/`: manuscript tables, figure inputs and frozen PDF figures.
- `provenance/`: source hashes, release checksums and original diagnostics.
- `tests/`: regression checks for exact +2 outcome rules and missingness bounds.

Country means affiliation country, not nationality. PRI compares accepted-paper
and observed production shares; it is not an acceptance probability. Exact +2
reappearance is a publication-window outcome, not career survival. Adjusted
coauthor contrasts are observational associations.

Code is provided under the existing [MIT license](LICENSE). Upstream DBLP,
OpenAlex and ETO materials retain their own terms. No archive DOI or final paper
citation is asserted by this release.

한국어 안내: [`docs/README_KO.md`](docs/README_KO.md).

# Reproducibility and provenance

## Frozen inputs and outputs

The manuscript PDF supplied for this update is identified in
`provenance/manuscript.json`; it is not redistributed. The release uses the v06
machine-readable-table archive: its 5 main CSVs, 17 supplementary CSVs other
than S10b, and 4 figure-data CSVs are preserved byte for byte. S10b's two threshold
columns are refreshed using the boundary-aware closed form; its original bytes
are retained in `provenance/baselines/table_s10b_brentq.csv`. The table mapping's
S10b note records this change. The additional RQ3 sample-flow CSV resolves the
mapping's intermediate sample of 5,207 people.

`provenance/source_manifest.json` records the original project-relative path and
SHA-256 of each copied source asset. Entries marked `unchanged` are exact copies.
The accepted numerator is an explicit exception: paper rows were aggregated by
year, country-mapping rule and country before export. No paper/person identifier
is needed to recompute the 12 grouped specifications.

`provenance/release_manifest.json` checksums the files in this release, excluding
itself, Git metadata and generated output. `python reproduce.py` checks these
hashes before computation, and checks regenerated numerical tables afterwards.
PDF bytes need not be identical across environments: font libraries, rendering
and timestamps can change. Frozen PDF figures are supplied as reference assets.

## Public computation

The three archived OpenAlex country queries are:

| Legacy identifier | Filter before adding `publication_year` |
| --- | --- |
| `ai_primary_peer_reviewed` | `primary_topic.subfield.id:1702,type:article\|conference-paper` |
| `ai_primary_all_types` | `primary_topic.subfield.id:1702` |
| `cs_primary_peer_reviewed` | `primary_topic.field.id:17,type:article\|conference-paper` |

All use `group_by=authorships.institutions.country_code`, `per-page=200` and years
2018–2024. Exact request URLs, retrieval timestamps, filters and API responses
are retained in `data/openalex_country_api/`. The legacy phrase `peer_reviewed`
means an article/conference-paper type filter; individual peer-review status
was not independently verified.

The 12 specifications combine two accepted-country mapping rules, two numerator
counting rules and three denominators. The original country calculator operates
unchanged on public country sums. Its 420 annual rows include a descriptive
South Korea series; the manuscript's four exhaustive groups supply 336 rows and
48 endpoints in the 44-specification dashboard. Country code `NA` is Namibia
and is parsed as a string rather than a missing value.

S10b reconstructs annual missingness rates from S10 and verifies all eight rows.
At δ=0 it reproduces the observed change; at δ=1 it reproduces the worst-case
bound. The code also checks increasing interval widths and loss of sign
identification as the permitted allocation range expands. The common-additive-
bias path sets the **same absolute bias** in the two years. Its no-reversal
result does not establish that every pair of same-sign, unequal biases preserves
the direction. Critical deltas now use the piecewise closed form documented in
`docs/CLOSED_FORM_CHECK_KO.md`; the reconstruction of missingness rates from S10
still uses the original `fsolve` step. `scripts/audit_closed_form.py` compares the
new thresholds with the archived Brent calculation. The maximum absolute
threshold difference is 7.913 × 10⁻¹²; all displayed rounding is unchanged.
In the portable script, an overbroad original diagnostic label
was corrected to refer specifically to the extremizing corners. Frozen source
bytes, CSV column names, delta-grid bounds and sign-identification flags are
unchanged. Only the two S10b threshold columns have the precision-level update.

## Individual-level analysis source

The `source/` tree retains the original `scripts/`, `config/` and `docs/` layout.
It is an archival analysis source distribution; the default `reproduce.py` does
not run collection, embedding generation, model fitting or manuscript editing.
Some legacy scripts have top-level side effects and should be inspected before
running them. Use an independent analysis workspace when rerunning source code.

The core full-analysis pipeline requires this directory arrangement:

```text
work/
  bibliometrics/
    top_ai_entry/data/processed/
      accepted_authorships_marked.csv
      accepted_papers_dblp.csv
    dblp/
      dblp.xml.gz
      dblp.dtd
  ai_geographic_entry/             # copy the contents of source/ here
    scripts/
    config/
    docs/
    outputs/                      # generated individual/aggregate results
```

With independently obtained inputs and `requirements-analysis.txt` installed,
run the following **inside that independent `ai_geographic_entry/` directory**:

```bash
python scripts/81_build_epj_distance_full_cohort.py
python scripts/82_scan_epj_distance_full_dblp_histories.py
python scripts/83_compute_epj_distance_full_embeddings.py
python scripts/84_analyze_epj_distance_full.py
```

Step 81 constructs the observed entrant cohort and exact +2 outcomes; 82 scans
DBLP histories; 83 generates SPECTER2 and alternative title distances; 84 fits
the distance, diagnostic coauthor and reappearance models using 2,000 bootstrap
replicates. `source/config/epj_distance_based_analysis_v10.json` fixes the seed,
windows, venue set, team-size rule and measurement specification. The original
software environment is recorded in `provenance/software_versions.json`; its
Python version was 3.11.13. The original GPU run used PyTorch 2.6.0+cu124; select
the appropriate PyTorch wheel for your hardware when using the analysis extras.

Additional inputs for the archived supplementary and figure-building scripts:

| Source code | Required material beyond the core pipeline |
| --- | --- |
| `74_*`, `75_*`, `78_*`, `80_*` | Pilot cohort/source results from the earlier recurrence analysis, DBLP histories and pilot configs |
| `43_calculate_pri_construct_audit_v04.py` | Work-level OpenAlex intermediate and accepted-paper country data; the original work-level snapshot is not preserved |
| `59_facct_missingness_tipping_point.py`, `70_facct_numerator_missingness_and_roster_audit.py` | Historical country/roster inputs and closeout outputs referenced in each script |
| `90_build_reviewer_additions.py` | Individual full-analysis dataset, entry papers, accepted authorships, country panel and ETO v1.11.0 `publications_yearly_articles.csv` |
| `86_build_epj_submission_assets.py` | Full aggregate analysis outputs plus the individual 306-person calibration pilot; this is the historical figure/table builder |
| `91_epjds_targeted_revision_audit.py` | Full-analysis dataset, country panel and historical manuscript files; includes the endpoint/bootstrap and selection audit |

The original source is included for inspecting the actual analyses, including
steps whose historical inputs are unavailable. This repository does not claim
that the 32 work-level specifications or all historical collection stages can
be reconstructed from its distributed aggregate data.

## Access and unresolved provenance

- DBLP supplies bibliographic records and identities: <https://dblp.org/>.
- OpenAlex supplies works and institutional-country metadata: <https://openalex.org/>.
- ETO Country AI Activity Metrics v1.11.0 is the external comparison cited by the
  manuscript: <https://doi.org/10.5281/zenodo.19103157>.
- Original input SHA-256 values survive in the cohort/history diagnostics.
  Historical server paths in those records are provenance, not portable inputs.
- DBLP snapshot date and SPECTER2 model/adapter revision are not established by
  the preserved metadata. Configs specify model names but no immutable revision.
  Downloading their latest versions is therefore not a bitwise reconstruction.
- No archival DOI or final author/citation metadata has been supplied for this
  code release. These are not fabricated from the template placeholders.

Model refits and fresh data collection were not run for this packaging update.
The validated claim is offline aggregate reproduction and consistency with the
frozen manuscript results.

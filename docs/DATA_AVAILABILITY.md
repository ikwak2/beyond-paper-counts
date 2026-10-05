# Data sources and available materials

The repository distributes analysis inputs and frozen outputs supporting the manuscript. Source identifiers, collection dates and remaining limits are documented here so the paper can refer readers to the archive without repeating implementation details.

## Included data

- Accepted-paper country credits and annual country aggregates.
- 21 preserved OpenAlex country API responses: three definitions × 2018–2024, collected 11 August 2026, with request metadata.
- Public model input: 12,094 rows, original distance measurements, covariates and outcomes. Names, DBLP PIDs, author keys and original entrant IDs are omitted; all remaining source cells and original row order are preserved for seeded resampling.
- Figure 3: 249 observed calibration coordinates, without author identities or titles.
- Original main and supplementary numerical tables, including Supplementary Table S3 and the complete 210-code country expansion for 2018–2024.
- Frozen final figures, protocols, configurations and quality checks.

The public exports remove direct identifiers; they are not described as irreversibly anonymous. Variable definitions are in DATA_DICTIONARY.md.

## Original sources

### DBLP

Source: https://dblp.org/xml/dblp.xml.gz . The retained full dump was generated on 28 July 2026 at 21:22:51 UTC. Its SHA-256 is `158c35460ae2f39bcc29ca6d12ee77c28cbf7b24835a047dd5ae40111e7dbd1f`. The exact download date is not recorded, and this daily dump is not labelled as an official monthly release. Conference XML records were retrieved on 31 July 2026, as documented in the original source metadata. The complete daily dump and raw author histories are not included in this repository.

DBLP citation guidance: https://dblp.org/faq/4621382.html . Upstream metadata are CC0 1.0.

### OpenAlex

Source: https://openalex.org/ . Country responses and work-level collection are distinct inputs. The archived country responses support recalculation of the 12 aggregate PRI specifications.

The separate work-level collection contained 863,752 unique works, confirmed by execution output on 13 August 2026 and zero-duplicate QC output on 14 August. That intermediate is not currently available in the checked workspace. Its 32 specifications and role-matched endpoints are therefore preserved as frozen outputs. Later API requests may yield different records.

### ETO

Country AI Activity Metrics v1.11.0: https://doi.org/10.5281/zenodo.19103157 . This identifies the external source dataset; it is not the DOI of this study's code/data archive. Upstream source terms remain applicable.

### SPECTER2

Original model names: `allenai/specter2_base` and `allenai/specter2`. Execution output confirms 86,598 titles embedded on 27 August 2026; diagnostics record 768 dimensions and active adapter `Stack[[PRX]]`. The embedding cache was removed, and the immutable base/adapter revisions are not established. Model refits use the preserved distances and do not regenerate these embeddings.

## Validation and scope

The exported inputs reproduce 12 compared original model-result tables using the original seed and 2,000 resamples, within the recorded numerical tolerance. Aggregate reproduction, the existing tests and pixel-identical Figure 3 rendering were checked. Reports are in `provenance/final_verification/`.

Neither the remaining 32 work-level specifications nor all upstream collection, linking and embedding stages are claimed to be exactly reconstructible from this repository. See REPRODUCIBILITY.md for boundaries.

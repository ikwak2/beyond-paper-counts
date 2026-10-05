# Reproducibility and source provenance

The current public model input supports model refits from frozen measurements; the older original-release document is retained as historical documentation and does not describe the expanded current public data scope. See README.md for the current three reproduction commands.

## Data sources and dates

- DBLP full input: daily XML dump generated 28 July 2026, 21:22:51 UTC. Source: https://dblp.org/xml/dblp.xml.gz. SHA-256: 158c35460ae2f39bcc29ca6d12ee77c28cbf7b24835a047dd5ae40111e7dbd1f. Exact full-dump download date is not recorded. This file is not falsely labelled as an official monthly snapshot and is not redistributed in this archive.
- OpenAlex country API: 21 responses, 3 definitions × 2018–2024, retained with request metadata; collected 11 August 2026. Country grouping is `authorships.institutions.country_code`.
- Separate work-level OpenAlex collection: Codex execution output on 13 August 2026 confirms 863,752 cached unique works. QC output on 14 August records zero duplicate IDs. Work-level source inputs are not retained in the current checked workspace. This date does not replace the country API collection date.
- SPECTER2 execution output on 27 August 2026 confirms 86,598/86,598 titles embedded. Diagnostics record 768 dimensions and active adapter Stack[[PRX]]. Original model names: allenai/specter2_base and allenai/specter2. Immutable revisions are not known. Cached embeddings have been removed. The recorded matrix SHA-256 is 07b0f719738ece5face6f9979e997ac3fea31813829781acf65287b75fe97551.
- ETO Country AI Activity Metrics v1.11.0: https://doi.org/10.5281/zenodo.19103157. This is the external source DOI, not this study's archive DOI.

## Export and verification

`data/model_inputs/entrant_model_inputs.csv` removes the four direct identifier columns from the original 12,094-row analysis input and preserves the other source cells and row order. No new model measurement or outcome is introduced. The export is recorded in `provenance/public_input_export.json`.

`data/figure_inputs/figure3_calibration_coordinates.csv` contains the 249 observed pilot rows and only the four plotting columns. Raw titles, names and author links are excluded.

Preserving the source row order preserves seeded bootstrap resampling. Refits use the unchanged original model functions with paths redirected to the public input and outputs folder. A synthetic row counter is added in memory solely for existing size aggregations. The original seed is 20260827, resamples 2000, follow-up entry window 2018–2022, and primary lookback five years.

The package supports the 12 aggregate PRI specifications, restricted bounds, frozen-distance model refits and full Figure 3 rendering. It does not support exact reconstruction of the missing 32 work-level PRI specifications, original title embeddings or all upstream author/paper selection stages. No claim of complete upstream reproducibility is made.

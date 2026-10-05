# Beyond paper counts

Code and supporting data for **Beyond Paper Counts: Measuring Representation, Entry, and Persistence at ICML and NeurIPS, 2018–2024**.

The repository contains country-level aggregates, model inputs with direct identifiers removed, supplementary tables, and publication figures. A Zenodo archive DOI will be added after the submission release is published.

## Run the analyses

Use Python 3.11 on Linux. Install the relevant dependencies, then run:

```bash
# Country aggregates, restricted missingness bounds, and frozen-result checks
python -m pip install -r requirements.txt
python reproduce.py

# Distance, coauthor, and exact +2 models, including 2,000 bootstrap resamples
python -m pip install -r requirements-models.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/refit_models.py

# Complete Figure 3, using the preserved calibration coordinates
python scripts/render_publication_figure3.py
```

Generated outputs are written to `outputs/`. Current submission figures are in [`publication_assets/figures/`](publication_assets/figures/); the aggregate renderer retains the original plotting layouts.

## Files and documentation

| Material | Location |
| --- | --- |
| Country aggregates and 21 preserved API responses | [`data/`](data/) |
| Public model inputs: 12,094 rows | [`data/model_inputs/`](data/model_inputs/) |
| Figure 3 calibration coordinates: 249 points | [`data/figure_inputs/`](data/figure_inputs/) |
| Main and supplementary CSV tables | [`results/`](results/) |
| Supplementary PDF, table ZIP, and final figures | [`publication_assets/`](publication_assets/) |
| Original analysis code, protocols, and configurations | [`source/`](source/) |

See the [data dictionary](docs/DATA_DICTIONARY.md), [data sources and availability](docs/DATA_AVAILABILITY.md), and [reproduction scope](docs/REPRODUCIBILITY.md) for details. [한국어 안내](docs/README_KO.md).

Model refits start from preserved distances. Exact reconstruction of the original SPECTER2 embeddings and the 32 work-level OpenAlex specifications is not supported by the retained inputs. Verification results are in [`provenance/final_verification/`](provenance/final_verification/).

Code is [MIT licensed](LICENSE). The newly exported numeric model inputs and calibration coordinates are CC0 1.0; upstream datasets retain their own terms. Release and citation instructions are in [docs/RELEASE.md](docs/RELEASE.md).

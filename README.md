# Beyond paper counts

Code and supporting data for **Beyond Paper Counts: Measuring Representation, Entry, and Persistence at ICML and NeurIPS, 2018–2024**.

The repository contains country-level aggregates, model inputs with direct identifiers removed, supplementary tables, and publication figures. The submission archive is available on [Zenodo](https://doi.org/10.5281/zenodo.23175869) (version 1.0.2).

## Run the analyses

Use Python 3.11 on Linux. Install the relevant dependencies, then run:

```bash
# Country aggregates, restricted missingness bounds, and frozen-result checks
python -m pip install -r requirements.txt
python reproduce.py

# Distance, coauthor, and exact +2 models, including 2,000 bootstrap resamples
python -m pip install -r requirements-models.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/refit_models.py

# Final Figure 1–4, including the preserved calibration coordinates
python scripts/render_publication_figures.py
```

Generated outputs are written to `outputs/`. Submission figures are in [`publication_assets/figures/`](publication_assets/figures/); the default reproduction command regenerates these layouts.

## Files and documentation

| Material | Location |
| --- | --- |
| Country aggregates and 21 preserved OpenAlex API responses | [`data/`](data/) |
| Public model inputs: 12,094 rows | [`data/model_inputs/`](data/model_inputs/) |
| Figure 3 calibration coordinates: 249 points | [`data/figure_inputs/`](data/figure_inputs/) |
| Main and supplementary CSV tables | [`results/`](results/) |
| Supplementary PDF, table ZIP, and publication figures | [`publication_assets/`](publication_assets/) |
| Original analysis code, protocols, and configurations | [`source/`](source/) |

See the [data dictionary](docs/DATA_DICTIONARY.md), [data sources and availability](docs/DATA_AVAILABILITY.md), and [reproducibility documentation](docs/REPRODUCIBILITY.md) for details. [한국어 안내](docs/README_KO.md).

## Software and reproducibility

The analysis used Python 3.11.13 on Linux. Dependencies for aggregate reproduction and model refitting are listed in `requirements.txt` and `requirements-models.txt`, respectively. Installation and execution details are provided in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The archive preserves country-level OpenAlex API responses, analysis inputs with direct identifiers removed, frozen results, and figure inputs. Model refits start from the preserved distance values.

Work-level OpenAlex intermediate data and the SPECTER2 embedding cache are not preserved, and the exact SPECTER2 model revision was not recorded. The retained inputs therefore support reproduction from preserved analysis inputs, but not complete reconstruction of the original upstream data collection and embedding pipeline. The results of the 32 work-level OpenAlex specifications remain available as frozen aggregates.

Detailed provenance and reproduction limits are documented in [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Verification results are in [`provenance/final_verification/`](provenance/final_verification/).

## License

The code is licensed under [MIT](LICENSE), with no additional restrictions on non-academic use. Newly exported numeric model inputs and calibration coordinates are released under CC0 1.0. Third-party datasets retain their original terms.

## Citation

Ryu, Jungchan, and Kwak, Il-Youp (2026). *Code and supporting data for Beyond Paper Counts: Measuring Representation, Entry, and Persistence at ICML and NeurIPS, 2018–2024*. Version 1.0.2. Zenodo. https://doi.org/10.5281/zenodo.23175869

Release and citation instructions are provided in [docs/RELEASE.md](docs/RELEASE.md).

The Zenodo 1.0.2 archive corresponds to Git commit `9b8b962eddd87f7279504c3f59d8de34b230e90c` and includes the revised Figure 2C and renderer. Later documentation commits do not change this archived code version.

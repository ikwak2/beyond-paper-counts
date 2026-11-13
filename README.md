# Beyond paper counts

Code and supporting data for **Beyond Paper Counts: Measuring Representation, Entry, and Persistence at ICML and NeurIPS, 2018–2024**.

The repository contains country-level aggregates, model inputs with direct identifiers removed, supplementary tables, and publication figures. The original analysis archive is available on [Zenodo](https://doi.org/10.5281/zenodo.23175869) (version 1.0.2).

## Current manuscript supporting materials

Download the [supplementary reproducibility bundle](supplementary/Supplementary_Data_reproducibility.zip) for the current manuscript's country comparisons, aggregate diagnostics, source hashes, display-to-code crosswalk, and figure-reproduction materials. See the [bundle guide](supplementary/README.md) for contents, offline commands, and limits.

This single download includes the original `Additional_file_2.zip` and all files from `Supplementary_Data_country_comparison.zip`; separate downloads are unnecessary. The original v1.0.2 archive is preserved inside the bundle. Materials outside its `archive_v1.0.2/` directory are later additions, do not involve new outcome-model fits, and are **not** represented as part of Zenodo v1.0.2.

The root-level analysis commands and `publication_assets/` below document the original release, including its older supplementary PDF and Figure 1–4 numbering. For the current manuscript's Figure 1–5 materials, use the bundle guide. The revised Supplementary Information PDF is supplied with the manuscript; the historical PDF in this repository is not its replacement.

## Run the original analyses

Use Python 3.11 on Linux. Install the relevant dependencies, then run:

```bash
# Country aggregates, restricted missingness bounds, and frozen-result checks
python -m pip install -r requirements.txt
python reproduce.py

# Distance, coauthor, and exact +2 models, including 2,000 bootstrap resamples
python -m pip install -r requirements-models.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/refit_models.py

# Original-release Figure 1–4, including the preserved calibration coordinates
python scripts/render_publication_figures.py
```

Generated outputs are written to `outputs/`. Original-release figures are in [`publication_assets/figures/`](publication_assets/figures/); the default reproduction command regenerates these layouts.

## Files and documentation

| Material | Location |
| --- | --- |
| Country aggregates and 21 preserved OpenAlex API responses | [`data/`](data/) |
| Public model inputs: 12,094 rows | [`data/model_inputs/`](data/model_inputs/) |
| Figure 3 calibration coordinates: 249 points | [`data/figure_inputs/`](data/figure_inputs/) |
| Main and supplementary CSV tables | [`results/`](results/) |
| Original-release supplementary PDF, table ZIP, and figures | [`publication_assets/`](publication_assets/) |
| Current manuscript supporting data and figure materials | [`supplementary/`](supplementary/) |
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

For the additional materials, also cite this GitHub repository at the fixed commit used. The Zenodo citation above identifies only the original archive. Release and citation instructions are provided in [docs/RELEASE.md](docs/RELEASE.md).

The Zenodo 1.0.2 archive corresponds to Git commit `9b8b962eddd87f7279504c3f59d8de34b230e90c` and includes that release's revised Figure 2C and renderer. Subsequent repository updates, including the separately identified supplementary bundle, do not change the contents of that archived version.

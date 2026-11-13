# Current manuscript supporting materials

[Download Supplementary_Data_reproducibility.zip](Supplementary_Data_reproducibility.zip).

This is the single supporting-data download for the current Scientific Reports manuscript. It preserves the supplied ZIP unchanged. Its checksum is recorded in [SHA256SUMS.txt](SHA256SUMS.txt). The bundle's internal manifest and SHA-256 list cover its 251 payload files, excluding the two manifest files themselves.

## What is included

| Bundle path | Contents |
| --- | --- |
| `archive_v1.0.2/` | 173 preserved files from the original release, including analysis inputs, results, scripts, and historical publication assets |
| `archive_v1.0.2/publication_assets/Additional_file_2.zip` | The unchanged original supplementary table archive |
| `data/country_comparison/` | All eight files from the separately packaged country-comparison ZIP, including selected-country and complete-country tables |
| `data/author_checks/` and `data/provenance/` | Aggregate diagnostics, source hashes, and provenance evidence |
| `data/display_source_code_map.csv` | Current display-to-input-and-code crosswalk |
| `figures/`, `scripts/`, and `provenance_source/` | Current and retained figure assets, downstream utilities, and source-provenance code |

There is no need to download `Additional_file_2.zip` or `Supplementary_Data_country_comparison.zip` separately. References inside the preserved bundle to those files being "separately supplied" describe the earlier submission packaging. The bundle already contains their full contents. Its internal files and manifests have not been rewritten for this hosting change.

The original archive is also deposited at [Zenodo version 1.0.2](https://doi.org/10.5281/zenodo.23175869). The additional material outside `archive_v1.0.2/` is not part of that Zenodo version. These additions summarize frozen inputs and estimates and do not introduce new outcome-model fits.

## Using the bundle

Extract the ZIP into its own directory and run commands from that extracted root. Keep its directory structure: scripts and the Supplementary Information use paths relative to this root. In the file crosswalk, `[A]` denotes `archive_v1.0.2/` and `[R]` denotes the extracted bundle root.

Read the included `README.md` and `REPRODUCTION.md`, then install `requirements-figures.txt` in a suitable isolated Python environment. With those dependencies installed, the documented offline rendering commands include:

```bash
# Current Figures 2 and 4 from frozen CSVs
python scripts/render_figures_2_4_from_csv.py --bundle-root . --output-dir outputs/figures_2_4

# Current Figure 5 from frozen probabilities and contrasts
python scripts/render_figure5_v5.py --bundle-root . --output-dir outputs/figure5

# Retained Figure 3 and S1 vector-label utility
python scripts/standardize_panel_labels_v5.py --output-dir outputs/vector_figures --qa-dir outputs/vector_validation
```

These commands reproduce the operations documented in the bundle, not every historical collection or computation. Source-manuscript PDFs needed by the historical `recover_vector_assets.py` utility are not distributed; that utility is not required by the current Figures 2/4 CSV renderer. Some preserved scripts contain original local paths and serve as provenance records rather than portable commands.

Linked person–paper–affiliation records, some local row-level inputs, the complete upstream SPECTER2 cache and immutable model revisions, and the separate work-level OpenAlex intermediate are not supplied. The 32 work-level specification results remain frozen outputs. The bundle README and the Supplementary Information describe the reconstruction limits and distinguish preserved original scripts from reconstructed utilities.

## Manuscript and historical files

The revised Supplementary Information PDF accompanies the manuscript. PDFs under the repository's root `publication_assets/` and the bundle's `archive_v1.0.2/publication_assets/` are historical release assets. Their table and figure numbering must not be interpreted as the current manuscript numbering. Use the current display crosswalk when locating numerical results.

This repository addition has been checked for ZIP integrity, internal file checksums, and inclusion of the two previously separate data packages. Hosting these materials does not constitute a fresh model refit or independent rerun of every bundled generator.

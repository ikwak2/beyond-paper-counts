# Repository versions, citation, and archived materials

## Original archive and current materials

The original analysis is deposited in [Zenodo version 1.0.2](https://doi.org/10.5281/zenodo.23175869), corresponding to Git commit `9b8b962eddd87f7279504c3f59d8de34b230e90c`. Later repository changes do not alter that deposit.

The public GitHub repository also supplies [the supplementary reproducibility bundle](../supplementary/README.md): current figure utilities, country comparisons, aggregate diagnostics and provenance. Its `archive_v1.0.2/` directory preserves the original files. Other directories contain later additions, not contents of the v1.0.2 deposit. These additions do not involve new outcome-model fits.

The revised Supplementary Information PDF accompanies the manuscript. Root `publication_assets/` PDFs are historical assets, not the current submission PDF. Use the bundle crosswalk for current numbering.

## Citation

Cite the original archive and current repository separately:

1. Ryu, Jungchan, and Kwak, Il-Youp (2026). *Code and supporting data for Beyond Paper Counts: Measuring Representation, Entry, and Persistence at ICML and NeurIPS, 2018–2024*. Version 1.0.2. Zenodo. https://doi.org/10.5281/zenodo.23175869
2. The same authors and title, GitHub repository, specifying the full Git commit used and actual access date. Link to `https://github.com/ikwak2/beyond-paper-counts/tree/<full-commit>`. Obtain the final commit with `git rev-parse HEAD` after changes are committed.

The root `CITATION.cff` describes the repository without assigning the old archive DOI or version to later additions. Add the selected commit to the manuscript reference. No new Zenodo DOI or formal release version is asserted for the current repository.

## Integrity

`provenance/release_manifest.json` retains the original file inventory with hashes updated for current repository contents; it excludes itself. When a listed file changes, update its hash using committed Git bytes, then run `python reproduce.py --check-only`. On Windows, LF/CRLF checkout conversion can affect raw hashes; use the documented Linux environment or export with `git -c core.autocrlf=false archive --format=zip HEAD` to check committed bytes.

The supplementary ZIP is checked separately by `supplementary/SHA256SUMS.txt` and its internal manifests. Its original `archive_v1.0.2/` files are preserved unchanged. Current bundle documentation changes are described in the supplementary guide.

## Optional future archive

To archive the combined materials, create a new version of the existing Zenodo record or use GitHub–Zenodo integration for a reviewed release. Choose one route. Assign the actual new version, review `.zenodo.json` and `CITATION.cff`, and inspect all files before publishing. `.zenodo.json` currently describes the repository scope and deliberately omits a new version number.

Use a new version-specific DOI only after that deposit is published. Do not relabel the original v1.0.2 DOI as including later additions.

Official instructions: [GitHub archiving](https://help.zenodo.org/docs/github/archive-software/github-upload/) and [creating an upload](https://help.zenodo.org/docs/deposit/create-new-upload/).

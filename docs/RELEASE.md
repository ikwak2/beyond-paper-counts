# Submission release and Zenodo archive

The public repository has a manually deposited Zenodo archive, version 1.0.2: https://doi.org/10.5281/zenodo.23175869. Its code and assets correspond to Git commit `9b8b962eddd87f7279504c3f59d8de34b230e90c`. The archive includes the compact Figure 1 and the revised Figure 2C with distinct legend symbols and dark reference marks. Analysis inputs and frozen numerical results are unchanged. Later citation and documentation updates on main do not alter the archived code version or require a new archive.

## GitHub-to-Zenodo workflow

1. After the owner makes the repository public, sign into Zenodo and enable its GitHub integration for `ikwak2/beyond-paper-counts`.
2. Check the authors, title, actual release version, licenses and description in `.zenodo.json` and `CITATION.cff`. They describe the software and supporting data; no manuscript publication DOI is asserted.
3. Create a GitHub release from the reviewed commit, using the chosen version tag. All intended data must be committed in the repository before this release. The reviewed model inputs, coordinates and publication assets are included here.
4. Wait for Zenodo processing, inspect the archived files and record, and obtain the DOI for that specific version.
5. Add the real DOI to `CITATION.cff`, the README citation section and the manuscript's BibTeX/Availability statement. Cite the specific submission version rather than silently pointing to a moving branch. Do not reuse the external ETO DOI as the study archive DOI.

Alternatively, upload the complete reviewed source archive manually to Zenodo. Use one chosen route for this submission so the same materials are not needlessly published as two unrelated records. Manual upload does not automatically fill its form from `.zenodo.json`.

Official instructions: https://help.zenodo.org/docs/github/archive-software/github-upload/ . The DOI is registered on publication, not merely by reserving it in a draft: https://help.zenodo.org/docs/deposit/create-new-upload/ .

## Paper availability wording

After an actual archive has been published, the paper can keep its access statement concise and cite the repository/archive using numbered BibTeX references. The essential limitations must remain accurate; detailed filters, fingerprints, filenames and reproduction instructions belong in the repository documentation.

Suggested starting sentence after publication:

> The code and supporting datasets are available in the Zenodo repository [archive reference].

Retain the original manuscript's essential scope limitation for the unavailable work-level inputs. Software/platform/language/license information must still meet the current journal instructions; shortening the statement is not a reason to omit required information.

# Frozen tables and current numbering

This directory contains five original main-table CSVs (tables/), twenty-one supplementary
CSV datasets (additional_files/), and main_table_source_mapping.csv.
The current manuscript has six main tables: Table 3(a-b) relocates frozen supplementary
CSV S10; Table 5(c) relocates semantic_primary results from supplementary CSV S5.
Frozen CSV bytes, filenames, and estimates are unchanged.

table_s10b_restricted_missingness_bounds.csv (Table 3c) is the one CSV added since v05.
It is not a new data collection. Observed group shares, accepted shares, and annual
missingness rates are reconstructed from frozen CSV S10 and re-verified against it:
delta = 0 reproduces the observed endpoint change and delta = 1 reproduces the frozen
worst-case bound. No OpenAlex query was re-run.
Four figure-data CSVs provide the plot inputs for the three-panel Figure 2 and for
Figure S3. They supersede the two-panel Figure 2 inputs shipped with v05 and are
separate from the 5+21 tables.

IMPORTANT: table_2_geographic_endpoints.csv retains the legacy field
direction_identified_under_denominator_missingness_bounds. Do not interpret that
field as a missingness verdict for the primary all-authorship country API PRI.
Current Table 2 omits it. Table 3 and CSV S10 distinguish fractional and full
numerators under the separate strict first-position fractional denominator.
China is sign-identified only in the fractional-numerator denominator-only case;
the full-numerator bound spans zero. Separate numerator-only bounds allow reversal.
Neither analysis jointly varies both sources of missingness.

Three country-expansion CSVs add Supplementary Table S3 (30 countries/regions),
all 210 country/region endpoints, and the annual 2018–2024 series. They use the
same frozen primary aggregates.

The mapping also references qa/sample_flow_rq3.csv for the intermediate N=5207.
That diagnostic is in the full source bundle, not among the 5+21 frozen CSVs.

Work-level provenance: the 863,752-record work-level intermediate that produced the
32 work-level specifications and the role-matched denominators is not preserved.
Its results survive only as frozen values in table_s9_pri_specification_dashboard.csv,
which is why the manuscript reports the role-matched specification as a 2018/2024
endpoint sensitivity rather than as an annual trajectory.

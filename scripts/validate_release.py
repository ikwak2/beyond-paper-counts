"""Verify asset hashes, manuscript arithmetic and optional regenerated results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(generated=False):
    manifest = json.loads((ROOT / "provenance/release_manifest.json").read_text())
    for name, digest in manifest.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Changed asset: {name}")
    require(len(list((ROOT / "results/tables").glob("*.csv"))) == 5, "Expected 5 legacy main-table CSVs")
    require(len(list((ROOT / "results/additional_files").glob("*.csv"))) == 18, "Expected 18 supplementary CSVs")
    mapping = pd.read_csv(ROOT / "results/main_table_source_mapping.csv")
    for name in mapping.package_source_path:
        require((ROOT / "results" / name).is_file(), f"Missing table mapping target: {name}")
    populations = pd.read_csv(ROOT / "results/tables/table_1_analysis_populations.csv")
    require(populations.primary_n.tolist() == [26872, 9639, 9521, 5122], "Primary sample counts")
    flow = pd.read_csv(ROOT / "results/qa/sample_flow_rq3.csv")
    require(flow.included_n.tolist() == [6704, 5207, 5122], "RQ3 sample flow")
    require(flow.excluded_at_stage.iloc[1:].sum() == 6704 - 5122, "RQ3 exclusion arithmetic")
    annual = pd.read_csv(ROOT / "data/distance/rq1_year_distance_descriptive.csv")
    require(annual.entrants_with_distance.sum() == 9639, "Distance-observed cohort count")
    endpoint = annual.set_index("index_year").median_distance
    require(round(endpoint.loc[2024] - endpoint.loc[2018], 5) == -.00130, "Distance endpoint in abstract")
    prob = pd.read_csv(ROOT / "data/distance/rq3_exact_plus2_probabilities.csv")
    np.testing.assert_allclose(prob.groupby(["specification", "distance_level", "experienced_coauthor"])
                              .adjusted_probability.sum(), 1, atol=1e-10)
    con = pd.read_csv(ROOT / "data/distance/rq3_exact_plus2_contrasts.csv")
    con = con.loc[con.specification.eq("semantic_primary") &
                  con.contrast_family.eq("experienced_minus_no_experienced")]
    np.testing.assert_allclose(con.groupby("distance_level").probability_difference.sum(), 0, atol=1e-10)
    absence = con.loc[con.pathway.eq("no_recurrence")].set_index("distance_level")
    require(round(-100*absence.loc["Q25", "probability_difference"], 1) == 10.7, "Q25 contrast in abstract")
    require(round(-100*absence.loc["Q75", "probability_difference"], 1) == 13.4, "Q75 contrast in abstract")
    dashboard = pd.read_csv(ROOT / "results/additional_files/table_s9_pri_specification_dashboard.csv")
    require(len(dashboard) == 176 and dashboard.spec_id.nunique() == 44, "44 specifications × 4 groups")
    require(dashboard.groupby("group").direction.nunique().eq(1).all(), "Observed direction agreement")
    bounds = pd.read_csv(ROOT / "results/additional_files/table_s10_denominator_missingness_bounds.csv")
    crossing = bounds.assumption_free_change_low.le(0) & bounds.assumption_free_change_high.ge(0)
    require(len(bounds) == 8 and crossing.sum() == 7, "7 of 8 worst-case bounds cross zero")
    row = bounds.loc[~crossing].iloc[0]
    require(row.group == "China" and row.numerator_counting == "fractional", "Only China fractional is sign identified")
    # Check distributed tables, not code variable names or protocol field definitions.
    forbidden = {"author_name", "author_key", "author_id", "dblp_pid", "pid", "paper_id",
                 "paper_key", "entrant_id", "email", "title", "abstract", "raw_author_name"}
    csv_count = 0
    for directory in (ROOT / "data", ROOT / "results"):
        for path in directory.rglob("*.csv"):
            cols = set(pd.read_csv(path, nrows=0).columns.str.lower())
            require(not (cols & forbidden), f"Individual identifier column in {path.relative_to(ROOT)}")
            csv_count += 1
    if generated:
        for name in ["table_s10b_restricted_missingness_bounds.csv"]:
            expected = pd.read_csv(ROOT / "results/additional_files" / name)
            actual = pd.read_csv(OUT / name)
            pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-9, atol=5e-9)
        for name in ["figure_2_panelA_aggregate_data.csv", "figure_2_panelB_four_group_data.csv",
                     "figure_2_panelC_sensitivity_data.csv", "figure_s3_plot_data.csv"]:
            pd.testing.assert_frame_equal(pd.read_csv(OUT / name),
                pd.read_csv(ROOT / "results/figure_data" / name), check_exact=False, rtol=1e-10, atol=1e-9)
        for name in ["fig1.pdf", "fig2.pdf", "figure3b_distance_trajectory.pdf", "fig4.pdf"]:
            require((OUT / name).read_bytes().startswith(b"%PDF"), f"Missing generated figure: {name}")
        for name in ["restricted_bounds_qa.json", "figure2_v06_qa.json"]:
            report = json.loads((OUT / name).read_text())
            require(all(report["checks"].values()), f"Failed generator check: {name}")
        require(json.loads((OUT / "pri_qa.json").read_text())["status"] == "PASS", "PRI reproduction failed")
    return {"status": "PASS", "hashed_files": len(manifest), "aggregate_csvs_checked": csv_count,
            "manuscript_sample_counts": True, "probability_arithmetic": True,
            "44_specifications_and_8_bounds": True, "table_mapping_targets_exist": True,
            "generated_outputs_verified": generated,
            "scope": "Aggregate reproduction and frozen results; no person-level model refit"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", action="store_true")
    args = parser.parse_args()
    result = validate(args.generated)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

"""Recalculate 12 annual PRI specifications from preserved country aggregates.

No network calls: the archived collector is imported only for response parsing.
The numerator is a country-year sum, never a person/paper-level input.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load_script(name):
    path = ROOT / "source/scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_numeric(actual, expected, keys, columns, atol=1e-9):
    """Require identical keys as well as equal values; never drop unmatched rows."""
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError(f"Mismatched or duplicated keys: {keys}")
    np.testing.assert_allclose(a[columns], b[columns], rtol=1e-12, atol=atol)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    collector = load_script("22_fetch_openalex_capacity_denominators.py")
    calculator = load_script("67_journal_korea_pri_v01.py")
    records = []
    query_counts = {}
    paths = sorted((ROOT / "data/openalex_country_api").glob("*.json"))
    if len(paths) != 21:
        raise ValueError("Expected 3 denominator definitions × 7 annual responses")
    for path in paths:
        wrapped = json.loads(path.read_text())
        records.extend(collector.records_from_payload(wrapped))
        if wrapped["specification"] == "ai_primary_peer_reviewed":
            query_counts[str(wrapped["year"])] = wrapped["response"]["meta"]["count"]
    capacity = pd.DataFrame(records)
    frozen = pd.read_csv(
        ROOT / "data/aggregate/openalex_capacity_country_year_v01.csv",
        keep_default_na=False,  # "NA" is Namibia, not a missing value.
    )
    frozen = frozen.loc[frozen.year.between(2018, 2024)]
    compare_numeric(capacity, frozen, ["specification", "year", "country_code"],
                    ["country_work_count_full", "openalex_work_count", "country_grouped_count_total"])
    capacity.to_csv(OUT / "openalex_capacity.csv", index=False)
    top = pd.read_csv(ROOT / "data/aggregate/accepted_country_credits.csv", keep_default_na=False)
    annual = calculator.calculate(top, OUT / "openalex_capacity.csv")
    expected = pd.read_csv(ROOT / "data/aggregate/pri_annual_frozen.csv")
    compare_numeric(annual, expected, ["spec_id", "year", "group"],
                    ["top_count", "top_credit_total", "top_share", "broad_count",
                     "broad_country_credit_total", "broad_share", "pri"])
    dashboard = pd.read_csv(ROOT / "results/additional_files/table_s9_pri_specification_dashboard.csv")
    dashboard = dashboard.loc[dashboard.specification_family.eq("existing_inspected_12spec")]
    ends = annual.loc[annual.group.ne("South Korea")].pivot(
        index=["spec_id", "group"], columns="year", values="pri")
    ends = ends.assign(change_2018_2024=ends[2024] - ends[2018]).reset_index()
    compare_numeric(ends, dashboard, ["spec_id", "group"], ["change_2018_2024"], atol=2e-8)
    if sum(query_counts.values()) != 864215 or annual.spec_id.nunique() != 12:
        raise ValueError("Snapshot count or specification count differs from the manuscript")
    annual.to_csv(OUT / "pri_annual.csv", index=False)
    report = {"status": "PASS", "annual_specifications": 12, "annual_rows": len(annual),
              "api_query_target_count": sum(query_counts.values()), "query_counts": query_counts,
              "frozen_annual_values_reproduced": True, "dashboard_endpoints_reproduced": True,
              "work_level_32_specifications": "frozen results only; intermediate not preserved"}
    (OUT / "pri_qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

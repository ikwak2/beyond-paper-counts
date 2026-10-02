#!/usr/bin/env python3
"""Audit numerator-country missingness, country-credit expansion, and roster scope.

This is a deterministic aggregate-only audit. It performs no network calls and
does not expose paper-level identifiers in its outputs.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COUNTRY = ROOT / "data/processed/country_measurement_combined_v01_first_author.csv"
PRI = ROOT / "outputs/tables/facct_rq1_v01_group_year.csv"
ROSTER_SOURCE = ROOT.parent / "bibliometrics/top_ai_entry/data/processed/accepted_papers_dblp.csv"
FROZEN_ROSTER = ROOT / "data/interim/openalex_pri_v04/private/top_target_roster_2018_2024.csv"
OUT = ROOT / "outputs/facct_final_closeout/numerator_missingness"
YEARS = tuple(range(2018, 2025))
GROUPS = ("China", "Core Anglophone", "Other non-core", "India")


def parse_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return sorted({str(item).strip() for item in parsed if str(item).strip()})
    return []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "increase"
    if value < -tolerance:
        return "decrease"
    return "zero"


def country_credit_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    geo = pd.read_csv(COUNTRY, keep_default_na=False)
    geo = geo.loc[geo["venue"].isin(["ICML", "NeurIPS"]) & geo["year"].isin(YEARS)].copy()
    geo["country_codes"] = geo["primary_country_codes"].map(parse_list)
    geo["country_count"] = geo["country_codes"].map(len)
    geo["covered"] = geo["primary_country_covered"].astype(bool) & geo["country_count"].gt(0)

    coverage = (
        geo.groupby(["venue", "year"], as_index=False)
        .agg(
            roster_papers=("paper_id", "nunique"),
            covered_unique_papers=("covered", "sum"),
            country_credits=("country_count", "sum"),
            multi_country_first_authors=("country_count", lambda s: int((s > 1).sum())),
            maximum_countries_per_first_author=("country_count", "max"),
        )
        .sort_values(["year", "venue"])
    )
    coverage["missing_unique_papers"] = coverage["roster_papers"] - coverage["covered_unique_papers"]

    annual = (
        coverage.groupby("year", as_index=False)
        .agg(
            roster_papers=("roster_papers", "sum"),
            covered_unique_papers=("covered_unique_papers", "sum"),
            country_credits=("country_credits", "sum"),
            multi_country_first_authors=("multi_country_first_authors", "sum"),
            missing_unique_papers=("missing_unique_papers", "sum"),
            maximum_countries_per_first_author=("maximum_countries_per_first_author", "max"),
        )
    )
    annual["extra_country_credits"] = annual["country_credits"] - annual["covered_unique_papers"]
    annual["country_credit_per_covered_paper"] = annual["country_credits"] / annual["covered_unique_papers"]
    annual["multi_country_first_author_share"] = annual["multi_country_first_authors"] / annual["covered_unique_papers"]
    return coverage, annual


def boundary_audit(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pri = pd.read_csv(PRI, keep_default_na=False)
    pri = pri.loc[
        pri["country_specification"].eq("conservative")
        & pri["capacity_specification"].eq("ai_primary_peer_reviewed")
        & pri["year"].isin([2018, 2024])
        & pri["group"].isin(GROUPS)
    ].copy()
    annual_index = annual.set_index("year")
    corner_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []

    for counting in ("full", "fractional"):
        indexed = pri.loc[pri["counting_specification"].eq(counting)].set_index(["group", "year"])
        for group in GROUPS:
            values: dict[int, dict[str, float]] = {}
            for year in (2018, 2024):
                row = indexed.loc[(group, year)]
                values[year] = {
                    "count": float(row["first_author_count"]),
                    "total": float(row["first_author_count_total"]),
                    "broad_share": float(row["capacity_share"]),
                    "observed_pri": float(row["first_author_pri"]),
                    "missing": float(annual_index.loc[year, "missing_unique_papers"]),
                }

            def completed_pri(year: int, q: float) -> float:
                v = values[year]
                return ((v["count"] + q * v["missing"]) / (v["total"] + v["missing"])) / v["broad_share"]

            corner_changes: dict[tuple[int, int], float] = {}
            for q18 in (0, 1):
                for q24 in (0, 1):
                    p18 = completed_pri(2018, q18)
                    p24 = completed_pri(2024, q24)
                    change = p24 - p18
                    corner_changes[(q18, q24)] = change
                    corner_rows.append(
                        {
                            "counting_specification": counting,
                            "completion_rule": "one total country credit per unresolved paper",
                            "group": group,
                            "q_2018": q18,
                            "q_2024": q24,
                            "completed_pri_2018": p18,
                            "completed_pri_2024": p24,
                            "endpoint_change": change,
                            "direction": direction(change),
                        }
                    )

            base18 = completed_pri(2018, 0.0)
            base24 = completed_pri(2024, 0.0)
            slope18 = completed_pri(2018, 1.0) - base18
            slope24 = completed_pri(2024, 1.0) - base24
            intersections = {
                "q_2018_at_q_2024_0": (base24 - base18) / slope18,
                "q_2018_at_q_2024_1": (base24 + slope24 - base18) / slope18,
                "q_2024_at_q_2018_0": (base18 - base24) / slope24,
                "q_2024_at_q_2018_1": (base18 + slope18 - base24) / slope24,
            }
            feasible = {key: value for key, value in intersections.items() if -1e-12 <= value <= 1 + 1e-12}
            signs = {direction(value) for value in corner_changes.values()}
            boundary_rows.append(
                {
                    "counting_specification": counting,
                    "completion_rule": "one total country credit per unresolved paper",
                    "group": group,
                    "observed_pri_2018": values[2018]["observed_pri"],
                    "observed_pri_2024": values[2024]["observed_pri"],
                    "observed_direction": direction(values[2024]["observed_pri"] - values[2018]["observed_pri"]),
                    **intersections,
                    "feasible_boundary_intersections": json.dumps(feasible, sort_keys=True),
                    "direction_identified_over_full_unit_square": len(signs) == 1 and "zero" not in signs,
                    "corner_directions": json.dumps(sorted(signs)),
                }
            )
    return pd.DataFrame(boundary_rows), pd.DataFrame(corner_rows)


def roster_audit() -> pd.DataFrame:
    source = pd.read_csv(ROSTER_SOURCE, keep_default_na=False)
    source = source.loc[source["venue"].isin(["ICML", "NeurIPS"]) & source["year"].isin(YEARS)].copy()

    def track(row: pd.Series) -> str:
        urls = " ".join(parse_list(row["electronic_editions"]))
        if row["venue"] == "ICML":
            return "ICML proceedings"
        if "Datasets_and_Benchmarks" in urls:
            return "NeurIPS Datasets & Benchmarks"
        if "Abstract-Conference" in urls:
            return "NeurIPS Main Conference"
        return "NeurIPS legacy proceedings (track not separated in URL)"

    source["track"] = source.apply(track, axis=1)
    counts = source.groupby(["year", "track"]).size().unstack(fill_value=0)
    rows: list[dict[str, Any]] = []
    for year in YEARS:
        get = lambda name: int(counts.loc[year, name]) if name in counts.columns else 0
        icml = get("ICML proceedings")
        legacy = get("NeurIPS legacy proceedings (track not separated in URL)")
        main = get("NeurIPS Main Conference")
        datasets = get("NeurIPS Datasets & Benchmarks")
        neurips = legacy + main + datasets
        rows.append(
            {
                "year": year,
                "icml_frozen_proceedings": icml,
                "neurips_legacy_unseparated": legacy,
                "neurips_main_conference": main,
                "neurips_datasets_benchmarks": datasets,
                "neurips_frozen_total": neurips,
                "pooled_frozen_total": icml + neurips,
                "track_basis": "frozen DBLP electronic_editions URL suffix",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    by_venue, annual = country_credit_audit()
    boundary, corners = boundary_audit(annual)
    roster = roster_audit()

    china_full = boundary.loc[
        boundary["counting_specification"].eq("full") & boundary["group"].eq("China")
    ].iloc[0]
    china_fractional = boundary.loc[
        boundary["counting_specification"].eq("fractional") & boundary["group"].eq("China")
    ].iloc[0]
    endpoint = annual.set_index("year")
    checks = {
        "frozen_roster_26872": int(roster["pooled_frozen_total"].sum()) == 26_872,
        "roster_2024_7104": int(roster.loc[roster["year"].eq(2024), "pooled_frozen_total"].iloc[0]) == 7_104,
        "neurips_2024_4035_plus_459": bool(
            int(roster.loc[roster["year"].eq(2024), "neurips_main_conference"].iloc[0]) == 4_035
            and int(roster.loc[roster["year"].eq(2024), "neurips_datasets_benchmarks"].iloc[0]) == 459
        ),
        "numerator_endpoint_counts": bool(
            int(endpoint.loc[2018, "covered_unique_papers"]) == 1_435
            and int(endpoint.loc[2018, "country_credits"]) == 1_470
            and int(endpoint.loc[2018, "missing_unique_papers"]) == 196
            and int(endpoint.loc[2024, "covered_unique_papers"]) == 6_444
            and int(endpoint.loc[2024, "country_credits"]) == 6_602
            and int(endpoint.loc[2024, "missing_unique_papers"]) == 660
        ),
        "china_full_boundary_reproduced": math.isclose(float(china_full["q_2018_at_q_2024_0"]), 0.8678351885893426, abs_tol=1e-12),
        "china_fractional_boundary_reproduced": math.isclose(float(china_fractional["q_2018_at_q_2024_0"]), 0.8679970367645898, abs_tol=1e-12),
        "credit_multiplier_not_mechanically_constant": annual["country_credit_per_covered_paper"].nunique() > 1,
        "observed_three_country_case_retained": int(annual["maximum_countries_per_first_author"].max()) == 3,
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "model_scope": {
            "full": "conditional on every unresolved paper contributing exactly one total country credit",
            "fractional": "each unresolved paper contributes total weight one; removes unknown multi-country credit expansion",
            "not_claimed": "assumption-free identification under arbitrary unresolved multi-country multiplicity",
        },
        "china": {
            "full_boundary_q2018_when_q2024_zero": float(china_full["q_2018_at_q_2024_0"]),
            "full_boundary_q2024_when_q2018_one": float(china_full["q_2024_at_q_2018_1"]),
            "fractional_boundary_q2018_when_q2024_zero": float(china_fractional["q_2018_at_q_2024_0"]),
            "fractional_boundary_q2024_when_q2018_one": float(china_fractional["q_2024_at_q_2018_1"]),
            "interpretation": "not sign-identified over the full unit square; reversal is confined to the high-2018/low-2024 missing-China corner under both checks",
        },
        "credit_multiplier": {
            "2018": float(endpoint.loc[2018, "country_credit_per_covered_paper"]),
            "2024": float(endpoint.loc[2024, "country_credit_per_covered_paper"]),
            "annual_minimum": float(annual["country_credit_per_covered_paper"].min()),
            "annual_maximum": float(annual["country_credit_per_covered_paper"].max()),
            "mechanically_fixed": False,
        },
        "roster_provenance": {
            "source": str(ROSTER_SOURCE),
            "source_sha256": sha256(ROSTER_SOURCE),
            "frozen_subset": str(FROZEN_ROSTER),
            "frozen_subset_sha256": sha256(FROZEN_ROSTER),
            "neurips_2024_decision_time_comparison": "official fact sheet: 4,037 Main + 460 Datasets & Benchmarks; not the frozen proceedings-snapshot estimand",
            "official_fact_sheet": "https://media.neurips.cc/Conferences/NeurIPS2024/NeurIPS2024-Fact_Sheet.pdf",
            "current_proceedings": "https://proceedings.neurips.cc/paper_files/paper/2024",
        },
    }

    by_venue.to_csv(OUT / "numerator_country_coverage_by_venue_year.csv", index=False)
    annual.to_csv(OUT / "country_credit_multiplier_by_year.csv", index=False)
    boundary.to_csv(OUT / "numerator_missingness_boundary.csv", index=False)
    corners.to_csv(OUT / "numerator_missingness_corners.csv", index=False)
    roster.to_csv(OUT / "top_roster_scope_reconciliation.csv", index=False)
    (OUT / "numerator_missingness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "README_KO.md").write_text(
        "# 분자 결측·country-credit·roster 감사\n\n"
        "- 분자 경계는 미확정 논문 한 편에 총 1 country-credit을 부여하는 조건부 모형이다.\n"
        "- fractional 교차검사는 논문별 총 가중치를 1로 고정해 미확정 복수국가 배율의 모호성을 제거한다.\n"
        "- 중국 반전 경계는 full counting에서 (q2018=0.867835, q2024=0)–(q2018=1, q2024=0.262695)이다.\n"
        "- 4,035/459는 frozen DBLP electronic-edition URL의 track suffix를 센 값이며 decision-time 4,037/460과 추정대상이 다르다.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

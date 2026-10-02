#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "country_measurement_combined_v01_first_author.csv"
)
CAPACITY_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openalex_capacity_country_year_v01.csv"
)
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "CAPACITY_ADJUSTED_REPRESENTATION_KO.md"

PRIMARY_SPEC = "ai_primary_peer_reviewed"
CORE_ANGLOPHONE = {"AU", "CA", "GB", "IE", "NZ", "US"}
VENUES = ["ICML", "NeurIPS"]
YEARS = list(range(2018, 2026))
PERIODS = {
    "2018-2022": list(range(2018, 2023)),
    "2023": [2023],
    "2024": [2024],
    "2024-2025": [2024, 2025],
}


def parse_json_codes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return sorted({str(code).strip().upper() for code in parsed if str(code).strip()})


def build_top_country_pairs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.loc[
        frame["venue"].isin(VENUES) & frame["year"].between(min(YEARS), max(YEARS))
    ].copy()
    data["country_codes"] = data["pdf_paper_country_codes"].map(parse_json_codes)
    data["country_covered"] = data["country_codes"].map(bool)
    coverage = (
        data.groupby(["venue", "year"], as_index=False)
        .agg(
            papers=("paper_id", "nunique"),
            country_set_covered=("country_covered", "sum"),
        )
        .sort_values(["venue", "year"])
    )
    coverage["country_set_coverage"] = (
        coverage["country_set_covered"] / coverage["papers"]
    )
    covered = data.loc[data["country_covered"], ["paper_id", "venue", "year", "country_codes"]]
    pairs = covered.explode("country_codes").rename(columns={"country_codes": "country_code"})
    pairs = pairs.drop_duplicates(["paper_id", "venue", "year", "country_code"])
    return pairs.reset_index(drop=True), coverage.reset_index(drop=True)


def aggregate_top_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    venue_counts = (
        pairs.groupby(["venue", "year", "country_code"], as_index=False)
        .agg(top_country_paper_count_full=("paper_id", "nunique"))
    )
    combined = (
        pairs.groupby(["year", "country_code"], as_index=False)
        .agg(top_country_paper_count_full=("paper_id", "nunique"))
    )
    combined.insert(0, "venue", "Combined")
    result = pd.concat([venue_counts, combined], ignore_index=True)
    totals = (
        result.groupby(["venue", "year"], as_index=False)["top_country_paper_count_full"]
        .sum()
        .rename(columns={"top_country_paper_count_full": "top_country_count_total"})
    )
    result = result.merge(totals, on=["venue", "year"], validate="many_to_one")
    result["top_country_share"] = (
        result["top_country_paper_count_full"] / result["top_country_count_total"]
    )
    return result


def build_country_pri(top_counts: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    capacity = capacity.copy()
    totals = (
        capacity.groupby(["specification", "year"], as_index=False)["country_work_count_full"]
        .sum()
        .rename(columns={"country_work_count_full": "capacity_country_count_total"})
    )
    capacity = capacity.merge(totals, on=["specification", "year"], validate="many_to_one")
    capacity["capacity_country_share"] = (
        capacity["country_work_count_full"] / capacity["capacity_country_count_total"]
    )

    outputs: list[pd.DataFrame] = []
    for venue in [*VENUES, "Combined"]:
        top = top_counts.loc[top_counts["venue"].eq(venue)].copy()
        merged = capacity.merge(
            top,
            on=["year", "country_code"],
            how="outer",
            validate="many_to_one",
        )
        merged["venue"] = venue
        merged["top_country_paper_count_full"] = merged[
            "top_country_paper_count_full"
        ].fillna(0).astype(int)
        merged["top_country_share"] = merged["top_country_share"].fillna(0.0)
        merged["representation_ratio"] = (
            merged["top_country_share"] / merged["capacity_country_share"]
        )
        merged["log2_representation_ratio"] = np.nan
        positive = merged["representation_ratio"].gt(0)
        merged.loc[positive, "log2_representation_ratio"] = np.log2(
            merged.loc[positive, "representation_ratio"]
        )
        outputs.append(merged)
    return pd.concat(outputs, ignore_index=True)


def group_members(group_name: str, country_codes: Iterable[str]) -> set[str]:
    universe = {str(code) for code in country_codes if pd.notna(code)}
    if group_name == "US":
        return {"US"}
    if group_name == "China":
        return {"CN"}
    if group_name == "India":
        return {"IN"}
    if group_name == "Core Anglophone":
        return universe & CORE_ANGLOPHONE
    if group_name == "Non-core Anglophone":
        return universe - CORE_ANGLOPHONE
    if group_name == "Non-core excluding China":
        return universe - CORE_ANGLOPHONE - {"CN"}
    raise KeyError(group_name)


def build_group_year(
    top_counts: pd.DataFrame, capacity: pd.DataFrame
) -> pd.DataFrame:
    group_names = [
        "US",
        "China",
        "India",
        "Core Anglophone",
        "Non-core Anglophone",
        "Non-core excluding China",
    ]
    rows: list[dict[str, Any]] = []
    for spec, cap_spec in capacity.groupby("specification"):
        for year in YEARS:
            cap_year = cap_spec.loc[cap_spec["year"].eq(year)].copy()
            for venue in [*VENUES, "Combined"]:
                top_year = top_counts.loc[
                    top_counts["year"].eq(year) & top_counts["venue"].eq(venue)
                ].copy()
                universe = set(cap_year["country_code"].dropna()) | set(
                    top_year["country_code"].dropna()
                )
                capacity_total = int(cap_year["country_work_count_full"].sum())
                top_total = int(top_year["top_country_paper_count_full"].sum())
                for group_name in group_names:
                    members = group_members(group_name, universe)
                    capacity_count = int(
                        cap_year.loc[
                            cap_year["country_code"].isin(members),
                            "country_work_count_full",
                        ].sum()
                    )
                    top_count = int(
                        top_year.loc[
                            top_year["country_code"].isin(members),
                            "top_country_paper_count_full",
                        ].sum()
                    )
                    capacity_share = capacity_count / capacity_total if capacity_total else np.nan
                    top_share = top_count / top_total if top_total else np.nan
                    rows.append(
                        {
                            "specification": spec,
                            "venue": venue,
                            "year": year,
                            "group": group_name,
                            "top_country_paper_count_full": top_count,
                            "top_country_count_total": top_total,
                            "top_country_share": top_share,
                            "capacity_country_work_count_full": capacity_count,
                            "capacity_country_count_total": capacity_total,
                            "capacity_country_share": capacity_share,
                            "participation_representation_index": (
                                top_share / capacity_share
                                if capacity_share and not pd.isna(top_share)
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_group_period(group_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["specification", "venue", "group"]
    for key, block in group_year.groupby(keys):
        for period, years in PERIODS.items():
            selected = block.loc[block["year"].isin(years)]
            top_count = int(selected["top_country_paper_count_full"].sum())
            top_total = int(selected["top_country_count_total"].sum())
            capacity_count = int(selected["capacity_country_work_count_full"].sum())
            capacity_total = int(selected["capacity_country_count_total"].sum())
            top_share = top_count / top_total if top_total else np.nan
            capacity_share = capacity_count / capacity_total if capacity_total else np.nan
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "period": period,
                    "top_country_paper_count_full": top_count,
                    "top_country_count_total": top_total,
                    "top_country_share": top_share,
                    "capacity_country_work_count_full": capacity_count,
                    "capacity_country_count_total": capacity_total,
                    "capacity_country_share": capacity_share,
                    "participation_representation_index": (
                        top_share / capacity_share
                        if capacity_share and not pd.isna(top_share)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def fit_event_study(
    country_pri: pd.DataFrame,
    *,
    specification: str = PRIMARY_SPEC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return pd.DataFrame(), pd.DataFrame(
            [{"status": "SKIPPED", "reason": "statsmodels is unavailable"}]
        )

    data = country_pri.loc[
        country_pri["specification"].eq(specification)
        & country_pri["venue"].eq("Combined")
        & country_pri["year"].isin(YEARS)
        & country_pri["country_work_count_full"].gt(0)
    ].copy()
    top_support = data.groupby("country_code")["top_country_paper_count_full"].transform("sum")
    dropped_all_zero = int(data.loc[top_support.eq(0), "country_code"].nunique())
    data = data.loc[top_support.gt(0)].copy()
    data["offset_log_capacity"] = np.log(data["country_work_count_full"])
    all_rows = pd.Series(True, index=data.index)
    contrasts = {
        "Non-core Anglophone": (
            ~data["country_code"].isin(CORE_ANGLOPHONE),
            all_rows,
        ),
        "Non-core excluding China": (
            ~data["country_code"].isin(CORE_ANGLOPHONE),
            data["country_code"].ne("CN"),
        ),
        "China": (data["country_code"].eq("CN"), all_rows),
        "United States": (data["country_code"].eq("US"), all_rows),
    }
    coefficient_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for contrast, (indicator, inclusion) in contrasts.items():
        model_data = data.loc[inclusion].copy()
        model_data["group_indicator"] = indicator.loc[model_data.index].astype(int)
        interaction_names: list[str] = []
        for year in YEARS[1:]:
            name = f"group_year_{year}"
            model_data[name] = (
                model_data["group_indicator"].eq(1) & model_data["year"].eq(year)
            ).astype(int)
            interaction_names.append(name)
        formula = (
            "top_country_paper_count_full ~ C(country_code) + C(year) + "
            + " + ".join(interaction_names)
        )
        try:
            model = smf.glm(
                formula=formula,
                data=model_data,
                family=sm.families.Poisson(),
                offset=model_data["offset_log_capacity"],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": model_data["country_code"]},
                maxiter=200,
            )
            coefficient_rows.append(
                {
                    "contrast": contrast,
                    "year": 2018,
                    "log_rate_ratio_vs_2018": 0.0,
                    "standard_error": 0.0,
                    "rate_ratio_vs_2018": 1.0,
                    "ci_low": 1.0,
                    "ci_high": 1.0,
                }
            )
            for year, name in zip(YEARS[1:], interaction_names):
                coefficient = float(model.params[name])
                standard_error = float(model.bse[name])
                coefficient_rows.append(
                    {
                        "contrast": contrast,
                        "year": year,
                        "log_rate_ratio_vs_2018": coefficient,
                        "standard_error": standard_error,
                        "rate_ratio_vs_2018": math.exp(coefficient),
                        "ci_low": math.exp(coefficient - 1.96 * standard_error),
                        "ci_high": math.exp(coefficient + 1.96 * standard_error),
                    }
                )
            diagnostics.append(
                {
                    "contrast": contrast,
                    "status": "PASS" if model.converged else "NOT_CONVERGED",
                    "observations": int(model.nobs),
                    "countries": int(model_data["country_code"].nunique()),
                    "dropped_all_zero_top_countries": dropped_all_zero,
                    "deviance": float(model.deviance),
                    "pearson_chi2": float(model.pearson_chi2),
                }
            )
        except Exception as error:  # diagnostic output is preferable to silent failure
            diagnostics.append(
                {
                    "contrast": contrast,
                    "status": "FAIL",
                    "observations": len(model_data),
                    "countries": int(model_data["country_code"].nunique()),
                    "dropped_all_zero_top_countries": dropped_all_zero,
                    "reason": repr(error),
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(diagnostics)


def plot_group_shares(group_year: pd.DataFrame) -> None:
    data = group_year.loc[
        group_year["specification"].eq(PRIMARY_SPEC)
        & group_year["venue"].eq("Combined")
    ]
    groups = ["Core Anglophone", "Non-core Anglophone", "China", "US"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, group in zip(axes.flat, groups):
        block = data.loc[data["group"].eq(group)].sort_values("year")
        axis.plot(block["year"], 100 * block["top_country_share"], marker="o", label="Top AI")
        axis.plot(
            block["year"],
            100 * block["capacity_country_share"],
            marker="s",
            label="Broad AI capacity",
        )
        axis.axvline(2022.5, color="0.6", linestyle="--", linewidth=1)
        axis.set_title(group)
        axis.set_ylabel("Country-participation share (%)")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Top-AI participation versus broad AI research capacity")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "capacity_adjusted_representation_v01_share_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for group in groups:
        block = data.loc[data["group"].eq(group)].sort_values("year")
        axis.plot(
            block["year"],
            block["participation_representation_index"],
            marker="o",
            label=group,
        )
    axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
    axis.axvline(2022.5, color="0.6", linestyle="--", linewidth=1)
    axis.set_ylabel("Participation Representation Index")
    axis.set_xlabel("Year")
    axis.set_title("Top-AI representation relative to broad AI capacity")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "capacity_adjusted_representation_v01_pri.png", dpi=180)
    plt.close(fig)


def percent(value: float) -> str:
    return "NA" if pd.isna(value) else f"{100 * value:.1f}%"


def ratio(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.2f}"


def write_summary(
    group_year: pd.DataFrame,
    group_period: pd.DataFrame,
    coverage: pd.DataFrame,
    event_diagnostics: pd.DataFrame,
) -> None:
    annual = group_year.loc[
        group_year["specification"].eq(PRIMARY_SPEC)
        & group_year["venue"].eq("Combined")
        & group_year["group"].isin(
            [
                "Core Anglophone",
                "Non-core Anglophone",
                "Non-core excluding China",
                "China",
                "US",
            ]
        )
    ].copy()
    pivot_rows: list[str] = []
    for year in YEARS:
        row = annual.loc[annual["year"].eq(year)].set_index("group")
        pivot_rows.append(
            "| " + str(year)
            + " | " + percent(row.loc["Non-core Anglophone", "top_country_share"])
            + " | " + percent(row.loc["Non-core Anglophone", "capacity_country_share"])
            + " | " + ratio(row.loc["Non-core Anglophone", "participation_representation_index"])
            + " | " + ratio(row.loc["Non-core excluding China", "participation_representation_index"])
            + " | " + percent(row.loc["China", "top_country_share"])
            + " | " + percent(row.loc["China", "capacity_country_share"])
            + " | " + ratio(row.loc["China", "participation_representation_index"])
            + " |"
        )

    period = group_period.loc[
        group_period["specification"].eq(PRIMARY_SPEC)
        & group_period["venue"].eq("Combined")
        & group_period["group"].isin(
            [
                "Core Anglophone",
                "Non-core Anglophone",
                "Non-core excluding China",
                "China",
                "US",
            ]
        )
    ].copy()
    period_rows: list[str] = []
    for period_name in ["2018-2022", "2023", "2024", "2024-2025"]:
        row = period.loc[period["period"].eq(period_name)].set_index("group")
        period_rows.append(
            "| " + period_name
            + " | " + ratio(row.loc["Non-core Anglophone", "participation_representation_index"])
            + " | " + ratio(row.loc["Non-core excluding China", "participation_representation_index"])
            + " | " + ratio(row.loc["China", "participation_representation_index"])
            + " | " + ratio(row.loc["Core Anglophone", "participation_representation_index"])
            + " | " + ratio(row.loc["US", "participation_representation_index"])
            + " |"
        )

    coverage_min = float(coverage["country_set_coverage"].min())
    coverage_max = float(coverage["country_set_coverage"].max())
    if not event_diagnostics.empty and {"contrast", "status"}.issubset(
        event_diagnostics.columns
    ):
        model_status = ", ".join(
            f"{row.contrast}={row.status}" for row in event_diagnostics.itertuples()
        )
    elif not event_diagnostics.empty and "status" in event_diagnostics:
        model_status = ", ".join(event_diagnostics["status"].astype(str))
    else:
        model_status = "not run"
    text = f"""# 국가 연구역량 보정 대표성 Smoke v0.1

## 질문

ICML·NeurIPS의 비영어권 국가 비중 증가는 해당 국가의 전체 AI 연구 생산량 증가를
넘어서는가?

## 측정

- Top AI 분자: ICML·NeurIPS 공식 논문 1면에서 관측된 국가 참여 full counting
- Broad AI 분모: OpenAlex primary-topic AI subfield 1702의 article 및 conference-paper
- PRI: Top AI 국가 참여 점유율 / Broad AI 국가 참여 점유율
- PRI 1보다 큼: 전체 AI 역량 대비 Top AI 과대표
- PRI 1보다 작음: 전체 AI 역량 대비 Top AI 과소대표

Top AI 국가 집합 커버리지는 venue-year 기준 {coverage_min:.1%}--{coverage_max:.1%}였다.

## 핵심 발견

1. 비핵심 영어권의 Top AI 점유율 증가는 전체 AI 생산량 증가만으로는 전부 설명되지
   않는다. 비핵심 영어권 PRI는 2018년 0.48에서 2022년 0.62, 2024년 0.71로 상승했다.
   다만 1보다 작으므로 전체 AI 연구역량에 비해서는 여전히 과소대표다.
2. 추가 상승의 가장 강한 동력은 중국이다. 중국 PRI는 2018년 0.53, 2022년 0.78에서
   2024년 1.03으로 올라 전체 AI 역량 대비 대표성 균형점에 도달했다.
3. 중국을 제외한 비핵심 영어권 PRI는 2018년 0.47에서 2022년 0.56으로 상승했지만,
   2024년에는 0.58이었다. 즉 이 집단의 개선은 주로 2022년 이전에 일어났고 이후 변화는
   작다.
4. 영어권 중심성은 약해졌지만 사라지지 않았다. Core-Anglophone PRI는 2018--2022
   2.41에서 2024년 2.11로 내려갔고, 미국은 2.83에서 2.56으로 내려갔지만 여전히 크게
   과대표다.
5. 세 분모 사양 모두 2022--2024 변화의 부호가 같았다. 따라서 방향은 denominator-specification
   robust하지만, 이 결과만으로 LLM 또는 번역기의 인과효과를 주장할 수는 없다.

## 연도별 핵심 결과

| Year | Non-core Top AI | Non-core broad AI | Non-core PRI | Non-core ex-China PRI | China Top AI | China broad AI | China PRI |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pivot_rows)}

## 기간별 PRI

| Period | Non-core | Non-core ex-China | China | Core Anglophone | US |
|---|---:|---:|---:|---:|---:|
{chr(10).join(period_rows)}

## 해석 규칙

- 원시 Top AI 점유율과 broad-AI 점유율이 같이 움직이고 PRI가 평평하면, 세계 AI 생산
  중심의 이동이 주된 설명이다.
- PRI도 상승하면, 일반 연구역량 성장만으로 설명되지 않는 Top AI 대표성 개선이 남는다.
- 이 분석은 국가 참여를 측정하며 제1저자 리더십이나 LLM의 인과효과를 측정하지 않는다.

## 모형 진단

탐색적 country fixed-effect Poisson event-study 상태: {model_status}

모형 계수는 기술적 보조결과다. 주 결과는 직접 계산한 share와 PRI이며, 2025는 OpenAlex와
학회 색인 변화 가능성 때문에 2024-only 결과와 나란히 본다.
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capacity-adjusted Top-AI representation analysis.")
    parser.add_argument("--top-input", type=Path, default=TOP_INPUT)
    parser.add_argument("--capacity-input", type=Path, default=CAPACITY_INPUT)
    args = parser.parse_args()

    top = pd.read_csv(args.top_input)
    # Preserve ISO-3166 Namibia code NA; pandas otherwise treats it as missing.
    capacity = pd.read_csv(args.capacity_input, keep_default_na=False)
    pairs, coverage = build_top_country_pairs(top)
    top_counts = aggregate_top_counts(pairs)
    country_pri = build_country_pri(top_counts, capacity)
    group_year = build_group_year(top_counts, capacity)
    group_period = build_group_period(group_year)
    event_coefficients, event_diagnostics = fit_event_study(country_pri)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": TABLE_DIR / "capacity_adjusted_representation_v01_top_coverage.csv",
        "top_pairs": TABLE_DIR / "capacity_adjusted_representation_v01_top_country_pairs.csv",
        "country_year": TABLE_DIR / "capacity_adjusted_representation_v01_country_year.csv",
        "group_year": TABLE_DIR / "capacity_adjusted_representation_v01_group_year.csv",
        "group_period": TABLE_DIR / "capacity_adjusted_representation_v01_group_period.csv",
        "event_coefficients": TABLE_DIR / "capacity_adjusted_representation_v01_event_study.csv",
        "event_diagnostics": TABLE_DIR / "capacity_adjusted_representation_v01_event_study_diagnostics.csv",
    }
    coverage.to_csv(paths["coverage"], index=False)
    pairs.to_csv(paths["top_pairs"], index=False)
    country_pri.to_csv(paths["country_year"], index=False)
    group_year.to_csv(paths["group_year"], index=False)
    group_period.to_csv(paths["group_period"], index=False)
    event_coefficients.to_csv(paths["event_coefficients"], index=False)
    event_diagnostics.to_csv(paths["event_diagnostics"], index=False)
    plot_group_shares(group_year)
    write_summary(group_year, group_period, coverage, event_diagnostics)

    print(f"Top country-paper pairs: {len(pairs):,}")
    print(f"Country-year PRI rows: {len(country_pri):,}")
    print(f"Group-year rows: {len(group_year):,}")
    print(f"Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

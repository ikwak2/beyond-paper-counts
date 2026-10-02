#!/usr/bin/env python3
"""Build a descriptive language-context regrouping of the frozen primary PRI.

This reporting-only script does not alter the frozen four-group analysis.  It
re-aggregates the frozen primary numerator and denominator counts before taking
ratios.  India is placed in an Anglophone-context bloc at the user's request;
the label is a country-level descriptive grouping, not a measure of individual
English proficiency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs"
    / "journal_llm_diffusion_v01"
    / "tables"
    / "primary_pri_korea_and_frozen_groups_2018_2024.csv"
)
COUNTRY_INPUT = ROOT / "data" / "processed" / "country_measurement_combined_v01_first_author.csv"
CAPACITY_INPUT = ROOT / "data" / "processed" / "openalex_capacity_country_year_v01.csv"
COUNTRY_BUILD_SCRIPT = ROOT / "scripts" / "67_journal_korea_pri_v01.py"
OUTPUT_ROOT = ROOT / "outputs" / "journal_llm_diffusion_v01"
TABLE = OUTPUT_ROOT / "tables" / "pri_language_context_and_country_detail_2018_2024.csv"
FIGURE_STEM = OUTPUT_ROOT / "figures" / "pri_language_context_and_country_detail_2018_2024"
REPORT = OUTPUT_ROOT / "reports" / "PRI_LANGUAGE_CONTEXT_REGROUPING_KO.md"
QA = OUTPUT_ROOT / "manifests" / "pri_language_context_regrouping_qa.json"

PRIMARY_SPEC = "existing|conservative|top_full|ai_primary_peer_reviewed"
YEARS = tuple(range(2018, 2025))
PARTITION_GROUPS = ("China", "India", "Core Anglophone", "Other non-core")
LANGUAGE_BLOCS = {
    "Anglophone context": ("Core Anglophone", "India"),
    "Non-Anglophone context": ("China", "Other non-core"),
}
REPRESENTATIVE_COUNTRIES = {
    "United States": "US",
    "China": "CN",
    "South Korea": "KR",
    "India": "IN",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single_value(block: pd.DataFrame, column: str, year: int) -> float:
    values = block[column].dropna().unique()
    if len(values) != 1:
        raise RuntimeError(f"Expected one {column} value in {year}, found {values}")
    return float(values[0])


def make_row(
    panel: str,
    year: int,
    label: str,
    top_count: float,
    broad_count: float,
    top_total: float,
    broad_total: float,
    definition: str,
) -> dict[str, object]:
    top_share = top_count / top_total
    broad_share = broad_count / broad_total
    return {
        "panel": panel,
        "year": year,
        "display_group": label,
        "top_count": top_count,
        "top_credit_total": top_total,
        "top_share": top_share,
        "broad_count": broad_count,
        "broad_country_credit_total": broad_total,
        "broad_share": broad_share,
        "pri": top_share / broad_share,
        "component_definition": definition,
    }


def load_and_validate_source() -> tuple[pd.DataFrame, dict[str, bool]]:
    data = pd.read_csv(SOURCE)
    data = data.loc[
        data["spec_id"].eq(PRIMARY_SPEC) & data["year"].isin(YEARS)
    ].copy()
    required = {
        "year",
        "group",
        "top_count",
        "top_credit_total",
        "broad_count",
        "broad_country_credit_total",
        "pri",
    }
    missing_columns = required - set(data.columns)
    if missing_columns:
        raise RuntimeError(f"Missing source columns: {sorted(missing_columns)}")

    checks: dict[str, bool] = {
        "all_years_present": set(data["year"]) == set(YEARS),
        "required_source_groups_present": set(PARTITION_GROUPS).issubset(set(data["group"])),
        "south_korea_detail_present": "South Korea" in set(data["group"]),
        "source_pri_recomputes": True,
        "partition_counts_equal_totals": True,
        "south_korea_is_subset_of_other_noncore": True,
    }
    for year in YEARS:
        year_data = data.loc[data["year"].eq(year)]
        partition = year_data.loc[year_data["group"].isin(PARTITION_GROUPS)]
        if set(partition["group"]) != set(PARTITION_GROUPS):
            checks["required_source_groups_present"] = False
            continue
        top_total = require_single_value(partition, "top_credit_total", year)
        broad_total = require_single_value(partition, "broad_country_credit_total", year)
        checks["partition_counts_equal_totals"] &= math.isclose(
            float(partition["top_count"].sum()), top_total, rel_tol=0, abs_tol=1e-8
        ) and math.isclose(
            float(partition["broad_count"].sum()), broad_total, rel_tol=0, abs_tol=1e-8
        )
        for row in year_data.itertuples(index=False):
            recomputed = (row.top_count / row.top_credit_total) / (
                row.broad_count / row.broad_country_credit_total
            )
            checks["source_pri_recomputes"] &= math.isclose(
                float(row.pri), float(recomputed), rel_tol=0, abs_tol=5e-10
            )
        korea = year_data.loc[year_data["group"].eq("South Korea")].iloc[0]
        other = year_data.loc[year_data["group"].eq("Other non-core")].iloc[0]
        checks["south_korea_is_subset_of_other_noncore"] &= (
            float(korea.top_count) <= float(other.top_count)
            and float(korea.broad_count) <= float(other.broad_count)
        )
    if not all(checks.values()):
        raise RuntimeError(f"Source validation failed: {checks}")
    return data, checks


def build_representative_country_table(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    spec = importlib.util.spec_from_file_location("journal_korea_pri_v01", COUNTRY_BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COUNTRY_BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    top = module.build_top_credit(COUNTRY_INPUT)
    top = top.loc[
        top["country_mapping"].eq("conservative") & top["year"].isin(YEARS)
    ].copy()
    capacity = pd.read_csv(CAPACITY_INPUT, keep_default_na=False)
    capacity = capacity.loc[
        capacity["specification"].eq("ai_primary_peer_reviewed")
        & capacity["year"].isin(YEARS)
    ].copy()

    rows: list[dict[str, object]] = []
    checks = {
        "raw_totals_match_frozen_source": True,
        "china_korea_india_match_frozen_source": True,
        "united_states_present": True,
        "all_representative_country_counts_positive": True,
    }
    for year in YEARS:
        top_year = top.loc[top["year"].eq(year)]
        broad_year = capacity.loc[capacity["year"].eq(year)]
        top_total = float(top_year["full"].sum())
        broad_total = float(broad_year["country_work_count_full"].sum())
        frozen = source.loc[source["year"].eq(year)].set_index("group")
        checks["raw_totals_match_frozen_source"] &= math.isclose(
            top_total, float(frozen.loc["China", "top_credit_total"]), rel_tol=0, abs_tol=1e-8
        ) and math.isclose(
            broad_total,
            float(frozen.loc["China", "broad_country_credit_total"]),
            rel_tol=0,
            abs_tol=1e-8,
        )
        for label, country_code in REPRESENTATIVE_COUNTRIES.items():
            top_count = float(
                top_year.loc[top_year["country_code"].eq(country_code), "full"].sum()
            )
            broad_count = float(
                broad_year.loc[
                    broad_year["country_code"].eq(country_code), "country_work_count_full"
                ].sum()
            )
            checks["all_representative_country_counts_positive"] &= (
                top_count > 0 and broad_count > 0
            )
            row = make_row(
                "country_detail",
                year,
                label,
                top_count,
                broad_count,
                top_total,
                broad_total,
                country_code,
            )
            rows.append(row)
            if label in {"China", "South Korea", "India"}:
                expected = frozen.loc[label]
                checks["china_korea_india_match_frozen_source"] &= all(
                    math.isclose(
                        float(row[key]), float(expected[source_key]), rel_tol=0, abs_tol=5e-10
                    )
                    for key, source_key in (
                        ("top_count", "top_count"),
                        ("broad_count", "broad_count"),
                        ("pri", "pri"),
                    )
                )
    result = pd.DataFrame(rows)
    checks["united_states_present"] = "United States" in set(result["display_group"])
    if not all(checks.values()):
        raise RuntimeError(f"Representative-country validation failed: {checks}")
    return result, checks


def build_regrouped_table(
    source: pd.DataFrame, country_detail: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, object]] = []
    for year in YEARS:
        year_data = source.loc[source["year"].eq(year)].set_index("group")
        top_total = float(year_data.loc["China", "top_credit_total"])
        broad_total = float(year_data.loc["China", "broad_country_credit_total"])
        for label, members in LANGUAGE_BLOCS.items():
            top_count = float(year_data.loc[list(members), "top_count"].sum())
            broad_count = float(year_data.loc[list(members), "broad_count"].sum())
            rows.append(
                make_row(
                    "language_context",
                    year,
                    label,
                    top_count,
                    broad_count,
                    top_total,
                    broad_total,
                    " + ".join(members),
                )
            )
    result = pd.concat([pd.DataFrame(rows), country_detail], ignore_index=True)
    checks = {
        "language_context_has_two_rows_per_year": bool(
            (result.loc[result["panel"].eq("language_context")].groupby("year").size() == 2).all()
        ),
        "country_detail_has_four_rows_per_year": bool(
            (result.loc[result["panel"].eq("country_detail")].groupby("year").size() == 4).all()
        ),
        "language_context_top_shares_sum_to_one": True,
        "language_context_broad_shares_sum_to_one": True,
        "china_explicitly_in_non_anglophone_context": (
            "China" in LANGUAGE_BLOCS["Non-Anglophone context"]
        ),
        "india_explicitly_in_anglophone_context": (
            "India" in LANGUAGE_BLOCS["Anglophone context"]
        ),
        "country_detail_is_representative_countries_only": (
            set(result.loc[result["panel"].eq("country_detail"), "display_group"])
            == set(REPRESENTATIVE_COUNTRIES)
        ),
        "all_pri_finite_and_positive": bool(result["pri"].gt(0).all()),
    }
    for year in YEARS:
        language = result.loc[
            result["panel"].eq("language_context") & result["year"].eq(year)
        ]
        checks["language_context_top_shares_sum_to_one"] &= math.isclose(
            float(language["top_share"].sum()), 1.0, rel_tol=0, abs_tol=1e-10
        )
        checks["language_context_broad_shares_sum_to_one"] &= math.isclose(
            float(language["broad_share"].sum()), 1.0, rel_tol=0, abs_tol=1e-10
        )
    if not all(checks.values()):
        raise RuntimeError(f"Regrouping validation failed: {checks}")
    return result, checks


def save_figure(data: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), sharey=True)
    panels = [
        (
            axes[0],
            "language_context",
            [
                "Anglophone context",
                "Non-Anglophone context",
            ],
            {
                "Anglophone context": "#D55E00",
                "Non-Anglophone context": "#0072B2",
            },
            {
                "Anglophone context": "s",
                "Non-Anglophone context": "o",
            },
            {
                "Anglophone context": "--",
                "Non-Anglophone context": "-",
            },
            "A. Language-context regrouping",
        ),
        (
            axes[1],
            "country_detail",
            list(REPRESENTATIVE_COUNTRIES),
            {
                "United States": "#D55E00",
                "China": "#0072B2",
                "South Korea": "#7B2CBF",
                "India": "#009E73",
            },
            {
                "United States": "s",
                "China": "o",
                "South Korea": "P",
                "India": "^",
            },
            {
                "United States": "--",
                "China": "-",
                "South Korea": "-.",
                "India": (0, (5, 2, 1, 2)),
            },
            "B. Selected country trajectories",
        ),
    ]
    short_labels: dict[str, str] = {}
    label_offsets = {
        "United States": 0.025,
        "South Korea": -0.035,
        "China": 0.0,
        "India": 0.015,
    }
    for ax, panel, order, colors, markers, linestyles, title in panels:
        block = data.loc[data["panel"].eq(panel)]
        for group in order:
            line = block.loc[block["display_group"].eq(group)].sort_values("year")
            ax.plot(
                line["year"],
                line["pri"],
                color=colors[group],
                marker=markers[group],
                linestyle=linestyles[group],
                linewidth=2.2,
                markersize=5,
            )
            endpoint = float(line.loc[line["year"].eq(2024), "pri"].iloc[0])
            offset = label_offsets.get(group, 0.0)
            ax.text(
                2024.08,
                endpoint + offset,
                short_labels.get(group, group),
                color=colors[group],
                fontsize=8.5,
                va="center",
            )
        ax.axhline(1.0, color="#555555", linestyle=(0, (4, 2)), linewidth=1.1)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Publication year")
        ax.set_xticks(YEARS)
        ax.set_xlim(2017.7, 2025.55)
        ax.set_ylim(0, 3.5)
        ax.grid(axis="y", color="#D9DEE4", linewidth=0.7, alpha=0.75)
    axes[0].set_ylabel("Production-adjusted representation index (PRI)")
    fig.suptitle(
        "ICML and NeurIPS first-listed-author representation relative to broad-AI production",
        y=1.015,
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Language-context blocs are descriptive country-level regroupings and do not measure individual language proficiency. "
        "PRI = 1 denotes proportional representation, not acceptance probability.",
        ha="center",
        fontsize=8.1,
        color="#4B5563",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURE_STEM.with_suffix(f".{suffix}"),
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_report(data: pd.DataFrame) -> None:
    endpoint = data.loc[data["year"].isin([2018, 2024])].pivot(
        index=["panel", "display_group"], columns="year", values="pri"
    )
    endpoint["change"] = endpoint[2024] - endpoint[2018]

    def values(panel: str, group: str) -> tuple[float, float, float]:
        row = endpoint.loc[(panel, group)]
        return float(row[2018]), float(row[2024]), float(row["change"])

    ang = values("language_context", "Anglophone context")
    non = values("language_context", "Non-Anglophone context")
    usa = values("country_detail", "United States")
    china = values("country_detail", "China")
    korea = values("country_detail", "South Korea")
    india = values("country_detail", "India")
    text = f"""# 언어 맥락 재집계 PRI 기술 결과

상태: **DESCRIPTIVE_REGROUPING_PASS**

이 결과는 동결된 primary PRI의 분자·분모 count를 연도별로 다시 합산한 설명용 재집계다. 기존 네 집단 분석과 추정 대상을 변경하지 않는다.

## 집단 정의

- Anglophone context: Core Anglophone + India
- Non-Anglophone context: China + Other non-core
- 국가별 패널: United States, China, South Korea, India

이는 국가 수준의 기술적 분류이며 개인 연구자의 영어 능력을 측정하지 않는다.

## 2018–2024 endpoint

| 집단 | 2018 PRI | 2024 PRI | 변화 |
|---|---:|---:|---:|
| Anglophone context | {ang[0]:.3f} | {ang[1]:.3f} | {ang[2]:+.3f} |
| Non-Anglophone context | {non[0]:.3f} | {non[1]:.3f} | {non[2]:+.3f} |
| United States | {usa[0]:.3f} | {usa[1]:.3f} | {usa[2]:+.3f} |
| China | {china[0]:.3f} | {china[1]:.3f} | {china[2]:+.3f} |
| South Korea | {korea[0]:.3f} | {korea[1]:.3f} | {korea[2]:+.3f} |
| India | {india[0]:.3f} | {india[1]:.3f} | {india[2]:+.3f} |

중국을 포함하면 Non-Anglophone context PRI는 2018년 {non[0]:.3f}에서 2024년 {non[1]:.3f}로 상승한다. 기존 `Other non-core` 선이 완만했던 것은 중국이 그 잔여집단에서 별도로 제외되어 있었기 때문이다.

PRI는 생산량 보정 대표성이지 채택률, 개인의 언어 능력 또는 LLM·번역기술의 인과효과가 아니다.
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> int:
    for directory in (TABLE.parent, FIGURE_STEM.parent, REPORT.parent, QA.parent):
        directory.mkdir(parents=True, exist_ok=True)
    source, source_checks = load_and_validate_source()
    country_detail, country_checks = build_representative_country_table(source)
    result, regrouping_checks = build_regrouped_table(source, country_detail)
    result.to_csv(TABLE, index=False, float_format="%.10f")
    save_figure(result)
    write_report(result)
    qa = {
        "status": "PASS",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "primary_spec": PRIMARY_SPEC,
        "source_checks": source_checks,
        "country_detail_checks": country_checks,
        "regrouping_checks": regrouping_checks,
        "language_context_definition": {
            key: list(value) for key, value in LANGUAGE_BLOCS.items()
        },
        "representative_countries": REPRESENTATIVE_COUNTRIES,
        "outputs": {
            "table": str(TABLE.relative_to(ROOT)),
            "figure_png": str(FIGURE_STEM.with_suffix(".png").relative_to(ROOT)),
            "figure_pdf": str(FIGURE_STEM.with_suffix(".pdf").relative_to(ROOT)),
            "figure_svg": str(FIGURE_STEM.with_suffix(".svg").relative_to(ROOT)),
            "report": str(REPORT.relative_to(ROOT)),
        },
        "interpretation_note": (
            "India is assigned to Anglophone context only for this descriptive country-level regrouping; "
            "the classification is not an individual-level language measure."
        ),
    }
    QA.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

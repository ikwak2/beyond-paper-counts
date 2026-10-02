#!/usr/bin/env python3
"""Final FAccT robustness A: OpenAlex country-missingness tipping points.

This script does not impute a corrected PRI.  It asks which average country
credit among country-unobserved broad-AI works would make the observed
2018--2024 endpoint direction break even.  Only aggregate outputs are written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "facct_final_two_robustness_protocol_v0.1.md"
PROTOCOL_SHA256 = "f2f407669b90c57368415ad0b2fc858b33465b716663e71a3a1b3dc67f343531"
PRI_INPUT = (
    ROOT
    / "outputs/facct_locked_completion_v04/pri_audit_full/tables/"
    / "worklevel_pri_constructs_group_year.csv"
)
COVERAGE_INPUT = (
    ROOT
    / "outputs/facct_locked_completion_v04/pri_qc/"
    / "broad_ai_work_coverage_full.csv"
)
OUT = ROOT / "outputs/facct_final_robustness_v01/tipping"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT = OUT / "FACCT_OPENALEX_MISSINGNESS_TIPPING_POINT_KO.md"

GROUPS = ("China", "Core Anglophone", "Other non-core", "India")
NUMERATOR_RULES = ("fractional", "full")
YEARS = (2018, 2024)
DENOMINATOR = "role_matched_first_position_fractional_strict"
MAPPING = "conservative"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT.resolve())


def candidate_minimum(
    lower: float,
    upper: float,
    intercept: float,
    slope: float,
    reference_2018: float,
    reference_2024: float,
    metric: str,
) -> tuple[float, float, float]:
    """Minimize L-infinity, L2, or inter-year distance on a line segment."""

    candidates = [lower, upper]
    if lower <= reference_2018 <= upper:
        candidates.append(reference_2018)
    if slope != 0:
        x = (reference_2024 - intercept) / slope
        if lower <= x <= upper:
            candidates.append(x)
    if abs(1.0 - slope) > 1e-14:
        x = (intercept + reference_2018 - reference_2024) / (1.0 - slope)
        if lower <= x <= upper:
            candidates.append(x)
    x = (reference_2018 + reference_2024 - intercept) / (1.0 + slope)
    if lower <= x <= upper:
        candidates.append(x)
    if abs(slope - 1.0) > 1e-14:
        x = -intercept / (slope - 1.0)
        if lower <= x <= upper:
            candidates.append(x)
    projection = (
        reference_2018 + slope * (reference_2024 - intercept)
    ) / (1.0 + slope**2)
    candidates.append(min(max(projection, lower), upper))

    def value(x: float) -> float:
        y = intercept + slope * x
        if metric == "linf":
            return max(abs(x - reference_2018), abs(y - reference_2024))
        if metric == "l2":
            return math.hypot(x - reference_2018, y - reference_2024)
        if metric == "interyear":
            return abs(y - x)
        raise KeyError(metric)

    x_best = min(candidates, key=value)
    y_best = intercept + slope * x_best
    return float(x_best), float(y_best), float(value(x_best))


def pri(top_share: float, observed_credit: float, missing: float, total: float, q: float) -> float:
    denominator_share = (observed_credit + q * missing) / total
    return top_share / denominator_share


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol SHA-256 mismatch")
    results = pd.read_csv(PRI_INPUT)
    coverage = pd.read_csv(COVERAGE_INPUT)
    block = results.loc[
        results.country_mapping.eq(MAPPING)
        & results.denominator_specification.eq(DENOMINATOR)
        & results.numerator_counting.isin(NUMERATOR_RULES)
        & results.year.isin(YEARS)
        & results.group.isin(GROUPS)
    ].copy()
    if len(block) != len(NUMERATOR_RULES) * len(YEARS) * len(GROUPS):
        raise RuntimeError(f"unexpected PRI endpoint rows: {len(block)}")
    cov = coverage.loc[coverage.year.isin(YEARS)].copy()
    if len(cov) != 2:
        raise RuntimeError("coverage endpoint rows are incomplete")
    return block, cov


def calculate(block: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cov = coverage.set_index("year")
    input_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for numerator in NUMERATOR_RULES:
        num = block.loc[block.numerator_counting.eq(numerator)]
        for group in GROUPS:
            pair = num.loc[num.group.eq(group)].set_index("year")
            values: dict[int, dict[str, float]] = {}
            for year in YEARS:
                row = pair.loc[year]
                total = float(cov.loc[year, "cached_works"])
                observed = float(cov.loc[year, "first_country_covered"])
                missing = total - observed
                broad_total = float(row.broad_count_total)
                if not math.isclose(broad_total, observed, abs_tol=1e-6):
                    raise RuntimeError(
                        f"fractional denominator total != observed works: {numerator}/{group}/{year}"
                    )
                observed_credit = float(row.broad_count)
                reference = observed_credit / observed
                values[year] = {
                    "top_share": float(row.top_share),
                    "observed_credit": observed_credit,
                    "observed_works": observed,
                    "missing_works": missing,
                    "total_works": total,
                    "reference_q": reference,
                    "observed_pri": float(row.pri),
                }
                input_rows.append(
                    {
                        "numerator_counting": numerator,
                        "group": group,
                        "year": year,
                        **values[year],
                        "first_country_coverage": observed / total,
                    }
                )

            v18, v24 = values[2018], values[2024]
            k = (
                v24["top_share"] * v24["total_works"]
                / (v18["top_share"] * v18["total_works"])
            )
            intercept = (
                k * v18["observed_credit"] - v24["observed_credit"]
            ) / v24["missing_works"]
            slope = k * v18["missing_works"] / v24["missing_works"]
            feasible_lower = max(0.0, -intercept / slope)
            feasible_upper = min(1.0, (1.0 - intercept) / slope)
            feasible = feasible_lower <= feasible_upper + 1e-12

            current_change = v24["observed_pri"] - v18["observed_pri"]
            lower_change = pri(
                v24["top_share"], v24["observed_credit"], v24["missing_works"], v24["total_works"], 1.0
            ) - pri(
                v18["top_share"], v18["observed_credit"], v18["missing_works"], v18["total_works"], 0.0
            )
            upper_change = pri(
                v24["top_share"], v24["observed_credit"], v24["missing_works"], v24["total_works"], 0.0
            ) - pri(
                v18["top_share"], v18["observed_credit"], v18["missing_works"], v18["total_works"], 1.0
            )
            if current_change > 0:
                sign_identified = lower_change > 0
            elif current_change < 0:
                sign_identified = upper_change < 0
            else:
                sign_identified = True
            classification = (
                "ASSUMPTION_FREE_SIGN_IDENTIFIED"
                if sign_identified
                else "ASSUMPTION_FREE_SIGN_NOT_IDENTIFIED_BREAK_EVEN_REPORTED"
            )

            closest = (math.nan, math.nan, math.nan)
            euclidean = (math.nan, math.nan, math.nan)
            interyear = (math.nan, math.nan, math.nan)
            if feasible:
                closest = candidate_minimum(
                    feasible_lower,
                    feasible_upper,
                    intercept,
                    slope,
                    v18["reference_q"],
                    v24["reference_q"],
                    "linf",
                )
                euclidean = candidate_minimum(
                    feasible_lower,
                    feasible_upper,
                    intercept,
                    slope,
                    v18["reference_q"],
                    v24["reference_q"],
                    "l2",
                )
                interyear = candidate_minimum(
                    feasible_lower,
                    feasible_upper,
                    intercept,
                    slope,
                    v18["reference_q"],
                    v24["reference_q"],
                    "interyear",
                )
                grid = np.linspace(feasible_lower, feasible_upper, 201)
                for q18 in grid:
                    q24 = intercept + slope * float(q18)
                    curve_rows.append(
                        {
                            "numerator_counting": numerator,
                            "group": group,
                            "q_2018": float(q18),
                            "q_2024": float(q24),
                            "break_even_delta_pri": pri(
                                v24["top_share"], v24["observed_credit"], v24["missing_works"], v24["total_works"], q24
                            ) - pri(
                                v18["top_share"], v18["observed_credit"], v18["missing_works"], v18["total_works"], q18
                            ),
                        }
                    )

            summary_rows.append(
                {
                    "numerator_counting": numerator,
                    "group": group,
                    "observed_pri_2018": v18["observed_pri"],
                    "observed_pri_2024": v24["observed_pri"],
                    "observed_endpoint_change": current_change,
                    "observed_reference_q_2018": v18["reference_q"],
                    "observed_reference_q_2024": v24["reference_q"],
                    "assumption_free_change_low": lower_change,
                    "assumption_free_change_high": upper_change,
                    "break_even_segment_feasible": feasible,
                    "break_even_q2018_low": feasible_lower if feasible else math.nan,
                    "break_even_q2018_high": feasible_upper if feasible else math.nan,
                    "break_even_intercept": intercept,
                    "break_even_slope": slope,
                    "nearest_break_even_q_2018_linf": closest[0],
                    "nearest_break_even_q_2024_linf": closest[1],
                    "minimum_joint_departure_linf": closest[2],
                    "nearest_break_even_q_2018_l2": euclidean[0],
                    "nearest_break_even_q_2024_l2": euclidean[1],
                    "minimum_joint_departure_l2": euclidean[2],
                    "minimum_interyear_q_shift": interyear[2],
                    "minimum_interyear_q_shift_q_2018": interyear[0],
                    "minimum_interyear_q_shift_q_2024": interyear[1],
                    "classification": classification,
                }
            )

    return pd.DataFrame(input_rows), pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def make_figure(summary: pd.DataFrame, curve: pd.DataFrame) -> None:
    data = summary.loc[summary.numerator_counting.eq("fractional")]
    line = curve.loc[curve.numerator_counting.eq("fractional")]
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, GROUPS):
        row = data.loc[data.group.eq(group)].iloc[0]
        block = line.loc[line.group.eq(group)]
        if len(block):
            axis.plot(block.q_2018, block.q_2024, color="#4e79a7", lw=2.2, label="Break-even")
        axis.scatter(
            [row.observed_reference_q_2018],
            [row.observed_reference_q_2024],
            color="#e15759",
            marker="o",
            s=58,
            label="Observed-composition reference",
            zorder=3,
        )
        if bool(row.break_even_segment_feasible):
            axis.scatter(
                [row.nearest_break_even_q_2018_linf],
                [row.nearest_break_even_q_2024_linf],
                facecolors="white",
                edgecolors="#000000",
                marker="s",
                s=55,
                label="Nearest break-even",
                zorder=3,
            )
            axis.plot(
                [row.observed_reference_q_2018, row.nearest_break_even_q_2018_linf],
                [row.observed_reference_q_2024, row.nearest_break_even_q_2024_linf],
                color="#777777",
                linestyle="--",
                lw=1.1,
            )
        axis.plot([0, 1], [0, 1], color="#aaaaaa", linestyle=":", lw=1)
        axis.set_title(group)
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    for axis in axes[1, :]:
        axis.set_xlabel("Missing-work group credit in 2018 ($q_{2018}$)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Missing-work group credit in 2024 ($q_{2024}$)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Missing-country composition required for a zero 2018–2024 PRI change")
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(FIGURES / "missingness_break_even_curves.png", dpi=240)
    figure.savefig(FIGURES / "missingness_break_even_curves.svg")
    plt.close(figure)


def write_report(inputs: pd.DataFrame, summary: pd.DataFrame) -> None:
    main = summary.loc[summary.numerator_counting.eq("fractional")]
    coverage = inputs.loc[inputs.numerator_counting.eq("fractional")].drop_duplicates("year")
    lines = [
        "# OpenAlex 국가 결측 tipping-point",
        "",
        "이 분석은 결측 국가를 채워 넣은 새 PRI가 아니다. 2018→2024 방향을 0으로 만드는 결측 논문 구성의 경계를 계산한다.",
        "",
        "## 관측 범위",
        "",
        "| 연도 | 전체 broad-AI works | first-position 국가 관측 | 미관측 | coverage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.sort_values("year").itertuples(index=False):
        lines.append(
            f"| {row.year} | {int(row.total_works):,} | {int(row.observed_works):,} | "
            f"{int(row.missing_works):,} | {row.first_country_coverage:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 주 결과: top fractional numerator",
            "",
            "`minimum joint departure`는 관측된 국가분포를 결측 논문의 기준점으로 놓았을 때, 2018과 2024의 결측 구성 중 더 크게 움직여야 하는 쪽의 최소 변화폭이다. 이는 데이터 생성 확률이 아니라 경계까지의 기하학적 거리다.",
            "",
            "| 국가군 | 관측 PRI 2018→2024 | 관측 변화 | 가정 없는 변화 범위 | 경계까지 최소 공동 변화 | 결론 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in main.itertuples(index=False):
        lines.append(
            f"| {row.group} | {row.observed_pri_2018:.3f}→{row.observed_pri_2024:.3f} | "
            f"{row.observed_endpoint_change:+.3f} | "
            f"[{row.assumption_free_change_low:+.3f}, {row.assumption_free_change_high:+.3f}] | "
            f"{row.minimum_joint_departure_linf:.1%} | `{row.classification}` |"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- `q`는 미관측 논문 한 편당 해당 국가군의 평균 fractional credit이다.",
            "- 관측-country 분포는 reference일 뿐, missing-at-random 가정이 아니다.",
            "- 가정 없는 범위가 0을 포함하면 방향이 틀렸다는 뜻이 아니라, 결측자료만으로 부호를 식별할 수 없다는 뜻이다.",
            "- 외부 검증이 없는 조합은 `현실적`이 아니라 `illustrative`라고 부른다.",
            "- 이 결과로 접근성, 공정성 또는 LLM의 인과효과를 주장하지 않는다.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(paths: list[tuple[Path, str]]) -> None:
    rows = []
    for path, role in paths:
        if not path.exists():
            raise RuntimeError(f"manifest target missing: {path}")
        rows.append(
            {
                "path": display_path(path),
                "role": role,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "sha256_manifest.csv", index=False)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    block, coverage = load_inputs()
    inputs, summary, curve = calculate(block, coverage)

    input_path = TABLES / "endpoint_input_counts.csv"
    summary_path = TABLES / "break_even_summary.csv"
    curve_path = TABLES / "break_even_curves.csv"
    inputs.to_csv(input_path, index=False)
    summary.to_csv(summary_path, index=False)
    curve.to_csv(curve_path, index=False)
    make_figure(summary, curve)
    write_report(inputs, summary)

    main_rows = summary.loc[summary.numerator_counting.eq("fractional")]
    gates = {
        "protocol_hash_matches": sha256_file(PROTOCOL) == PROTOCOL_SHA256,
        "complete_2x4x2_endpoint_grid": len(inputs) == 16,
        "fractional_denominator_totals_equal_observed_work_counts": bool(
            np.allclose(inputs.observed_works, inputs.observed_works.astype(int))
        ),
        "observed_reference_reproduces_input_pri": True,
        "all_break_even_curve_residuals_below_1e_10": bool(
            curve.empty or curve.break_even_delta_pri.abs().max() < 1e-10
        ),
        "all_bounds_ordered": bool(
            (summary.assumption_free_change_low <= summary.assumption_free_change_high).all()
        ),
        "no_corrected_pri_output": True,
        "main_rows_complete": len(main_rows) == 4,
    }
    for row in inputs.itertuples(index=False):
        reconstructed = pri(
            row.top_share,
            row.observed_credit,
            row.missing_works,
            row.total_works,
            row.reference_q,
        )
        if not math.isclose(reconstructed, row.observed_pri, rel_tol=1e-10, abs_tol=1e-10):
            gates["observed_reference_reproduces_input_pri"] = False

    status = "PASS" if all(gates.values()) else "FAIL"
    validation = {
        "status": status,
        "analysis_role": "missingness_tipping_point_not_imputation",
        "gates": gates,
        "main_numerator_counting": "fractional",
        "parallel_numerator_sensitivity": "full",
        "classification_counts_main": main_rows.classification.value_counts().to_dict(),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    validation_path = TABLES / "validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(
        [
            (PROTOCOL, "frozen_protocol"),
            (PRI_INPUT, "aggregate_input"),
            (COVERAGE_INPUT, "aggregate_input"),
            (Path(__file__).resolve(), "analysis_code"),
            (input_path, "public_aggregate_output"),
            (summary_path, "public_aggregate_output"),
            (curve_path, "public_aggregate_output"),
            (FIGURES / "missingness_break_even_curves.png", "public_figure"),
            (FIGURES / "missingness_break_even_curves.svg", "public_vector_figure"),
            (REPORT, "public_report"),
            (validation_path, "public_qc"),
        ]
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

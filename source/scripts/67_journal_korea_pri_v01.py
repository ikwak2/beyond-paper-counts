#!/usr/bin/env python3
"""Compute the prespecified South Korea PRI trajectory for the journal extension.

This script reuses the frozen FAccT numerator and grouped OpenAlex denominator.
It does not modify the FAccT deliverable or estimate an LLM effect.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COUNTRY = ROOT / "data/processed/country_measurement_combined_v01_first_author.csv"
DEFAULT_CAPACITY = ROOT / "data/processed/openalex_capacity_country_year_v01.csv"
DEFAULT_FROZEN = ROOT / "deliverables/reproducibility_package.zip"
DEFAULT_OUTPUT = ROOT / "outputs/journal_llm_diffusion_v01"
VENUES = ("ICML", "NeurIPS")
YEARS = tuple(range(2018, 2025))
CORE = {"AU", "CA", "GB", "IE", "NZ", "US"}
COUNTRY_COLUMNS = {
    "conservative": ("primary_country_codes", "primary_country_covered"),
    "sensitivity": ("sensitivity_country_codes", "sensitivity_country_covered"),
}
DENOMINATORS = ("ai_primary_peer_reviewed", "ai_primary_all_types", "cs_primary_peer_reviewed")
GROUPS = ("South Korea", "China", "India", "Core Anglophone", "Other non-core")
PRIMARY_SPEC = "existing|conservative|top_full|ai_primary_peer_reviewed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_codes(value: Any) -> list[str]:
    if isinstance(value, list):
        return sorted({str(x).strip().upper() for x in value if str(x).strip()})
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(str(value).strip())
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return sorted({str(x).strip().upper() for x in parsed if str(x).strip()})
    return []


def members(group: str, universe: set[str]) -> set[str]:
    if group == "South Korea":
        return {"KR"}
    if group == "China":
        return {"CN"}
    if group == "India":
        return {"IN"}
    if group == "Core Anglophone":
        return universe & CORE
    if group == "Other non-core":
        return universe - CORE - {"CN", "IN"}
    raise KeyError(group)


def build_top_credit(path: Path) -> pd.DataFrame:
    cols = ["paper_id", "venue", "year", *[x for pair in COUNTRY_COLUMNS.values() for x in pair]]
    data = pd.read_csv(path, usecols=cols)
    data = data.loc[data["venue"].isin(VENUES) & data["year"].between(min(YEARS), max(YEARS))].copy()
    if data["paper_id"].duplicated().any():
        raise RuntimeError("Duplicate paper_id in frozen numerator input")
    rows: list[dict[str, Any]] = []
    for record in data.to_dict("records"):
        for mapping, (code_col, covered_col) in COUNTRY_COLUMNS.items():
            codes = parse_codes(record[code_col])
            covered = str(record[covered_col]).strip().casefold() in {"true", "1"} and bool(codes)
            if not covered:
                continue
            for code in codes:
                rows.append({
                    "paper_id": record["paper_id"],
                    "venue": record["venue"],
                    "year": int(record["year"]),
                    "country_mapping": mapping,
                    "country_code": code,
                    "full": 1.0,
                    "fractional": 1.0 / len(codes),
                })
    return pd.DataFrame(rows)


def calculate(top: pd.DataFrame, capacity_path: Path) -> pd.DataFrame:
    capacity = pd.read_csv(capacity_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for mapping in COUNTRY_COLUMNS:
        top_mapping = top.loc[top["country_mapping"].eq(mapping)]
        for counting in ("full", "fractional"):
            for denominator_spec in DENOMINATORS:
                broad = capacity.loc[
                    capacity["specification"].eq(denominator_spec) & capacity["year"].isin(YEARS),
                    ["year", "country_code", "country_work_count_full"],
                ]
                for year in YEARS:
                    top_year = top_mapping.loc[top_mapping["year"].eq(year)]
                    broad_year = broad.loc[broad["year"].eq(year)]
                    universe = set(top_year["country_code"]) | set(broad_year["country_code"])
                    top_total = float(top_year[counting].sum())
                    broad_total = float(broad_year["country_work_count_full"].sum())
                    for group in GROUPS:
                        group_codes = members(group, universe)
                        top_count = float(top_year.loc[top_year["country_code"].isin(group_codes), counting].sum())
                        broad_count = float(broad_year.loc[broad_year["country_code"].isin(group_codes), "country_work_count_full"].sum())
                        top_share = top_count / top_total if top_total else math.nan
                        broad_share = broad_count / broad_total if broad_total else math.nan
                        rows.append({
                            "spec_id": f"existing|{mapping}|top_{counting}|{denominator_spec}",
                            "country_mapping": mapping,
                            "numerator_counting": counting,
                            "denominator_specification": denominator_spec,
                            "year": year,
                            "group": group,
                            "top_count": top_count,
                            "top_credit_total": top_total,
                            "top_share": top_share,
                            "broad_count": broad_count,
                            "broad_country_credit_total": broad_total,
                            "broad_share": broad_share,
                            "pri": top_share / broad_share if broad_share > 0 else math.nan,
                        })
    return pd.DataFrame(rows)


def validate_frozen(results: pd.DataFrame, frozen_zip: Path) -> dict[str, Any]:
    member = "reproducibility_package/data/aggregate/S12_pri_specification_dashboard.csv"
    with zipfile.ZipFile(frozen_zip) as archive:
        with archive.open(member) as handle:
            frozen = pd.read_csv(handle)
    ours = results.loc[
        results["spec_id"].eq(PRIMARY_SPEC)
        & results["group"].isin(["China", "India", "Core Anglophone", "Other non-core"])
        & results["year"].isin([2018, 2024]),
        ["group", "year", "pri"],
    ]
    expected = frozen.loc[
        frozen["spec_id"].eq(PRIMARY_SPEC)
        & frozen["group"].isin(ours["group"]),
        ["group", "pri_2018", "pri_2024"],
    ].melt(id_vars="group", var_name="year_name", value_name="expected_pri")
    expected["year"] = expected["year_name"].map({"pri_2018": 2018, "pri_2024": 2024})
    check = ours.merge(expected[["group", "year", "expected_pri"]], on=["group", "year"], validate="one_to_one")
    check["absolute_difference"] = (check["pri"] - check["expected_pri"]).abs()
    return {
        "status": "PASS" if len(check) == 8 and check["absolute_difference"].max() < 5e-8 else "FAIL",
        "rows_compared": int(len(check)),
        "maximum_absolute_difference": float(check["absolute_difference"].max()),
        "primary_spec": PRIMARY_SPEC,
    }


def save_figure(primary: pd.DataFrame, output: Path) -> None:
    order = ["South Korea", "China", "Core Anglophone", "Other non-core", "India"]
    colors = {
        "South Korea": "#8E44AD", "China": "#D35400", "Core Anglophone": "#21618C",
        "Other non-core": "#148F77", "India": "#7F8C8D",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for group in order:
        block = primary.loc[primary["group"].eq(group)].sort_values("year")
        ax.plot(block["year"], block["pri"], marker="o", linewidth=2.2 if group == "South Korea" else 1.7,
                label=group, color=colors[group])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_xlabel("Year")
    ax.set_ylabel("Production-adjusted representation index (PRI)")
    ax.set_title("ICML and NeurIPS first-listed-author representation")
    ax.set_xticks(YEARS)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country-input", type=Path, default=DEFAULT_COUNTRY)
    parser.add_argument("--capacity-input", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--frozen-package", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    reports = args.output_root / "reports"
    manifests = args.output_root / "manifests"
    for directory in (tables, figures, reports, manifests):
        directory.mkdir(parents=True, exist_ok=True)

    top = build_top_credit(args.country_input)
    results = calculate(top, args.capacity_input)
    validation = validate_frozen(results, args.frozen_package)
    if validation["status"] != "PASS":
        raise RuntimeError(f"Frozen FAccT endpoint reproduction failed: {validation}")

    all_path = tables / "korea_pri_all_12specifications_2018_2024.csv"
    results.to_csv(all_path, index=False)
    primary = results.loc[results["spec_id"].eq(PRIMARY_SPEC)].copy()
    primary_path = tables / "primary_pri_korea_and_frozen_groups_2018_2024.csv"
    primary.to_csv(primary_path, index=False)

    endpoints = primary.loc[primary["year"].isin([2018, 2024])].pivot(
        index="group", columns="year", values="pri"
    ).reset_index().rename(columns={2018: "pri_2018", 2024: "pri_2024"})
    endpoints["change_2018_2024"] = endpoints["pri_2024"] - endpoints["pri_2018"]
    endpoints["direction"] = endpoints["change_2018_2024"].map(lambda x: "increase" if x > 0 else ("decrease" if x < 0 else "flat"))
    endpoint_path = tables / "primary_pri_endpoints_with_korea.csv"
    endpoints.to_csv(endpoint_path, index=False)

    venue = top.loc[top["country_mapping"].eq("conservative")].copy()
    venue["is_korea"] = venue["country_code"].eq("KR")
    venue_counts = venue.groupby(["venue", "year"], as_index=False).agg(
        korea_first_listed_credit=("is_korea", "sum"), total_country_credit=("full", "sum")
    )
    venue_counts["korea_top_share"] = venue_counts["korea_first_listed_credit"] / venue_counts["total_country_credit"]
    venue_path = tables / "korea_top_venue_counts_by_venue_year.csv"
    venue_counts.to_csv(venue_path, index=False)

    save_figure(primary, figures / "korea_and_comparator_pri_trajectory")
    validation_path = manifests / "korea_pri_frozen_reproduction_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = pd.DataFrame([
        {"kind": "input", "path": str(args.country_input.relative_to(ROOT)), "sha256": sha256_file(args.country_input)},
        {"kind": "input", "path": str(args.capacity_input.relative_to(ROOT)), "sha256": sha256_file(args.capacity_input)},
        {"kind": "input_frozen", "path": str(args.frozen_package.relative_to(ROOT)), "sha256": sha256_file(args.frozen_package)},
        *[
            {"kind": "output", "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for path in (all_path, primary_path, endpoint_path, venue_path, validation_path,
                         figures / "korea_and_comparator_pri_trajectory.png",
                         figures / "korea_and_comparator_pri_trajectory.pdf")
        ],
    ])
    manifest.to_csv(manifests / "korea_pri_file_manifest.csv", index=False)

    korea = primary.loc[primary["group"].eq("South Korea")].sort_values("year")
    first, last = korea.iloc[0], korea.iloc[-1]
    report = f"""# 한국 PRI 기술 결과 v0.1

상태: **DESCRIPTIVE_KOREA_GO**  
FAccT 동결 결과 재현: **{validation['status']}** (최대 절대차 {validation['maximum_absolute_difference']:.2e})

한국의 ICML·NeurIPS 채택논문 제1저자 PRI는 2018년 **{first.pri:.3f}**에서 2024년 **{last.pri:.3f}**로 변했다(차이 **{last.pri-first.pri:+.3f}**). 이 값은 broad-AI 논문 생산 비중과 비교한 상대 대표성이며, 채택률이나 LLM의 효과가 아니다.

현재 단계에서 말할 수 있는 것은 한국의 생산량 보정 대표성 궤적뿐이다. 생성형 AI 확산과의 연관성은 노출 시점보다 뒤에 있는 학회 결과가 확보되고 시간 정합성 gate가 통과된 뒤에만 추정한다.
"""
    (reports / "KOREA_PRI_RESULT_KO.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "DESCRIPTIVE_KOREA_GO",
        "frozen_validation": validation,
        "korea_pri_2018": float(first.pri),
        "korea_pri_2024": float(last.pri),
        "change": float(last.pri - first.pri),
        "output_root": str(args.output_root),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

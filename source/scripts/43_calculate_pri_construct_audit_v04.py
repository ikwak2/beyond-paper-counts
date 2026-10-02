#!/usr/bin/env python3
"""Calculate the locked PRI construct audit when its inputs pass QC.

The script always reproduces the previously inspected 12-specification hybrid
reference.  Work-level role/topic constructs are computed only from a complete,
schema-valid compact cache.  Topic constructs additionally require the frozen
top-roster linkage gate.  Incomplete smoke inputs produce availability and QC
artifacts, never pseudo-estimates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COUNTRY = ROOT / "data/processed/country_measurement_combined_v01_first_author.csv"
DEFAULT_CAPACITY = ROOT / "data/processed/openalex_capacity_country_year_v01.csv"
DEFAULT_ROSTER = ROOT / "data/interim/openalex_pri_v04/private/top_target_roster_2018_2024.csv"
DEFAULT_CACHE = ROOT / "data/interim/openalex_pri_v04/private/broad_ai_full_v041"
DEFAULT_CACHE_QC = ROOT / "outputs/facct_locked_completion_v04/pri_qc/broad_ai_validation_full.json"
DEFAULT_MATCH = ROOT / "data/interim/openalex_pri_v04/private/top_batch_match_results_v041.jsonl"
DEFAULT_MATCH_VALIDATION = ROOT / "outputs/facct_locked_completion_v04/pri_qc/top_batch_match_validation.json"
DEFAULT_OUTPUT = ROOT / "outputs/facct_locked_completion_v04/pri_audit_smoke"
EXPECTED_SCHEMA = "openalex_pri_v04.1"
VENUES = ("ICML", "NeurIPS")
YEARS = tuple(range(2018, 2025))
CORE = {"AU", "CA", "GB", "IE", "NZ", "US"}
GROUPS = (
    "Core Anglophone",
    "Non-core Anglophone",
    "Non-core excluding China",
    "Other non-core",
    "China",
    "India",
    "United States",
)
# Mutually exclusive groups are required for cross-group ordering. Legacy "Non-core Anglophone" means all non-core.
FOCUS_DECISION_GROUPS = ("China", "India", "Core Anglophone", "Other non-core")
COUNTRY_COLUMNS = {
    "conservative": ("primary_country_codes", "primary_country_covered"),
    "sensitivity": ("sensitivity_country_codes", "sensitivity_country_covered"),
}
EXISTING_CAPACITY_SPECS = (
    "ai_primary_peer_reviewed",
    "ai_primary_all_types",
    "cs_primary_peer_reviewed",
)
WORK_METRICS = {
    "hybrid_paper_country_full": ("hybrid", "paper_country_full"),
    "hybrid_paper_country_fractional": ("hybrid", "paper_country_fractional"),
    "hybrid_authorship_fractional_all_authors": ("hybrid", "authorship_country_fractional_all_authors"),
    "hybrid_authorship_fractional_covered_authors": ("hybrid", "authorship_country_fractional_covered_authors"),
    "role_matched_first_position_full_strict": ("role_matched_strict", "first_author_full_strict"),
    "role_matched_first_position_fractional_strict": ("role_matched_strict", "first_author_fractional_strict"),
    "role_matched_row0_fallback_full": ("role_matched_fallback", "first_author_full_fallback"),
    "role_matched_row0_fallback_fractional": ("role_matched_fallback", "first_author_fractional_fallback"),
}
REQUIRED_WORK_FIELDS = {
    "schema_version", "openalex_work_id", "publication_year", "primary_topic_id",
    "primary_subfield_id", "all_country_codes", "paper_country_fractional",
    "first_author_country_codes_strict", "first_author_country_codes_row0_fallback",
    "first_author_country_fractional_strict", "first_author_country_fractional_row0_fallback",
    "authorship_country_fractional_all_authors", "authorship_country_fractional_covered_authors",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_codes(value: Any) -> list[str]:
    if isinstance(value, list):
        return sorted({str(item).strip().upper() for item in value if str(item).strip()})
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return sorted({str(item).strip().upper() for item in parsed if str(item).strip()})
    return []


def group_members(group: str, universe: Iterable[str]) -> set[str]:
    codes = {str(code) for code in universe if pd.notna(code)}
    if group == "Core Anglophone":
        return codes & CORE
    if group == "Non-core Anglophone":
        return codes - CORE
    if group == "Non-core excluding China":
        return codes - CORE - {"CN"}
    if group == "Other non-core":
        return codes - CORE - {"CN", "IN"}
    if group == "China":
        return {"CN"}
    if group == "India":
        return {"IN"}
    if group == "United States":
        return {"US"}
    raise KeyError(group)


def build_top_paper_country(country_path: Path) -> pd.DataFrame:
    usecols = ["paper_id", "venue", "year", *[value for pair in COUNTRY_COLUMNS.values() for value in pair]]
    source = pd.read_csv(country_path, usecols=usecols)
    source = source.loc[source["venue"].isin(VENUES) & source["year"].between(2018, 2024)].copy()
    if source["paper_id"].duplicated().any():
        raise RuntimeError("Top country input contains duplicate paper_id rows")
    rows: list[dict[str, Any]] = []
    for item in source.itertuples(index=False):
        values = item._asdict()
        for mapping, (code_column, covered_column) in COUNTRY_COLUMNS.items():
            codes = parse_codes(values[code_column])
            covered = bool(values[covered_column]) and bool(codes)
            rows.append({
                "paper_key": values["paper_id"], "venue": values["venue"], "year": int(values["year"]),
                "country_mapping": mapping, "country_codes": codes if covered else [], "country_covered": covered,
            })
    return pd.DataFrame(rows)


def aggregate_top_country(papers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in papers.loc[papers["country_covered"]].itertuples(index=False):
        count = len(item.country_codes)
        for code in item.country_codes:
            rows.append({
                "paper_key": item.paper_key, "year": int(item.year), "venue": item.venue,
                "country_mapping": item.country_mapping, "country_code": code,
                "full": 1.0, "fractional": 1.0 / count,
            })
    return pd.DataFrame(rows)


def country_to_group_pri(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    *,
    spec_columns: dict[str, Any],
    numerator_value: str,
    denominator_value: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in YEARS:
        top = numerator.loc[numerator["year"].eq(year)]
        broad = denominator.loc[denominator["year"].eq(year)]
        universe = set(top["country_code"].dropna()) | set(broad["country_code"].dropna())
        top_total = float(top[numerator_value].sum())
        broad_total = float(broad[denominator_value].sum())
        for group in GROUPS:
            members = group_members(group, universe)
            top_count = float(top.loc[top["country_code"].isin(members), numerator_value].sum())
            broad_count = float(broad.loc[broad["country_code"].isin(members), denominator_value].sum())
            top_share = top_count / top_total if top_total else math.nan
            broad_share = broad_count / broad_total if broad_total else math.nan
            rows.append({
                **spec_columns, "year": year, "group": group,
                "top_count": top_count, "top_count_total": top_total, "top_share": top_share,
                "broad_count": broad_count, "broad_count_total": broad_total, "broad_share": broad_share,
                "pri": top_share / broad_share if broad_share > 0 and not math.isnan(top_share) else math.nan,
            })
    return rows


def read_capacity_country(path: Any) -> pd.DataFrame:
    """Read ISO-2 country codes without treating Namibia's NA as missing."""
    capacity = pd.read_csv(path, keep_default_na=False)
    if capacity["country_code"].eq("").any():
        raise RuntimeError("capacity input contains blank country codes")
    return capacity


def existing_hybrid_reference(top_counts: pd.DataFrame, capacity_path: Path) -> pd.DataFrame:
    capacity = read_capacity_country(capacity_path)
    outputs: list[dict[str, Any]] = []
    for mapping in COUNTRY_COLUMNS:
        mapped = top_counts.loc[top_counts["country_mapping"].eq(mapping)]
        for count_spec in ("full", "fractional"):
            for capacity_spec in EXISTING_CAPACITY_SPECS:
                denominator = capacity.loc[
                    capacity["specification"].eq(capacity_spec) & capacity["year"].isin(YEARS),
                    ["year", "country_code", "country_work_count_full"],
                ].copy()
                outputs.extend(country_to_group_pri(
                    mapped, denominator,
                    spec_columns={
                        "audit_family": "existing_inspected_hybrid_reference",
                        "country_mapping": mapping,
                        "numerator_counting": count_spec,
                        "denominator_specification": capacity_spec,
                        "topic_specification": "none",
                        "status": "EXISTING_INSPECTED_REFERENCE",
                        "spec_id": f"existing|{mapping}|top_{count_spec}|{capacity_spec}",
                    },
                    numerator_value=count_spec,
                    denominator_value="country_work_count_full",
                ))
    return pd.DataFrame(outputs)


def iter_cache(root: Path):
    for path in sorted(root.glob("20*/page_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def work_credit_maps(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    def full(codes: Any) -> dict[str, float]:
        return {str(code).upper(): 1.0 for code in (codes or [])}
    return {
        "paper_country_full": full(row.get("all_country_codes")),
        "paper_country_fractional": {str(k).upper(): float(v) for k, v in (row.get("paper_country_fractional") or {}).items()},
        "authorship_country_fractional_all_authors": {str(k).upper(): float(v) for k, v in (row.get("authorship_country_fractional_all_authors") or {}).items()},
        "authorship_country_fractional_covered_authors": {str(k).upper(): float(v) for k, v in (row.get("authorship_country_fractional_covered_authors") or {}).items()},
        "first_author_full_strict": full(row.get("first_author_country_codes_strict")),
        "first_author_fractional_strict": {str(k).upper(): float(v) for k, v in (row.get("first_author_country_fractional_strict") or {}).items()},
        "first_author_full_fallback": full(row.get("first_author_country_codes_row0_fallback")),
        "first_author_fractional_fallback": {str(k).upper(): float(v) for k, v in (row.get("first_author_country_fractional_row0_fallback") or {}).items()},
    }


def aggregate_work_cache(cache_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregate: defaultdict[tuple[int, str, str, str], float] = defaultdict(float)
    works = 0
    duplicate = 0
    missing_fields: defaultdict[str, int] = defaultdict(int)
    bad_schema = 0
    seen: set[str] = set()
    for row in iter_cache(cache_root):
        works += 1
        missing = REQUIRED_WORK_FIELDS - set(row)
        for field in missing:
            missing_fields[field] += 1
        bad_schema += int(row.get("schema_version") != EXPECTED_SCHEMA)
        work_id = str(row.get("openalex_work_id") or "")
        duplicate += int(work_id in seen)
        seen.add(work_id)
        year = int(row.get("publication_year"))
        topic = str(row.get("primary_topic_id") or "MISSING")
        for metric, credits in work_credit_maps(row).items():
            for country, value in credits.items():
                aggregate[(year, topic, metric, country)] += float(value)
    data = pd.DataFrame([
        {"year": year, "primary_topic_id": topic, "metric": metric, "country_code": country, "credit": credit}
        for (year, topic, metric, country), credit in aggregate.items()
    ])
    diagnostic = {
        "cached_works": works, "unique_work_ids": len(seen), "duplicate_work_ids": duplicate,
        "bad_schema_version_rows": bad_schema, "missing_required_fields_by_field": dict(missing_fields),
        "schema_valid": bool(works and duplicate == 0 and bad_schema == 0 and not missing_fields),
    }
    return data, diagnostic


def cache_full_complete(cache_qc_path: Path, aggregate_diagnostic: dict[str, Any]) -> bool:
    if not cache_qc_path.exists():
        return False
    qc = json.loads(cache_qc_path.read_text(encoding="utf-8"))
    return bool((qc.get("status") == "PASS" or qc.get("full_cache_complete")) and qc.get("schema_valid", True) and aggregate_diagnostic["schema_valid"])


def load_matches(path: Path, roster_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        return pd.DataFrame(), {"available": False, "reason": "match_file_missing", "linkage_gate_pass": False}
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                work = item.get("matched_work") or {}
                rows.append({
                    "paper_key": item.get("paper_key"), "venue": item.get("venue"), "year": item.get("year"),
                    "match_status": item.get("match_status"), "primary_topic_id": work.get("primary_topic_id"),
                    "primary_subfield_id": work.get("primary_subfield_id"),
                })
        matches = pd.DataFrame(rows)
        accepted_status = {"accepted_high", "accepted_medium"}
    else:
        matches = pd.read_csv(path)
        accepted_status = {"accepted_exact_title_year_author"}
    roster = pd.read_csv(roster_path, usecols=["paper_key", "venue", "year"])
    status = matches[["paper_key", "match_status", "primary_topic_id", "primary_subfield_id"]].drop_duplicates("paper_key")
    merged = roster.merge(status, on="paper_key", how="left", validate="one_to_one")
    merged["accepted"] = merged["match_status"].isin(accepted_status)
    cell = merged.groupby(["venue", "year"], as_index=False).agg(roster=("paper_key", "size"), accepted=("accepted", "sum"))
    cell["match_rate"] = cell["accepted"] / cell["roster"]
    minimum = float(cell["match_rate"].min()) if len(cell) == 14 else math.nan
    maximum = float(cell["match_rate"].max()) if len(cell) == 14 else math.nan
    spread = maximum - minimum if len(cell) == 14 else math.nan
    gate = bool(len(cell) == 14 and minimum >= 0.80 and spread <= 0.10)
    return merged, {
        "available": True, "linkage_gate_pass": gate, "minimum_venue_year_match_rate": minimum,
        "maximum_venue_year_match_rate": maximum, "venue_year_match_rate_spread": spread,
        "overall_match_rate": float(merged["accepted"].mean()), "venue_year_cells": len(cell),
        "no_silent_first_result_required": True,
    }


def worklevel_non_topic_pri(top_counts: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mapping in COUNTRY_COLUMNS:
        top = top_counts.loc[top_counts["country_mapping"].eq(mapping)]
        for numerator_counting in ("full", "fractional"):
            for denominator_name, (family, metric) in WORK_METRICS.items():
                denominator = aggregate.loc[aggregate["metric"].eq(metric)].groupby(
                    ["year", "country_code"], as_index=False
                )["credit"].sum()
                rows.extend(country_to_group_pri(
                    top, denominator,
                    spec_columns={
                        "audit_family": family, "country_mapping": mapping,
                        "numerator_counting": numerator_counting,
                        "denominator_specification": denominator_name,
                        "topic_specification": "none", "status": "VALID_WORKLEVEL",
                        "spec_id": f"work|{mapping}|top_{numerator_counting}|{denominator_name}",
                    },
                    numerator_value=numerator_counting, denominator_value="credit",
                ))
    return pd.DataFrame(rows)


def topic_pri(top_papers: pd.DataFrame, matches: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    matched_topics = matches.loc[matches["accepted"] & matches["primary_topic_id"].notna(), ["paper_key", "primary_topic_id"]]
    broad_topics = aggregate.groupby("year")["primary_topic_id"].agg(lambda values: set(values)).to_dict()
    paper_year = top_papers[["paper_key", "year"]].drop_duplicates("paper_key")
    fixed_topic_population = matched_topics.merge(paper_year, on="paper_key", how="inner", validate="one_to_one")
    fixed_topic_population["compatible_topic"] = fixed_topic_population.apply(lambda row: row["primary_topic_id"] in broad_topics.get(int(row["year"]), set()), axis=1)
    fixed_topic_population = fixed_topic_population.loc[fixed_topic_population["compatible_topic"]].copy()
    fixed_topic_weights = {int(year): block["primary_topic_id"].value_counts(normalize=True).to_dict() for year, block in fixed_topic_population.groupby("year")}
    data = top_papers.merge(fixed_topic_population[["paper_key", "primary_topic_id"]], on="paper_key", how="inner", validate="many_to_one")
    data = data.loc[data["country_covered"]].copy()
    top_credit_rows = []
    for item in data.itertuples(index=False):
        for counting in ("full", "fractional"):
            weight = 1.0 if counting == "full" else 1.0 / len(item.country_codes)
            for code in item.country_codes:
                top_credit_rows.append({
                    "paper_key": item.paper_key, "year": int(item.year), "country_mapping": item.country_mapping,
                    "topic": item.primary_topic_id, "country_code": code, "counting": counting, "credit": weight,
                })
    top_credit = pd.DataFrame(top_credit_rows)
    if top_credit.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    all_top_counts = aggregate_top_country(top_papers)
    for mapping in COUNTRY_COLUMNS:
        for counting in ("full", "fractional"):
            top = top_credit.loc[top_credit["country_mapping"].eq(mapping) & top_credit["counting"].eq(counting)]
            all_top = all_top_counts.loc[all_top_counts["country_mapping"].eq(mapping), ["year", "country_code", counting]].rename(columns={counting: "credit"})
            compatible_by_year = top.groupby("year")["topic"].agg(lambda values: set(values)).to_dict()
            for denominator_name, (family, metric) in WORK_METRICS.items():
                broad_metric = aggregate.loc[aggregate["metric"].eq(metric)].copy()
                # Topic-restricted: retain exact primary-topic support shared with matched top papers.
                restricted_broad = broad_metric.loc[broad_metric.apply(
                    lambda row: row["primary_topic_id"] in compatible_by_year.get(int(row["year"]), set()), axis=1
                )].groupby(["year", "country_code"], as_index=False)["credit"].sum()
                restricted_top = top.groupby(["year", "country_code"], as_index=False)["credit"].sum()
                rows.extend(country_to_group_pri(
                    restricted_top, restricted_broad,
                    spec_columns={
                        "audit_family": family, "country_mapping": mapping, "numerator_counting": counting,
                        "denominator_specification": denominator_name, "topic_specification": "compatible_topic_restricted",
                        "status": "VALID_TOPIC_LINKED", "spec_id": f"topic_restricted|{mapping}|top_{counting}|{denominator_name}",
                    }, numerator_value="credit", denominator_value="credit",
                ))
                # Topic-standardized denominator: fixed within-year top-paper topic mix.
                for year in YEARS:
                    top_year = top.loc[top["year"].eq(year)]
                    if top_year.empty:
                        continue
                    # Weights are fixed before country coverage or mapping.
                    weights = fixed_topic_weights.get(year, {})
                    if not weights:
                        continue
                    broad_year = broad_metric.loc[broad_metric["year"].eq(year) & broad_metric["primary_topic_id"].isin(weights)]
                    all_top_year = all_top.loc[all_top["year"].eq(year)]
                    universe = set(all_top_year["country_code"]) | set(broad_year["country_code"])
                    top_country = all_top_year.groupby("country_code")["credit"].sum()
                    matched_top_country = top_year.groupby("country_code")["credit"].sum()
                    top_total = float(top_country.sum())
                    matched_top_total = float(matched_top_country.sum())
                    for group in GROUPS:
                        members = group_members(group, universe)
                        top_count = float(top_country.reindex(list(members), fill_value=0).sum())
                        top_share = top_count / top_total if top_total else math.nan
                        matched_top_count = float(matched_top_country.reindex(list(members), fill_value=0).sum())
                        matched_top_share = matched_top_count / matched_top_total if matched_top_total else math.nan
                        standardized = 0.0
                        used_weight = 0.0
                        for topic, weight in weights.items():
                            cell = broad_year.loc[broad_year["primary_topic_id"].eq(topic)]
                            total = float(cell["credit"].sum())
                            if total <= 0:
                                continue
                            group_credit = float(cell.loc[cell["country_code"].isin(members), "credit"].sum())
                            standardized += float(weight) * group_credit / total
                            used_weight += float(weight)
                        standardized = standardized / used_weight if used_weight else math.nan
                        rows.append({
                            "audit_family": family, "country_mapping": mapping, "numerator_counting": counting,
                            "denominator_specification": denominator_name,
                            "topic_specification": "fixed_top_mix_standardized",
                            "status": "VALID_TOPIC_LINKED", "spec_id": f"topic_standardized_fixed_numerator|{mapping}|top_{counting}|{denominator_name}",
                            "year": year, "group": group, "top_count": top_count, "top_count_total": top_total,
                            "top_share": top_share, "broad_count": math.nan, "broad_count_total": math.nan,
                            "broad_share": standardized, "pri": top_share / standardized if standardized > 0 else math.nan,
                        })
                        rows.append({
                            "audit_family": family, "country_mapping": mapping, "numerator_counting": counting,
                            "denominator_specification": denominator_name,
                            "topic_specification": "matched_compatible_numerator_standardized_SELECTION_SENSITIVE",
                            "status": "SELECTION_SENSITIVE_DIAGNOSTIC", "spec_id": f"topic_standardized_matched_numerator_selection_sensitive|{mapping}|top_{counting}|{denominator_name}",
                            "year": year, "group": group, "top_count": matched_top_count, "top_count_total": matched_top_total,
                            "top_share": matched_top_share, "broad_count": math.nan, "broad_count_total": math.nan,
                            "broad_share": standardized, "pri": matched_top_share / standardized if standardized > 0 else math.nan,
                        })
    return pd.DataFrame(rows)


def sign(value: float, tolerance: float = 1e-12) -> int:
    return 1 if value > tolerance else (-1 if value < -tolerance else 0)


def classify_pri(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {"classification": "NOT_EVALUATED_INCOMPLETE", "reason": "no_valid_specifications"}
    valid = results.loc[results["status"].isin(["EXISTING_INSPECTED_REFERENCE", "VALID_WORKLEVEL", "VALID_TOPIC_LINKED"])].copy()
    pivot = valid.loc[valid["year"].isin([2018, 2024]) & valid["group"].isin(FOCUS_DECISION_GROUPS)].pivot_table(
        index=["spec_id"], columns=["group", "year"], values="pri", aggfunc="first"
    ).dropna()
    if len(pivot) < 2:
        return {"classification": "NOT_EVALUATED_INCOMPLETE", "reason": "fewer_than_two_complete_valid_specifications", "complete_specs": len(pivot)}
    direction_signatures = {}
    ordering_signatures = {}
    changes: defaultdict[str, list[float]] = defaultdict(list)
    spread_values: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    for spec, row in pivot.iterrows():
        directions = []
        values_2024 = []
        for group in FOCUS_DECISION_GROUPS:
            change = float(row[(group, 2024)] - row[(group, 2018)])
            directions.append((group, sign(change)))
            changes[group].append(abs(change))
            values_2024.append((group, float(row[(group, 2024)])))
            spread_values[(group, 2018)].append(float(row[(group, 2018)]))
            spread_values[(group, 2024)].append(float(row[(group, 2024)]))
        direction_signatures[str(spec)] = directions
        ordering_signatures[str(spec)] = [group for group, _ in sorted(values_2024, key=lambda item: (-item[1], item[0]))]
    direction_reversal = len({tuple(value) for value in direction_signatures.values()}) > 1
    ordering_reversal = len({tuple(value) for value in ordering_signatures.values()}) > 1
    ratios = {}
    ratio_material = False
    for group, values in changes.items():
        maximum, minimum = max(values), min(values)
        ratio = math.inf if minimum <= 1e-12 and maximum > 1e-12 else (maximum / minimum if minimum > 0 else 1.0)
        ratios[group] = ratio
        ratio_material |= ratio >= 2.0
    spreads = {f"{group}|{year}": max(values) - min(values) for (group, year), values in spread_values.items()}
    spread_material = bool(spreads and max(spreads.values()) >= 0.25)
    if direction_reversal or ordering_reversal:
        classification = "PRI_CONSTRUCT_SENSITIVE_WARNING"
    elif ratio_material or spread_material:
        classification = "DIRECTIONALLY_ROBUST_CONSTRUCT_SENSITIVE"
    else:
        classification = "DIRECTIONALLY_ROBUST"
    return {
        "classification": classification, "complete_specs": len(pivot),
        "direction_reversal": direction_reversal, "cross_group_ordering_reversal": ordering_reversal,
        "magnitude_change_ratios": ratios, "pri_spreads": spreads,
        "material_ratio_at_least_2": ratio_material, "material_spread_at_least_0_25": spread_material,
        "direction_signatures": direction_signatures, "ordering_signatures": ordering_signatures,
    }


def write_hash_manifest(path: Path, inputs: list[Path], outputs: list[Path]) -> None:
    rows = []
    for kind, paths in [("input", inputs), ("output_public_aggregate", outputs)]:
        for item in paths:
            if item.exists() and item.is_file():
                rows.append({"kind": kind, "path": str(item), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country-input", type=Path, default=DEFAULT_COUNTRY)
    parser.add_argument("--capacity-input", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cache-qc", type=Path, default=DEFAULT_CACHE_QC)
    parser.add_argument("--top-match", type=Path, default=DEFAULT_MATCH)
    parser.add_argument("--top-match-validation", type=Path, default=DEFAULT_MATCH_VALIDATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    top_papers = build_top_paper_country(args.country_input)
    roster_population = pd.read_csv(args.roster, usecols=["paper_key", "venue", "year"])
    roster_keys = set(roster_population["paper_key"].astype(str))
    country_keys = set(top_papers["paper_key"].astype(str))
    population_qc = {
        "roster_rows": int(len(roster_population)),
        "roster_unique_papers": int(roster_population["paper_key"].nunique()),
        "country_input_unique_papers": int(top_papers["paper_key"].nunique()),
        "missing_from_country_input": int(len(roster_keys - country_keys)),
        "extra_in_country_input": int(len(country_keys - roster_keys)),
    }
    population_qc["pass"] = bool(population_qc["roster_rows"] == 26872 and population_qc["roster_unique_papers"] == 26872 and population_qc["missing_from_country_input"] == 0 and population_qc["extra_in_country_input"] == 0)
    top_counts = aggregate_top_country(top_papers)
    existing = existing_hybrid_reference(top_counts, args.capacity_input)
    existing_path = tables / "existing_hybrid_reference_12spec_group_year.csv"
    existing.to_csv(existing_path, index=False)

    aggregate, aggregate_qc = aggregate_work_cache(args.cache_root)
    full_cache = cache_full_complete(args.cache_qc, aggregate_qc)
    matches, linkage_qc = load_matches(args.top_match, args.roster)
    batch_validation = json.loads(args.top_match_validation.read_text(encoding="utf-8")) if args.top_match_validation.exists() else {"status": "MISSING"}
    linkage_qc["calculated_rate_gate_pass"] = bool(linkage_qc.get("linkage_gate_pass"))
    linkage_qc["batch_validation_status"] = batch_validation.get("status")
    linkage_qc["linkage_gate_pass"] = bool(linkage_qc.get("calculated_rate_gate_pass") and batch_validation.get("status") == "PASS")
    availability = [
        {"construct": "frozen_top_population_alignment", "available": population_qc["pass"], "status": "PASS" if population_qc["pass"] else "INCOMPLETE_NOT_ESTIMAND", "reason": "26,872 frozen roster papers must align one-to-one with the country panel"},
        {"construct": "existing_hybrid_12spec_reference", "available": True, "status": "EXISTING_INSPECTED_REFERENCE", "reason": "frozen grouped OpenAlex denominator and top-country panel available"},
        {"construct": "worklevel_hybrid_and_role_matched", "available": full_cache, "status": "AVAILABLE" if full_cache else "INCOMPLETE_NOT_ESTIMAND", "reason": "complete schema-valid broad work cache required"},
        {"construct": "topic_restricted_and_standardized", "available": bool(full_cache and linkage_qc["linkage_gate_pass"]), "status": "AVAILABLE" if full_cache and linkage_qc["linkage_gate_pass"] else "INCOMPLETE_NOT_ESTIMAND", "reason": "complete broad cache plus >=80% every venue-year top linkage and <=10pp spread required"},
    ]
    worklevel = worklevel_non_topic_pri(top_counts, aggregate) if full_cache else pd.DataFrame()
    topics = topic_pri(top_papers, matches, aggregate) if full_cache and linkage_qc["linkage_gate_pass"] else pd.DataFrame()
    work_columns = list(existing.columns)
    worklevel = worklevel.reindex(columns=work_columns)
    topics = topics.reindex(columns=work_columns)
    work_path = tables / "worklevel_pri_constructs_group_year.csv"
    topic_path = tables / "topic_pri_constructs_group_year.csv"
    worklevel.to_csv(work_path, index=False)
    topics.to_csv(topic_path, index=False)
    availability_path = tables / "construct_availability.csv"
    pd.DataFrame(availability).to_csv(availability_path, index=False)

    existing_decision = classify_pri(existing)
    overall_complete = bool(population_qc["pass"] and full_cache and linkage_qc["linkage_gate_pass"] and not worklevel.empty and not topics.empty)
    combined = pd.concat([existing, worklevel, topics], ignore_index=True)
    overall_decision = classify_pri(combined) if overall_complete else {
        "classification": "NOT_EVALUATED_INCOMPLETE",
        "reason": "locked role/topic construct audit inputs have not passed all gates",
    }
    partial_role_only = bool(population_qc["pass"] and full_cache and not linkage_qc["linkage_gate_pass"] and not worklevel.empty)
    decision_status = (
        "PASS" if overall_complete else
        "PARTIAL_PASS_ROLE_MATCHED_TOPIC_BLOCKED" if partial_role_only else
        "SMOKE_INCOMPLETE_NOT_ESTIMAND"
    )
    decision = {
        "generated_at_utc": utc_now(), "status": decision_status,
        "existing_inspected_reference_decision": existing_decision,
        "locked_full_construct_audit_decision": overall_decision,
        "broad_cache_qc": aggregate_qc, "broad_cache_full_complete": full_cache,
        "frozen_population_qc": population_qc,
        "top_linkage_qc": linkage_qc,
        "rule": {
            "direction_or_order_reversal": "PRI_CONSTRUCT_SENSITIVE_WARNING",
            "same_direction_but_abs_change_ratio_ge_2_or_pri_spread_ge_0.25": "DIRECTIONALLY_ROBUST_CONSTRUCT_SENSITIVE",
            "otherwise": "DIRECTIONALLY_ROBUST",
        },
        "adverse_specs_retained": True, "no_favorable_specification_selection": True,
    }
    decision_path = tables / "pri_construct_decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    availability_text = "\n".join(
        f"| {row['construct']} | {row['status']} | {row['reason']} |" for row in availability
    )
    report = "# PRI 구성개념 감사 결과\n\n"
    if overall_complete:
        report += "현재 상태는 **PASS**입니다. role-matched와 topic-linked PRI가 모두 사전 게이트를 통과했습니다.\n\n"
    elif partial_role_only:
        report += "현재 상태는 **PARTIAL_PASS_ROLE_MATCHED_TOPIC_BLOCKED**입니다. 완전한 broad-AI work cache로 role-matched PRI는 유효하게 계산했지만, top-paper linkage의 80%/10%p 게이트 실패로 topic-restricted 및 topic-standardized PRI는 계산하지 않았습니다.\n\n"
    else:
        report += "현재 상태는 **SMOKE_INCOMPLETE_NOT_ESTIMAND**입니다. 필요한 입력 게이트가 완성되지 않아 새 work-level/topic PRI를 계산하지 않았습니다.\n\n"
    report += "| 구성개념 | 상태 | 이유 |\n|---|---|---|\n" + availability_text + "\n\n"
    report += "## 해석 제한\n\n- 빈 topic 표는 0 효과가 아니라 **top-linkage 게이트 미통과**를 뜻합니다.\n- role-matched 결과의 유효성은 완전한 broad work cache에 근거하며, 실패한 topic linkage를 복권하지 않습니다.\n- 기존 12개 조합의 분류는 이미 관찰된 reference를 포함한 측정감사입니다.\n- 이름·논문 ID·제목·개별 매칭 행은 PRIVATE 디렉터리에만 존재합니다.\n- 사전 80%/10%p linkage gate는 결과를 보고 낮추지 않았습니다.\n- 레거시 `Non-core Anglophone` 표기는 실제로 모든 non-core 국가를 뜻하는 기존 명명 부채이며, 국가군 순위 판정에는 배타적 `Other non-core`를 사용합니다.\n"
    report_path = args.output_root / "PRI_CONSTRUCT_AUDIT_SMOKE_KO.md"
    report_path.write_text(report, encoding="utf-8")
    software_path = tables / "pri_software_versions.json"
    software_path.write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__}, indent=2), encoding="utf-8")
    privacy_path = tables / "pri_public_private_manifest.csv"
    pd.DataFrame([
        {"path": str(path), "classification": "PUBLIC_AGGREGATE_OR_QC", "contains_names_titles_or_record_ids": False}
        for path in [existing_path, work_path, topic_path, availability_path, decision_path, report_path, software_path]
    ] + [
        {"path": str(args.top_match), "classification": "PRIVATE_INPUT_NOT_RELEASED", "contains_names_titles_or_record_ids": True}
    ]).to_csv(privacy_path, index=False)
    manifest_path = tables / "pri_reproducibility_sha256.csv"
    input_paths = [Path(__file__), ROOT / "docs/facct_locked_completion_protocol_v0.4.md", args.country_input, args.capacity_input, args.roster, args.cache_qc, args.top_match, args.top_match_validation, *sorted(args.cache_root.glob("20*/page_*.jsonl")), *sorted(args.cache_root.glob("20*/page_*.manifest.json"))]
    output_paths = [existing_path, work_path, topic_path, availability_path, decision_path, report_path, software_path, privacy_path, tables / "synthetic_assertions.json"]
    write_hash_manifest(manifest_path, input_paths, output_paths)
    print(json.dumps({"status": decision["status"], "existing_reference": existing_decision["classification"], "locked_full_audit": overall_decision["classification"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

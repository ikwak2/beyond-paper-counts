#!/usr/bin/env python3
"""Deterministically match the frozen top-venue roster to a compact cache.

This is a legacy/smoke-only local linkage diagnostic, not an API client and not a prerequisite for the main PRI. The final linkage is the PRIVATE script44 OR-batch ±1-year result.  Exact normalized title and
year define the candidate set; author agreement and explicit ambiguity rules
decide whether a candidate is accepted.  The first candidate is never silently
accepted.  Record-level output is PRIVATE, while public output is aggregated.
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
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROSTER = ROOT / "data/interim/openalex_pri_v04/private/top_target_roster_2018_2024.csv"
DEFAULT_CACHE = ROOT / "data/interim/openalex_pri_v04/private/broad_ai_smoke_v041"
DEFAULT_COUNTRY = ROOT / "data/processed/country_measurement_combined_v01_first_author.csv"
DEFAULT_OUTPUT = ROOT / "outputs/facct_locked_completion_v04/pri_audit_smoke"
EXPECTED_SCHEMA = "openalex_pri_v04.1"
YEARS = tuple(range(2018, 2025))
VENUES = ("ICML", "NeurIPS")
MIN_MATCH_RATE = 0.80
MAX_CELL_SPREAD = 0.10
K_PUBLIC = 20

REQUIRED_CACHE_FIELDS = {
    "schema_version",
    "openalex_work_id",
    "title_normalized_PRIVATE",
    "publication_year",
    "primary_topic_id",
    "primary_subfield_id",
    "author_names_normalized_PRIVATE",
    "authorship_positions",
    "authorship_country_sets",
    "all_country_codes",
    "first_author_country_codes_strict",
    "first_author_country_codes_row0_fallback",
    "paper_country_fractional",
    "first_author_country_fractional_strict",
    "first_author_country_fractional_row0_fallback",
    "authorship_country_fractional_all_authors",
    "authorship_country_fractional_covered_authors",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: Any) -> str:
    text = str(value or "").encode("utf-8", "replace").decode("utf-8")
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
    return []


def iter_cache_rows(root: Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in sorted(root.glob("20*/page_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def validate_cache(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_counts: defaultdict[str, int] = defaultdict(int)
    bad_schema = 0
    duplicate_ids = 0
    seen: set[str] = set()
    parse_error: str | None = None
    try:
        for path, line_number, row in iter_cache_rows(root):
            rows.append(row)
            missing = REQUIRED_CACHE_FIELDS - set(row)
            for field in missing:
                missing_counts[field] += 1
            bad_schema += int(row.get("schema_version") != EXPECTED_SCHEMA)
            work_id = str(row.get("openalex_work_id") or "")
            duplicate_ids += int(bool(work_id) and work_id in seen)
            if work_id:
                seen.add(work_id)
    except (OSError, json.JSONDecodeError) as error:
        parse_error = f"{type(error).__name__}: {error}"

    manifests = sorted(root.glob("20*/page_*.manifest.json"))
    year_completion: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        paths = [path for path in manifests if path.parent.name == str(year)]
        if not paths:
            year_completion[str(year)] = {"complete": False, "reason": "no_manifest"}
            continue
        last = json.loads(paths[-1].read_text(encoding="utf-8"))
        terminal = last.get("next_cursor") in {None, "", "None"}
        count_equal = int(last.get("cumulative_works", -1)) == int(last.get("openalex_count", -2))
        year_completion[str(year)] = {
            "complete": bool(terminal and count_equal),
            "terminal_cursor": terminal,
            "cached": int(last.get("cumulative_works", 0)),
            "expected": int(last.get("openalex_count", 0)),
        }
    schema_valid = bool(rows and not missing_counts and bad_schema == 0 and parse_error is None)
    full_cache_complete = bool(
        schema_valid
        and duplicate_ids == 0
        and len(year_completion) == len(YEARS)
        and all(item.get("complete") for item in year_completion.values())
    )
    diagnostic = {
        "generated_at_utc": utc_now(),
        "cache_root": str(root),
        "rows": len(rows),
        "unique_work_ids": len(seen),
        "duplicate_work_ids": duplicate_ids,
        "expected_schema_version": EXPECTED_SCHEMA,
        "bad_schema_version_rows": bad_schema,
        "missing_required_fields_by_field": dict(sorted(missing_counts.items())),
        "parse_error": parse_error,
        "schema_valid": schema_valid,
        "full_cache_complete": full_cache_complete,
        "year_completion": year_completion,
    }
    return rows, diagnostic


def candidate_decision(expected_authors: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [normalize(name) for name in expected_authors if normalize(name)]
    expected_set = set(expected)
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        observed = [normalize(name) for name in candidate.get("author_names_normalized_PRIVATE") or [] if normalize(name)]
        observed_set = set(observed)
        overlap = len(expected_set & observed_set)
        union = len(expected_set | observed_set)
        jaccard = overlap / union if union else 0.0
        first_exact = bool(expected and observed and expected[0] == observed[0])
        credible = bool(first_exact and overlap >= 1 and jaccard >= 0.15)
        scored.append(
            {
                "candidate": candidate,
                "first_author_exact": first_exact,
                "author_overlap": overlap,
                "author_jaccard": jaccard,
                "credible": credible,
            }
        )
    scored.sort(
        key=lambda item: (item["credible"], item["first_author_exact"], item["author_jaccard"], item["author_overlap"]),
        reverse=True,
    )
    credible = [item for item in scored if item["credible"]]
    if len(credible) == 1:
        best = credible[0]
        return {"status": "accepted_exact_title_year_author", "reason": "unique_author_credible_candidate", **best}
    if len(credible) > 1:
        return {
            "status": "ambiguous_multiple_credible",
            "reason": "multiple_exact_title_year_candidates_with_author_agreement",
            "candidate": None,
            "first_author_exact": None,
            "author_overlap": None,
            "author_jaccard": None,
            "credible": False,
        }
    best = scored[0] if scored else None
    return {
        "status": "rejected_author_disagreement",
        "reason": "exact_title_year_candidate_without_sufficient_author_agreement",
        "candidate": best["candidate"] if best else None,
        "first_author_exact": best["first_author_exact"] if best else None,
        "author_overlap": best["author_overlap"] if best else None,
        "author_jaccard": best["author_jaccard"] if best else None,
        "credible": False,
    }


def match_roster(roster: pd.DataFrame, cache_rows: list[dict[str, Any]]) -> pd.DataFrame:
    index: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cache_rows:
        year = pd.to_numeric(row.get("publication_year"), errors="coerce")
        title = normalize(row.get("title_normalized_PRIVATE"))
        if not pd.isna(year) and title:
            index[(int(year), title)].append(row)

    output: list[dict[str, Any]] = []
    for source in roster.itertuples(index=False):
        title_normalized = normalize(source.title)
        candidates = index.get((int(source.year), title_normalized), [])
        if not candidates:
            decision = {
                "status": "unmatched_no_exact_title_year",
                "reason": "no_normalized_title_year_candidate",
                "candidate": None,
                "first_author_exact": None,
                "author_overlap": None,
                "author_jaccard": None,
            }
        else:
            decision = candidate_decision(parse_list(source.author_names), candidates)
        matched = decision.get("candidate") if decision["status"].startswith("accepted_") else None
        output.append(
            {
                "paper_key": source.paper_key,
                "venue": source.venue,
                "year": int(source.year),
                "match_status": decision["status"],
                "match_reason": decision["reason"],
                "candidate_count_exact_title_year": len(candidates),
                "first_author_exact": decision.get("first_author_exact"),
                "author_overlap": decision.get("author_overlap"),
                "author_jaccard": decision.get("author_jaccard"),
                "matched_openalex_work_id": matched.get("openalex_work_id") if matched else None,
                "primary_topic_id": matched.get("primary_topic_id") if matched else None,
                "primary_topic_name": matched.get("primary_topic_name") if matched else None,
                "primary_subfield_id": matched.get("primary_subfield_id") if matched else None,
                "primary_subfield_name": matched.get("primary_subfield_name") if matched else None,
            }
        )
    return pd.DataFrame(output)


def build_cell_gate(matches: pd.DataFrame, cache_diagnostic: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    matches = matches.copy()
    matches["accepted"] = matches["match_status"].eq("accepted_exact_title_year_author")
    rows: list[dict[str, Any]] = []
    for venue in VENUES:
        for year in YEARS:
            block = matches.loc[matches["venue"].eq(venue) & matches["year"].eq(year)]
            counts = block["match_status"].value_counts()
            rows.append(
                {
                    "venue": venue,
                    "year": year,
                    "roster_papers": len(block),
                    "accepted": int(block["accepted"].sum()),
                    "match_rate": float(block["accepted"].mean()) if len(block) else math.nan,
                    "unmatched_no_exact_title_year": int(counts.get("unmatched_no_exact_title_year", 0)),
                    "rejected_author_disagreement": int(counts.get("rejected_author_disagreement", 0)),
                    "ambiguous_multiple_credible": int(counts.get("ambiguous_multiple_credible", 0)),
                }
            )
    cells = pd.DataFrame(rows)
    rates = cells["match_rate"].dropna()
    min_rate = float(rates.min()) if len(rates) else math.nan
    max_rate = float(rates.max()) if len(rates) else math.nan
    spread = max_rate - min_rate if len(rates) else math.nan
    linkage_gate = bool(len(rates) == 14 and min_rate >= MIN_MATCH_RATE and spread <= MAX_CELL_SPREAD)
    inferential_gate = bool(linkage_gate and cache_diagnostic["schema_valid"] and cache_diagnostic["full_cache_complete"])
    gate = {
        "generated_at_utc": utc_now(),
        "status": "PASS" if inferential_gate else "INCOMPLETE_NOT_ESTIMAND",
        "overall_match_rate": float(matches["accepted"].mean()) if len(matches) else math.nan,
        "minimum_venue_year_match_rate": min_rate,
        "maximum_venue_year_match_rate": max_rate,
        "venue_year_match_rate_spread": spread,
        "all_venue_year_rates_at_least_0_80": bool(len(rates) == 14 and min_rate >= MIN_MATCH_RATE),
        "venue_year_spread_at_most_0_10": bool(len(rates) == 14 and spread <= MAX_CELL_SPREAD),
        "linkage_rate_gate_pass": linkage_gate,
        "cache_schema_valid": bool(cache_diagnostic["schema_valid"]),
        "cache_full_complete": bool(cache_diagnostic["full_cache_complete"]),
        "inferential_gate_pass": inferential_gate,
        "no_silent_first_result": True,
        "candidate_definition": "exact normalized title and exact publication year",
        "acceptance_definition": "unique credible candidate with first-author agreement and author Jaccard >= 0.15",
        "selection_sensitive_if_failed": True,
        "diagnostic_only_not_final_linkage": True,
        "does_not_substitute_for_script44_or_batch_plus_minus_one_year": True,
    }
    return cells, gate


def country_linkage_table(matches: pd.DataFrame, country_path: Path) -> pd.DataFrame:
    columns = ["paper_id", "primary_country_codes", "sensitivity_country_codes"]
    country = pd.read_csv(country_path, usecols=columns).drop_duplicates("paper_id")
    data = matches.merge(country, left_on="paper_key", right_on="paper_id", how="left", validate="one_to_one")
    data["accepted"] = data["match_status"].eq("accepted_exact_title_year_author")
    rows: list[dict[str, Any]] = []
    for mapping, column in [("conservative", "primary_country_codes"), ("sensitivity", "sensitivity_country_codes")]:
        temp = data[["paper_key", "year", "accepted", column]].copy()
        temp["country_code"] = temp[column].map(parse_list)
        temp = temp.explode("country_code")
        temp["country_code"] = temp["country_code"].fillna("UNRESOLVED").astype(str).str.upper()
        grouped = temp.groupby(["year", "country_code"], as_index=False).agg(
            roster_papers=("paper_key", "nunique"), matched=("accepted", "sum")
        )
        grouped["country_mapping"] = mapping
        grouped["public_k20"] = grouped["roster_papers"].ge(K_PUBLIC)
        grouped["match_rate"] = grouped["matched"] / grouped["roster_papers"]
        grouped.loc[~grouped["public_k20"], ["matched", "match_rate"]] = math.nan
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)[
        ["country_mapping", "year", "country_code", "roster_papers", "matched", "match_rate", "public_k20"]
    ]


def write_manifest(output: Path, inputs: list[Path], products: list[Path]) -> None:
    rows = []
    for visibility, paths in [("input_private_or_internal", inputs), ("output", products)]:
        for path in paths:
            if path.exists() and path.is_file():
                rows.append({"visibility": visibility, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--country-input", type=Path, default=DEFAULT_COUNTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output_root
    tables, private = output / "tables", output / "private"
    tables.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    roster = pd.read_csv(args.roster)
    required_roster = {"paper_key", "venue", "year", "title", "author_names"}
    if not required_roster.issubset(roster.columns):
        raise RuntimeError(f"Roster missing columns: {sorted(required_roster - set(roster.columns))}")
    frozen_population = bool(
        len(roster) == 26_872
        and roster["paper_key"].nunique() == 26_872
        and set(roster["venue"]) == set(VENUES)
        and set(roster["year"].astype(int)) == set(YEARS)
    )
    cache_rows, cache_diagnostic = validate_cache(args.cache_root)
    matches = match_roster(roster, cache_rows)
    cells, gate = build_cell_gate(matches, cache_diagnostic)
    gate["frozen_roster_valid"] = frozen_population
    if not frozen_population:
        gate["status"] = "INCOMPLETE_NOT_ESTIMAND"
        gate["inferential_gate_pass"] = False

    private_match = private / "local_top_match_records_PRIVATE.csv"
    schema_path = tables / "local_cache_schema_validation.json"
    cell_path = tables / "local_match_dispositions_by_venue_year.csv"
    gate_path = tables / "local_match_gate.json"
    country_path = tables / "local_match_rates_by_country_year_k20.csv"
    software_path = tables / "software_versions.json"
    matches.to_csv(private_match, index=False)
    cells.to_csv(cell_path, index=False)
    country_linkage_table(matches, args.country_input).to_csv(country_path, index=False)
    schema_path.write_text(json.dumps(cache_diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    software_path.write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__}, indent=2),
        encoding="utf-8",
    )
    privacy_path = tables / "local_match_public_private_manifest.csv"
    pd.DataFrame([
        {"path": str(private_match), "classification": "PRIVATE_RECORD_LEVEL", "contains_names_titles_or_ids": True},
        {"path": str(cell_path), "classification": "PUBLIC_AGGREGATE", "contains_names_titles_or_ids": False},
        {"path": str(country_path), "classification": "PUBLIC_AGGREGATE_K20", "contains_names_titles_or_ids": False},
        {"path": str(gate_path), "classification": "PUBLIC_QC", "contains_names_titles_or_ids": False},
    ]).to_csv(privacy_path, index=False)
    manifest_path = tables / "local_match_reproducibility_sha256.csv"
    write_manifest(
        manifest_path,
        [Path(__file__), ROOT / "docs/facct_locked_completion_protocol_v0.4.md", args.roster, args.country_input, *sorted(args.cache_root.glob("20*/page_*.jsonl")), *sorted(args.cache_root.glob("20*/page_*.manifest.json"))],
        [private_match, schema_path, cell_path, gate_path, country_path, software_path, privacy_path],
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

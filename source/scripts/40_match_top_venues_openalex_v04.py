#!/usr/bin/env python3
"""Resumable OpenAlex matching for the frozen ICML/NeurIPS 2018--2024 roster.

The matcher deliberately does *not* accept the first search result.  It records
the top candidate set, scores title and author agreement, checks the score
margin, and leaves weak or competing matches unresolved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSTER_PATH = (
    PROJECT_ROOT.parent
    / "bibliometrics"
    / "top_ai_entry"
    / "data"
    / "processed"
    / "accepted_papers_dblp.csv"
)
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "openalex_pri_v04" / "private"
QC_DIR = PROJECT_ROOT / "outputs" / "facct_locked_completion_v04" / "pri_qc"
COUNTRY_PATH = PROJECT_ROOT / "data" / "processed" / "country_measurement_combined_v01_first_author.csv"
API_URL = "https://api.openalex.org/works"
VENUES = ("ICML", "NeurIPS")
YEARS = tuple(range(2018, 2025))
SELECT = (
    "id,doi,title,publication_year,type,primary_topic,authorships,"
    "is_authors_truncated,primary_location"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_api_key(api_key_file: str | None, env_value: str | None) -> str | None:
    if api_key_file:
        path = Path(api_key_file)
        if path.stat().st_mode & 0o077:
            raise PermissionError(f"API key file must be mode 0600: {path}")
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"Empty API key file: {path}")
        return value
    return env_value


def clean_text(value: Any) -> str:
    text = str(value or "")
    # Invalid lone surrogates occasionally occur in bibliographic metadata.
    return text.encode("utf-8", "replace").decode("utf-8")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value)).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def name_set(values: Iterable[Any]) -> set[str]:
    return {normalized for value in values if (normalized := normalize(value))}


def token_jaccard(left: str, right: str) -> float:
    a, b = set(normalize(left).split()), set(normalize(right).split())
    return len(a & b) / max(1, len(a | b))


def parse_json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return [clean_text(item) for item in parsed] if isinstance(parsed, list) else []


def work_author_names(work: dict[str, Any]) -> list[str]:
    return [
        clean_text(authorship.get("author", {}).get("display_name"))
        for authorship in work.get("authorships") or []
    ]


def source_details(work: dict[str, Any]) -> tuple[str | None, str | None]:
    source = ((work.get("primary_location") or {}).get("source") or {})
    return source.get("id"), source.get("display_name")


def score_candidate(
    title: str, expected_authors: list[str], expected_year: int, work: dict[str, Any]
) -> dict[str, Any]:
    observed_authors = work_author_names(work)
    expected_set, observed_set = name_set(expected_authors), name_set(observed_authors)
    author_jaccard = len(expected_set & observed_set) / max(1, len(expected_set | observed_set))
    any_author = bool(expected_set & observed_set)
    first_author = bool(
        expected_authors
        and observed_authors
        and normalize(expected_authors[0]) == normalize(observed_authors[0])
    )
    title_sequence = SequenceMatcher(None, normalize(title), normalize(work.get("title"))).ratio()
    title_tokens = token_jaccard(title, clean_text(work.get("title")))
    year = pd.to_numeric(work.get("publication_year"), errors="coerce")
    year_exact = bool(not pd.isna(year) and int(year) == int(expected_year))
    combined = (
        0.72 * title_sequence
        + 0.10 * title_tokens
        + 0.13 * author_jaccard
        + 0.04 * float(first_author)
        + 0.01 * float(year_exact)
    )
    return {
        "title_sequence": float(title_sequence),
        "title_token_jaccard": float(title_tokens),
        "author_jaccard": float(author_jaccard),
        "any_author_exact": any_author,
        "first_author_exact": first_author,
        "year_exact": year_exact,
        "combined_score": float(combined),
    }


def candidate_summary(work: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    source_id, source_name = source_details(work)
    authors = work_author_names(work)
    return {
        "openalex_work_id": work.get("id"),
        "doi": work.get("doi"),
        "title": clean_text(work.get("title")),
        "publication_year": work.get("publication_year"),
        "first_author_name": authors[0] if authors else None,
        "author_count": len(authors),
        "source_id": source_id,
        "source_name": clean_text(source_name) if source_name else None,
        **scores,
    }


def compact_work(work: dict[str, Any]) -> dict[str, Any]:
    topic = work.get("primary_topic") or {}
    subfield = topic.get("subfield") or {}
    field = topic.get("field") or {}
    authorships = work.get("authorships") or []
    author_country_sets: list[list[str]] = []
    for authorship in authorships:
        countries = {
            clean_text(code).upper()
            for code in (authorship.get("countries") or [])
            if clean_text(code)
        }
        if not countries:
            countries = {
                clean_text(institution.get("country_code")).upper()
                for institution in (authorship.get("institutions") or [])
                if clean_text(institution.get("country_code"))
            }
        author_country_sets.append(sorted(countries))
    first = next(
        (
            author_country_sets[index]
            for index, authorship in enumerate(authorships)
            if authorship.get("author_position") == "first"
        ),
        author_country_sets[0] if author_country_sets else [],
    )
    all_countries = sorted({code for codes in author_country_sets for code in codes})
    source_id, source_name = source_details(work)
    return {
        "openalex_work_id": work.get("id"),
        "doi": work.get("doi"),
        "openalex_title": clean_text(work.get("title")),
        "openalex_year": work.get("publication_year"),
        "openalex_type": work.get("type"),
        "primary_topic_id": topic.get("id"),
        "primary_topic_name": clean_text(topic.get("display_name")),
        "primary_subfield_id": subfield.get("id"),
        "primary_subfield_name": clean_text(subfield.get("display_name")),
        "primary_field_id": field.get("id"),
        "primary_field_name": clean_text(field.get("display_name")),
        "source_id": source_id,
        "source_name": clean_text(source_name) if source_name else None,
        "author_count_openalex": len(authorships),
        "is_authors_truncated": bool(work.get("is_authors_truncated")),
        "first_author_country_codes": first,
        "all_authorship_country_codes": all_countries,
        "authorship_country_sets": author_country_sets,
    }


def classify_match(ranked: list[tuple[dict[str, Any], dict[str, Any]]]) -> tuple[str, str, float | None]:
    if not ranked:
        return "rejected", "no_candidates", None
    top_work, top = ranked[0]
    del top_work
    second_score = ranked[1][1]["combined_score"] if len(ranked) > 1 else None
    margin = top["combined_score"] - second_score if second_score is not None else math.inf
    second_credible = bool(
        len(ranked) > 1
        and ranked[1][1]["title_sequence"] >= 0.94
        and ranked[1][1]["any_author_exact"]
    )
    if second_credible and margin < 0.025:
        return "ambiguous", "competing_credible_candidates", float(margin)
    high = (
        top["title_sequence"] >= 0.98
        and top["title_token_jaccard"] >= 0.94
        and top["first_author_exact"]
        and top["author_jaccard"] >= 0.20
        and margin >= 0.025
    )
    if high:
        return "accepted_high", "strong_title_author_margin", float(margin)
    medium = (
        top["title_sequence"] >= 0.94
        and top["title_token_jaccard"] >= 0.85
        and top["first_author_exact"]
        and top["any_author_exact"]
        and margin >= 0.015
    )
    if medium:
        return "accepted_medium", "adequate_title_author_margin", float(margin)
    review = top["title_sequence"] >= 0.92 and (
        top["any_author_exact"] or top["title_sequence"] >= 0.985
    )
    if review:
        return "review", "plausible_but_not_autoaccepted", None if math.isinf(margin) else float(margin)
    return "rejected", "insufficient_title_author_agreement", None if math.isinf(margin) else float(margin)


class Client:
    def __init__(self, *, mailto: str | None, api_key: str | None, pause: float) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "ai-geographic-entry/0.4 (public bibliometric research)"}
        )
        self.mailto, self.api_key, self.pause = mailto, api_key, pause
        self.last_request = 0.0

    def search(self, title: str, year: int, retries: int = 6) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params: dict[str, Any] = {
            "search": clean_text(title),
            "filter": (f"from_publication_date:{year - 1}-01-01," f"to_publication_date:{year + 1}-12-31"),
            "per_page": 10,
            "select": SELECT,
        }
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        for attempt in range(retries):
            wait = self.pause - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            response = self.session.get(API_URL, params=params, timeout=90)
            self.last_request = time.monotonic()
            if response.status_code == 200:
                payload = response.json()
                return payload.get("results") or [], payload.get("meta") or {}
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            if attempt + 1 == retries:
                response.raise_for_status()
            retry = response.headers.get("Retry-After")
            time.sleep(float(retry) if retry else min(30.0, 2**attempt + random.random()))
        raise RuntimeError("unreachable retry loop")


def load_roster() -> pd.DataFrame:
    columns = ["paper_key", "dblp_key", "venue", "year", "title", "author_names", "electronic_editions"]
    frame = pd.read_csv(ROSTER_PATH, usecols=columns)
    frame = frame.loc[
        frame["venue"].isin(VENUES) & frame["year"].between(min(YEARS), max(YEARS))
    ].copy()
    if len(frame) != 26_872 or frame["paper_key"].nunique() != len(frame):
        raise RuntimeError(
            f"Frozen target roster mismatch: rows={len(frame):,}, unique={frame['paper_key'].nunique():,}"
        )
    return frame.sort_values(["venue", "year", "paper_key"]).reset_index(drop=True)


def select_run(frame: pd.DataFrame, mode: str, sample_per_cell: int, seed: int) -> pd.DataFrame:
    if mode == "full":
        return frame
    return (
        frame.groupby(["venue", "year"], group_keys=False)
        .sample(n=sample_per_cell, random_state=seed)
        .sort_values(["venue", "year", "paper_key"])
        .reset_index(drop=True)
    )


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Corrupt resumable cache at {path}:{line_number}; refusing silent recovery") from error
                records[str(record["paper_key"])] = record
    return records


def write_qc(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    *,
    mode: str,
    cache_path: Path,
) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    selected_keys = set(selected["paper_key"])
    rows = [records[key] for key in selected_keys if key in records]
    result = pd.DataFrame(rows)
    status_order = ["accepted_high", "accepted_medium", "review", "ambiguous", "rejected", "api_error"]
    cells: list[dict[str, Any]] = []
    for (venue, year), block in selected.groupby(["venue", "year"]):
        block_records = result.loc[
            result.get("venue", pd.Series(dtype=str)).eq(venue)
            & pd.to_numeric(result.get("year", pd.Series(dtype=float)), errors="coerce").eq(year)
        ] if not result.empty else pd.DataFrame()
        counts = block_records.get("match_status", pd.Series(dtype=str)).value_counts()
        row = {"venue": venue, "year": year, "selected": len(block), "cached": len(block_records)}
        for status in status_order:
            row[status] = int(counts.get(status, 0))
        row["autoaccepted"] = row["accepted_high"] + row["accepted_medium"]
        row["autoaccept_rate_of_cached"] = row["autoaccepted"] / max(1, row["cached"])
        cells.append(row)
    cell_frame = pd.DataFrame(cells)
    cell_frame.to_csv(QC_DIR / f"top_match_status_by_cell_{mode}.csv", index=False)
    sample_complete = bool(len(cell_frame) == len(VENUES) * len(YEARS) and cell_frame["cached"].eq(cell_frame["selected"]).all())
    gate_pass = bool(sample_complete and cell_frame["autoaccept_rate_of_cached"].ge(0.80).all())
    country = pd.read_csv(COUNTRY_PATH, usecols=["paper_id", "primary_country_codes"]).drop_duplicates("paper_id")
    country["country_codes"] = country["primary_country_codes"].map(parse_json_list)
    match_rows = result[["paper_key", "venue", "year", "match_status"]].copy() if not result.empty else pd.DataFrame(columns=["paper_key", "venue", "year", "match_status"])
    match_rows["autoaccepted"] = match_rows["match_status"].isin(["accepted_high", "accepted_medium"])
    match_rows = match_rows.merge(country[["paper_id", "country_codes"]], left_on="paper_key", right_on="paper_id", how="left")
    match_rows["country_codes"] = match_rows["country_codes"].map(lambda value: value if isinstance(value, list) and value else ["UNRESOLVED"])
    country_rows = match_rows.explode("country_codes").rename(columns={"country_codes": "country_code"})
    country_qc = country_rows.groupby(["country_code", "year"], as_index=False).agg(cached=("paper_key", "nunique"), autoaccepted=("autoaccepted", "sum"))
    country_qc["autoaccept_rate"] = country_qc["autoaccepted"] / country_qc["cached"]
    country_qc.to_csv(QC_DIR / f"top_match_status_by_country_year_{mode}.csv", index=False)
    ambiguity = result.loc[result.get("match_status", pd.Series(dtype=str)).isin(["review", "ambiguous", "rejected"])].copy() if not result.empty else result
    private_dir = QC_DIR / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_columns = ["paper_key", "venue", "year", "source_title", "source_first_author", "match_status", "match_reason", "score_margin", "best_openalex_work_id", "best_candidate_title", "best_combined_score", "best_title_sequence", "best_author_jaccard", "best_first_author_exact", "candidate_summaries"]
    ambiguity.reindex(columns=private_columns).to_csv(private_dir / f"top_match_review_queue_{mode}_PRIVATE.csv", index=False)
    public_columns = ["paper_key", "venue", "year", "match_status", "match_reason", "score_margin", "best_combined_score", "best_title_sequence", "best_author_jaccard", "best_first_author_exact"]
    ambiguity.reindex(columns=public_columns).to_csv(QC_DIR / f"top_match_review_queue_{mode}.csv", index=False)
    status_counts = result.get("match_status", pd.Series(dtype=str)).value_counts().to_dict()
    manifest = {
        "generated_at_utc": utc_now(),
        "mode": mode,
        "roster_path": str(ROSTER_PATH),
        "roster_sha256": sha256_file(ROSTER_PATH),
        "population_rows": len(population),
        "selected_rows": len(selected),
        "roster_doi_available": 0,
        "doi_match_attempted": 0,
        "fallback_reason": "no_doi_in_frozen_roster",
        "sample_complete": sample_complete,
        "full_run_gate_min_cell_autoaccept_rate": 0.80,
        "full_run_gate_pass": gate_pass,
        "full_run_gate_status": ("PASS" if gate_pass else ("FULL_MATCH_GATE_FAIL" if sample_complete else "INCOMPLETE")),
        "cached_selected_rows": len(result),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path) if cache_path.exists() else None,
        "no_silent_first_result": True,
        "candidate_count_requested": 10,
        "selection_rules_version": "v0.4.1",
    }
    (QC_DIR / f"top_match_manifest_{mode}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--sample-per-cell", type=int, default=10)
    parser.add_argument("--api-key-file")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--mailto")
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--max-new-requests", type=int)
    args = parser.parse_args()
    args.api_key = read_api_key(args.api_key_file, os.environ.get("OPENALEX_API_KEY"))

    population = load_roster()
    selected = select_run(population, args.mode, args.sample_per_cell, args.seed)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    (INTERIM_DIR / "top_target_roster_2018_2024.csv").write_text(
        population.to_csv(index=False), encoding="utf-8"
    )
    cache_path = INTERIM_DIR / "top_match_records.jsonl"
    existing = load_cache(cache_path)
    retryable = {key for key, record in existing.items() if record.get("match_status") == "api_error"}
    pending = selected.loc[~selected["paper_key"].isin(existing) | selected["paper_key"].isin(retryable)].copy()
    if args.max_new_requests is not None:
        pending = pending.head(args.max_new_requests)
    print(
        f"population={len(population):,} selected={len(selected):,} "
        f"cached_selected={selected['paper_key'].isin(existing).sum():,} pending_now={len(pending):,}"
    )
    client = Client(mailto=args.mailto, api_key=args.api_key, pause=args.pause)
    with cache_path.open("a", encoding="utf-8") as output:
        for position, row in enumerate(pending.itertuples(index=False), start=1):
            authors = parse_json_list(row.author_names)
            try:
                candidates, meta = client.search(row.title, int(row.year))
                ranked = [(work, score_candidate(row.title, authors, int(row.year), work)) for work in candidates]
                ranked.sort(key=lambda item: item[1]["combined_score"], reverse=True)
                status, reason, margin = classify_match(ranked)
                summaries = [candidate_summary(work, scores) for work, scores in ranked]
                best_work, best_scores = ranked[0] if ranked else ({}, {})
                accepted = status in {"accepted_high", "accepted_medium"}
                record: dict[str, Any] = {
                    "paper_key": row.paper_key,
                    "venue": row.venue,
                    "year": int(row.year),
                    "source_title": clean_text(row.title),
                    "source_first_author": authors[0] if authors else None,
                    "queried_at_utc": utc_now(),
                    "match_status": status,
                    "match_reason": reason,
                    "score_margin": margin,
                    "candidate_count": len(ranked),
                    "query_result_count": meta.get("count"),
                    "candidate_summaries": summaries,
                    "best_openalex_work_id": best_work.get("id"),
                    "best_candidate_title": clean_text(best_work.get("title")),
                    "best_combined_score": best_scores.get("combined_score"),
                    "best_title_sequence": best_scores.get("title_sequence"),
                    "best_author_jaccard": best_scores.get("author_jaccard"),
                    "best_first_author_exact": best_scores.get("first_author_exact"),
                    "matched_work": compact_work(best_work) if accepted else None,
                }
            except Exception as error:  # cache the failure explicitly; never silently match.
                record = {
                    "paper_key": row.paper_key,
                    "venue": row.venue,
                    "year": int(row.year),
                    "source_title": clean_text(row.title),
                    "source_first_author": authors[0] if authors else None,
                    "queried_at_utc": utc_now(),
                    "match_status": "api_error",
                    "match_reason": f"{type(error).__name__}: {clean_text(error)}",
                    "matched_work": None,
                }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            existing[row.paper_key] = record
            if position % 25 == 0 or position == len(pending):
                print(f"processed={position:,}/{len(pending):,} last={row.paper_key} status={record['match_status']}")

    write_qc(population, selected, existing, mode=args.mode, cache_path=cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

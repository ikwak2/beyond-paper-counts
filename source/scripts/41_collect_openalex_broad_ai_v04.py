#!/usr/bin/env python3
"""Collect a compact, resumable OpenAlex broad-AI work-level denominator."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_ROOT = PROJECT_ROOT / "data" / "interim" / "openalex_pri_v04" / "private"
QC_DIR = PROJECT_ROOT / "outputs" / "facct_locked_completion_v04" / "pri_qc"
API_URL = "https://api.openalex.org/works"
YEARS = tuple(range(2018, 2025))
FILTER_BASE = "primary_topic.subfield.id:1702,type:article|conference-paper"
SELECT = "id,title,publication_year,type,primary_topic,authorships,is_authors_truncated,primary_location"
SCHEMA_VERSION = "openalex_pri_v04.1"
REQUIRED_FIELDS = {"openalex_work_id", "schema_version", "title_normalized_PRIVATE", "publication_year", "work_type", "primary_topic_id", "authorship_positions", "authorship_country_sets", "all_country_codes", "first_author_country_codes_strict", "first_author_country_codes_row0_fallback", "paper_country_fractional", "first_author_country_fractional_strict", "first_author_country_fractional_row0_fallback", "authorship_country_fractional_all_authors", "authorship_country_fractional_covered_authors", "author_count", "authors_with_country", "is_authors_truncated", "missing_author_position_count"}


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



def clean(value: Any) -> str:
    return str(value or "").encode("utf-8", "replace").decode("utf-8")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value)).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def author_countries(authorship: dict[str, Any]) -> list[str]:
    countries = {clean(code).upper() for code in authorship.get("countries") or [] if clean(code)}
    if not countries:
        countries = {clean(inst.get("country_code")).upper() for inst in authorship.get("institutions") or [] if clean(inst.get("country_code"))}
    return sorted(countries)


def compact_work(work: dict[str, Any]) -> dict[str, Any]:
    topic = work.get("primary_topic") or {}
    subfield = topic.get("subfield") or {}
    source = ((work.get("primary_location") or {}).get("source") or {})
    authorships = work.get("authorships") or []
    positions = [clean(item.get("author_position")) or "MISSING" for item in authorships]
    country_sets = [author_countries(item) for item in authorships]
    first_position_indices = [i for i, item in enumerate(authorships) if item.get("author_position") == "first"]
    first_position_found = bool(first_position_indices)
    strict_first_index = first_position_indices[0] if first_position_found else None
    fallback_first_index = strict_first_index if strict_first_index is not None else (0 if authorships else None)
    first_position_fallback_to_row0 = bool(authorships and not first_position_found)
    missing_author_position_count = sum(not item.get("author_position") for item in authorships)
    strict_first_codes = country_sets[strict_first_index] if strict_first_index is not None else []
    fallback_first_codes = country_sets[fallback_first_index] if fallback_first_index is not None else []
    all_codes = sorted({code for codes in country_sets for code in codes})
    n_authors = len(authorships)
    n_covered = sum(bool(codes) for codes in country_sets)
    paper_fractional = {code: 1.0 / len(all_codes) for code in all_codes} if all_codes else {}
    strict_first_fractional = {code: 1.0 / len(strict_first_codes) for code in strict_first_codes} if strict_first_codes else {}
    fallback_first_fractional = {code: 1.0 / len(fallback_first_codes) for code in fallback_first_codes} if fallback_first_codes else {}
    auth_all: dict[str, float] = defaultdict(float)
    auth_covered: dict[str, float] = defaultdict(float)
    for codes in country_sets:
        if not codes:
            continue
        for code in codes:
            auth_all[code] += 1.0 / max(1, n_authors) / len(codes)
            auth_covered[code] += 1.0 / max(1, n_covered) / len(codes)
    return {
        "schema_version": SCHEMA_VERSION,
        "openalex_work_id": work.get("id"),
        "title_normalized_PRIVATE": normalize(work.get("title")),
        "publication_year": work.get("publication_year"),
        "work_type": work.get("type"),
        "primary_topic_id": topic.get("id"),
        "primary_topic_name": clean(topic.get("display_name")),
        "primary_subfield_id": subfield.get("id"),
        "primary_subfield_name": clean(subfield.get("display_name")),
        "source_id": source.get("id"),
        "source_name": clean(source.get("display_name")),
        "author_count": n_authors,
        "authors_with_country": n_covered,
        "is_authors_truncated": bool(work.get("is_authors_truncated")),
        "missing_author_position_count": missing_author_position_count,
        "authorship_positions": positions,
        "authorship_country_sets": country_sets,
        "first_position_found": first_position_found,
        "first_position_fallback_to_row0": first_position_fallback_to_row0,
        "first_author_country_codes_strict": strict_first_codes,
        "first_author_country_codes_row0_fallback": fallback_first_codes,
        "paper_country_fractional": paper_fractional,
        "first_author_country_fractional_strict": strict_first_fractional,
        "all_country_codes": all_codes,
        "first_author_country_fractional_row0_fallback": fallback_first_fractional,
        "authorship_country_fractional_all_authors": dict(auth_all),
        "authorship_country_fractional_covered_authors": dict(auth_covered),
    }


class Client:
    def __init__(self, *, mailto: str | None, api_key: str | None, pause: float) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "ai-geographic-entry/0.4 (public bibliometric research)"}
        )
        self.mailto, self.api_key, self.pause = mailto, api_key, pause
        self.last_request = 0.0

    def page(self, *, year: int, cursor: str, retries: int = 7) -> dict[str, Any]:
        params: dict[str, Any] = {
            "filter": f"{FILTER_BASE},publication_year:{year}",
            "per_page": 100,
            "cursor": cursor,
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
            response = self.session.get(API_URL, params=params, timeout=180)
            self.last_request = time.monotonic()
            if response.status_code == 200:
                return response.json()
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            if attempt + 1 == retries:
                response.raise_for_status()
            retry = response.headers.get("Retry-After")
            time.sleep(float(retry) if retry else min(60.0, 2**attempt + random.random()))
        raise RuntimeError("unreachable retry loop")


def year_state(year_dir: Path) -> tuple[int, str, int]:
    manifests = sorted(year_dir.glob("page_*.manifest.json"))
    if not manifests:
        return 1, "*", 0
    cumulative = 0
    last: dict[str, Any] = {}
    for expected_page, manifest_path in enumerate(manifests, start=1):
        item = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(item.get("page_number", -1)) != expected_page:
            raise RuntimeError(f"Non-contiguous page manifest sequence at {manifest_path}")
        expected_name = f"page_{expected_page:05d}.jsonl"
        if item.get("page_file") != expected_name:
            raise RuntimeError(f"Manifest/page name mismatch at {manifest_path}")
        page_path = year_dir / expected_name
        if not page_path.exists() or sha256_file(page_path) != item.get("sha256"):
            raise RuntimeError(f"Incomplete or altered cached page: {page_path}")
        cumulative += int(item.get("works_on_page", 0))
        if cumulative != int(item.get("cumulative_works", -1)):
            raise RuntimeError(f"Cumulative count mismatch at {manifest_path}")
        last = item
    return len(manifests) + 1, str(last.get("next_cursor")), cumulative

def write_page(year_dir: Path, page_number: int, payload: dict[str, Any], cumulative: int) -> tuple[str, int]:
    results = payload.get("results") or []
    compact = [compact_work(work) for work in results]
    final_path = year_dir / f"page_{page_number:05d}.jsonl"
    temp_path = final_path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in compact:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(final_path)
    cumulative += len(compact)
    next_cursor = (payload.get("meta") or {}).get("next_cursor")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "retrieved_at_utc": utc_now(),
        "page_number": page_number,
        "page_file": final_path.name,
        "sha256": sha256_file(final_path),
        "works_on_page": len(compact),
        "cumulative_works": cumulative,
        "openalex_count": (payload.get("meta") or {}).get("count"),
        "next_cursor": next_cursor,
        "cost_usd": (payload.get("meta") or {}).get("cost_usd"),
        "filter": FILTER_BASE,
        "year": int(year_dir.name),
        "select": SELECT,
    }
    manifest_path = year_dir / f"page_{page_number:05d}.manifest.json"
    manifest_tmp = manifest_path.with_suffix(".manifest.json.tmp")
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    manifest_tmp.replace(manifest_path)
    return str(next_cursor), cumulative


def iter_cached_rows(base: Path):
    for page in sorted(base.glob("20*/page_*.jsonl")):
        with page.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def migrate_smoke_cache_v041(base: Path) -> dict[str, Any]:
    provenance: list[dict[str, Any]] = []
    for page in sorted(base.glob("20*/page_*.jsonl")):
        source_sha = sha256_file(page)
        rows: list[dict[str, Any]] = []
        with page.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["schema_version"] = SCHEMA_VERSION
                missing = REQUIRED_FIELDS - set(row)
                if missing:
                    raise RuntimeError(f"Migration cannot derive required fields at {page}:{line_number}: {sorted(missing)}")
                rows.append(row)
        temp = page.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(page)
        target_sha = sha256_file(page)
        manifest_path = page.with_name(page.stem + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = SCHEMA_VERSION
        manifest["sha256"] = target_sha
        manifest["schema_migration"] = "v0.4.1 add explicit schema_version after required-field verification"
        manifest_temp = manifest_path.with_suffix(".manifest.json.tmp")
        with manifest_temp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        manifest_temp.replace(manifest_path)
        provenance.append({"page": str(page.relative_to(PROJECT_ROOT)), "source_sha256": source_sha, "target_sha256": target_sha, "rows": len(rows)})
    output = {"generated_at_utc": utc_now(), "migration": "openalex_pri_v04.1", "network_requests": 0, "pages": provenance}
    QC_DIR.mkdir(parents=True, exist_ok=True)
    (QC_DIR / "broad_ai_smoke_v041_migration_provenance.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def finalize_qc(base: Path, mode: str) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    aggregates: dict[tuple[int, str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    seen: set[str] = set()
    duplicate = 0
    for row in iter_cached_rows(base):
        missing_fields = REQUIRED_FIELDS - set(row)
        if missing_fields:
            raise RuntimeError(f"Cache schema gate failed for {row.get('openalex_work_id')}: missing={sorted(missing_fields)}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Cache schema version mismatch: {row.get('schema_version')} != {SCHEMA_VERSION}")
        work_id = str(row["openalex_work_id"])
        duplicate += int(work_id in seen)
        seen.add(work_id)
        year = int(row["publication_year"])
        topic_id = str(row.get("primary_topic_id") or "MISSING")
        topic_name = str(row.get("primary_topic_name") or "MISSING")
        rows.append({
            "year": year,
            "work_type": row.get("work_type"),
            "has_topic": topic_id != "MISSING",
            "has_any_country": bool(row.get("all_country_codes")),
            "has_first_country": bool(row.get("first_author_country_codes_strict")),
            "authors_truncated": bool(row.get("is_authors_truncated")),
            "first_position_fallback_to_row0": bool(row.get("first_position_fallback_to_row0")),
            "missing_author_position_count": int(row.get("missing_author_position_count") or 0),
        })
        metrics: dict[str, dict[str, float]] = {
            "paper_country_full": {code: 1.0 for code in row.get("all_country_codes") or []},
            "paper_country_fractional": row.get("paper_country_fractional") or {},
            "first_author_full": {code: 1.0 for code in row.get("first_author_country_codes_strict") or []},
            "first_author_full_row0_fallback": {code: 1.0 for code in row.get("first_author_country_codes_row0_fallback") or []},
            "first_author_fractional_row0_fallback": row.get("first_author_country_fractional_row0_fallback") or {},
            "first_author_fractional": row.get("first_author_country_fractional_strict") or {},
            "authorship_fractional_all_authors": row.get("authorship_country_fractional_all_authors") or {},
            "authorship_fractional_covered_authors": row.get("authorship_country_fractional_covered_authors") or {},
        }
        for metric, credits in metrics.items():
            for country, value in credits.items():
                key = (year, topic_id, topic_name, country)
                aggregates[key][metric] += float(value)

    frame = pd.DataFrame(rows)
    coverage = (
        frame.groupby("year", as_index=False).agg(
            cached_works=("year", "size"),
            topic_covered=("has_topic", "sum"),
            any_country_covered=("has_any_country", "sum"),
            first_country_covered=("has_first_country", "sum"),
            authors_truncated=("authors_truncated", "sum"),
            first_position_fallback_to_row0=("first_position_fallback_to_row0", "sum"),
            missing_author_position_count=("missing_author_position_count", "sum"),
        ) if not frame.empty else pd.DataFrame(columns=["year", "cached_works"])
    )
    if not frame.empty:
        frame.groupby(["year", "work_type"], dropna=False).size().rename("works").reset_index().to_csv(QC_DIR / f"broad_ai_work_type_counts_{mode}.csv", index=False)
    if not coverage.empty:
        for numerator, output in [
            ("topic_covered", "topic_coverage"),
            ("any_country_covered", "any_country_coverage"),
            ("first_country_covered", "first_country_coverage"),
        ]:
            coverage[output] = coverage[numerator] / coverage["cached_works"]
    coverage.to_csv(QC_DIR / f"broad_ai_work_coverage_{mode}.csv", index=False)
    aggregate_rows = []
    for (year, topic_id, topic_name, country), metrics in aggregates.items():
        aggregate_rows.append(
            {"year": year, "primary_topic_id": topic_id, "primary_topic_name": topic_name, "country_code": country, **metrics}
        )
    pd.DataFrame(aggregate_rows).to_csv(
        QC_DIR / f"broad_ai_country_topic_aggregates_{mode}.csv", index=False
    )

    manifest_paths = sorted(base.glob("20*/page_*.manifest.json"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    year_completion = {}
    for year in YEARS:
        year_items = [item for path, item in zip(manifest_paths, manifests) if path.parent.name == str(year)]
        if not year_items:
            year_completion[str(year)] = {"complete": False, "reason": "no_pages"}
            continue
        last = year_items[-1]
        terminal = last.get("next_cursor") in {None, "", "None"}
        count_equal = int(last.get("cumulative_works", -1)) == int(last.get("openalex_count", -2))
        year_completion[str(year)] = {"complete": bool(terminal and count_equal), "terminal_cursor": terminal, "cumulative_works": int(last.get("cumulative_works", 0)), "openalex_count": int(last.get("openalex_count", 0))}
    schema_gate_pass = bool(not frame.empty and coverage["any_country_coverage"].gt(0).all())
    full_complete = bool(mode == "full" and len(year_completion) == len(YEARS) and all(item["complete"] for item in year_completion.values()) and duplicate == 0 and schema_gate_pass)
    validation_status = "PASS" if full_complete else ("SCHEMA_GATE_FAIL" if not schema_gate_pass and not frame.empty else "INCOMPLETE")
    validation = {
        "generated_at_utc": utc_now(),
        "mode": mode,
        "status": validation_status,
        "filter": FILTER_BASE,
        "select": SELECT,
        "cached_unique_works": len(seen),
        "schema_version": SCHEMA_VERSION,
        "schema_gate_pass": schema_gate_pass,
        "required_fields": sorted(REQUIRED_FIELDS),
        "duplicate_work_ids": duplicate,
        "cached_pages": len(manifests),
        "year_completion": year_completion,
        "expected_full_works_from_frozen_v01_metadata_reference_only": 864215,
        "expected_full_requests_at_100_per_page_reference": 8643,
        "person_names_or_ids_retained_in_broad_cache": False,
        "ordered_authorship_positions_and_country_sets_retained": True,
        "raw_affiliation_strings_retained": False,
        "work_level_country_and_topic_available": True,
        "role_matched_first_author_denominator_available": True,
        "fractional_denominators_available": True,
    }
    (QC_DIR / f"broad_ai_validation_{mode}.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--pages-per-year", type=int, default=1)
    parser.add_argument("--mailto")
    parser.add_argument("--api-key-file")
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--migrate-smoke-cache", action="store_true")
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()
    args.api_key = read_api_key(args.api_key_file, os.environ.get("OPENALEX_API_KEY"))
    if not set(args.years).issubset(YEARS):
        raise ValueError(f"Years must be within {YEARS}")
    if args.migrate_smoke_cache:
        base = INTERIM_ROOT / "broad_ai_smoke_v041"
        print(json.dumps(migrate_smoke_cache_v041(base), ensure_ascii=False, indent=2))
        finalize_qc(base, "smoke")
        return 0
    if args.mode == "full" and not args.api_key:
        QC_DIR.mkdir(parents=True, exist_ok=True)
        status = {"generated_at_utc": utc_now(), "mode": "full", "status": "API_KEY_MISSING_FULL_NOT_STARTED", "estimated_or_title_batch_cost_usd": 0.59, "estimated_broad_list_cost_usd": 0.8643, "paid_run_started": False}
        (QC_DIR / "broad_ai_validation_full.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return 78
    branch = "broad_ai_smoke_v041" if args.mode == "smoke" else "broad_ai_full_v041"
    base = INTERIM_ROOT / branch
    base.mkdir(parents=True, exist_ok=True)
    client = Client(mailto=args.mailto, api_key=args.api_key, pause=args.pause)
    total_new_pages = 0
    for year in sorted(args.years):
        year_dir = base / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        page_number, cursor, cumulative = year_state(year_dir)
        limit = args.pages_per_year if args.mode == "smoke" else None
        pages_this_year = 0
        if limit is not None and page_number > limit:
            print(f"year={year} smoke cache already has {page_number - 1} page(s); skipping")
            continue
        while cursor and cursor != "None":
            if limit is not None and pages_this_year >= limit:
                break
            if args.max_pages is not None and total_new_pages >= args.max_pages:
                break
            payload = client.page(year=year, cursor=cursor)
            cursor, cumulative = write_page(year_dir, page_number, payload, cumulative)
            pages_this_year += 1
            total_new_pages += 1
            print(
                f"year={year} page={page_number:,} rows={len(payload.get('results') or []):,} "
                f"cumulative={cumulative:,} expected={(payload.get('meta') or {}).get('count'):,}"
            )
            page_number += 1
        if args.max_pages is not None and total_new_pages >= args.max_pages:
            break
    finalize_qc(base, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

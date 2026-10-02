#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.openalex.org/works"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "openalex" / "capacity_v01"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "openalex_capacity_country_year_v01.csv"
)
MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "tables" / "openalex_capacity_v01_manifest.csv"
)

SPECS = {
    "ai_primary_peer_reviewed": (
        "primary_topic.subfield.id:1702,type:article|conference-paper"
    ),
    "ai_primary_all_types": "primary_topic.subfield.id:1702",
    "cs_primary_peer_reviewed": (
        "primary_topic.field.id:17,type:article|conference-paper"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(
    session: requests.Session,
    params: dict[str, Any],
    *,
    retries: int = 6,
) -> tuple[dict[str, Any], str]:
    for attempt in range(retries):
        response = session.get(API_URL, params=params, timeout=60)
        if response.status_code == 200:
            return response.json(), response.url
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        if attempt + 1 == retries:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        wait_seconds = float(retry_after) if retry_after else min(30, 2**attempt)
        time.sleep(wait_seconds)
    raise RuntimeError("OpenAlex request retry loop ended unexpectedly")


def fetch_or_load(
    session: requests.Session,
    *,
    spec: str,
    year: int,
    refresh: bool,
    mailto: str | None,
) -> tuple[dict[str, Any], Path, bool]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{spec}_{year}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8")), path, True

    filters = f"{SPECS[spec]},publication_year:{year}"
    params: dict[str, Any] = {
        "filter": filters,
        "group_by": "authorships.institutions.country_code",
        "per-page": 200,
    }
    if mailto:
        params["mailto"] = mailto
    payload, request_url = get_json(session, params)
    wrapped = {
        "retrieved_at_utc": utc_now(),
        "specification": spec,
        "year": year,
        "request_url": request_url,
        "filter": filters,
        "response": payload,
    }
    path.write_text(
        json.dumps(wrapped, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return wrapped, path, False


def country_code_from_key(key: str) -> str:
    return str(key).rstrip("/").split("/")[-1].upper()


def records_from_payload(wrapped: dict[str, Any]) -> list[dict[str, Any]]:
    payload = wrapped["response"]
    meta = payload.get("meta", {})
    groups = payload.get("group_by", [])
    expected_groups = int(meta.get("groups_count") or 0)
    if expected_groups > len(groups):
        raise RuntimeError(
            f"Truncated grouped response: expected {expected_groups}, got {len(groups)}"
        )
    grouped_total = sum(int(item.get("count") or 0) for item in groups)
    rows: list[dict[str, Any]] = []
    for item in groups:
        rows.append(
            {
                "specification": wrapped["specification"],
                "year": int(wrapped["year"]),
                "country_code": country_code_from_key(item.get("key", "")),
                "country_name": item.get("key_display_name"),
                "country_work_count_full": int(item.get("count") or 0),
                "openalex_work_count": int(meta.get("count") or 0),
                "country_grouped_count_total": grouped_total,
                "countries_returned": len(groups),
                "openalex_groups_count": expected_groups,
                "retrieved_at_utc": wrapped["retrieved_at_utc"],
                "request_url": wrapped["request_url"],
            }
        )
    return rows


def parse_years(raw: str) -> list[int]:
    if "-" in raw:
        start, end = (int(value) for value in raw.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return sorted({int(value) for value in raw.split(",") if value.strip()})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch country-year broad AI/CS capacity counts from OpenAlex."
    )
    parser.add_argument("--years", default="2018-2025")
    parser.add_argument("--specifications", nargs="+", choices=sorted(SPECS), default=list(SPECS))
    parser.add_argument("--mailto")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--pause", type=float, default=0.15)
    args = parser.parse_args()

    years = parse_years(args.years)
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "ai-geographic-entry/0.1 (public bibliometric research)"}
    )

    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    total = len(args.specifications) * len(years)
    position = 0
    for spec in args.specifications:
        for year in years:
            position += 1
            wrapped, raw_path, from_cache = fetch_or_load(
                session,
                spec=spec,
                year=year,
                refresh=args.refresh,
                mailto=args.mailto,
            )
            rows = records_from_payload(wrapped)
            records.extend(rows)
            meta = wrapped["response"].get("meta", {})
            manifest.append(
                {
                    "specification": spec,
                    "year": year,
                    "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(raw_path),
                    "from_cache": from_cache,
                    "openalex_work_count": int(meta.get("count") or 0),
                    "countries_returned": len(rows),
                    "request_url": wrapped["request_url"],
                    "retrieved_at_utc": wrapped["retrieved_at_utc"],
                }
            )
            print(
                f"[{position}/{total}] {spec} {year}: "
                f"works={int(meta.get('count') or 0):,}, countries={len(rows)}"
                f"{' (cache)' if from_cache else ''}"
            )
            if not from_cache and args.pause:
                time.sleep(args.pause)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values(
        ["specification", "year", "country_work_count_full"],
        ascending=[True, True, False],
    )
    frame.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(manifest).to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {len(frame):,} country-year rows to {OUTPUT_PATH}")
    print(f"Wrote request manifest to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

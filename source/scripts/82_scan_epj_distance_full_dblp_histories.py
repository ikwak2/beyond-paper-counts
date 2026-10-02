#!/usr/bin/env python3
"""Recover five-year pre-entry DBLP title histories using globally unique PID aliases."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_entrant_cohort_private.csv"
XML = ROOT.parent / "bibliometrics/dblp/dblp.xml.gz"
DTD = ROOT.parent / "bibliometrics/dblp/dblp.dtd"
CONFIG = ROOT / "config/epj_distance_based_analysis_v10.json"
PROTOCOL = ROOT / "docs/epj_distance_based_analysis_protocol_v1.0.md"
OUT = ROOT / "outputs/epj_distance_based_v10"
PRIVATE = OUT / "private"
TABLES = OUT / "tables"
MANIFESTS = OUT / "manifests"

PUBLICATION_TAGS = {
    "article", "inproceedings", "incollection", "book", "phdthesis", "mastersthesis"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_name(elem: etree._Element) -> str:
    return etree.QName(elem).localname if elem.tag is not etree.Comment else ""


def clear_record(elem: etree._Element) -> None:
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]


def normalize_title(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def person_pid(key: str) -> str:
    prefix = "homepages/"
    return key[len(prefix) :] if key.startswith(prefix) else ""


def iter_context():
    handle = gzip.open(XML, "rb")
    context = etree.iterparse(
        handle,
        events=("end",),
        load_dtd=True,
        resolve_entities=True,
        no_network=True,
        huge_tree=True,
        recover=True,
    )
    return handle, context


def collect_target_aliases(
    target_pids: set[str],
    progress_every: int,
) -> tuple[dict[str, list[str]], int]:
    aliases: dict[str, list[str]] = {}
    records = 0
    handle, context = iter_context()
    try:
        for _, elem in context:
            tag = local_name(elem)
            if tag == "www":
                records += 1
                pid = person_pid(elem.get("key") or "")
                if pid in target_pids:
                    names = [
                        "".join(node.itertext()).strip()
                        for node in elem.findall("author")
                        if "".join(node.itertext()).strip()
                    ]
                    if names:
                        aliases[pid] = list(dict.fromkeys(names))
                if progress_every and records % progress_every == 0:
                    print(
                        f"pass=target_person_records www={records:,} targets={len(aliases):,}",
                        flush=True,
                    )
                clear_record(elem)
            elif tag in PUBLICATION_TAGS or tag == "proceedings":
                clear_record(elem)
    finally:
        handle.close()
    return aliases, records


def count_alias_owners(
    target_aliases: set[str],
    progress_every: int,
) -> tuple[dict[str, set[str]], int]:
    owners: dict[str, set[str]] = defaultdict(set)
    records = 0
    handle, context = iter_context()
    try:
        for _, elem in context:
            tag = local_name(elem)
            if tag == "www":
                records += 1
                pid = person_pid(elem.get("key") or "")
                if pid:
                    for node in elem.findall("author"):
                        name = "".join(node.itertext()).strip()
                        if name in target_aliases:
                            owners[name].add(pid)
                if progress_every and records % progress_every == 0:
                    print(
                        f"pass=global_alias_owners www={records:,} aliases_seen={len(owners):,}",
                        flush=True,
                    )
                clear_record(elem)
            elif tag in PUBLICATION_TAGS or tag == "proceedings":
                clear_record(elem)
    finally:
        handle.close()
    return owners, records


def scan_publications(
    alias_to_pid: dict[str, str],
    windows: dict[str, tuple[int, int]],
    progress_every: int,
) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, object]] = []
    records = 0
    raw_matches = 0
    min_year = min(start for start, _ in windows.values())
    max_year = max(end for _, end in windows.values())
    alias_keys = set(alias_to_pid)
    handle, context = iter_context()
    try:
        for _, elem in context:
            tag = local_name(elem)
            if tag in PUBLICATION_TAGS:
                records += 1
                year_node = elem.find("year")
                title_node = elem.find("title")
                try:
                    year = int(year_node.text.strip()) if year_node is not None and year_node.text else -1
                except ValueError:
                    year = -1
                if min_year <= year <= max_year:
                    matched_aliases = {
                        name
                        for node in elem.findall("author")
                        if (name := "".join(node.itertext()).strip()) in alias_keys
                    }
                    if matched_aliases:
                        title = "" if title_node is None else "".join(title_node.itertext()).strip()
                        normalized_title = normalize_title(title)
                    else:
                        normalized_title = ""
                    if normalized_title:
                        matched_pids = {alias_to_pid[name] for name in matched_aliases}
                        for pid in matched_pids:
                            start, end = windows[pid]
                            if start <= year <= end:
                                aliases_used = sorted(
                                    name for name in matched_aliases if alias_to_pid[name] == pid
                                )
                                raw_matches += 1
                                rows.append(
                                    {
                                        "author_pid": pid,
                                        "prior_year": year,
                                        "dblp_key": elem.get("key") or "",
                                        "publication_type": tag,
                                        "title": title,
                                        "normalized_title": normalized_title,
                                        "matched_aliases": json.dumps(aliases_used, ensure_ascii=False),
                                    }
                                )
                if progress_every and records % progress_every == 0:
                    print(
                        f"pass=publications records={records:,} raw_matches={raw_matches:,}",
                        flush=True,
                    )
                clear_record(elem)
            elif tag == "www" or tag == "proceedings":
                clear_record(elem)
    finally:
        handle.close()
    history = pd.DataFrame(rows)
    if history.empty:
        history = pd.DataFrame(
            columns=[
                "author_pid", "prior_year", "dblp_key", "publication_type",
                "title", "normalized_title", "matched_aliases",
            ]
        )
    history = history.drop_duplicates(["author_pid", "dblp_key"])
    return history, records, raw_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-every", type=int, default=500_000)
    args = parser.parse_args()
    for directory in (PRIVATE, TABLES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cohort = pd.read_csv(COHORT, keep_default_na=False)
    cohort["author_pid"] = cohort.author_pid.astype(str)
    if cohort.author_pid.eq("").any() or cohort.author_pid.duplicated().any():
        raise RuntimeError("cohort PID must be nonblank and unique")
    target_pids = set(cohort.author_pid)
    windows = {
        str(row.author_pid): (int(row.history_start_year), int(row.history_end_year))
        for row in cohort.itertuples(index=False)
    }

    aliases_by_pid, target_person_records_scanned = collect_target_aliases(
        target_pids, args.progress_every
    )
    # Preserve an observed canonical label only when it is also owned by the same PID globally.
    for row in cohort.itertuples(index=False):
        aliases_by_pid.setdefault(str(row.author_pid), []).append(str(row.author_name))
        aliases_by_pid[str(row.author_pid)] = list(
            dict.fromkeys(name for name in aliases_by_pid[str(row.author_pid)] if name)
        )
    target_aliases = {name for names in aliases_by_pid.values() for name in names}
    owners, owner_person_records_scanned = count_alias_owners(
        target_aliases, args.progress_every
    )

    alias_to_pid: dict[str, str] = {}
    colliding_aliases: dict[str, list[str]] = {}
    for pid, names in aliases_by_pid.items():
        for name in names:
            observed_owners = owners.get(name, set())
            if observed_owners == {pid}:
                alias_to_pid[name] = pid
            else:
                colliding_aliases[name] = sorted(observed_owners)
    pids_with_unique_alias = set(alias_to_pid.values())

    history, publication_records_scanned, raw_matches = scan_publications(
        alias_to_pid, windows, args.progress_every
    )
    history = history.merge(
        cohort[["entrant_id", "author_pid", "author_key", "index_year"]],
        on="author_pid",
        how="left",
        validate="many_to_one",
    )
    history = history.sort_values(["index_year", "entrant_id", "prior_year", "dblp_key"])
    history_path = PRIVATE / "epj_distance_full_dblp_history_private.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")

    primary_types = set(config["primary_history_record_types"])
    sensitivity_types = set(config["sensitivity_history_record_types"])
    primary = history.loc[history.publication_type.isin(primary_types)].copy()
    sensitivity = history.loc[history.publication_type.isin(sensitivity_types)].copy()
    primary_counts = primary.groupby("entrant_id").size()
    sensitivity_counts = sensitivity.groupby("entrant_id").size()
    profile = cohort.copy()
    profile["person_record_found"] = profile.author_pid.isin(aliases_by_pid)
    profile["has_globally_unique_alias"] = profile.author_pid.isin(pids_with_unique_alias)
    profile["n_person_aliases"] = profile.author_pid.map(
        lambda pid: len(aliases_by_pid.get(str(pid), []))
    )
    profile["n_globally_unique_aliases"] = profile.author_pid.map(
        lambda pid: sum(alias_to_pid.get(name) == str(pid) for name in aliases_by_pid.get(str(pid), []))
    )
    profile["n_primary_prior_titles"] = profile.entrant_id.map(primary_counts).fillna(0).astype(int)
    profile["n_sensitivity_prior_titles"] = profile.entrant_id.map(sensitivity_counts).fillna(0).astype(int)
    profile["primary_history_observed"] = profile.n_primary_prior_titles.gt(0)
    profile["sensitivity_history_observed"] = profile.n_sensitivity_prior_titles.gt(0)
    profile_path = PRIVATE / "epj_distance_full_history_profiles_private.csv"
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    coverage = (
        profile.groupby(["index_year", "entry_with_experienced_top4_coauthor"], as_index=False)
        .agg(
            entrants=("entrant_id", "size"),
            person_record_coverage=("person_record_found", "mean"),
            unique_alias_coverage=("has_globally_unique_alias", "mean"),
            primary_history_coverage=("primary_history_observed", "mean"),
            sensitivity_history_coverage=("sensitivity_history_observed", "mean"),
            median_primary_prior_titles=("n_primary_prior_titles", "median"),
        )
        .sort_values(["index_year", "entry_with_experienced_top4_coauthor"])
    )
    coverage_path = TABLES / "epj_distance_full_history_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    rng = np.random.default_rng(int(config["random_seed"]))
    audit_n = min(int(config["audit_authors"]), len(profile))
    audit_ids = rng.choice(profile.entrant_id.to_numpy(), size=audit_n, replace=False)
    audit_people = profile.loc[
        profile.entrant_id.isin(audit_ids),
        [
            "entrant_id", "author_pid", "author_name", "index_year",
            "history_start_year", "history_end_year", "n_person_aliases",
            "n_globally_unique_aliases", "n_primary_prior_titles",
        ],
    ]
    audit = audit_people.merge(
        primary[
            [
                "entrant_id", "prior_year", "dblp_key", "publication_type", "title",
                "matched_aliases",
            ]
        ],
        on="entrant_id",
        how="left",
        validate="one_to_many",
    ).sort_values(["entrant_id", "prior_year", "dblp_key"])
    audit_path = PRIVATE / "epj_distance_history_automated_audit_100_private.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    coverage_by_year = profile.groupby("index_year").primary_history_observed.mean()
    diagnostics = {
        "status": "PASS",
        "target_entrants": int(len(cohort)),
        "target_person_records_found": int(len(set(aliases_by_pid).intersection(target_pids))),
        "target_person_record_coverage": float(profile.person_record_found.mean()),
        "target_alias_strings": int(len(target_aliases)),
        "globally_unique_alias_strings": int(len(alias_to_pid)),
        "colliding_or_unowned_alias_strings": int(len(colliding_aliases)),
        "entrants_with_globally_unique_alias": int(profile.has_globally_unique_alias.sum()),
        "unique_history_rows_all_types": int(len(history)),
        "unique_history_rows_primary_types": int(len(primary)),
        "raw_history_matches_before_pid_key_deduplication": int(raw_matches),
        "primary_history_coverage_overall": float(profile.primary_history_observed.mean()),
        "primary_history_coverage_by_year": {str(k): float(v) for k, v in coverage_by_year.items()},
        "publication_type_counts": {
            str(k): int(v) for k, v in history.publication_type.value_counts().items()
        },
        "records_scanned": {
            "target_person_pass_www": int(target_person_records_scanned),
            "global_owner_pass_www": int(owner_person_records_scanned),
            "publication_pass": int(publication_records_scanned),
        },
        "audit_authors": int(audit_n),
        "input_sha256": {
            str(COHORT.relative_to(ROOT)): sha256_file(COHORT),
            str(XML): sha256_file(XML),
            str(DTD): sha256_file(DTD),
            str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG),
            str(PROTOCOL.relative_to(ROOT)): sha256_file(PROTOCOL),
        },
        "output_sha256": {
            str(history_path.relative_to(ROOT)): sha256_file(history_path),
            str(profile_path.relative_to(ROOT)): sha256_file(profile_path),
            str(coverage_path.relative_to(ROOT)): sha256_file(coverage_path),
            str(audit_path.relative_to(ROOT)): sha256_file(audit_path),
        },
    }
    if not profile.has_globally_unique_alias.all():
        # This is allowed: unresolved aliases stay missing, never force-linked.
        diagnostics["unresolved_identity_handling"] = "retained_in_cohort_with_no_forced_history_link"
    if len(primary) == 0:
        diagnostics["status"] = "FAIL"
    diagnostics_path = MANIFESTS / "epj_distance_full_history_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if diagnostics["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

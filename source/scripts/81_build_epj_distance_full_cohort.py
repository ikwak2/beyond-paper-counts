#!/usr/bin/env python3
"""Build the frozen EPJ ICML/NeurIPS entrant cohort and exact +2 outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT.parent / "bibliometrics/top_ai_entry/data/processed/accepted_authorships_marked.csv"
PAPERS = ROOT.parent / "bibliometrics/top_ai_entry/data/processed/accepted_papers_dblp.csv"
CONFIG = ROOT / "config/epj_distance_based_analysis_v10.json"
PROTOCOL = ROOT / "docs/epj_distance_based_analysis_protocol_v1.0.md"
OUT = ROOT / "outputs/epj_distance_based_v10"
PRIVATE = OUT / "private"
TABLES = OUT / "tables"
MANIFESTS = OUT / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def venue_label(values: pd.Series) -> str:
    venues = sorted(set(values))
    return venues[0] if len(venues) == 1 else "Both"


def classify_exact_plus2(
    cohort: pd.DataFrame,
    auth: pd.DataFrame,
    index_coauthors: dict[str, set[str]],
    venues: tuple[str, ...],
) -> tuple[pd.Series, dict[str, object]]:
    eligible = cohort.loc[cohort.index_year.le(2022), ["author_key", "index_year"]].copy()
    result = pd.Series("not_observed", index=cohort.author_key, dtype="object")
    result.loc[eligible.author_key] = "no_recurrence"

    appearances = auth.loc[
        auth.author_key.isin(set(eligible.author_key)) & auth.venue.isin(venues),
        ["paper_key", "venue", "year", "author_key"],
    ].merge(eligible, on="author_key", validate="many_to_one")
    appearances = appearances.loc[appearances.year.sub(appearances.index_year).eq(2)].copy()
    exact_papers = appearances[["author_key", "paper_key", "venue", "year"]].drop_duplicates()
    if exact_papers.empty:
        return result.reindex(cohort.author_key).reset_index(drop=True), {
            "eligible_entrants": int(len(eligible)),
            "entrants_with_exact_plus2": 0,
            "exact_plus2_papers": 0,
        }

    members = exact_papers.merge(
        auth[["paper_key", "venue", "year", "author_key"]].rename(
            columns={"author_key": "paper_member_key"}
        ),
        on=["paper_key", "venue", "year"],
        how="left",
        validate="many_to_many",
    )
    members["is_index_coauthor"] = [
        member in index_coauthors.get(entrant, set())
        for entrant, member in zip(members.author_key, members.paper_member_key)
    ]
    overlap = (
        members.groupby(["author_key", "paper_key", "venue", "year"], as_index=False)
        .agg(shares_index_coauthor=("is_index_coauthor", "max"))
    )
    categories = overlap.groupby("author_key").shares_index_coauthor.agg(
        lambda values: (
            "coauthor_continuity_only"
            if bool(values.all())
            else "at_least_one_no_index_coauthor"
        )
    )
    result.loc[categories.index] = categories
    diagnostics = {
        "eligible_entrants": int(len(eligible)),
        "entrants_with_exact_plus2": int(exact_papers.author_key.nunique()),
        "exact_plus2_papers": int(exact_papers.paper_key.nunique()),
    }
    return result.reindex(cohort.author_key).reset_index(drop=True), diagnostics


def main() -> None:
    for directory in (PRIVATE, TABLES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    index_venues = tuple(config["index_venues"])
    top4 = tuple(config["top4_venues"])
    year_start = int(config["index_year_start"])
    year_end = int(config["index_year_end"])

    auth = pd.read_csv(
        AUTH,
        usecols=[
            "paper_key", "venue", "year", "author_order", "author_name",
            "author_pid", "author_key", "identity_source", "top4_newcomer_5y",
        ],
        keep_default_na=False,
    )
    auth["year"] = pd.to_numeric(auth.year, errors="raise").astype(int)
    auth["author_order"] = pd.to_numeric(auth.author_order, errors="raise").astype(int)
    auth["top4_newcomer_5y"] = as_bool(auth.top4_newcomer_5y)
    papers = pd.read_csv(
        PAPERS,
        usecols=["paper_key", "venue", "year", "title", "author_count"],
        keep_default_na=False,
    )
    papers["year"] = pd.to_numeric(papers.year, errors="raise").astype(int)
    papers["author_count"] = pd.to_numeric(papers.author_count, errors="coerce")
    if papers.duplicated(["paper_key", "venue", "year"]).any():
        raise RuntimeError("duplicate accepted-paper keys")

    scope = auth.loc[
        auth.venue.isin(index_venues) & auth.year.between(year_start, year_end)
    ].copy()
    first = scope.loc[scope.author_order.eq(1) & scope.top4_newcomer_5y].copy()
    first["stable_pid"] = (
        first.identity_source.eq("dblp_pid")
        & first.author_pid.ne("")
        & first.author_key.ne("")
    )
    stable_coverage = (
        first.groupby(["venue", "year"], as_index=False)
        .agg(
            eligible_first_listed_papers=("paper_key", "size"),
            stable_pid_papers=("stable_pid", "sum"),
        )
        .sort_values(["venue", "year"])
    )
    stable_coverage["stable_pid_coverage"] = (
        stable_coverage.stable_pid_papers / stable_coverage.eligible_first_listed_papers
    )

    stable_first = first.loc[first.stable_pid].copy()
    if stable_first.duplicated(["paper_key", "venue", "year"]).any():
        raise RuntimeError("multiple eligible first-listed rows on one paper")
    coauthors = scope.loc[scope.author_order.gt(1)].copy()
    coauthors["coauthor_recent_top4_experienced"] = ~coauthors.top4_newcomer_5y
    coauthor_feature = (
        coauthors.groupby(["paper_key", "venue", "year"], as_index=False)
        .agg(entry_paper_has_experienced=("coauthor_recent_top4_experienced", "max"))
    )
    index_papers = stable_first.merge(
        papers,
        on=["paper_key", "venue", "year"],
        how="left",
        validate="one_to_one",
    ).merge(
        coauthor_feature,
        on=["paper_key", "venue", "year"],
        how="left",
        validate="one_to_one",
    )
    index_papers["entry_paper_has_experienced"] = (
        index_papers.entry_paper_has_experienced.fillna(False).astype(bool)
    )
    index_papers["team_size"] = index_papers.author_count
    if index_papers.title.eq("").any() or index_papers.team_size.isna().any():
        raise RuntimeError("missing index-paper title or team size")

    first_year = index_papers.groupby("author_key").year.min().rename("index_year")
    index_papers = index_papers.merge(first_year, on="author_key", validate="many_to_one")
    index_papers = index_papers.loc[index_papers.year.eq(index_papers.index_year)].copy()
    cohort = (
        index_papers.groupby(["author_key", "index_year"], as_index=False)
        .agg(
            author_name=("author_name", "first"),
            author_pid=("author_pid", "first"),
            n_index_papers=("paper_key", "size"),
            entry_team_size_mean=("team_size", "mean"),
            index_venue=("venue", venue_label),
            entry_with_experienced_top4_coauthor=("entry_paper_has_experienced", "max"),
        )
        .sort_values(["index_year", "author_key"])
        .reset_index(drop=True)
    )
    cohort["history_start_year"] = cohort.index_year - int(config["lookback_years"])
    cohort["history_end_year"] = cohort.index_year - 1

    index_key_rows = index_papers[["author_key", "paper_key", "venue", "year"]]
    index_members = index_key_rows.merge(
        auth[["paper_key", "venue", "year", "author_key"]].rename(
            columns={"author_key": "paper_member_key"}
        ),
        on=["paper_key", "venue", "year"],
        how="left",
        validate="many_to_many",
    )
    index_members = index_members.loc[
        index_members.paper_member_key.ne(index_members.author_key)
    ].copy()
    index_coauthors = (
        index_members.groupby("author_key").paper_member_key.agg(lambda x: set(x)).to_dict()
    )

    return_specs = {
        "exact_plus2_top4": top4,
        "exact_plus2_exclude_aaai": tuple(v for v in top4 if v != "AAAI"),
        "exact_plus2_icml_neurips": index_venues,
    }
    classification_diagnostics: dict[str, dict[str, object]] = {}
    for column, venues in return_specs.items():
        cohort[column], diagnostics = classify_exact_plus2(
            cohort, auth, index_coauthors, venues
        )
        diagnostics["venues"] = list(venues)
        classification_diagnostics[column] = diagnostics

    cohort.insert(0, "entrant_id", [f"EPJ{value:05d}" for value in range(1, len(cohort) + 1)])
    id_lookup = cohort[["entrant_id", "author_key"]]
    index_papers = index_papers.merge(id_lookup, on="author_key", validate="many_to_one")
    index_papers = index_papers[
        [
            "entrant_id", "author_key", "author_name", "author_pid", "index_year",
            "paper_key", "venue", "year", "title", "team_size",
            "entry_paper_has_experienced",
        ]
    ].sort_values(["index_year", "entrant_id", "paper_key"])

    cohort_path = PRIVATE / "epj_distance_full_entrant_cohort_private.csv"
    index_path = PRIVATE / "epj_distance_full_index_papers_private.csv"
    cohort.to_csv(cohort_path, index=False, encoding="utf-8-sig")
    index_papers.to_csv(index_path, index=False, encoding="utf-8-sig")
    coverage_path = TABLES / "epj_distance_full_stable_pid_coverage.csv"
    stable_coverage.to_csv(coverage_path, index=False)
    cohort_summary_all = (
        cohort.groupby(["index_year", "index_venue", "entry_with_experienced_top4_coauthor"], as_index=False)
        .agg(
            entrants=("entrant_id", "size"),
            median_team_size=("entry_team_size_mean", "median"),
            mean_team_size=("entry_team_size_mean", "mean"),
            median_index_papers=("n_index_papers", "median"),
            mean_index_papers=("n_index_papers", "mean"),
        )
        .sort_values(["index_year", "index_venue", "entry_with_experienced_top4_coauthor"])
    )
    public_minimum_cell_size = int(config["public_minimum_cell_size"])
    suppressed_public_cells = int(cohort_summary_all.entrants.lt(public_minimum_cell_size).sum())
    cohort_summary = cohort_summary_all.loc[
        cohort_summary_all.entrants.ge(public_minimum_cell_size)
    ].copy()
    summary_path = TABLES / "epj_distance_full_cohort_summary.csv"
    cohort_summary.to_csv(summary_path, index=False)

    observed = cohort.loc[cohort.index_year.le(int(config["exact_plus2_last_index_year"]))]
    diagnostics = {
        "status": "PASS",
        "entrants_2018_2024": int(len(cohort)),
        "entrants_2018_2022_exact_plus2_eligible": int(len(observed)),
        "unique_index_papers": int(index_papers.paper_key.nunique()),
        "multiple_index_paper_entrants": int(cohort.n_index_papers.gt(1).sum()),
        "stable_pid_coverage_minimum_cell": float(stable_coverage.stable_pid_coverage.min()),
        "duplicate_author_index_year": int(cohort.duplicated(["author_key", "index_year"]).sum()),
        "blank_pid": int(cohort.author_pid.eq("").sum()),
        "public_minimum_cell_size": public_minimum_cell_size,
        "suppressed_public_cohort_summary_cells": suppressed_public_cells,
        "exact_plus2_classification": classification_diagnostics,
        "input_sha256": {
            str(AUTH): sha256_file(AUTH),
            str(PAPERS): sha256_file(PAPERS),
            str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG),
            str(PROTOCOL.relative_to(ROOT)): sha256_file(PROTOCOL),
        },
        "output_sha256": {
            str(cohort_path.relative_to(ROOT)): sha256_file(cohort_path),
            str(index_path.relative_to(ROOT)): sha256_file(index_path),
            str(coverage_path.relative_to(ROOT)): sha256_file(coverage_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
        },
    }
    if diagnostics["duplicate_author_index_year"] or diagnostics["blank_pid"]:
        diagnostics["status"] = "FAIL"
    diagnostic_path = MANIFESTS / "epj_distance_full_cohort_diagnostics.json"
    diagnostic_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if diagnostics["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

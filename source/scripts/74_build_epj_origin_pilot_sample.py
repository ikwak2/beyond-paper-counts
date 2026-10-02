#!/usr/bin/env python3
"""Build the frozen ICML/NeurIPS disciplinary-origin feasibility sample.

Private output contains author identifiers and names. Public outputs contain only
aggregate sampling diagnostics. This script does not query a network service.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUTH = (
    ROOT.parent
    / "bibliometrics"
    / "top_ai_entry"
    / "data"
    / "processed"
    / "accepted_authorships_marked.csv"
)
PAPERS = (
    ROOT.parent
    / "bibliometrics"
    / "top_ai_entry"
    / "data"
    / "processed"
    / "accepted_papers_dblp.csv"
)
CONFIG = ROOT / "config" / "epj_origin_feasibility_v01.json"
PROTOCOL = ROOT / "docs" / "epj_origin_feasibility_protocol_v0.1.md"
OUT = ROOT / "outputs" / "epj_origin_feasibility_v01"
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


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    target_venues = tuple(config["target_venues"])
    pilot_years = tuple(int(value) for value in config["pilot_index_years"])
    n_per = int(config["sample_per_stratum"])
    seed = int(config["sample_seed"])

    for directory in (PRIVATE, TABLES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)

    auth = pd.read_csv(
        AUTH,
        usecols=[
            "paper_key",
            "venue",
            "year",
            "author_order",
            "author_name",
            "author_pid",
            "author_key",
            "identity_source",
            "top4_newcomer_5y",
        ],
        keep_default_na=False,
    )
    auth["year"] = pd.to_numeric(auth["year"], errors="raise").astype(int)
    auth["author_order"] = pd.to_numeric(auth["author_order"], errors="raise").astype(int)
    auth["top4_newcomer_5y"] = as_bool(auth["top4_newcomer_5y"])

    scope = auth.loc[
        auth.venue.isin(target_venues) & auth.year.between(min(pilot_years), max(pilot_years))
    ].copy()
    sizes = (
        scope.groupby(["paper_key", "venue", "year"], as_index=False)
        .size()
        .rename(columns={"size": "team_size"})
    )
    first = scope.loc[
        scope.author_order.eq(1)
        & scope.top4_newcomer_5y
        & scope.identity_source.eq("dblp_pid")
        & scope.author_key.ne("")
    ].copy()
    coauthors = scope.loc[scope.author_order.gt(1)].copy()
    coauthors["coauthor_experienced"] = ~coauthors.top4_newcomer_5y
    coauthor_feature = (
        coauthors.groupby(["paper_key", "venue", "year"], as_index=False)
        .agg(entry_paper_has_experienced=("coauthor_experienced", "max"))
    )
    first = first.merge(sizes, on=["paper_key", "venue", "year"], validate="one_to_one")
    first = first.merge(
        coauthor_feature,
        on=["paper_key", "venue", "year"],
        how="left",
        validate="one_to_one",
    )
    first["entry_paper_has_experienced"] = (
        first.entry_paper_has_experienced.fillna(False).astype(bool)
    )

    first_year = first.groupby("author_key")["year"].min().rename("index_year")
    index_papers = first.merge(first_year, on="author_key", validate="many_to_one")
    index_papers = index_papers.loc[index_papers.year.eq(index_papers.index_year)].copy()

    cohort = (
        index_papers.groupby(["author_key", "index_year"], as_index=False)
        .agg(
            author_name=("author_name", "first"),
            author_pid=("author_pid", "first"),
            index_venue=("venue", venue_label),
            n_index_papers=("paper_key", "size"),
            entry_team_size_mean=("team_size", "mean"),
            entry_with_experienced_top4_coauthor=(
                "entry_paper_has_experienced",
                "max",
            ),
        )
        .sort_values(["index_year", "index_venue", "author_key"])
    )
    cohort["entry_with_experienced_top4_coauthor"] = cohort[
        "entry_with_experienced_top4_coauthor"
    ].astype(bool)
    cohort = cohort.loc[cohort.index_year.isin(pilot_years)].copy()

    anchor = (
        index_papers.sort_values(["author_key", "year", "venue", "paper_key"])
        .drop_duplicates(["author_key", "index_year"])
        [["author_key", "index_year", "paper_key", "venue"]]
        .rename(columns={"paper_key": "index_paper_key", "venue": "anchor_venue"})
    )
    paper_meta = pd.read_csv(
        PAPERS,
        usecols=["paper_key", "title", "electronic_editions", "author_names"],
        keep_default_na=False,
    )
    paper_meta = paper_meta.drop_duplicates("paper_key")
    anchor = anchor.merge(
        paper_meta,
        left_on="index_paper_key",
        right_on="paper_key",
        how="left",
        validate="many_to_one",
    ).drop(columns="paper_key")
    cohort = cohort.merge(anchor, on=["author_key", "index_year"], validate="one_to_one")

    rng = np.random.default_rng(seed)
    selected_parts: list[pd.DataFrame] = []
    strata_columns = [
        "index_year",
        "index_venue",
        "entry_with_experienced_top4_coauthor",
    ]
    for _, block in cohort.groupby(strata_columns, sort=True, dropna=False):
        ordered = block.sort_values("author_key").reset_index(drop=True)
        take = min(n_per, len(ordered))
        positions = np.sort(rng.choice(len(ordered), size=take, replace=False))
        selected_parts.append(ordered.iloc[positions].copy())
    sample = pd.concat(selected_parts, ignore_index=True)
    sample = sample.sort_values(strata_columns + ["author_key"]).reset_index(drop=True)
    sample.insert(0, "pilot_id", [f"EPO{i:04d}" for i in range(1, len(sample) + 1)])
    sample["history_start_year"] = sample.index_year - int(config["history_lookback_years"])
    sample["history_end_year"] = sample.index_year - 1

    private_path = PRIVATE / "epj_origin_pilot_sample_private.csv"
    sample.to_csv(private_path, index=False, encoding="utf-8-sig")

    frame_summary = (
        cohort.groupby(strata_columns, as_index=False)
        .agg(frame_authors=("author_key", "nunique"))
        .sort_values(strata_columns)
    )
    selected_summary = (
        sample.groupby(strata_columns, as_index=False)
        .agg(sampled_authors=("author_key", "nunique"))
        .sort_values(strata_columns)
    )
    summary = frame_summary.merge(selected_summary, on=strata_columns, how="left")
    summary["sampled_authors"] = summary.sampled_authors.fillna(0).astype(int)
    summary["inclusion_fraction"] = summary.sampled_authors / summary.frame_authors
    summary_path = TABLES / "epj_origin_pilot_sampling_summary.csv"
    summary.to_csv(summary_path, index=False)

    checks = {
        "protocol_version": config["protocol_version"],
        "sample_seed": seed,
        "frame_unique_authors": int(cohort.author_key.nunique()),
        "sample_unique_authors": int(sample.author_key.nunique()),
        "sample_rows": int(len(sample)),
        "duplicate_author_index_year": int(sample.duplicated(["author_key", "index_year"]).sum()),
        "blank_author_key": int(sample.author_key.eq("").sum()),
        "all_identity_source_dblp_pid_by_construction": True,
        "pilot_years": list(pilot_years),
        "target_venues": list(target_venues),
        "private_output": str(private_path.relative_to(ROOT)),
        "public_summary": str(summary_path.relative_to(ROOT)),
        "input_sha256": {
            str(AUTH): sha256_file(AUTH),
            str(PAPERS): sha256_file(PAPERS),
            str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG),
            str(PROTOCOL.relative_to(ROOT)): sha256_file(PROTOCOL),
        },
        "output_sha256": {
            str(private_path.relative_to(ROOT)): sha256_file(private_path),
            str(summary_path.relative_to(ROOT)): sha256_file(summary_path),
        },
    }
    manifest_path = MANIFESTS / "epj_origin_pilot_sample_manifest.json"
    manifest_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


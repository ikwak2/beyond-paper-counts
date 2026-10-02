#!/usr/bin/env python3
"""Reviewer-requested aggregate sensitivities for the EPJ Data Science manuscript.

This script does not modify frozen primary estimates. It creates three explicitly
post-review supplementary analyses:
1. external production-share triangulation using ETO Country AI Activity Metrics;
2. country-group descriptive distance trajectories; and
3. a cumulative within-two-cycle reappearance sensitivity.

No person-level output is written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "epj_reviewer_additions_v12"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MANIFESTS = OUT / "manifests"

DISTANCE_DATA = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_analysis_dataset_private.csv"
INDEX_PAPERS = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_index_papers_private.csv"
COUNTRY_PANEL = ROOT / "data/processed/country_measurement_combined_v01_first_author.csv"
AUTH = ROOT.parent / "bibliometrics/top_ai_entry/data/processed/accepted_authorships_marked.csv"
OPENALEX_PRIMARY = ROOT / "outputs/journal_llm_diffusion_v01/tables/primary_pri_korea_and_frozen_groups_2018_2024.csv"
ETO_ARTICLES = ROOT / "data/external/eto_cat_v1_11_0/cat/publications_yearly_articles.csv"
ETO_ZIP = ROOT / "data/external/eto_cat_v1_11_0/cat.zip"
FROZEN_ANALYSIS = ROOT / "scripts/84_analyze_epj_distance_full.py"

CORE_CODES = {"AU", "CA", "IE", "NZ", "GB", "US"}
CORE_NAMES = {
    "Australia", "Canada", "Ireland", "New Zealand", "United Kingdom", "United States"
}
ETO_AGGREGATE_ENTITIES = {
    "ASEAN",
    "Africa",
    "Asia",
    "EU",
    "Europe",
    "Five Eyes",
    "Global Partnership on Artificial Intelligence",
    "NATO",
    "North America and the Caribbean",
    "OECD",
    "Oceania",
    "Quad",
    "South and Central America",
}
EXTERNAL_GROUPS = ["China", "Core Anglophone", "India"]
GROUP_ORDER = ["China", "Core Anglophone", "Other non-core", "India"]
COLORS = {
    "China": "#0072B2",
    "Core Anglophone": "#D55E00",
    "Other non-core": "#CC79A7",
    "India": "#009E73",
}
RNG_SEED = 20260901
N_BOOT = 2000
MIN_PUBLIC_CELL = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_module():
    spec = importlib.util.spec_from_file_location("epj_frozen_analysis", FROZEN_ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen analysis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_code_list(raw: str) -> str:
    codes = set(json.loads(raw))
    if not codes:
        return "Missing"
    groups = {
        "China" if code == "CN" else
        "India" if code == "IN" else
        "Core Anglophone" if code in CORE_CODES else
        "Other non-core"
        for code in codes
    }
    return next(iter(groups)) if len(groups) == 1 else "Mixed"


def external_triangulation() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare production-share directions using country-credit denominators.

    The ETO country column mixes individual countries/locations with regional
    and political aggregates. The denominator must exclude the latter or the
    same articles are counted again through Asia, OECD, NATO, Quad, and similar
    rows. Both sources then use full country counting: one credit for every
    distinct represented country on a work, summed across country rows.
    """
    primary = pd.read_csv(OPENALEX_PRIMARY)
    spec_id = "existing|conservative|top_full|ai_primary_peer_reviewed"
    openalex = primary.loc[
        primary.spec_id.eq(spec_id)
        & primary.group.isin(EXTERNAL_GROUPS)
        & primary.year.between(2018, 2024),
        ["year", "group", "broad_count", "broad_country_credit_total", "broad_share"],
    ].copy()
    openalex = openalex.rename(
        columns={
            "broad_count": "country_credit",
            "broad_country_credit_total": "all_country_credit",
            "broad_share": "production_share",
        }
    )
    openalex["raw_all_entity_credit"] = openalex["all_country_credit"]
    openalex["aggregate_entity_credit_excluded"] = 0.0
    openalex["source"] = "OpenAlex broad-AI primary"
    openalex["complete"] = True
    openalex["denominator_rule"] = (
        "sum of full-counted distinct work-country credits across all authorships"
    )

    eto_raw = pd.read_csv(ETO_ARTICLES)
    eto_raw = eto_raw.loc[
        eto_raw.field.eq("All") & eto_raw.year.between(2018, 2024)
    ].copy()
    found_aggregates = set(eto_raw.country) & ETO_AGGREGATE_ENTITIES
    if found_aggregates != ETO_AGGREGATE_ENTITIES:
        raise RuntimeError(
            "ETO aggregate-entity set changed: "
            f"missing={sorted(ETO_AGGREGATE_ENTITIES - found_aggregates)}"
        )
    if not bool(eto_raw.complete.all()):
        raise RuntimeError("ETO 2018-2024 publication rows are not all marked complete")

    raw_totals = eto_raw.groupby("year").num_articles.sum().rename("raw_all_entity_credit")
    eto = eto_raw.loc[~eto_raw.country.isin(ETO_AGGREGATE_ENTITIES)].copy()
    totals = eto.groupby("year").num_articles.sum().rename("all_country_credit")
    excluded = raw_totals.sub(totals).rename("aggregate_entity_credit_excluded")

    group_frames = []
    for label, country_names in (
        ("China", {"China (mainland)"}),
        ("Core Anglophone", CORE_NAMES),
        ("India", {"India"}),
    ):
        block = (
            eto.loc[eto.country.isin(country_names)]
            .groupby("year", as_index=False)
            .agg(num_articles=("num_articles", "sum"), complete=("complete", "min"))
        )
        block["group"] = label
        group_frames.append(block)
    eto_groups = pd.concat(group_frames, ignore_index=True)
    eto_groups = eto_groups.rename(columns={"num_articles": "country_credit"})
    eto_groups = (
        eto_groups.merge(totals, on="year", validate="many_to_one")
        .merge(raw_totals, on="year", validate="many_to_one")
        .merge(excluded, on="year", validate="many_to_one")
    )
    eto_groups["production_share"] = (
        eto_groups.country_credit / eto_groups.all_country_credit
    )
    eto_groups["source"] = "ETO Merged Academic Corpus v1.11.0"
    eto_groups["denominator_rule"] = (
        "sum of individual country/location credits; 13 regional/political aggregate rows excluded"
    )

    columns = [
        "source", "year", "group", "country_credit", "all_country_credit",
        "raw_all_entity_credit", "aggregate_entity_credit_excluded",
        "production_share", "complete", "denominator_rule",
    ]
    trajectories = pd.concat([openalex[columns], eto_groups[columns]], ignore_index=True)
    trajectories = trajectories.sort_values(["group", "source", "year"]).reset_index(drop=True)

    endpoints = []
    for (source, group), block in trajectories.groupby(["source", "group"]):
        lookup = block.set_index("year")
        start = float(lookup.loc[2018, "production_share"])
        end = float(lookup.loc[2024, "production_share"])
        endpoints.append(
            {
                "source": source,
                "group": group,
                "share_2018": start,
                "share_2024": end,
                "change_percentage_points": 100 * (end - start),
                "relative_change_percent": 100 * (end / start - 1),
                "direction": "increase" if end > start else "decrease" if end < start else "no change",
                "denominator_rule": str(lookup.loc[2024, "denominator_rule"]),
            }
        )
    summary = pd.DataFrame(endpoints).sort_values(["group", "source"]).reset_index(drop=True)
    diagnostics = {
        "aggregate_entities_excluded": sorted(ETO_AGGREGATE_ENTITIES),
        "n_aggregate_entities_excluded": len(ETO_AGGREGATE_ENTITIES),
        "raw_all_entity_credit_2018": int(raw_totals.loc[2018]),
        "country_only_credit_2018": int(totals.loc[2018]),
        "raw_all_entity_credit_2024": int(raw_totals.loc[2024]),
        "country_only_credit_2024": int(totals.loc[2024]),
        "duplicate_country_year_rows": int(
            eto.duplicated(["country", "field", "year"]).sum()
        ),
        "duplicate_note": (
            "Rows labeled Congo represent two distinct country entities collapsed to one display label; "
            "their credits remain separate and are summed in the country-credit denominator."
        ),
    }
    return trajectories, summary, diagnostics

def distance_by_country_group() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    analysis = pd.read_csv(
        DISTANCE_DATA,
        usecols=["entrant_id", "index_year", "specter2_title_distance_primary"],
    )
    index = pd.read_csv(INDEX_PAPERS, usecols=["entrant_id", "paper_key"])
    country = pd.read_csv(
        COUNTRY_PANEL,
        usecols=["paper_id", "primary_country_codes"],
        keep_default_na=False,
    ).rename(columns={"paper_id": "paper_key"})
    joined = index.merge(country, on="paper_key", how="left", validate="one_to_one")
    if joined.primary_country_codes.isna().any():
        raise RuntimeError("country panel failed to match every index paper")
    joined["paper_group"] = joined.primary_country_codes.map(classify_code_list)
    entrant_group = joined.groupby("entrant_id").paper_group.agg(
        lambda values: next(iter(set(values))) if len(set(values)) == 1
        else "Mixed across index papers"
    ).rename("group")
    data = analysis.merge(entrant_group, on="entrant_id", validate="one_to_one")
    data = data.loc[data.specter2_title_distance_primary.notna()].copy()

    exclusions = {
        str(group): int(count)
        for group, count in data.loc[~data.group.isin(GROUP_ORDER), "group"].value_counts().items()
    }
    eligible = data.loc[data.group.isin(GROUP_ORDER)].copy()
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (group, year), block in eligible.groupby(["group", "index_year"]):
        values = block.specter2_title_distance_primary.to_numpy(dtype=float)
        boot = np.median(
            values[rng.integers(0, len(values), size=(N_BOOT, len(values)))],
            axis=1,
        )
        rows.append(
            {
                "group": group,
                "index_year": int(year),
                "n": int(len(values)),
                "median_distance": float(np.median(values)),
                "q25_distance": float(np.quantile(values, 0.25)),
                "q75_distance": float(np.quantile(values, 0.75)),
                "median_ci_low": float(np.quantile(boot, 0.025)),
                "median_ci_high": float(np.quantile(boot, 0.975)),
                "interval": "entrant_bootstrap",
                "small_cell": bool(len(values) < 10),
            }
        )
    annual = pd.DataFrame(rows).sort_values(["group", "index_year"]).reset_index(drop=True)

    pooled = (
        eligible.groupby("group")
        .specter2_title_distance_primary
        .agg(
            n="size",
            median_distance="median",
            mean_distance="mean",
            q25_distance=lambda x: x.quantile(0.25),
            q75_distance=lambda x: x.quantile(0.75),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    return annual, pooled, exclusions


def classify_within_two(
    analysis: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    auth = pd.read_csv(
        AUTH,
        usecols=["paper_key", "venue", "year", "author_key"],
        keep_default_na=False,
    )
    auth["year"] = pd.to_numeric(auth.year, errors="raise").astype(int)
    index = pd.read_csv(
        INDEX_PAPERS,
        usecols=["entrant_id", "author_key", "index_year", "paper_key", "venue", "year"],
        keep_default_na=False,
    )
    index["year"] = pd.to_numeric(index.year, errors="raise").astype(int)
    eligible = (
        analysis.loc[analysis.index_year.le(2022), ["entrant_id", "author_key", "index_year"]]
        .drop_duplicates("entrant_id")
        .copy()
    )
    result = pd.Series("not_observed", index=analysis.entrant_id, dtype="object")
    result.loc[eligible.entrant_id] = "no_recurrence"

    index_members = index.merge(
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
        index_members.groupby("entrant_id").paper_member_key.agg(lambda x: set(x)).to_dict()
    )

    appearances = auth.loc[
        auth.author_key.isin(set(eligible.author_key))
        & auth.venue.isin(("AAAI", "ICLR", "ICML", "NeurIPS")),
        ["paper_key", "venue", "year", "author_key"],
    ].merge(eligible, on="author_key", validate="many_to_one")
    appearances = appearances.loc[
        appearances.year.sub(appearances.index_year).isin([1, 2])
    ].copy()
    return_papers = appearances[
        ["entrant_id", "paper_key", "venue", "year"]
    ].drop_duplicates()
    members = return_papers.merge(
        auth[["paper_key", "venue", "year", "author_key"]].rename(
            columns={"author_key": "paper_member_key"}
        ),
        on=["paper_key", "venue", "year"],
        how="left",
        validate="many_to_many",
    )
    members["is_index_coauthor"] = [
        member in index_coauthors.get(entrant, set())
        for entrant, member in zip(members.entrant_id, members.paper_member_key)
    ]
    overlap = (
        members.groupby(["entrant_id", "paper_key", "venue", "year"], as_index=False)
        .agg(shares_index_coauthor=("is_index_coauthor", "max"))
    )
    categories = overlap.groupby("entrant_id").shares_index_coauthor.agg(
        lambda values: (
            "coauthor_continuity_only"
            if bool(values.all())
            else "at_least_one_no_index_coauthor"
        )
    )
    result.loc[categories.index] = categories
    diagnostics = {
        "eligible_entrants_2018_2022": int(len(eligible)),
        "entrants_with_within_two_record": int(return_papers.entrant_id.nunique()),
        "within_two_papers": int(return_papers.paper_key.nunique()),
    }
    return result.reindex(analysis.entrant_id).reset_index(drop=True), diagnostics


def within_two_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frozen = load_frozen_module()
    data = pd.read_csv(DISTANCE_DATA, keep_default_na=False, na_values=[""])
    data["within_two_top4"], classification = classify_within_two(data)
    probabilities, contrasts, model = frozen.rq3_specification(
        data,
        specification="postreview_within_two_top4",
        distance_column="specter2_title_distance_primary",
        outcome_column="within_two_top4",
        n_bootstrap=N_BOOT,
        seed=RNG_SEED + 3000,
    )
    model["estimand"] = "any accepted Top-4 record at index+1 or index+2"
    model["postreview_sensitivity"] = True
    return probabilities, contrasts, {"classification": classification, "model": model}


def make_figures(
    triangulation: pd.DataFrame,
    annual_distance: pd.DataFrame,
    within_contrasts: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharex=True)
    source_styles = {
        "OpenAlex broad-AI primary": ("#0072B2", "o", "-"),
        "ETO Merged Academic Corpus v1.11.0": ("#D55E00", "s", "--"),
    }
    for axis, group in zip(axes, EXTERNAL_GROUPS):
        block = triangulation.loc[triangulation.group.eq(group)]
        for source, source_block in block.groupby("source"):
            color, marker, linestyle = source_styles[source]
            axis.plot(
                source_block.year,
                100 * source_block.production_share,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                label=source,
            )
        axis.set_title(group)
        axis.set_xlabel("Publication year")
        axis.set_ylabel("Share of country credits (%)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.2)
    fig.suptitle("External triangulation of broad-AI production-share directions", y=1.01)
    fig.text(
        0.5, -0.01,
        "ETO denominator excludes 13 regional/political aggregate rows. "
        "Corpora and classifiers differ; comparison tests direction, not numerical equivalence.",
        ha="center", fontsize=7.3, color="#555555",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_s6_external_production_triangulation.png", dpi=260, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_s6_external_production_triangulation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 5.1))
    for group in GROUP_ORDER:
        block = annual_distance.loc[annual_distance.group.eq(group)].sort_values("index_year")
        color = COLORS[group]
        axis.plot(
            block.index_year,
            block.median_distance,
            marker="o",
            linewidth=2,
            color=color,
            label=group,
        )
        axis.fill_between(
            block.index_year.to_numpy(),
            block.median_ci_low.to_numpy(),
            block.median_ci_high.to_numpy(),
            color=color,
            alpha=0.12,
        )
        small = block.loc[block.small_cell]
        if not small.empty:
            axis.scatter(
                small.index_year,
                small.median_distance,
                s=55,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                zorder=4,
            )
    axis.set_xlabel("Index year")
    axis.set_ylabel("Median SPECTER2 title-only distance")
    axis.set_title("Distance trajectories by unambiguous first-author affiliation-country group")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.text(
        0.5, 0.01,
        "Shading: 95% entrant bootstrap interval. Open markers: N < 10. Descriptive; observed-distance entrants only.",
        ha="center", fontsize=7.5, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "figure_s7_country_group_distance.png", dpi=260)
    fig.savefig(FIGURES / "figure_s7_country_group_distance.pdf")
    plt.close(fig)

    exact = pd.read_csv(ROOT / "outputs/epj_distance_based_v10/tables/rq3_exact_plus2_contrasts.csv")
    exact = exact.loc[
        exact.specification.eq("semantic_primary")
        & exact.contrast_family.eq("experienced_minus_no_experienced")
        & exact.pathway.eq("no_recurrence")
    ].copy()
    exact["estimand"] = "Exact +2"
    within = within_contrasts.loc[
        within_contrasts.contrast_family.eq("experienced_minus_no_experienced")
        & within_contrasts.pathway.eq("no_recurrence")
    ].copy()
    within["estimand"] = "Within +2"
    compare = pd.concat([exact, within], ignore_index=True)
    fig, axis = plt.subplots(figsize=(7.4, 4.3))
    positions = {"Exact +2": -0.10, "Within +2": 0.10}
    colors = {"Exact +2": "#0072B2", "Within +2": "#D55E00"}
    for estimand, block in compare.groupby("estimand"):
        xs = np.arange(2) + positions[estimand]
        block = block.set_index("distance_level").loc[["Q25", "Q75"]].reset_index()
        values = 100 * block.probability_difference.to_numpy()
        lows = 100 * block.ci_low.to_numpy()
        highs = 100 * block.ci_high.to_numpy()
        axis.errorbar(
            xs,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            marker="o",
            capsize=4,
            linewidth=1.8,
            color=colors[estimand],
            label=estimand,
        )
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xticks([0, 1], ["Distance Q25", "Distance Q75"])
    axis.set_ylabel("Experienced − no-experienced difference\nin probability of no record (percentage points)")
    axis.set_title("Fixed-horizon and cumulative reappearance estimands")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_s8_within_two_sensitivity.png", dpi=260)
    fig.savefig(FIGURES / "figure_s8_within_two_sensitivity.pdf")
    plt.close(fig)


def main() -> int:
    for directory in (TABLES, FIGURES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories, endpoint_summary, external_diagnostics = external_triangulation()
    annual_distance, pooled_distance, exclusions = distance_by_country_group()
    within_probabilities, within_contrasts, within_diagnostics = within_two_sensitivity()

    trajectories.to_csv(TABLES / "table_s12_external_production_trajectories.csv", index=False)
    endpoint_summary.to_csv(TABLES / "table_s13_external_production_endpoints.csv", index=False)
    annual_distance.to_csv(TABLES / "table_s14_country_group_distance_by_year.csv", index=False)
    pooled_distance.to_csv(TABLES / "table_s15_country_group_distance_pooled.csv", index=False)
    within_probabilities.to_csv(TABLES / "table_s16_within_two_probabilities.csv", index=False)
    within_contrasts.to_csv(TABLES / "table_s17_within_two_contrasts.csv", index=False)

    make_figures(trajectories, annual_distance, within_contrasts)

    china = endpoint_summary.loc[endpoint_summary.group.eq("China")].set_index("source")
    core = endpoint_summary.loc[endpoint_summary.group.eq("Core Anglophone")].set_index("source")
    india = endpoint_summary.loc[endpoint_summary.group.eq("India")].set_index("source")
    gates = {
        "eto_2018_2024_complete": bool(
            trajectories.loc[trajectories.source.str.startswith("ETO"), "complete"].all()
        ),
        "eto_aggregate_entities_excluded": bool(
            external_diagnostics["n_aggregate_entities_excluded"] == 13
            and external_diagnostics["country_only_credit_2024"]
            < external_diagnostics["raw_all_entity_credit_2024"]
        ),
        "external_groups_complete": bool(
            set(endpoint_summary.group) == set(EXTERNAL_GROUPS)
            and endpoint_summary.groupby("group").source.nunique().eq(2).all()
        ),
        "china_direction_agrees": bool(china.direction.nunique() == 1 and china.direction.iloc[0] == "increase"),
        "core_direction_agrees": bool(core.direction.nunique() == 1 and core.direction.iloc[0] == "decrease"),
        "india_direction_agrees": bool(india.direction.nunique() == 1 and india.direction.iloc[0] == "increase"),
        "distance_public_cells_ge_5": bool(annual_distance.n.min() >= MIN_PUBLIC_CELL),
        "distance_groups_present": bool(set(annual_distance.group) == set(GROUP_ORDER)),
        "within_two_n_matches_primary": bool(within_diagnostics["model"]["n"] == 5122),
        "within_two_bootstrap_complete": bool(
            within_diagnostics["model"]["bootstrap_successful"] == N_BOOT
        ),
    }
    outputs = sorted(TABLES.glob("*.csv")) + sorted(FIGURES.glob("*"))
    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "postreview_scope": True,
        "gates": gates,
        "external_triangulation": external_diagnostics,
        "distance_exclusions": exclusions,
        "within_two": within_diagnostics,
        "source": {
            "eto_dataset": "Country AI Activity Metrics v1.11.0",
            "eto_doi": "10.5281/zenodo.19103157",
            "eto_documentation": "https://eto.tech/dataset-docs/country-ai-activity-metrics/index.html",
            "eto_zip_sha256": sha256_file(ETO_ZIP),
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in outputs
        },
    }
    (MANIFESTS / "reviewer_additions_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

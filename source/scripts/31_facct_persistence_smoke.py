#!/usr/bin/env python3
"""FAccT v0.3: accepted-newcomer two-cycle persistence smoke analysis.

Implements the new RQ3 frozen in ``docs/facct_access_persistence_plan_v0.3.md``
and operationalized in ``docs/facct_persistence_technical_spec_v0.3.1.md``.
Only aggregate-safe outputs are written.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
AUTH = (
    ROOT.parent
    / "bibliometrics"
    / "top_ai_entry"
    / "data"
    / "processed"
    / "accepted_authorships_marked.csv"
)
GEO = ROOT / "data" / "processed" / "country_measurement_combined_v01_first_author.csv"
PLAN = ROOT / "docs" / "facct_access_persistence_plan_v0.3.md"
TECH_SPEC = ROOT / "docs" / "facct_persistence_technical_spec_v0.3.1.md"
OUT = ROOT / "outputs" / "facct_persistence_v03"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT = OUT / "FACCT_PERSISTENCE_SMOKE_KO.md"

INDEX_VENUES = ("ICML", "NeurIPS")
TOP4 = ("AAAI", "ICLR", "ICML", "NeurIPS")
INDEX_YEARS = tuple(range(2018, 2023))
FOLLOWUP_CEILING = 2024
CORE = {"AU", "CA", "GB", "IE", "NZ", "US"}
COUNTRY_SPECS = {
    "conservative": "primary_country_codes",
    "sensitivity": "sensitivity_country_codes",
}
OUTCOMES = {
    "any_author_return": "any_top4_return_2cycle",
    "first_author_return": "first_author_top4_return_2cycle",
}
RNG_SEED = 202603
N_DRAWS = 2000


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


def parse_codes(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, (list, tuple, set)):
        return []
    return sorted({str(code).strip().upper() for code in parsed if str(code).strip()})


def code_group(code: str) -> str:
    if code == "CN":
        return "China"
    if code == "IN":
        return "India"
    if code in CORE:
        return "Core Anglophone"
    return "Other non-core"


def exclusive_group(codes: list[str]) -> str:
    if not codes:
        return "Unresolved"
    groups = {code_group(code) for code in codes}
    return next(iter(groups)) if len(groups) == 1 else "Mixed-group"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    auth = pd.read_csv(
        AUTH,
        usecols=[
            "paper_key",
            "venue",
            "year",
            "author_order",
            "author_key",
            "identity_source",
            "top4_newcomer_5y",
        ],
        keep_default_na=False,
    )
    auth["top4_newcomer_5y"] = as_bool(auth["top4_newcomer_5y"])
    auth["year"] = pd.to_numeric(auth["year"], errors="raise").astype(int)
    auth["author_order"] = pd.to_numeric(auth["author_order"], errors="raise").astype(int)
    geo = pd.read_csv(
        GEO,
        usecols=[
            "paper_id",
            "venue",
            "year",
            "primary_country_codes",
            "sensitivity_country_codes",
        ],
        keep_default_na=False,
    )
    geo["year"] = pd.to_numeric(geo["year"], errors="raise").astype(int)
    if geo.duplicated(["paper_id", "venue", "year"]).any():
        raise RuntimeError("duplicate geography paper keys")
    return auth, geo


def build_index_papers(auth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope = auth.loc[
        auth.venue.isin(INDEX_VENUES) & auth.year.isin(INDEX_YEARS)
    ].copy()
    sizes = (
        scope.groupby(["paper_key", "venue", "year"], as_index=False)
        .size()
        .rename(columns={"size": "team_size"})
    )
    first = scope.loc[scope.author_order.eq(1)].copy()
    if first.duplicated(["paper_key", "venue", "year"]).any():
        raise RuntimeError("duplicate first-author rows")
    eligible_all = first.loc[first.top4_newcomer_5y].copy()
    eligible_all["stable_pid"] = (
        eligible_all.identity_source.eq("dblp_pid") & eligible_all.author_key.ne("")
    )
    coverage = (
        eligible_all.groupby(["venue", "year"], as_index=False)
        .agg(
            eligible_newcomer_papers=("paper_key", "size"),
            stable_pid_papers=("stable_pid", "sum"),
            unique_person_proxies=("author_key", "nunique"),
        )
        .sort_values(["venue", "year"])
    )
    coverage["stable_pid_coverage"] = (
        coverage.stable_pid_papers / coverage.eligible_newcomer_papers
    )

    coauthors = scope.loc[scope.author_order.gt(1)].copy()
    coauthors["coauthor_experienced"] = ~coauthors.top4_newcomer_5y
    coauthor_feature = (
        coauthors.groupby(["paper_key", "venue", "year"], as_index=False)
        .agg(entry_paper_has_experienced=("coauthor_experienced", "max"))
    )
    eligible = eligible_all.loc[eligible_all.stable_pid].copy()
    eligible = eligible.merge(
        sizes, on=["paper_key", "venue", "year"], validate="one_to_one"
    ).merge(
        coauthor_feature,
        on=["paper_key", "venue", "year"],
        how="left",
        validate="one_to_one",
    )
    eligible["entry_paper_has_experienced"] = (
        eligible.entry_paper_has_experienced.fillna(False).astype(bool)
    )
    return eligible, coverage


def aggregate_author_cohorts(index_papers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_year = index_papers.groupby("author_key")["year"].min().rename("index_year")
    papers = index_papers.merge(first_year, on="author_key", validate="many_to_one")
    papers = papers.loc[papers.year.eq(papers.index_year)].copy()

    def venue_label(values: pd.Series) -> str:
        venues = sorted(set(values))
        return venues[0] if len(venues) == 1 else "Both"

    cohort = (
        papers.groupby(["author_key", "index_year"], as_index=False)
        .agg(
            n_index_papers=("paper_key", "size"),
            entry_team_size_mean=("team_size", "mean"),
            index_venue=("venue", venue_label),
            entry_with_experienced_top4_coauthor=(
                "entry_paper_has_experienced",
                "max",
            ),
        )
        .sort_values(["index_year", "author_key"])
    )
    cohort["entry_with_experienced_top4_coauthor"] = (
        cohort.entry_with_experienced_top4_coauthor.astype(bool)
    )
    return cohort, papers


def attach_country_groups(
    cohort: pd.DataFrame, papers: pd.DataFrame, geo: pd.DataFrame
) -> pd.DataFrame:
    paper_country = papers[["author_key", "index_year", "paper_key", "venue", "year"]].merge(
        geo,
        left_on=["paper_key", "venue", "year"],
        right_on=["paper_id", "venue", "year"],
        how="left",
        validate="many_to_one",
    )
    result = cohort.copy()
    for specification, column in COUNTRY_SPECS.items():
        paper_country[f"codes_{specification}"] = paper_country[column].map(parse_codes)
        grouped = (
            paper_country.groupby(["author_key", "index_year"])[f"codes_{specification}"]
            .agg(lambda values: sorted({code for codes in values for code in codes}))
            .rename(f"country_codes_{specification}")
            .reset_index()
        )
        grouped[f"comparison_group_{specification}"] = grouped[
            f"country_codes_{specification}"
        ].map(exclusive_group)
        grouped[f"country_observed_{specification}"] = grouped[
            f"country_codes_{specification}"
        ].map(bool)
        result = result.merge(
            grouped,
            on=["author_key", "index_year"],
            how="left",
            validate="one_to_one",
        )
        result[f"comparison_group_{specification}"] = result[
            f"comparison_group_{specification}"
        ].fillna("Unresolved")
        result[f"country_observed_{specification}"] = result[
            f"country_observed_{specification}"
        ].fillna(False)
    return result


def attach_returns(cohort: pd.DataFrame, auth: pd.DataFrame) -> pd.DataFrame:
    keys = set(cohort.author_key)
    follow = auth.loc[
        auth.author_key.isin(keys)
        & auth.venue.isin(TOP4)
        & auth.year.le(FOLLOWUP_CEILING)
    ].copy()
    lookup = cohort[["author_key", "index_year"]]
    follow = follow.merge(lookup, on="author_key", validate="many_to_one")
    follow = follow.loc[
        follow.year.gt(follow.index_year) & follow.year.le(follow.index_year + 2)
    ].copy()
    follow["return_cycle"] = follow.year - follow.index_year
    any_return = (
        follow.groupby("author_key")["return_cycle"].min().rename("any_return_cycle")
    )
    first_return = (
        follow.loc[follow.author_order.eq(1)]
        .groupby("author_key")["return_cycle"]
        .min()
        .rename("first_author_return_cycle")
    )
    result = cohort.merge(any_return, on="author_key", how="left", validate="one_to_one")
    result = result.merge(first_return, on="author_key", how="left", validate="one_to_one")
    result["any_top4_return_2cycle"] = result.any_return_cycle.notna()
    result["first_author_top4_return_2cycle"] = result.first_author_return_cycle.notna()
    result["any_top4_return_1cycle"] = result.any_return_cycle.eq(1)
    result["first_author_top4_return_1cycle"] = result.first_author_return_cycle.eq(1)
    return result


def make_risk_panel(cohort: pd.DataFrame, cycle_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in cohort.to_dict("records"):
        event_cycle = record.get(cycle_column)
        event_cycle = int(event_cycle) if pd.notna(event_cycle) else None
        for cycle in (1, 2):
            rows.append(
                {
                    **record,
                    "followup_cycle": cycle,
                    "event": int(event_cycle == cycle),
                }
            )
            if event_cycle == cycle:
                break
    return pd.DataFrame(rows)


def inverse_cloglog(eta: np.ndarray) -> np.ndarray:
    safe = np.clip(eta, -30.0, 20.0)
    return 1.0 - np.exp(-np.exp(safe))


def nearest_psd(covariance: np.ndarray) -> np.ndarray:
    symmetric = (covariance + covariance.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, 1e-12, None)
    return (vectors * values) @ vectors.T


def standardized_two_cycle(
    model: Any,
    base: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    beta = np.asarray(model.params, dtype=float)
    covariance = nearest_psd(np.asarray(model.cov_params(), dtype=float))
    draws = rng.multivariate_normal(beta, covariance, size=N_DRAWS)
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        designs = []
        for cycle in (1, 2):
            new = base.copy()
            new["followup_cycle"] = cycle
            for key, value in scenario.items():
                new[key] = value
            design = np.asarray(
                build_design_matrices([model.model.data.design_info], new)[0],
                dtype=float,
            )
            designs.append(design)
        hazards = [inverse_cloglog(design @ beta) for design in designs]
        estimate = float(np.mean(1.0 - (1.0 - hazards[0]) * (1.0 - hazards[1])))

        simulated: list[np.ndarray] = []
        chunk = 100
        for start in range(0, N_DRAWS, chunk):
            block = draws[start : start + chunk].T
            h1 = inverse_cloglog(designs[0] @ block)
            h2 = inverse_cloglog(designs[1] @ block)
            simulated.append(np.mean(1.0 - (1.0 - h1) * (1.0 - h2), axis=0))
        distribution = np.concatenate(simulated)
        results.append(
            {
                **scenario,
                "adjusted_two_cycle_probability": estimate,
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "n_coefficient_draws": N_DRAWS,
            }
        )
    return results


def fit_primary_models(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RNG_SEED)
    formula = (
        "event ~ entry_with_experienced_top4_coauthor + C(followup_cycle) + "
        "C(index_year) + C(index_venue) + np.log1p(entry_team_size_mean)"
    )
    for outcome_name, outcome_column in OUTCOMES.items():
        cycle_column = (
            "any_return_cycle"
            if outcome_name == "any_author_return"
            else "first_author_return_cycle"
        )
        risk = make_risk_panel(cohort, cycle_column)
        try:
            model = smf.glm(
                formula=formula,
                data=risk,
                family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": risk.author_key},
                maxiter=300,
            )
            converged = bool(getattr(model, "converged", True))
            finite = bool(
                np.isfinite(np.asarray(model.params)).all()
                and np.isfinite(np.asarray(model.cov_params())).all()
            )
            for term in model.params.index:
                coefficient_rows.append(
                    {
                        "outcome": outcome_name,
                        "term": term,
                        "coefficient_cloglog": float(model.params[term]),
                        "standard_error_cluster": float(model.bse[term]),
                        "hazard_ratio": float(math.exp(model.params[term])),
                        "ci_low": float(math.exp(model.params[term] - 1.96 * model.bse[term])),
                        "ci_high": float(math.exp(model.params[term] + 1.96 * model.bse[term])),
                        "p_value": float(model.pvalues[term]),
                    }
                )
            base = cohort[
                ["index_year", "index_venue", "entry_team_size_mean"]
            ].copy()
            scenarios = [
                {"entry_with_experienced_top4_coauthor": False},
                {"entry_with_experienced_top4_coauthor": True},
            ]
            predicted = standardized_two_cycle(model, base, scenarios, rng=rng)
            for row in predicted:
                row["outcome"] = outcome_name
                prediction_rows.append(row)
            diagnostic_rows.append(
                {
                    "outcome": outcome_name,
                    "observations_person_cycle": int(len(risk)),
                    "unique_authors": int(risk.author_key.nunique()),
                    "events": int(risk.event.sum()),
                    "converged": converged,
                    "finite_parameters": finite,
                    "status": "PASS" if converged and finite else "FAIL",
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            diagnostic_rows.append(
                {
                    "outcome": outcome_name,
                    "observations_person_cycle": int(len(risk)),
                    "unique_authors": int(risk.author_key.nunique()),
                    "events": int(risk.event.sum()),
                    "converged": False,
                    "finite_parameters": False,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(diagnostic_rows),
    )


def fit_country_models(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rng = np.random.default_rng(RNG_SEED + 1)
    formula = (
        "event ~ entry_with_experienced_top4_coauthor * "
        "C(comparison_group, Treatment(reference='Other non-core')) + "
        "C(followup_cycle) + C(index_year) + C(index_venue) + "
        "np.log1p(entry_team_size_mean)"
    )
    for country_specification in COUNTRY_SPECS:
        group_column = f"comparison_group_{country_specification}"
        model_cohort = cohort.loc[
            cohort[group_column].isin(["China", "Other non-core"])
        ].copy()
        model_cohort["comparison_group"] = model_cohort[group_column]
        for outcome_name in OUTCOMES:
            cycle_column = (
                "any_return_cycle"
                if outcome_name == "any_author_return"
                else "first_author_return_cycle"
            )
            risk = make_risk_panel(model_cohort, cycle_column)
            try:
                model = smf.glm(
                    formula=formula,
                    data=risk,
                    family=sm.families.Binomial(link=sm.families.links.CLogLog()),
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": risk.author_key},
                    maxiter=300,
                )
                converged = bool(getattr(model, "converged", True))
                finite = bool(
                    np.isfinite(np.asarray(model.params)).all()
                    and np.isfinite(np.asarray(model.cov_params())).all()
                )
                for term in model.params.index:
                    coefficients.append(
                        {
                            "country_specification": country_specification,
                            "outcome": outcome_name,
                            "term": term,
                            "coefficient_cloglog": float(model.params[term]),
                            "standard_error_cluster": float(model.bse[term]),
                            "hazard_ratio": float(math.exp(model.params[term])),
                            "ci_low": float(math.exp(model.params[term] - 1.96 * model.bse[term])),
                            "ci_high": float(math.exp(model.params[term] + 1.96 * model.bse[term])),
                            "p_value": float(model.pvalues[term]),
                        }
                    )
                base = model_cohort[
                    ["index_year", "index_venue", "entry_team_size_mean"]
                ].copy()
                scenarios = [
                    {
                        "entry_with_experienced_top4_coauthor": experienced,
                        "comparison_group": group,
                    }
                    for group in ("China", "Other non-core")
                    for experienced in (False, True)
                ]
                predicted = standardized_two_cycle(model, base, scenarios, rng=rng)
                for row in predicted:
                    row["country_specification"] = country_specification
                    row["outcome"] = outcome_name
                    predictions.append(row)
                diagnostics.append(
                    {
                        "country_specification": country_specification,
                        "outcome": outcome_name,
                        "unique_authors": int(risk.author_key.nunique()),
                        "events": int(risk.event.sum()),
                        "converged": converged,
                        "finite_parameters": finite,
                        "status": "PASS" if converged and finite else "FAIL",
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                diagnostics.append(
                    {
                        "country_specification": country_specification,
                        "outcome": outcome_name,
                        "unique_authors": int(risk.author_key.nunique()),
                        "events": int(risk.event.sum()),
                        "converged": False,
                        "finite_parameters": False,
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(coefficients), pd.DataFrame(predictions), pd.DataFrame(diagnostics)


def aggregate_tables(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_network = (
        cohort.groupby(["index_year", "entry_with_experienced_top4_coauthor"], as_index=False)
        .agg(
            unique_authors=("author_key", "nunique"),
            index_papers=("n_index_papers", "sum"),
            any_return_1cycle_events=("any_top4_return_1cycle", "sum"),
            any_return_2cycle_events=("any_top4_return_2cycle", "sum"),
            first_author_return_1cycle_events=("first_author_top4_return_1cycle", "sum"),
            first_author_return_2cycle_events=("first_author_top4_return_2cycle", "sum"),
            any_return_1cycle=("any_top4_return_1cycle", "mean"),
            any_return_2cycle=("any_top4_return_2cycle", "mean"),
            first_author_return_1cycle=("first_author_top4_return_1cycle", "mean"),
            first_author_return_2cycle=("first_author_top4_return_2cycle", "mean"),
            mean_entry_team_size=("entry_team_size_mean", "mean"),
        )
        .sort_values(["index_year", "entry_with_experienced_top4_coauthor"])
    )
    country_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    for specification in COUNTRY_SPECS:
        group_column = f"comparison_group_{specification}"
        observed_column = f"country_observed_{specification}"
        coverage = (
            cohort.groupby("index_year", as_index=False)
            .agg(
                eligible_authors=("author_key", "nunique"),
                country_observed=(observed_column, "sum"),
                unresolved=(group_column, lambda x: int(x.eq("Unresolved").sum())),
                mixed_group=(group_column, lambda x: int(x.eq("Mixed-group").sum())),
            )
        )
        coverage["country_coverage"] = coverage.country_observed / coverage.eligible_authors
        coverage["country_specification"] = specification
        coverage_rows.append(coverage)

        block = cohort.copy()
        block["comparison_group"] = block[group_column]
        summary = (
            block.groupby(
                ["index_year", "comparison_group", "entry_with_experienced_top4_coauthor"],
                as_index=False,
            )
            .agg(
                unique_authors=("author_key", "nunique"),
                any_return_2cycle=("any_top4_return_2cycle", "mean"),
                first_author_return_2cycle=("first_author_top4_return_2cycle", "mean"),
            )
        )
        summary["country_specification"] = specification
        country_rows.append(summary)
    return raw_network, pd.concat(country_rows, ignore_index=True), pd.concat(coverage_rows, ignore_index=True)


def make_figure(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return
    plot = predictions.copy()
    plot["network"] = np.where(
        plot.entry_with_experienced_top4_coauthor,
        "Experienced coauthor",
        "All-new entry team",
    )
    outcomes = ["any_author_return", "first_author_return"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for axis, outcome in zip(axes, outcomes):
        block = plot.loc[plot.outcome.eq(outcome)].reset_index(drop=True)
        x = np.arange(len(block))
        axis.bar(x, block.adjusted_two_cycle_probability, color=["#8da0cb", "#fc8d62"])
        axis.errorbar(
            x,
            block.adjusted_two_cycle_probability,
            yerr=[
                block.adjusted_two_cycle_probability - block.ci_low,
                block.ci_high - block.adjusted_two_cycle_probability,
            ],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        axis.set_xticks(x, block.network, rotation=12, ha="right")
        axis.set_title(outcome.replace("_", " "))
        axis.set_ylim(0, min(1.0, max(0.5, float(block.ci_high.max()) + 0.08)))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Adjusted probability of return within two cycles")
    fig.suptitle("Accepted newcomer persistence by entry-network experience")
    fig.tight_layout()
    fig.savefig(FIGURES / "facct_v03_adjusted_two_cycle_persistence.png", dpi=220)
    plt.close(fig)


def validate(
    cohort: pd.DataFrame,
    stable_coverage: pd.DataFrame,
    primary_diagnostics: pd.DataFrame,
    country_diagnostics: pd.DataFrame,
    output_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    cohort_values = sorted(cohort.index_year.unique().tolist())
    exposure_by_cohort = cohort.groupby("index_year")[
        "entry_with_experienced_top4_coauthor"
    ].nunique()
    event_cells: dict[str, bool] = {}
    for outcome_name, column in OUTCOMES.items():
        check = cohort.groupby("entry_with_experienced_top4_coauthor")[column].nunique()
        event_cells[outcome_name] = bool((check >= 2).all() and len(check) == 2)
    banned = ("author_name", "author_pid", "author_key", "title", "profile_id")
    aggregate_safe = all(
        not any(any(token in str(column).casefold() for token in banned) for column in table.columns)
        for table in output_tables.values()
    )
    stable_pass = bool(stable_coverage.stable_pid_coverage.ge(0.90).all())
    primary_models_pass = bool(
        len(primary_diagnostics) == 2
        and primary_diagnostics.status.eq("PASS").all()
    )
    country_models_pass = bool(
        len(country_diagnostics) == 4
        and country_diagnostics.status.eq("PASS").all()
    )
    gates = {
        "five_complete_index_cohorts": cohort_values == list(INDEX_YEARS),
        "stable_pid_coverage_ge_90pct": stable_pass,
        "both_entry_network_categories_each_cohort": bool((exposure_by_cohort == 2).all()),
        "event_and_nonevent_in_each_primary_exposure": bool(all(event_cells.values())),
        "primary_models_pass": primary_models_pass,
        "country_models_pass": country_models_pass,
        "aggregate_outputs_exclude_direct_identifiers": aggregate_safe,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "index_cohorts": cohort_values,
        "eligible_unique_authors": int(cohort.author_key.nunique()),
        "eligible_index_papers": int(cohort.n_index_papers.sum()),
        "multi_paper_index_authors": int(cohort.n_index_papers.gt(1).sum()),
        "stable_pid_min_venue_cohort_coverage": float(stable_coverage.stable_pid_coverage.min()),
        "input_sha256": {
            "authorship": sha256_file(AUTH),
            "geography": sha256_file(GEO),
            "analysis_plan": sha256_file(PLAN),
            "technical_spec": sha256_file(TECH_SPEC),
        },
        "warnings": [
            warning
            for condition, warning in [
                (not stable_pass, "V03_IDENTITY_COVERAGE_WARNING"),
                (not country_models_pass, "V03_COUNTRY_MODEL_WARNING"),
            ]
            if condition
        ],
    }


def write_report(
    validation: dict[str, Any],
    raw_network: pd.DataFrame,
    primary_predictions: pd.DataFrame,
    country_coverage: pd.DataFrame,
) -> None:
    overall = (
        raw_network.groupby("entry_with_experienced_top4_coauthor", as_index=False)
        .agg(
            authors=("unique_authors", "sum"),
            any_return_2cycle_events=("any_return_2cycle_events", "sum"),
            first_author_return_2cycle_events=("first_author_return_2cycle_events", "sum"),
        )
    )
    overall["any_return_2cycle"] = overall.any_return_2cycle_events / overall.authors
    overall["first_author_return_2cycle"] = (
        overall.first_author_return_2cycle_events / overall.authors
    )
    lines = [
        "# FAccT v0.3 신규 진입자 지속성 smoke 결과",
        "",
        f"**현재 판정: `{validation['status']}`**",
        "",
        "이 분석은 ICML·NeurIPS에서 채택된 newcomer 제1저자가 이후 두 cycle 안에",
        "AAAI·ICLR·ICML·NeurIPS 채택논문에 다시 등장하는지를 측정한다. 제출 또는",
        "채택 확률, 개인의 LLM 사용, 인과적 네트워크 효과를 측정하지 않는다.",
        "",
        "## 표본",
        "",
        f"- 고유 신규 제1저자: {validation['eligible_unique_authors']:,}명",
        f"- index 논문: {validation['eligible_index_papers']:,}편",
        f"- 같은 index cycle에 2편 이상인 저자: {validation['multi_paper_index_authors']:,}명",
        f"- venue-cohort stable PID 최저 coverage: {validation['stable_pid_min_venue_cohort_coverage']:.1%}",
        "",
        "## 원자료 기술통계",
        "",
        "| Entry network | Authors | Any-author return | First-author return |",
        "|---|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        label = "experienced coauthor" if row.entry_with_experienced_top4_coauthor else "all-new entry team"
        lines.append(
            f"| {label} | {int(row.authors):,} | {row.any_return_2cycle:.1%} | {row.first_author_return_2cycle:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 조정된 두-cycle 복귀확률",
            "",
            "| Outcome | Entry network | Probability | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in primary_predictions.itertuples(index=False):
        label = "experienced coauthor" if row.entry_with_experienced_top4_coauthor else "all-new entry team"
        lines.append(
            f"| {row.outcome} | {label} | {row.adjusted_two_cycle_probability:.1%} | [{row.ci_low:.1%}, {row.ci_high:.1%}] |"
        )
    min_country = country_coverage.groupby("country_specification").country_coverage.min()
    lines.extend(
        [
            "",
            "## 측정 상태",
            "",
            f"- conservative country coverage 최저값: {min_country.get('conservative', math.nan):.1%}",
            f"- sensitivity country coverage 최저값: {min_country.get('sensitivity', math.nan):.1%}",
            "- 모든 공개 산출물은 집계표이며 이름·PID·author key·논문 제목을 포함하지 않는다.",
            "",
            "## 해석 제한",
            "",
            "경험 공저자와 복귀의 차이는 선택·연구역량·기관자원 등 관측되지 않은 요인을",
            "포함한 연관성이다. 경험 공저자가 복귀를 인과적으로 만들었다는 뜻이 아니다.",
            "또한 accepted-only persistence이므로 학회 제출기회나 채택 장벽 전체를 대표하지 않는다.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    auth, geo = load_inputs()
    index_papers, stable_coverage = build_index_papers(auth)
    cohort, index_paper_subset = aggregate_author_cohorts(index_papers)
    cohort = attach_country_groups(cohort, index_paper_subset, geo)
    cohort = attach_returns(cohort, auth)

    raw_network, raw_country, country_coverage = aggregate_tables(cohort)
    primary_coef, primary_pred, primary_diag = fit_primary_models(cohort)
    country_coef, country_pred, country_diag = fit_country_models(cohort)

    output_tables = {
        "stable_pid_coverage": stable_coverage,
        "raw_network": raw_network,
        "raw_country": raw_country,
        "country_coverage": country_coverage,
        "primary_coefficients": primary_coef,
        "primary_predictions": primary_pred,
        "primary_diagnostics": primary_diag,
        "country_coefficients": country_coef,
        "country_predictions": country_pred,
        "country_diagnostics": country_diag,
    }
    validation = validate(
        cohort,
        stable_coverage,
        primary_diag,
        country_diag,
        output_tables,
    )

    for name, table in output_tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(primary_pred)
    write_report(validation, raw_network, primary_pred, country_coverage)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen EPJ distance, collaboration, and exact +2 analyses."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices, cr, dmatrix
from scipy.special import expit
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
DATA = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_analysis_dataset_private.csv"
CONFIG = ROOT / "config/epj_distance_based_analysis_v10.json"
PROTOCOL = ROOT / "docs/epj_distance_based_analysis_protocol_v1.0.md"
IMPLEMENTATION = ROOT / "docs/epj_distance_model_implementation_note_v1.0a.md"
OUT = ROOT / "outputs/epj_distance_based_v10"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MANIFESTS = OUT / "manifests"
REPORT = OUT / "EPJ_DISTANCE_BASED_RESULTS_KO.md"

CATEGORIES = (
    "no_recurrence",
    "coauthor_continuity_only",
    "at_least_one_no_index_coauthor",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_parameter_draws(
    params: np.ndarray, covariance: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    clipped = np.clip(values, 0, None)
    root = vectors @ np.diag(np.sqrt(clipped))
    noise = rng.standard_normal((n, len(params)))
    return params[None, :] + noise @ root.T


def quantile_interval(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def build_new_design(design_info: Any, data: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        build_design_matrices([design_info], data, return_type="dataframe")[0],
        dtype=float,
    )


def rq1_analysis(
    data: pd.DataFrame, n_draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for year, block in data.groupby("index_year"):
        values = block.specter2_title_distance_primary.to_numpy(dtype=float)
        boot = [
            float(np.median(values[rng.integers(0, len(values), size=len(values))]))
            for _ in range(n_draws)
        ]
        low, high = quantile_interval(boot)
        rows.append(
            {
                "index_year": int(year),
                "entrants_with_distance": int(len(block)),
                "median_distance": float(np.median(values)),
                "q25_distance": float(np.quantile(values, 0.25)),
                "q75_distance": float(np.quantile(values, 0.75)),
                "median_ci_low": low,
                "median_ci_high": high,
                "interval": "entrant_bootstrap",
            }
        )
    descriptive = pd.DataFrame(rows).sort_values("index_year")

    formula = (
        "specter2_title_distance_primary ~ C(index_year) + C(index_venue) + "
        "np.log1p(entry_team_size_mean) + np.log1p(n_index_papers) + "
        "np.log1p(n_unique_primary_prior_titles)"
    )
    fit = smf.ols(formula, data=data).fit(cov_type="HC3")
    design_info = fit.model.data.design_info
    parameter_draws = safe_parameter_draws(
        np.asarray(fit.params), np.asarray(fit.cov_params()), n_draws, rng
    )
    adjusted_rows: list[dict[str, object]] = []
    for year in sorted(data.index_year.unique()):
        new = data.copy()
        new["index_year"] = int(year)
        design = build_new_design(design_info, new)
        mean_design = design.mean(axis=0)
        estimate = float(mean_design @ np.asarray(fit.params))
        draws = parameter_draws @ mean_design
        low, high = quantile_interval(draws)
        adjusted_rows.append(
            {
                "index_year": int(year),
                "adjusted_mean_distance": estimate,
                "ci_low": low,
                "ci_high": high,
                "interval": "HC3_coefficient_draw",
            }
        )
    diagnostics = {
        "n": int(len(data)),
        "formula": formula,
        "r_squared": float(fit.rsquared),
        "finite_parameters": bool(np.isfinite(np.asarray(fit.params)).all()),
        "design_columns": int(fit.model.exog.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(fit.model.exog)),
        "full_rank": bool(
            np.linalg.matrix_rank(fit.model.exog) == fit.model.exog.shape[1]
        ),
    }
    return descriptive, pd.DataFrame(adjusted_rows), diagnostics


def prepare_distance(
    data: pd.DataFrame,
    distance_column: str,
    *,
    year_end: int | None = None,
    team_overlap: bool = False,
    exclude_both: bool = False,
) -> tuple[pd.DataFrame, float, float, float, float]:
    result = data.loc[data[distance_column].notna()].copy()
    if year_end is not None:
        result = result.loc[result.index_year.le(year_end)].copy()
    result = result.loc[result.entry_team_size_mean.ge(2)].copy()
    if team_overlap:
        result = result.loc[result.entry_team_size_mean.le(13)].copy()
    if exclude_both:
        result = result.loc[result.index_venue.ne("Both")].copy()
    mean = float(result[distance_column].mean())
    std = float(result[distance_column].std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"invalid distance standard deviation for {distance_column}")
    q25 = float(result[distance_column].quantile(0.25))
    q75 = float(result[distance_column].quantile(0.75))
    result["z_distance"] = (result[distance_column] - mean) / std
    result["experienced"] = result.entry_with_experienced_top4_coauthor.astype(int)
    return result, mean, std, q25, q75


def rq2_specification(
    source: pd.DataFrame,
    *,
    specification: str,
    distance_column: str,
    n_draws: int,
    seed: int,
    distance_spline: bool = False,
    team_overlap: bool = False,
    exclude_both: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data, mean, std, q25, q75 = prepare_distance(
        source,
        distance_column,
        team_overlap=team_overlap,
        exclude_both=exclude_both,
    )
    distance_term = (
        "cr(z_distance, df=3, constraints='center')"
        if distance_spline
        else "z_distance"
    )
    prior_title_count_column = (
        "n_unique_alltype_prior_titles"
        if distance_column == "specter2_title_distance_alltypes_sensitivity"
        else "n_unique_primary_prior_titles"
    )
    formula = (
        f"experienced ~ {distance_term} + C(index_year) + C(index_venue) + "
        "cr(np.log1p(entry_team_size_mean), df=3, constraints='center') + "
        f"np.log1p(n_index_papers) + np.log1p({prior_title_count_column})"
    )
    fit = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(cov_type="HC3")
    design_info = fit.model.data.design_info
    rng = np.random.default_rng(seed)
    draws = safe_parameter_draws(
        np.asarray(fit.params), np.asarray(fit.cov_params()), n_draws, rng
    )
    probability_rows: list[dict[str, object]] = []
    distributions: dict[str, np.ndarray] = {}
    for label, raw_distance in (("Q25", q25), ("Q75", q75)):
        new = data.copy()
        new["z_distance"] = (raw_distance - mean) / std
        design = build_new_design(design_info, new)
        estimate = float(expit(design @ np.asarray(fit.params)).mean())
        distribution = expit(design @ draws.T).mean(axis=0)
        distributions[label] = distribution
        low, high = quantile_interval(distribution)
        probability_rows.append(
            {
                "specification": specification,
                "distance_measure": distance_column,
                "distance_level": label,
                "raw_distance": raw_distance,
                "adjusted_experienced_coauthor_probability": estimate,
                "ci_low": low,
                "ci_high": high,
                "interval": "HC3_coefficient_draw",
                "n": int(len(data)),
            }
        )
    contrast_distribution = distributions["Q75"] - distributions["Q25"]
    contrast_low, contrast_high = quantile_interval(contrast_distribution)
    probabilities = pd.DataFrame(probability_rows)
    point_lookup = probabilities.set_index("distance_level")
    contrasts = pd.DataFrame(
        [
            {
                "specification": specification,
                "distance_measure": distance_column,
                "contrast": "Q75_minus_Q25",
                "probability_difference": float(
                    point_lookup.loc["Q75", "adjusted_experienced_coauthor_probability"]
                    - point_lookup.loc["Q25", "adjusted_experienced_coauthor_probability"]
                ),
                "ci_low": contrast_low,
                "ci_high": contrast_high,
                "interval": "paired_HC3_coefficient_draw",
                "n": int(len(data)),
            }
        ]
    )
    diagnostics = {
        "specification": specification,
        "n": int(len(data)),
        "formula": formula,
        "converged": bool(fit.converged),
        "finite_parameters": bool(np.isfinite(np.asarray(fit.params)).all()),
        "design_columns": int(fit.model.exog.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(fit.model.exog)),
        "full_rank": bool(
            np.linalg.matrix_rank(fit.model.exog) == fit.model.exog.shape[1]
        ),
        "prior_title_count_column": prior_title_count_column,
        "distance_mean": mean,
        "distance_std": std,
        "distance_q25": q25,
        "distance_q75": q75,
    }
    return probabilities, contrasts, diagnostics


def fit_multinomial(data: pd.DataFrame, formula: str) -> tuple[Any, Any]:
    design = dmatrix(formula, data, return_type="dataframe")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise RuntimeError("multinomial design matrix is not full rank")
    outcome = pd.Categorical(data.outcome, categories=CATEGORIES).codes
    if set(np.unique(outcome)) != {0, 1, 2}:
        raise RuntimeError("multinomial outcome category missing")
    model = sm.MNLogit(outcome, design)
    try:
        fit = model.fit(
            method="newton", maxiter=200, disp=False, warn_convergence=False
        )
    except Exception:
        fit = None
    if fit is None or not bool(fit.mle_retvals.get("converged", False)):
        start_params = None if fit is None else np.asarray(fit.params)
        fit = model.fit(
            method="bfgs",
            maxiter=500,
            disp=False,
            warn_convergence=False,
            start_params=start_params,
        )
    if not bool(fit.mle_retvals.get("converged", False)):
        raise RuntimeError("multinomial model did not converge")
    return fit, design.design_info


def standardized_multinomial(
    fit: Any,
    design_info: Any,
    data: pd.DataFrame,
    *,
    z_distance: float,
    experienced: int,
) -> np.ndarray:
    new = data.copy()
    new["z_distance"] = z_distance
    new["experienced"] = experienced
    design = build_new_design(design_info, new)
    probabilities = np.asarray(fit.model.predict(fit.params, exog=design), dtype=float)
    return probabilities.mean(axis=0)


def rq3_specification(
    source: pd.DataFrame,
    *,
    specification: str,
    distance_column: str,
    outcome_column: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data, mean, std, q25, q75 = prepare_distance(
        source, distance_column, year_end=2022
    )
    data["outcome"] = data[outcome_column]
    data = data.loc[data.outcome.isin(CATEGORIES)].copy()
    prior_title_count_column = (
        "n_unique_alltype_prior_titles"
        if distance_column == "specter2_title_distance_alltypes_sensitivity"
        else "n_unique_primary_prior_titles"
    )
    formula = (
        "1 + z_distance * experienced + C(index_year) + C(index_venue) + "
        "cr(np.log1p(entry_team_size_mean), df=3, constraints='center') + "
        f"np.log1p(n_index_papers) + np.log1p({prior_title_count_column})"
    )
    fit, design_info = fit_multinomial(data, formula)
    levels = {"Q25": (q25 - mean) / std, "Q75": (q75 - mean) / std}
    points: dict[tuple[str, int], np.ndarray] = {
        (label, experienced): standardized_multinomial(
            fit,
            design_info,
            data,
            z_distance=z_value,
            experienced=experienced,
        )
        for label, z_value in levels.items()
        for experienced in (0, 1)
    }

    bootstrap_values: dict[tuple[str, int, int], list[float]] = {
        (label, experienced, category): []
        for label in levels
        for experienced in (0, 1)
        for category in range(3)
    }
    successful = 0
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        n = len(data)
        for iteration in range(n_bootstrap):
            sample = data.iloc[rng.integers(0, n, size=n)].copy()
            try:
                boot_fit, boot_design = fit_multinomial(sample, formula)
                if not np.isfinite(np.asarray(boot_fit.params)).all():
                    continue
                staged: dict[tuple[str, int], np.ndarray] = {}
                for label, z_value in levels.items():
                    for experienced in (0, 1):
                        staged[(label, experienced)] = standardized_multinomial(
                            boot_fit,
                            boot_design,
                            sample,
                            z_distance=z_value,
                            experienced=experienced,
                        )
                for (label, experienced), probabilities in staged.items():
                    for category in range(3):
                        bootstrap_values[(label, experienced, category)].append(
                            float(probabilities[category])
                        )
                successful += 1
            except Exception:
                continue
            if (iteration + 1) % 100 == 0:
                print(
                    f"rq3={specification} bootstrap={iteration + 1}/{n_bootstrap} successful={successful}",
                    flush=True,
                )

    probability_rows: list[dict[str, object]] = []
    for label, raw_distance in (("Q25", q25), ("Q75", q75)):
        for experienced in (0, 1):
            for category_index, category in enumerate(CATEGORIES):
                values = bootstrap_values[(label, experienced, category_index)]
                low, high = quantile_interval(values) if values else (np.nan, np.nan)
                probability_rows.append(
                    {
                        "specification": specification,
                        "distance_measure": distance_column,
                        "return_outcome": outcome_column,
                        "distance_level": label,
                        "raw_distance": raw_distance,
                        "experienced_coauthor": bool(experienced),
                        "pathway": category,
                        "adjusted_probability": float(points[(label, experienced)][category_index]),
                        "ci_low": low,
                        "ci_high": high,
                        "interval": "entrant_bootstrap" if n_bootstrap else "not_computed_sensitivity",
                        "n": int(len(data)),
                    }
                )

    contrast_rows: list[dict[str, object]] = []
    for label in levels:
        for category_index, category in enumerate(CATEGORIES):
            point = float(points[(label, 1)][category_index] - points[(label, 0)][category_index])
            true_values = np.asarray(bootstrap_values[(label, 1, category_index)])
            false_values = np.asarray(bootstrap_values[(label, 0, category_index)])
            values = true_values - false_values if len(true_values) else np.array([])
            low, high = quantile_interval(values) if len(values) else (np.nan, np.nan)
            contrast_rows.append(
                {
                    "specification": specification,
                    "contrast_family": "experienced_minus_no_experienced",
                    "distance_level": label,
                    "experienced_level": "difference",
                    "pathway": category,
                    "probability_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "interval": "paired_entrant_bootstrap" if n_bootstrap else "not_computed_sensitivity",
                }
            )
    for experienced in (0, 1):
        for category_index, category in enumerate(CATEGORIES):
            point = float(points[("Q75", experienced)][category_index] - points[("Q25", experienced)][category_index])
            high_values = np.asarray(bootstrap_values[("Q75", experienced, category_index)])
            low_values = np.asarray(bootstrap_values[("Q25", experienced, category_index)])
            values = high_values - low_values if len(high_values) else np.array([])
            low, high = quantile_interval(values) if len(values) else (np.nan, np.nan)
            contrast_rows.append(
                {
                    "specification": specification,
                    "contrast_family": "Q75_minus_Q25",
                    "distance_level": "difference",
                    "experienced_level": bool(experienced),
                    "pathway": category,
                    "probability_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "interval": "paired_entrant_bootstrap" if n_bootstrap else "not_computed_sensitivity",
                }
            )
    for category_index, category in enumerate(CATEGORIES):
        point = float(
            (points[("Q75", 1)][category_index] - points[("Q25", 1)][category_index])
            - (points[("Q75", 0)][category_index] - points[("Q25", 0)][category_index])
        )
        q75_true = np.asarray(bootstrap_values[("Q75", 1, category_index)])
        q25_true = np.asarray(bootstrap_values[("Q25", 1, category_index)])
        q75_false = np.asarray(bootstrap_values[("Q75", 0, category_index)])
        q25_false = np.asarray(bootstrap_values[("Q25", 0, category_index)])
        values = (
            (q75_true - q25_true) - (q75_false - q25_false)
            if len(q75_true)
            else np.array([])
        )
        low, high = quantile_interval(values) if len(values) else (np.nan, np.nan)
        contrast_rows.append(
            {
                "specification": specification,
                "contrast_family": "difference_in_differences",
                "distance_level": "Q75_minus_Q25",
                "experienced_level": "experienced_minus_no_experienced",
                "pathway": category,
                "probability_difference": point,
                "ci_low": low,
                "ci_high": high,
                "interval": "paired_entrant_bootstrap" if n_bootstrap else "not_computed_sensitivity",
            }
        )
    diagnostics = {
        "specification": specification,
        "n": int(len(data)),
        "formula": formula,
        "converged": bool(fit.mle_retvals.get("converged", True)),
        "finite_parameters": bool(np.isfinite(np.asarray(fit.params)).all()),
        "design_columns": int(fit.model.exog.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(fit.model.exog)),
        "full_rank": bool(
            np.linalg.matrix_rank(fit.model.exog) == fit.model.exog.shape[1]
        ),
        "prior_title_count_column": prior_title_count_column,
        "bootstrap_requested": int(n_bootstrap),
        "bootstrap_successful": int(successful),
        "distance_q25": q25,
        "distance_q75": q75,
        "all_categories_present": bool(set(data.outcome) == set(CATEGORIES)),
    }
    return pd.DataFrame(probability_rows), pd.DataFrame(contrast_rows), diagnostics


def measurement_robustness_decision(
    rq2: pd.DataFrame, rq3: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    measurement_specs = [
        "semantic_primary", "tfidf_word", "tfidf_char", "semantic_alltypes"
    ]
    rows: list[dict[str, object]] = []
    rq2_lookup = rq2.set_index("specification").probability_difference
    for specification in measurement_specs:
        rows.append(
            {
                "estimand": "RQ2_Q75_minus_Q25_experienced_probability",
                "specification": specification,
                "estimate": float(rq2_lookup.loc[specification]),
            }
        )
    focal = rq3.loc[
        rq3.specification.isin(measurement_specs)
        & (
            (
                rq3.contrast_family.eq("experienced_minus_no_experienced")
                & rq3.distance_level.eq("Q75")
            )
            | rq3.contrast_family.eq("Q75_minus_Q25")
        )
    ].copy()
    for row in focal.itertuples(index=False):
        rows.append(
            {
                "estimand": f"RQ3_{row.contrast_family}_{row.distance_level}_{row.experienced_level}_{row.pathway}",
                "specification": row.specification,
                "estimate": float(row.probability_difference),
            }
        )
    table = pd.DataFrame(rows)
    table["sign"] = np.sign(table.estimate).astype(int)
    sign_counts = table.groupby("estimand").sign.nunique()
    directionally_robust = bool(sign_counts.le(1).all())
    if not directionally_robust:
        decision = "DISTANCE_MEASUREMENT_SPECIFICATION_SENSITIVE_WARNING"
    else:
        grouped = table.groupby("estimand").estimate.agg(lambda x: float(x.max() - x.min()))
        decision = (
            "DIRECTIONALLY_ROBUST_QUANTITATIVELY_SENSITIVE"
            if bool((grouped > 0.05).any())
            else "DISTANCE_MEASUREMENT_DIRECTIONALLY_ROBUST"
        )
    return decision, table


def make_figures(
    descriptive: pd.DataFrame,
    adjusted: pd.DataFrame,
    rq2_probabilities: pd.DataFrame,
    rq3_probabilities: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.errorbar(
        descriptive.index_year,
        descriptive.median_distance,
        yerr=np.maximum(
            np.vstack(
                [
                    descriptive.median_distance - descriptive.median_ci_low,
                    descriptive.median_ci_high - descriptive.median_distance,
                ]
            ),
            0,
        ),
        marker="o",
        color="#4e79a7",
        label="Observed median",
    )
    axis.plot(
        adjusted.index_year,
        adjusted.adjusted_mean_distance,
        marker="s",
        color="#e15759",
        label="Adjusted mean",
    )
    axis.fill_between(
        adjusted.index_year,
        adjusted.ci_low,
        adjusted.ci_high,
        color="#e15759",
        alpha=0.18,
    )
    axis.set_xlabel("Index year")
    axis.set_ylabel("SPECTER2 title-only portfolio distance")
    axis.set_title("Prior-to-index portfolio distance among observed-history entrants")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "rq1_distance_trajectory.png", dpi=240)
    figure.savefig(FIGURES / "rq1_distance_trajectory.pdf")
    plt.close(figure)

    rq2 = rq2_probabilities.loc[rq2_probabilities.specification.eq("semantic_primary")]
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    axis.errorbar(
        rq2.distance_level,
        rq2.adjusted_experienced_coauthor_probability,
        yerr=np.maximum(
            np.vstack(
                [
                    rq2.adjusted_experienced_coauthor_probability - rq2.ci_low,
                    rq2.ci_high - rq2.adjusted_experienced_coauthor_probability,
                ]
            ),
            0,
        ),
        marker="o",
        color="#59a14f",
        capsize=4,
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Adjusted probability")
    axis.set_xlabel("Prior-to-index distance")
    axis.set_title("Observed recent Top-4-experienced coauthor at entry")
    figure.tight_layout()
    figure.savefig(FIGURES / "rq2_experienced_coauthor_probability.png", dpi=240)
    figure.savefig(FIGURES / "rq2_experienced_coauthor_probability.pdf")
    plt.close(figure)

    primary = rq3_probabilities.loc[rq3_probabilities.specification.eq("semantic_primary")].copy()
    scenario_order = [("Q25", False), ("Q25", True), ("Q75", False), ("Q75", True)]
    labels = ["Q25 / no exp.", "Q25 / exp.", "Q75 / no exp.", "Q75 / exp."]
    colors = {
        "no_recurrence": "#bab0ac",
        "coauthor_continuity_only": "#e15759",
        "at_least_one_no_index_coauthor": "#4e79a7",
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    bottom = np.zeros(len(scenario_order))
    for category in CATEGORIES:
        values = []
        for distance_level, experienced in scenario_order:
            value = primary.loc[
                primary.distance_level.eq(distance_level)
                & primary.experienced_coauthor.eq(experienced)
                & primary.pathway.eq(category),
                "adjusted_probability",
            ].iloc[0]
            values.append(float(value))
        axis.bar(
            np.arange(len(values)), values, bottom=bottom, color=colors[category],
            label=category.replace("_", " "),
        )
        bottom += np.asarray(values)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Adjusted pathway probability")
    axis.set_title("Exact +2 publication-cycle pathways")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES / "rq3_exact_plus2_pathways.png", dpi=240)
    figure.savefig(FIGURES / "rq3_exact_plus2_pathways.pdf")
    plt.close(figure)


def write_report(
    data: pd.DataFrame,
    descriptive: pd.DataFrame,
    rq2_contrasts: pd.DataFrame,
    rq3_contrasts: pd.DataFrame,
    decision: str,
    diagnostics: dict[str, object],
) -> None:
    rq2_primary = rq2_contrasts.loc[rq2_contrasts.specification.eq("semantic_primary")].iloc[0]
    rq3_primary = rq3_contrasts.loc[
        rq3_contrasts.specification.eq("semantic_primary")
        & rq3_contrasts.contrast_family.eq("experienced_minus_no_experienced")
        & rq3_contrasts.distance_level.eq("Q75")
    ]
    lines = [
        "# EPJ distance-based 본 분석 결과",
        "",
        f"**전체 QA 상태: `{diagnostics['status']}`**",
        f"**측정 사양 판정: `{decision}`**",
        "",
        "## 이 연구가 측정한 것",
        "",
        "ICML·NeurIPS 채택논문에서 first-listed observed Top-4-new entrant의 진입 전 5년 DBLP 논문 제목 포트폴리오와 index-year 논문 제목 포트폴리오 사이의 거리를 측정했다.",
        "이 값은 SPECTER2 title-only distance이며 실제 전공, 학위, 논문 전체 내용 또는 제출·채택 기회를 직접 나타내지 않는다.",
        "",
        "## 표본",
        "",
        f"- 전체 entrant: {len(data):,}명",
        f"- primary distance 관측 entrant: {int(data.primary_distance_observed.sum()):,}명 ({data.primary_distance_observed.mean():.1%})",
        "- exact +2 분석은 2018–2022 entrant와 관측된 prior portfolio에 조건부",
        "",
        "## RQ1. 연도별 거리",
        "",
        "| 연도 | N | 중앙값 | IQR | 95% bootstrap interval |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in descriptive.itertuples(index=False):
        lines.append(
            f"| {row.index_year} | {row.entrants_with_distance:,} | {row.median_distance:.3f} | "
            f"[{row.q25_distance:.3f}, {row.q75_distance:.3f}] | [{row.median_ci_low:.3f}, {row.median_ci_high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## RQ2. 거리와 경험공저자 구성",
            "",
            f"Q75 − Q25 조정확률 차이: {rq2_primary.probability_difference:+.1%} "
            f"[{rq2_primary.ci_low:+.1%}, {rq2_primary.ci_high:+.1%}]",
            "",
            "이 차이는 관측된 연관성이며 경험공저자가 진입을 일으켰다는 인과효과가 아니다.",
            "",
            "## RQ3. Q75 거리에서 경험공저자 있음 − 없음 exact +2 경로 차이",
            "",
            "| 경로 | 조정확률 차이 | 95% entrant-bootstrap interval |",
            "|---|---:|---:|",
        ]
    )
    for row in rq3_primary.itertuples(index=False):
        lines.append(
            f"| {row.pathway} | {row.probability_difference:+.1%} | [{row.ci_low:+.1%}, {row.ci_high:+.1%}] |"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- accepted-program data이므로 submission entry 또는 acceptance probability를 추정하지 않는다.",
            "- prior-title history가 관측된 entrant에 조건부인 거리 분석이다.",
            "- exact +2 conference cycle은 정확한 경과시간이 아니다.",
            "- coauthor continuity는 멘토링, 독립성 또는 네트워크 인과효과가 아니다.",
            "- LLM/ChatGPT 인과효과를 주장하지 않는다.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (TABLES, FIGURES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    data = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
    data["primary_distance_observed"] = data.specter2_title_distance_primary.notna()
    n_draws = int(config["bootstrap_resamples"])
    seed = int(config["random_seed"])
    observed = data.loc[data.primary_distance_observed].copy()

    descriptive, adjusted, rq1_diagnostics = rq1_analysis(observed, n_draws, seed)
    descriptive.to_csv(TABLES / "rq1_year_distance_descriptive.csv", index=False)
    adjusted.to_csv(TABLES / "rq1_year_adjusted_mean_distance.csv", index=False)

    rq2_specs = [
        ("semantic_primary", "specter2_title_distance_primary", False, False, False),
        ("tfidf_word", "tfidf_word_title_distance_sensitivity", False, False, False),
        ("tfidf_char", "tfidf_char_title_distance_sensitivity", False, False, False),
        ("semantic_alltypes", "specter2_title_distance_alltypes_sensitivity", False, False, False),
        ("semantic_team_overlap", "specter2_title_distance_primary", False, True, False),
        ("semantic_exclude_both", "specter2_title_distance_primary", False, False, True),
        ("semantic_distance_spline", "specter2_title_distance_primary", True, False, False),
    ]
    rq2_probabilities_list: list[pd.DataFrame] = []
    rq2_contrasts_list: list[pd.DataFrame] = []
    rq2_diagnostics: list[dict[str, object]] = []
    for offset, (spec, column, spline, overlap, exclude_both) in enumerate(rq2_specs):
        probabilities, contrasts, diagnostics = rq2_specification(
            data,
            specification=spec,
            distance_column=column,
            n_draws=n_draws,
            seed=seed + 100 + offset,
            distance_spline=spline,
            team_overlap=overlap,
            exclude_both=exclude_both,
        )
        rq2_probabilities_list.append(probabilities)
        rq2_contrasts_list.append(contrasts)
        rq2_diagnostics.append(diagnostics)
    rq2_probabilities = pd.concat(rq2_probabilities_list, ignore_index=True)
    rq2_contrasts = pd.concat(rq2_contrasts_list, ignore_index=True)
    rq2_probabilities.to_csv(TABLES / "rq2_experienced_coauthor_probabilities.csv", index=False)
    rq2_contrasts.to_csv(TABLES / "rq2_experienced_coauthor_contrasts.csv", index=False)
    pd.DataFrame(rq2_diagnostics).to_csv(TABLES / "rq2_model_diagnostics.csv", index=False)

    rq3_specs = [
        ("semantic_primary", "specter2_title_distance_primary", "exact_plus2_top4", n_draws),
        ("tfidf_word", "tfidf_word_title_distance_sensitivity", "exact_plus2_top4", 0),
        ("tfidf_char", "tfidf_char_title_distance_sensitivity", "exact_plus2_top4", 0),
        ("semantic_alltypes", "specter2_title_distance_alltypes_sensitivity", "exact_plus2_top4", 0),
        ("semantic_exclude_aaai_return", "specter2_title_distance_primary", "exact_plus2_exclude_aaai", 0),
        ("semantic_icml_neurips_return", "specter2_title_distance_primary", "exact_plus2_icml_neurips", 0),
    ]
    rq3_probabilities_list: list[pd.DataFrame] = []
    rq3_contrasts_list: list[pd.DataFrame] = []
    rq3_diagnostics: list[dict[str, object]] = []
    for offset, (spec, column, outcome, bootstrap) in enumerate(rq3_specs):
        probabilities, contrasts, diagnostics = rq3_specification(
            data,
            specification=spec,
            distance_column=column,
            outcome_column=outcome,
            n_bootstrap=bootstrap,
            seed=seed + 1000 + offset,
        )
        rq3_probabilities_list.append(probabilities)
        rq3_contrasts_list.append(contrasts)
        rq3_diagnostics.append(diagnostics)
    rq3_probabilities = pd.concat(rq3_probabilities_list, ignore_index=True)
    rq3_contrasts = pd.concat(rq3_contrasts_list, ignore_index=True)
    rq3_probabilities.to_csv(TABLES / "rq3_exact_plus2_probabilities.csv", index=False)
    rq3_contrasts.to_csv(TABLES / "rq3_exact_plus2_contrasts.csv", index=False)
    pd.DataFrame(rq3_diagnostics).to_csv(TABLES / "rq3_model_diagnostics.csv", index=False)

    missingness = (
        data.groupby("primary_distance_observed", as_index=False)
        .agg(
            entrants=("entrant_id", "size"),
            mean_team_size=("entry_team_size_mean", "mean"),
            experienced_coauthor_share=("entry_with_experienced_top4_coauthor", "mean"),
            mean_index_year=("index_year", "mean"),
        )
    )
    missingness.to_csv(TABLES / "distance_observation_missingness_comparison.csv", index=False)

    dimensions = ["index_year", "index_venue", "entry_with_experienced_top4_coauthor"]
    coverage_blocks: list[pd.DataFrame] = []
    distribution_blocks: list[pd.DataFrame] = []
    for dimension in dimensions:
        grouped = data.groupby(dimension, as_index=False, dropna=False)
        coverage = (
            grouped.agg(
                entrants=("entrant_id", "size"),
                distance_observed=("primary_distance_observed", "sum"),
                primary_distance_coverage=("primary_distance_observed", "mean"),
            )
            .rename(columns={dimension: "level"})
        )
        coverage.insert(0, "dimension", dimension)
        coverage_blocks.append(coverage)
        for observed, block in data.groupby("primary_distance_observed"):
            distribution = (
                block.groupby(dimension, dropna=False)
                .size()
                .rename("entrants")
                .reset_index()
                .rename(columns={dimension: "level"})
            )
            distribution.insert(0, "dimension", dimension)
            distribution.insert(0, "primary_distance_observed", bool(observed))
            distribution["within_observation_group_share"] = distribution.entrants / len(block)
            distribution_blocks.append(distribution)
    coverage_by_dimension = pd.concat(coverage_blocks, ignore_index=True)
    coverage_by_dimension["level"] = coverage_by_dimension.level.astype(str)
    missingness_distributions = pd.concat(distribution_blocks, ignore_index=True)
    missingness_distributions["level"] = missingness_distributions.level.astype(str)
    coverage_by_dimension.to_csv(TABLES / "distance_observation_coverage_by_dimension.csv", index=False)
    missingness_distributions.to_csv(TABLES / "distance_observation_missingness_distributions.csv", index=False)
    decision, robustness_table = measurement_robustness_decision(
        rq2_contrasts, rq3_contrasts
    )
    robustness_table.to_csv(TABLES / "measurement_robustness_focal_signs.csv", index=False)

    make_figures(descriptive, adjusted, rq2_probabilities, rq3_probabilities)
    primary_rq3_diag = next(
        row for row in rq3_diagnostics if row["specification"] == "semantic_primary"
    )
    gates = {
        "rq1_finite_full_rank": bool(
            rq1_diagnostics["finite_parameters"] and rq1_diagnostics["full_rank"]
        ),
        "all_rq2_converged_finite": bool(
            all(
                row["converged"] and row["finite_parameters"] and row["full_rank"]
                for row in rq2_diagnostics
            )
        ),
        "all_rq3_converged_finite": bool(
            all(
                row["converged"] and row["finite_parameters"] and row["full_rank"]
                for row in rq3_diagnostics
            )
        ),
        "primary_rq3_bootstrap_success_ge_90pct": bool(
            primary_rq3_diag["bootstrap_successful"] >= 0.90 * n_draws
        ),
        "probabilities_valid": bool(
            rq2_probabilities.adjusted_experienced_coauthor_probability.between(0, 1).all()
            and rq3_probabilities.adjusted_probability.between(0, 1).all()
        ),
        "coverage_public_cells_ge_minimum": bool(
            coverage_by_dimension.entrants.min() >= int(config["public_minimum_cell_size"])
        ),
        "missingness_public_cells_ge_minimum": bool(
            missingness_distributions.entrants.min() >= int(config["public_minimum_cell_size"])
        ),
    }
    diagnostics = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "measurement_robustness_decision": decision,
        "gates": gates,
        "rq1": rq1_diagnostics,
        "rq2": rq2_diagnostics,
        "rq3": rq3_diagnostics,
        "input_sha256": {
            str(SCRIPT.relative_to(ROOT)): sha256_file(SCRIPT),
            str(DATA.relative_to(ROOT)): sha256_file(DATA),
            str(CONFIG.relative_to(ROOT)): sha256_file(CONFIG),
            str(PROTOCOL.relative_to(ROOT)): sha256_file(PROTOCOL),
            str(IMPLEMENTATION.relative_to(ROOT)): sha256_file(IMPLEMENTATION),
        },
    }
    write_report(data, descriptive, rq2_contrasts, rq3_contrasts, decision, diagnostics)
    software = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **{
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scipy", "statsmodels", "patsy", "matplotlib"]
        },
        "seed": seed,
        "bootstrap_resamples": n_draws,
    }
    (MANIFESTS / "software_versions.json").write_text(
        json.dumps(software, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_paths = [
        TABLES / "rq1_year_distance_descriptive.csv",
        TABLES / "rq1_year_adjusted_mean_distance.csv",
        TABLES / "rq2_experienced_coauthor_probabilities.csv",
        TABLES / "rq2_experienced_coauthor_contrasts.csv",
        TABLES / "rq2_model_diagnostics.csv",
        TABLES / "rq3_exact_plus2_probabilities.csv",
        TABLES / "rq3_exact_plus2_contrasts.csv",
        TABLES / "rq3_model_diagnostics.csv",
        TABLES / "measurement_robustness_focal_signs.csv",
        TABLES / "distance_observation_missingness_comparison.csv",
        TABLES / "distance_observation_coverage_by_dimension.csv",
        TABLES / "distance_observation_missingness_distributions.csv",
        FIGURES / "rq1_distance_trajectory.png",
        FIGURES / "rq1_distance_trajectory.pdf",
        FIGURES / "rq2_experienced_coauthor_probability.png",
        FIGURES / "rq2_experienced_coauthor_probability.pdf",
        FIGURES / "rq3_exact_plus2_pathways.png",
        FIGURES / "rq3_exact_plus2_pathways.pdf",
        REPORT,
    ]
    diagnostics["output_sha256"] = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths
    }
    diagnostics_path = MANIFESTS / "epj_distance_full_analysis_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if diagnostics["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

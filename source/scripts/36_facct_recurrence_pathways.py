#!/usr/bin/env python3
"""FAccT v0.3.3: recurrence with versus beyond index-cycle coauthors."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "31_facct_persistence_smoke.py"
OUT = ROOT / "outputs" / "facct_recurrence_pathway_v033"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT = OUT / "FACCT_RECURRENCE_PATHWAY_KO.md"
TOP4 = ("AAAI", "ICLR", "ICML", "NeurIPS")
RETURN_SPECS = {
    "primary_top4": {"venues": TOP4, "drop_early_aaai_iclr": False},
    "exclude_AAAI": {"venues": ("ICLR", "ICML", "NeurIPS"), "drop_early_aaai_iclr": False},
    "ICML_NeurIPS_only": {"venues": ("ICML", "NeurIPS"), "drop_early_aaai_iclr": False},
    "strict_temporal_proxy": {"venues": TOP4, "drop_early_aaai_iclr": True},
}
CATEGORIES = (
    "no_recurrence",
    "coauthor_continuity_only",
    "at_least_one_no_index_coauthor",
)
FORMULA = (
    "1 + entry_with_experienced_top4_coauthor + C(index_year) + "
    "C(index_venue) + np.log1p(entry_team_size_mean) + np.log1p(n_index_papers)"
)
N_BOOTSTRAP = 1000
SEED = 202605


def load_base():
    spec = importlib.util.spec_from_file_location("persistence_v03", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cohort(base: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    auth, geo = base.load_inputs()
    index_papers, _ = base.build_index_papers(auth)
    cohort, index_subset = base.aggregate_author_cohorts(index_papers)
    cohort = base.attach_country_groups(cohort, index_subset, geo)

    index_keys = index_subset[["author_key", "index_year", "paper_key", "venue", "year"]]
    index_members = index_keys.merge(
        auth[["paper_key", "venue", "year", "author_key", "author_order"]],
        on=["paper_key", "venue", "year"],
        how="left",
        validate="many_to_many",
        suffixes=("_entrant", "_member"),
    )
    index_members = index_members.loc[
        index_members.author_key_member.ne(index_members.author_key_entrant)
    ].copy()
    coauthor_sets = (
        index_members.groupby("author_key_entrant").author_key_member
        .agg(lambda values: set(values))
        .to_dict()
    )
    return cohort, auth, coauthor_sets


def classify_recurrence(
    cohort: pd.DataFrame,
    auth: pd.DataFrame,
    coauthor_sets: dict[str, set[str]],
    *,
    venues: tuple[str, ...],
    drop_early_aaai_iclr: bool,
) -> pd.Series:
    entrant_lookup = cohort[["author_key", "index_year"]]
    appearances = auth.loc[
        auth.author_key.isin(set(cohort.author_key))
        & auth.venue.isin(venues)
        & auth.year.le(2024)
    ].merge(entrant_lookup, on="author_key", validate="many_to_one")
    appearances["return_cycle"] = appearances.year - appearances.index_year
    appearances = appearances.loc[appearances.return_cycle.isin([1, 2])].copy()
    if drop_early_aaai_iclr:
        appearances = appearances.loc[
            ~(appearances.return_cycle.eq(1) & appearances.venue.isin(["AAAI", "ICLR"]))
        ].copy()

    first_cycle = appearances.groupby("author_key").return_cycle.min().rename("first_cycle")
    appearances = appearances.merge(first_cycle, on="author_key", validate="many_to_one")
    first_appearances = appearances.loc[
        appearances.return_cycle.eq(appearances.first_cycle)
    ].copy()
    first_papers = first_appearances[
        ["author_key", "paper_key", "venue", "year"]
    ].drop_duplicates()

    paper_members = first_papers.merge(
        auth[["paper_key", "venue", "year", "author_key"]].rename(
            columns={"author_key": "paper_member_key"}
        ),
        on=["paper_key", "venue", "year"],
        how="left",
        validate="many_to_many",
    )
    paper_members["is_index_coauthor"] = paper_members.apply(
        lambda row: row.paper_member_key in coauthor_sets.get(row.author_key, set()),
        axis=1,
    )
    paper_overlap = (
        paper_members.groupby(["author_key", "paper_key", "venue", "year"], as_index=False)
        .agg(shares_index_coauthor=("is_index_coauthor", "max"))
    )
    recurrence = (
        paper_overlap.groupby("author_key").shares_index_coauthor
        .agg(lambda values: "coauthor_continuity_only" if bool(values.all()) else "at_least_one_no_index_coauthor")
    )
    result = pd.Series("no_recurrence", index=cohort.author_key, dtype="object")
    result.loc[recurrence.index] = recurrence
    return result.reindex(cohort.author_key).reset_index(drop=True)


def fit_multinomial(data: pd.DataFrame) -> tuple[Any, Any]:
    design = dmatrix(FORMULA, data, return_type="dataframe")
    outcome = pd.Categorical(data.outcome, categories=CATEGORIES).codes
    if set(np.unique(outcome)) != {0, 1, 2}:
        raise RuntimeError("missing multinomial outcome category")
    model = sm.MNLogit(outcome, design)
    try:
        fit = model.fit(method="newton", maxiter=200, disp=False)
    except Exception:
        fit = model.fit(method="bfgs", maxiter=500, disp=False)
    return fit, design.design_info


def standardized_probabilities(
    fit: Any,
    design_info: Any,
    base: pd.DataFrame,
    *,
    experienced: bool,
) -> np.ndarray:
    new = base.copy()
    new["entry_with_experienced_top4_coauthor"] = experienced
    design = np.asarray(
        build_design_matrices([design_info], new, return_type="dataframe")[0],
        dtype=float,
    )
    probabilities = np.asarray(fit.model.predict(fit.params, exog=design), dtype=float)
    return probabilities.mean(axis=0)


def analyze_specification(
    cohort: pd.DataFrame,
    *,
    specification: str,
    bootstrap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_data = cohort.loc[cohort.entry_team_size_mean.ge(2)].copy()
    fit, design_info = fit_multinomial(model_data)
    point: dict[bool, np.ndarray] = {}
    for experienced in (False, True):
        point[experienced] = standardized_probabilities(
            fit, design_info, model_data, experienced=experienced
        )

    prediction_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    bootstrap_predictions: dict[tuple[bool, int], list[float]] = {
        (experienced, category): []
        for experienced in (False, True)
        for category in range(3)
    }
    successful = 0
    if bootstrap:
        rng = np.random.default_rng(SEED)
        n = len(model_data)
        for _ in range(N_BOOTSTRAP):
            indices = rng.integers(0, n, size=n)
            sample = model_data.iloc[indices].copy()
            try:
                boot_fit, boot_design_info = fit_multinomial(sample)
                if not np.isfinite(np.asarray(boot_fit.params)).all():
                    continue
                for experienced in (False, True):
                    probabilities = standardized_probabilities(
                        boot_fit,
                        boot_design_info,
                        sample,
                        experienced=experienced,
                    )
                    for category in range(3):
                        bootstrap_predictions[(experienced, category)].append(
                            float(probabilities[category])
                        )
                successful += 1
            except Exception:
                continue

    for experienced in (False, True):
        for category_index, category in enumerate(CATEGORIES):
            values = np.asarray(bootstrap_predictions[(experienced, category_index)])
            prediction_rows.append(
                {
                    "specification": specification,
                    "entry_with_experienced_top4_coauthor": experienced,
                    "recurrence_pathway": category,
                    "adjusted_probability": float(point[experienced][category_index]),
                    "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "interval": "entrant_bootstrap" if bootstrap else "not_computed_sensitivity",
                }
            )
    for category_index, category in enumerate(CATEGORIES):
        difference = float(point[True][category_index] - point[False][category_index])
        true_values = np.asarray(bootstrap_predictions[(True, category_index)])
        false_values = np.asarray(bootstrap_predictions[(False, category_index)])
        distribution = true_values - false_values if len(true_values) else np.array([])
        contrast_rows.append(
            {
                "specification": specification,
                "recurrence_pathway": category,
                "experienced_minus_no_recent_experienced_difference": difference,
                "ci_low": float(np.quantile(distribution, 0.025)) if len(distribution) else np.nan,
                "ci_high": float(np.quantile(distribution, 0.975)) if len(distribution) else np.nan,
                "interval": "paired_entrant_bootstrap" if bootstrap else "not_computed_sensitivity",
            }
        )
    diagnostic = {
        "specification": specification,
        "unique_entrants_all": int(cohort.author_key.nunique()),
        "unique_entrants_common_support": int(model_data.author_key.nunique()),
        "solo_excluded_from_adjusted_model": int(cohort.entry_team_size_mean.lt(2).sum()),
        "model_converged": bool(fit.mle_retvals.get("converged", True)),
        "bootstrap_requested": int(N_BOOTSTRAP if bootstrap else 0),
        "bootstrap_successful": int(successful),
        "bootstrap_success_rate": float(successful / N_BOOTSTRAP) if bootstrap else np.nan,
    }
    return pd.DataFrame(prediction_rows), pd.DataFrame(contrast_rows), diagnostic


def raw_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    return (
        cohort.groupby(
            ["return_specification", "entry_with_experienced_top4_coauthor", "outcome"],
            as_index=False,
        )
        .agg(entrants=("author_key", "nunique"))
        .assign(
            share=lambda x: x.entrants
            / x.groupby(
                ["return_specification", "entry_with_experienced_top4_coauthor"]
            ).entrants.transform("sum")
        )
    )


def make_figure(predictions: pd.DataFrame) -> None:
    data = predictions.loc[predictions.specification.eq("primary_top4")].copy()
    labels = {
        False: "No observed recent\nTop-4-experienced coauthor",
        True: "Observed recent\nTop-4-experienced coauthor",
    }
    colors = {
        "no_recurrence": "#bab0ac",
        "coauthor_continuity_only": "#e15759",
        "at_least_one_no_index_coauthor": "#4e79a7",
    }
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(2)
    bottom = np.zeros(2)
    for category in CATEGORIES:
        block = data.loc[data.recurrence_pathway.eq(category)].set_index(
            "entry_with_experienced_top4_coauthor"
        )
        values = np.array([block.loc[False, "adjusted_probability"], block.loc[True, "adjusted_probability"]])
        axis.bar(x, values, bottom=bottom, label=category.replace("_", " "), color=colors[category])
        bottom += values
    axis.set_xticks(x, [labels[False], labels[True]])
    axis.set_ylabel("Adjusted probability")
    axis.set_ylim(0, 1)
    axis.set_title("Subsequent-cycle recurrence pathways")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "recurrence_pathway_adjusted_probabilities.png", dpi=220)
    plt.close(fig)


def write_report(
    raw: pd.DataFrame,
    predictions: pd.DataFrame,
    contrasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    primary_raw = raw.loc[raw.return_specification.eq("primary_top4")]
    primary_predictions = predictions.loc[predictions.specification.eq("primary_top4")]
    primary_contrasts = contrasts.loc[contrasts.specification.eq("primary_top4")]
    lines = [
        "# FAccT v0.3.3 후속 cycle 공동연구 경로",
        "",
        f"**판정: `{validation['status']}`**",
        "",
        "이 결과는 채택 newcomer의 subsequent publication-cycle recurrence를 기존 index-cycle 공저자와의",
        "관측된 연속성 여부로 분해한다. 인과적 네트워크 효과나 과학적 독립성을 뜻하지 않는다.",
        "",
        "## 원자료 비율",
        "",
        "| Entry network | Pathway | N | Share |",
        "|---|---|---:|---:|",
    ]
    for row in primary_raw.itertuples(index=False):
        exposure = "recent Top-4-experienced coauthor" if row.entry_with_experienced_top4_coauthor else "no observed recent experienced coauthor"
        lines.append(f"| {exposure} | {row.outcome} | {row.entrants:,} | {row.share:.1%} |")
    lines.extend(
        [
            "",
            "## 조정확률",
            "",
            "| Entry network | Pathway | Probability | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in primary_predictions.itertuples(index=False):
        exposure = "recent Top-4-experienced coauthor" if row.entry_with_experienced_top4_coauthor else "no observed recent experienced coauthor"
        lines.append(
            f"| {exposure} | {row.recurrence_pathway} | {row.adjusted_probability:.1%} | [{row.ci_low:.1%}, {row.ci_high:.1%}] |"
        )
    lines.extend(
        [
            "",
            "## 경험공저자 집단 − 비경험공저자 집단",
            "",
            "| Pathway | Difference | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in primary_contrasts.itertuples(index=False):
        lines.append(
            f"| {row.recurrence_pathway} | {row.experienced_minus_no_recent_experienced_difference:.1%} | [{row.ci_low:.1%}, {row.ci_high:.1%}] |"
        )
    lines.extend(
        [
            "",
            "## 정의 민감도",
            "",
            "각 민감도 사양의 점추정치는 tables/adjusted_probabilities.csv와 probability_differences.csv에 모두 보존했다.",
            f"Primary bootstrap 성공률은 {diagnostics.loc[diagnostics.specification.eq('primary_top4'), 'bootstrap_success_rate'].iloc[0]:.1%}이다.",
            "",
            validation["interpretation"],
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = load_base()
    cohort, auth, coauthor_sets = build_cohort(base)
    specification_cohorts: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    contrast_tables: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for specification, settings in RETURN_SPECS.items():
        data = cohort.copy()
        data["outcome"] = classify_recurrence(
            data,
            auth,
            coauthor_sets,
            venues=settings["venues"],
            drop_early_aaai_iclr=settings["drop_early_aaai_iclr"],
        )
        data["return_specification"] = specification
        specification_cohorts.append(data)
        predictions, contrasts, diagnostics = analyze_specification(
            data,
            specification=specification,
            bootstrap=specification == "primary_top4",
        )
        prediction_tables.append(predictions)
        contrast_tables.append(contrasts)
        diagnostic_rows.append(diagnostics)

    all_cohorts = pd.concat(specification_cohorts, ignore_index=True)
    raw = raw_summary(all_cohorts)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    contrasts = pd.concat(contrast_tables, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    min_cell = int(raw.entrants.min())
    primary_bootstrap = diagnostics.loc[diagnostics.specification.eq("primary_top4")].iloc[0]
    cell_gate = bool(min_cell >= 20)
    bootstrap_gate = bool(primary_bootstrap.bootstrap_success_rate >= 0.95)
    models_gate = bool(diagnostics.model_converged.all())
    sensitivity_direction = contrasts.loc[
        contrasts.recurrence_pathway.ne("no_recurrence")
    ].groupby("recurrence_pathway")[
        "experienced_minus_no_recent_experienced_difference"
    ].agg(lambda values: bool((values > 0).all() or (values < 0).all()))
    validation = {
        "status": "PASS" if cell_gate and bootstrap_gate and models_gate else "FAIL",
        "gates": {
            "public_raw_cells_k_ge_20": cell_gate,
            "primary_bootstrap_success_ge_95pct": bootstrap_gate,
            "all_multinomial_models_converged": models_gate,
        },
        "minimum_public_raw_cell": min_cell,
        "sensitivity_same_sign_by_recurrence_pathway": {str(key): bool(value) for key, value in sensitivity_direction.items()},
    }
    continuity = contrasts.loc[
        contrasts.specification.eq("primary_top4")
        & contrasts.recurrence_pathway.eq("coauthor_continuity_only"),
        "experienced_minus_no_recent_experienced_difference",
    ].iloc[0]
    beyond = contrasts.loc[
        contrasts.specification.eq("primary_top4")
        & contrasts.recurrence_pathway.eq("at_least_one_no_index_coauthor"),
        "experienced_minus_no_recent_experienced_difference",
    ].iloc[0]
    if continuity > 0 and beyond > 0 and float(contrasts.loc[contrasts.specification.eq("primary_top4") & contrasts.recurrence_pathway.eq("at_least_one_no_index_coauthor"), "ci_low"].iloc[0]) > 0:
        validation["interpretation"] = (
            "Observed recent Top-4-experienced coauthor entry is associated with both greater coauthor-continuity recurrence "
            "and greater recurrence containing at least one paper beyond the index coauthor set."
        )
    elif continuity > 0:
        validation["interpretation"] = (
            "The recurrence advantage is concentrated in observed index-coauthor continuity rather than recurrence beyond that set."
        )
    else:
        validation["interpretation"] = (
            "The observed recurrence decomposition does not support a continuity-concentrated network advantage."
        )

    raw.to_csv(TABLES / "raw_pathway_shares.csv", index=False)
    predictions.to_csv(TABLES / "adjusted_probabilities.csv", index=False)
    contrasts.to_csv(TABLES / "probability_differences.csv", index=False)
    diagnostics.to_csv(TABLES / "model_diagnostics.csv", index=False)
    (TABLES / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(predictions)
    write_report(raw, predictions, contrasts, diagnostics, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

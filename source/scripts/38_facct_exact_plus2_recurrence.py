#!/usr/bin/env python3
"""FAccT locked completion: exact index+2 publication-cycle recurrence.

This analysis keeps every eligible 2018--2022 ICML/NeurIPS first-listed
newcomer proxy in the denominator and classifies only accepted Top-venue
papers indexed in ``index_year + 2``.  It never conditions on recurrence in
``index_year + 1``.  The three mutually exclusive outcomes are:

1. no exact +2 recurrence;
2. exact +2 recurrence only on papers sharing an index-cycle coauthor; and
3. at least one exact +2 paper with no index-cycle coauthor.

Conference publication cycles are timing proxies rather than exact elapsed
calendar time.  Results are descriptive associations, not network effects.
Only aggregate, k>=20-safe outputs are written.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATHWAY_SCRIPT = ROOT / "scripts" / "36_facct_recurrence_pathways.py"
BASE_SCRIPT = ROOT / "scripts" / "31_facct_persistence_smoke.py"
PROTOCOL = ROOT / "docs" / "facct_locked_completion_protocol_v0.4.md"
OUT = ROOT / "outputs" / "facct_locked_completion_v04" / "plus2"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT = OUT / "FACCT_EXACT_PLUS2_RECURRENCE_KO.md"

TOP4 = ("AAAI", "ICLR", "ICML", "NeurIPS")
RETURN_SPECS = {
    "primary_top4": TOP4,
    "exclude_AAAI": ("ICLR", "ICML", "NeurIPS"),
    "ICML_NeurIPS_only": ("ICML", "NeurIPS"),
}
CATEGORIES = (
    "no_recurrence",
    "coauthor_continuity_only",
    "at_least_one_no_index_coauthor",
)
N_BOOTSTRAP = 1000
SEED = 20260813
K_PUBLIC = 20
EXPECTED_ENTRANTS = 6704


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT.resolve())


def load_pathway_module() -> Any:
    spec = importlib.util.spec_from_file_location("facct_pathway_v033", PATHWAY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PATHWAY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_exact_plus2(
    cohort: pd.DataFrame,
    auth: pd.DataFrame,
    coauthor_sets: dict[str, set[str]],
    *,
    venues: tuple[str, ...],
) -> tuple[pd.Series, dict[str, Any]]:
    """Classify exact index+2 papers without examining index+1 status."""

    entrant_lookup = cohort[["author_key", "index_year"]]
    appearances = auth.loc[
        auth.author_key.isin(set(cohort.author_key)) & auth.venue.isin(venues)
    ].merge(entrant_lookup, on="author_key", validate="many_to_one")
    appearances["return_cycle"] = appearances.year - appearances.index_year
    exact = appearances.loc[appearances.return_cycle.eq(2)].copy()

    exact_papers = exact[["author_key", "paper_key", "venue", "year"]].drop_duplicates()
    paper_members = exact_papers.merge(
        auth[["paper_key", "venue", "year", "author_key"]].rename(
            columns={"author_key": "paper_member_key"}
        ),
        on=["paper_key", "venue", "year"],
        how="left",
        validate="many_to_many",
    )
    if len(paper_members):
        paper_members["is_index_coauthor"] = [
            member in coauthor_sets.get(entrant, set())
            for entrant, member in zip(
                paper_members.author_key, paper_members.paper_member_key
            )
        ]
        paper_overlap = (
            paper_members.groupby(
                ["author_key", "paper_key", "venue", "year"], as_index=False
            )
            .agg(shares_index_coauthor=("is_index_coauthor", "max"))
        )
        recurrence = paper_overlap.groupby("author_key").shares_index_coauthor.agg(
            lambda values: (
                "coauthor_continuity_only"
                if bool(values.all())
                else "at_least_one_no_index_coauthor"
            )
        )
    else:
        paper_overlap = pd.DataFrame(
            columns=["author_key", "paper_key", "venue", "year", "shares_index_coauthor"]
        )
        recurrence = pd.Series(dtype="object")

    result = pd.Series("no_recurrence", index=cohort.author_key, dtype="object")
    result.loc[recurrence.index] = recurrence
    result = result.reindex(cohort.author_key).reset_index(drop=True)
    diagnostics = {
        "eligible_entrants": int(cohort.author_key.nunique()),
        "entrants_with_exact_plus2_paper": int(exact_papers.author_key.nunique()),
        "unique_exact_plus2_papers": int(
            exact_papers[["paper_key", "venue", "year"]].drop_duplicates().shape[0]
        ),
        "exact_plus2_appearance_rows": int(len(exact)),
        "all_classifying_appearances_are_cycle_2": bool(
            exact.empty or exact.return_cycle.eq(2).all()
        ),
        "plus1_status_used": False,
        "cohort_denominator_retained": bool(len(result) == len(cohort)),
    }
    return result, diagnostics


def analyze_specification(
    pathway: Any,
    cohort: pd.DataFrame,
    *,
    specification: str,
    bootstrap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit multinomial model and standardized probabilities on team-size support."""

    model_data = cohort.loc[cohort.entry_team_size_mean.ge(2)].copy()
    fit, design_info = pathway.fit_multinomial(model_data)
    point = {
        experienced: pathway.standardized_probabilities(
            fit, design_info, model_data, experienced=experienced
        )
        for experienced in (False, True)
    }

    bootstrap_predictions: dict[tuple[bool, int], list[float]] = {
        (experienced, category): []
        for experienced in (False, True)
        for category in range(len(CATEGORIES))
    }
    successful = 0
    if bootstrap:
        rng = np.random.default_rng(SEED)
        n = len(model_data)
        for iteration in range(N_BOOTSTRAP):
            sample = model_data.iloc[rng.integers(0, n, size=n)].copy()
            try:
                boot_fit, boot_design_info = pathway.fit_multinomial(sample)
                if not np.isfinite(np.asarray(boot_fit.params)).all():
                    continue
                staged: dict[bool, np.ndarray] = {}
                for experienced in (False, True):
                    staged[experienced] = pathway.standardized_probabilities(
                        boot_fit,
                        boot_design_info,
                        sample,
                        experienced=experienced,
                    )
                # Append both exposure predictions only after the paired fit succeeds.
                for experienced in (False, True):
                    for category in range(len(CATEGORIES)):
                        bootstrap_predictions[(experienced, category)].append(
                            float(staged[experienced][category])
                        )
                successful += 1
            except Exception:
                continue
            if bootstrap and (iteration + 1) % 100 == 0:
                print(
                    f"{specification}: bootstrap {iteration + 1}/{N_BOOTSTRAP}, "
                    f"successful={successful}",
                    flush=True,
                )

    prediction_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for experienced in (False, True):
        for category_index, category in enumerate(CATEGORIES):
            values = np.asarray(
                bootstrap_predictions[(experienced, category_index)], dtype=float
            )
            prediction_rows.append(
                {
                    "specification": specification,
                    "entry_with_experienced_top4_coauthor": experienced,
                    "exact_plus2_pathway": category,
                    "adjusted_probability": float(point[experienced][category_index]),
                    "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "interval": (
                        "entrant_bootstrap" if bootstrap else "not_computed_sensitivity"
                    ),
                }
            )
    for category_index, category in enumerate(CATEGORIES):
        false_values = np.asarray(
            bootstrap_predictions[(False, category_index)], dtype=float
        )
        true_values = np.asarray(
            bootstrap_predictions[(True, category_index)], dtype=float
        )
        paired = true_values - false_values if len(true_values) else np.array([])
        contrast_rows.append(
            {
                "specification": specification,
                "exact_plus2_pathway": category,
                "experienced_minus_no_recent_experienced_difference": float(
                    point[True][category_index] - point[False][category_index]
                ),
                "ci_low": float(np.quantile(paired, 0.025)) if len(paired) else np.nan,
                "ci_high": float(np.quantile(paired, 0.975)) if len(paired) else np.nan,
                "interval": (
                    "paired_entrant_bootstrap"
                    if bootstrap
                    else "not_computed_sensitivity"
                ),
            }
        )

    diagnostic = {
        "specification": specification,
        "unique_entrants_all": int(cohort.author_key.nunique()),
        "unique_entrants_common_support": int(model_data.author_key.nunique()),
        "solo_excluded_from_adjusted_model": int(
            cohort.entry_team_size_mean.lt(2).sum()
        ),
        "all_three_outcomes_present": bool(
            set(model_data.outcome.unique()) == set(CATEGORIES)
        ),
        "model_converged": bool(fit.mle_retvals.get("converged", True)),
        "finite_model_parameters": bool(np.isfinite(np.asarray(fit.params)).all()),
        "bootstrap_requested": int(N_BOOTSTRAP if bootstrap else 0),
        "bootstrap_successful": int(successful),
        "bootstrap_success_rate": (
            float(successful / N_BOOTSTRAP) if bootstrap else np.nan
        ),
    }
    return pd.DataFrame(prediction_rows), pd.DataFrame(contrast_rows), diagnostic


def raw_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    result = (
        cohort.groupby(
            ["return_specification", "entry_with_experienced_top4_coauthor", "outcome"],
            as_index=False,
        )
        .agg(entrants=("author_key", "nunique"))
        .sort_values(
            ["return_specification", "entry_with_experienced_top4_coauthor", "outcome"]
        )
    )
    result["share"] = result.entrants / result.groupby(
        ["return_specification", "entry_with_experienced_top4_coauthor"]
    ).entrants.transform("sum")
    return result


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
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(2)
    bottom = np.zeros(2)
    for category in CATEGORIES:
        block = data.loc[data.exact_plus2_pathway.eq(category)].set_index(
            "entry_with_experienced_top4_coauthor"
        )
        values = np.array(
            [
                block.loc[False, "adjusted_probability"],
                block.loc[True, "adjusted_probability"],
            ]
        )
        axis.bar(
            x,
            values,
            bottom=bottom,
            label=category.replace("_", " "),
            color=colors[category],
        )
        bottom += values
    axis.set_xticks(x, [labels[False], labels[True]])
    axis.set_ylabel("Adjusted probability")
    axis.set_ylim(0, 1)
    axis.set_title("Exact index+2 publication-cycle recurrence pathways")
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES / "exact_plus2_adjusted_probabilities.png", dpi=220)
    plt.close(figure)


def write_report(
    raw: pd.DataFrame,
    predictions: pd.DataFrame,
    contrasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    primary_raw = raw.loc[raw.return_specification.eq("primary_top4")]
    primary_predictions = predictions.loc[
        predictions.specification.eq("primary_top4")
    ]
    primary_contrasts = contrasts.loc[contrasts.specification.eq("primary_top4")]
    lines = [
        "# FAccT locked completion: exact +2-cycle 재등장",
        "",
        f"**판정: `{validation['status']}`**",
        "",
        "## 질문과 정의",
        "",
        "2018–2022년 ICML·NeurIPS first-listed observed Top-4-new entrant 전체를 분모로 두고,",
        "각 entrant의 index year보다 정확히 2 publication cycle 뒤에 색인된 채택 Top-venue 논문만 분류했다.",
        "+1 cycle 재등장 여부에는 조건화하지 않았다.",
        "",
        "- `no_recurrence`: 정확한 +2 논문 없음",
        "- `coauthor_continuity_only`: 모든 +2 논문이 index-cycle 공저자를 한 명 이상 포함",
        "- `at_least_one_no_index_coauthor`: +2 논문 중 적어도 한 편은 index-cycle 공저자를 포함하지 않음",
        "",
        "이 분류는 공동저자 집합의 관측된 연속성을 측정한다. 과학적 독립성, 멘토링 또는 네트워크의 인과효과가 아니다.",
        "학회 publication cycle은 정확한 경과시간이 아니라 timing proxy다.",
        "",
        "## Primary 원자료 비율",
        "",
        "| 진입 시 관측된 최근 Top-4 경험공저자 | Exact +2 경로 | N | 비율 |",
        "|---|---|---:|---:|",
    ]
    for row in primary_raw.itertuples(index=False):
        exposure = "있음" if row.entry_with_experienced_top4_coauthor else "없음"
        lines.append(
            f"| {exposure} | {row.outcome} | {row.entrants:,} | {row.share:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Primary 조정확률",
            "",
            "조정모형은 team size≥2 공통지지 표본에서 cohort, index venue, log team size, index-paper 수를 보정했다.",
            "",
            "| 진입 시 관측된 최근 Top-4 경험공저자 | Exact +2 경로 | 조정확률 | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in primary_predictions.itertuples(index=False):
        exposure = "있음" if row.entry_with_experienced_top4_coauthor else "없음"
        lines.append(
            f"| {exposure} | {row.exact_plus2_pathway} | {row.adjusted_probability:.1%} | "
            f"[{row.ci_low:.1%}, {row.ci_high:.1%}] |"
        )
    lines.extend(
        [
            "",
            "## 경험공저자 있음 − 없음 조정확률 차이",
            "",
            "| Exact +2 경로 | 차이 | Paired entrant-bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in primary_contrasts.itertuples(index=False):
        lines.append(
            f"| {row.exact_plus2_pathway} | {row.experienced_minus_no_recent_experienced_difference:.1%} | "
            f"[{row.ci_low:.1%}, {row.ci_high:.1%}] |"
        )
    lines.extend(
        [
            "",
            "## 사전 고정 민감도",
            "",
            "AAAI 제외 및 ICML·NeurIPS only return-universe에도 같은 outcome, 공통지지 모형,",
            "1,000회 paired entrant bootstrap을 적용했다.",
            "모든 사양은 tables/adjusted_probabilities.csv와 tables/probability_differences.csv에 나란히 보존했다.",
            "",
            "| Return universe | Exact +2 경로 | 차이 | 95% CI | Bootstrap 성공 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    diagnostic_lookup = diagnostics.set_index("specification")
    for row in contrasts.itertuples(index=False):
        if row.exact_plus2_pathway == "no_recurrence":
            continue
        success = diagnostic_lookup.loc[row.specification, "bootstrap_success_rate"]
        lines.append(
            f"| {row.specification} | {row.exact_plus2_pathway} | "
            f"{row.experienced_minus_no_recent_experienced_difference:.1%} | "
            f"[{row.ci_low:.1%}, {row.ci_high:.1%}] | {success:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            f"Protocol interpretation: `{validation['protocol_interpretation']}`",
            "",
            validation["interpretation_ko"],
            "",
            "이 결과는 채택논문과 DBLP 저자 PID에 조건부인 기술적 연관성이다. 제출·채택 가능성이나 실제 인간관계의 효과를 식별하지 않는다.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_reproducibility_files(pathway: Any) -> None:
    base = pathway.load_base()
    packages = ["numpy", "pandas", "scipy", "statsmodels", "patsy", "matplotlib"]
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **{package: importlib.metadata.version(package) for package in packages},
        "bootstrap_seed": SEED,
        "bootstrap_resamples_per_specification": N_BOOTSTRAP,
    }
    software_path = TABLES / "software_versions.json"
    software_path.write_text(
        json.dumps(versions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    files: list[tuple[Path, str, str]] = [
        (base.AUTH, "input", "PRIVATE_SOURCE"),
        (base.GEO, "input", "PRIVATE_SOURCE"),
        (PROTOCOL, "frozen_protocol", "PUBLIC_DOCUMENT"),
        (BASE_SCRIPT, "code_dependency", "CODE"),
        (PATHWAY_SCRIPT, "code_dependency", "CODE"),
        (Path(__file__).resolve(), "analysis_code", "CODE"),
        (TABLES / "raw_exact_plus2_pathway_shares.csv", "output", "PUBLIC_K20_AGGREGATE"),
        (TABLES / "adjusted_probabilities.csv", "output", "PUBLIC_AGGREGATE"),
        (TABLES / "probability_differences.csv", "output", "PUBLIC_AGGREGATE"),
        (TABLES / "model_diagnostics.csv", "output", "PUBLIC_AGGREGATE"),
        (TABLES / "classification_diagnostics.csv", "output", "PUBLIC_AGGREGATE"),
        (TABLES / "validation.json", "output", "PUBLIC_AGGREGATE"),
        (software_path, "environment", "PUBLIC_METADATA"),
        (REPORT, "report", "PUBLIC_AGGREGATE"),
        (FIGURES / "exact_plus2_adjusted_probabilities.png", "figure", "PUBLIC_AGGREGATE"),
    ]
    manifest_rows: list[dict[str, Any]] = []
    for path, category, public_status in files:
        if not path.exists():
            raise RuntimeError(f"manifest target does not exist: {path}")
        manifest_rows.append(
            {
                "path": display_path(path),
                "category": category,
                "public_status": public_status,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        TABLES / "sha256_manifest.csv", index=False
    )

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pathway = load_pathway_module()
    cohort, auth, coauthor_sets = pathway.build_cohort(pathway.load_base())

    specification_cohorts: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    contrast_tables: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    for specification, venues in RETURN_SPECS.items():
        data = cohort.copy()
        data["outcome"], classification = classify_exact_plus2(
            data, auth, coauthor_sets, venues=venues
        )
        data["return_specification"] = specification
        classification["specification"] = specification
        classification["return_venues"] = ",".join(venues)
        classification_rows.append(classification)
        specification_cohorts.append(data)
        predictions, contrasts, diagnostics = analyze_specification(
            pathway,
            data,
            specification=specification,
            bootstrap=True,
        )
        prediction_tables.append(predictions)
        contrast_tables.append(contrasts)
        diagnostic_rows.append(diagnostics)

    all_cohorts = pd.concat(specification_cohorts, ignore_index=True)
    raw = raw_summary(all_cohorts)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    contrasts = pd.concat(contrast_tables, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    classification_diagnostics = pd.DataFrame(classification_rows)

    primary_diag = diagnostics.loc[
        diagnostics.specification.eq("primary_top4")
    ].iloc[0]
    probability_sums = predictions.groupby(
        ["specification", "entry_with_experienced_top4_coauthor"]
    ).adjusted_probability.sum()
    raw_sums = raw.groupby(
        ["return_specification", "entry_with_experienced_top4_coauthor"]
    ).share.sum()
    sensitivity = contrasts.loc[
        contrasts.exact_plus2_pathway.ne("no_recurrence")
    ].groupby("exact_plus2_pathway")[
        "experienced_minus_no_recent_experienced_difference"
    ].agg(lambda values: bool((values > 0).all() or (values < 0).all()))
    specification_validation: dict[str, dict[str, Any]] = {}
    for specification in RETURN_SPECS:
        diagnostic = diagnostics.loc[
            diagnostics.specification.eq(specification)
        ].iloc[0]
        block = contrasts.loc[contrasts.specification.eq(specification)].set_index(
            "exact_plus2_pathway"
        )
        continuity_row = block.loc["coauthor_continuity_only"]
        expansion_row = block.loc["at_least_one_no_index_coauthor"]
        specification_validation[specification] = {
            "model_converged_and_finite": bool(
                diagnostic.model_converged and diagnostic.finite_model_parameters
            ),
            "bootstrap_successful": int(diagnostic.bootstrap_successful),
            "bootstrap_requested": int(diagnostic.bootstrap_requested),
            "bootstrap_success_ge_95pct": bool(
                diagnostic.bootstrap_success_rate >= 0.95
            ),
            "continuity_rd_positive_ci_excludes_zero": bool(
                continuity_row.experienced_minus_no_recent_experienced_difference > 0
                and continuity_row.ci_low > 0
            ),
            "expansion_rd_positive_ci_excludes_zero": bool(
                expansion_row.experienced_minus_no_recent_experienced_difference > 0
                and expansion_row.ci_low > 0
            ),
        }

    gates = {
        "all_6704_entrants_retained_each_specification": bool(
            diagnostics.unique_entrants_all.eq(EXPECTED_ENTRANTS).all()
        ),
        "adjusted_common_support_is_team_size_ge_2": bool(
            diagnostics.unique_entrants_common_support.add(
                diagnostics.solo_excluded_from_adjusted_model
            ).eq(diagnostics.unique_entrants_all).all()
        ),
        "classification_uses_exact_plus2_only": bool(
            classification_diagnostics.all_classifying_appearances_are_cycle_2.all()
        ),
        "classification_does_not_use_plus1_status": bool(
            (~classification_diagnostics.plus1_status_used).all()
        ),
        "cohort_denominator_retained": bool(
            classification_diagnostics.cohort_denominator_retained.all()
        ),
        "all_three_outcomes_present_in_adjusted_models": bool(
            diagnostics.all_three_outcomes_present.all()
        ),
        "all_multinomial_models_converged": bool(
            diagnostics.model_converged.all()
            and diagnostics.finite_model_parameters.all()
        ),
        "all_specification_bootstrap_success_ge_95pct": bool(
            all(
                value["bootstrap_success_ge_95pct"]
                for value in specification_validation.values()
            )
        ),
        "public_raw_cells_k_ge_20": bool(raw.entrants.min() >= K_PUBLIC),
        "raw_shares_sum_to_one": bool(np.allclose(raw_sums, 1.0)),
        "adjusted_probabilities_sum_to_one": bool(
            np.allclose(probability_sums, 1.0)
        ),
        "all_specification_continuity_rd_positive_ci_excludes_zero": bool(
            all(
                value["continuity_rd_positive_ci_excludes_zero"]
                for value in specification_validation.values()
            )
        ),
        "all_specification_expansion_rd_positive_ci_excludes_zero": bool(
            all(
                value["expansion_rd_positive_ci_excludes_zero"]
                for value in specification_validation.values()
            )
        ),
    }
    validation: dict[str, Any] = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "minimum_public_raw_cell": int(raw.entrants.min()),
        "primary_bootstrap_successful": int(primary_diag.bootstrap_successful),
        "primary_bootstrap_requested": int(primary_diag.bootstrap_requested),
        "specification_validation": specification_validation,
        "sensitivity_same_sign_by_nonzero_recurrence_pathway": {
            str(key): bool(value) for key, value in sensitivity.items()
        },
        "timing_warning": (
            "Exact index_year+2 publication cycle is a timing proxy, not exact elapsed calendar time."
        ),
        "causal_warning": (
            "Observed coauthor continuity is descriptive and does not identify a network, mentoring, or collaboration effect."
        ),
    }
    continuity = contrasts.loc[
        contrasts.specification.eq("primary_top4")
        & contrasts.exact_plus2_pathway.eq("coauthor_continuity_only")
    ].iloc[0]
    beyond = contrasts.loc[
        contrasts.specification.eq("primary_top4")
        & contrasts.exact_plus2_pathway.eq("at_least_one_no_index_coauthor")
    ].iloc[0]
    if (
        continuity.experienced_minus_no_recent_experienced_difference > 0
        and continuity.ci_low > 0
    ):
        if (
            beyond.experienced_minus_no_recent_experienced_difference > 0
            and beyond.ci_low > 0
        ):
            validation["protocol_interpretation"] = "PLUS2_CONTINUITY_AND_EXPANSION"
            validation["interpretation_ko"] = (
                "정확한 +2 publication cycle에서도 경험공저자와 함께 진입한 집단은 index-cycle 공저자 연속 경로와 "
                "index 공저자가 없는 논문을 포함한 경로 모두에서 더 높은 조정 재등장확률을 보였다."
            )
        else:
            validation["protocol_interpretation"] = "PLUS2_CONTINUITY_CONCENTRATED"
            validation["interpretation_ko"] = (
                "정확한 +2 publication cycle에서 관측된 재등장 격차는 주로 index-cycle 공저자 연속 경로에 집중되었다. "
                "index 공저자가 없는 논문을 포함한 경로의 차이는 95% bootstrap 구간에서 뚜렷하지 않았다."
            )
    elif continuity.experienced_minus_no_recent_experienced_difference > 0:
        validation["protocol_interpretation"] = "PLUS2_DIRECTIONALLY_ALIGNED_IMPRECISE"
        validation["interpretation_ko"] = (
            "정확한 +2 publication cycle의 continuity 차이는 기존 결과와 같은 방향이지만 구간이 0을 포함했다."
        )
    else:
        validation["protocol_interpretation"] = "PLUS2_TIMING_SENSITIVE_WARNING"
        validation["interpretation_ko"] = (
            "정확한 +2 publication cycle에서 continuity 점추정치의 부호가 기존 결과와 반대로 나타났다."
        )

    raw.to_csv(TABLES / "raw_exact_plus2_pathway_shares.csv", index=False)
    predictions.to_csv(TABLES / "adjusted_probabilities.csv", index=False)
    contrasts.to_csv(TABLES / "probability_differences.csv", index=False)
    diagnostics.to_csv(TABLES / "model_diagnostics.csv", index=False)
    classification_diagnostics.to_csv(
        TABLES / "classification_diagnostics.csv", index=False
    )
    (TABLES / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(predictions)
    write_report(raw, predictions, contrasts, diagnostics, validation)
    write_reproducibility_files(pathway)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

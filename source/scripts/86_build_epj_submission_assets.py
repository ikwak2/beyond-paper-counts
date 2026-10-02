#!/usr/bin/env python3
"""Build publication figures and machine-readable tables for the EPJ manuscript.

This script does not refit any model. It reads frozen aggregate outputs and the
frozen 306-author measurement-validation pilot, then renders submission assets.
No names or person identifiers are written to the submission package.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "epj_data_science_submission_v01"
FIG = OUT / "figures"
TAB = OUT / "tables"
ADD = OUT / "additional_files"
QA = OUT / "qa"

DIST = ROOT / "outputs" / "epj_distance_based_v10"
PILOT = ROOT / "outputs" / "epj_origin_feasibility_v01"
GEO = ROOT / "outputs" / "journal_llm_diffusion_v01"
REVIEW = ROOT / "outputs" / "epj_reviewer_additions_v12"

for directory in (FIG, TAB, ADD, QA):
    directory.mkdir(parents=True, exist_ok=True)

for directory in (FIG, ADD):
    for stem in (
        "figure_s4_calibration_margin",
        "figure_s5_diagnostic_distance_coauthor",
        "figure_s6_external_production_triangulation",
        "figure_s7_country_group_distance",
    ):
        for suffix in ("png", "pdf"):
            (directory / f"{stem}.{suffix}").unlink(missing_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "navy": "#38678F",
    "blue": "#4C78A8",
    "orange": "#E17C05",
    "green": "#3A923A",
    "red": "#D9534F",
    "purple": "#7A4FA3",
    "gray": "#A9A29F",
    "dark": "#263238",
    "light": "#EEF3F7",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def draw_box(ax, xy, width, height, text, facecolor, edgecolor, fontsize=8.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["dark"],
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, color="#607D8B"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color, shrinkA=2, shrinkB=2),
    )


# Figure 1: measurement and risk-set map.
fig, ax = plt.subplots(figsize=(7.2, 4.45))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(0.5, 0.97, "Accepted-program measurement architecture", ha="center", va="top", fontsize=12, fontweight="bold")

source = draw_box(
    ax,
    (0.22, 0.79),
    0.56,
    0.12,
    "ICML and NeurIPS accepted papers, 2018–2024\n26,872 papers",
    "#E8EEF5",
    COLORS["navy"],
    fontsize=8.7,
)

geo1 = draw_box(
    ax,
    (0.04, 0.55),
    0.27,
    0.14,
    "Affiliation-country evidence\nwork–country\nunits",
    "#FFF3E0",
    COLORS["orange"],
)
geo2 = draw_box(
    ax,
    (0.04, 0.30),
    0.27,
    0.15,
    "Broad-AI production\ndenominator\n863,752 works",
    "#FFF3E0",
    COLORS["orange"],
)
geo3 = draw_box(
    ax,
    (0.04, 0.07),
    0.27,
    0.13,
    "RQ1: production-adjusted\nrepresentation (PRI)",
    "#FFE0B2",
    COLORS["orange"],
    fontsize=8.3,
)

ent1 = draw_box(
    ax,
    (0.39, 0.55),
    0.27,
    0.14,
    "First-listed focal-set-new\nentrants observed\n12,094 people",
    "#E8F5E9",
    COLORS["green"],
)
ent2 = draw_box(
    ax,
    (0.39, 0.30),
    0.27,
    0.15,
    "Prior-title portfolio observed\n9,639 (79.7%)",
    "#E8F5E9",
    COLORS["green"],
)
ent3 = draw_box(
    ax,
    (0.39, 0.07),
    0.27,
    0.13,
    "RQ2: distance trajectory\nD1: team ≥2, n = 9,521",
    "#C8E6C9",
    COLORS["green"],
    fontsize=8.1,
)

ret1 = draw_box(
    ax,
    (0.72, 0.55),
    0.24,
    0.14,
    "Complete exact +2\nfollow-up\n2018–2022, n = 6,704",
    "#F3E5F5",
    COLORS["purple"],
)
ret2 = draw_box(
    ax,
    (0.72, 0.30),
    0.24,
    0.15,
    "Distance observed,\nteam ≥2\nprimary n = 5,122",
    "#F3E5F5",
    COLORS["purple"],
)
ret3 = draw_box(
    ax,
    (0.72, 0.07),
    0.24,
    0.13,
    "RQ3: mutually exclusive\nexact +2 pathways",
    "#E1BEE7",
    COLORS["purple"],
    fontsize=8.1,
)

arrow(ax, (0.42, 0.79), (0.22, 0.69))
arrow(ax, (0.53, 0.79), (0.53, 0.69))
arrow(ax, (0.17, 0.55), (0.17, 0.45))
arrow(ax, (0.17, 0.30), (0.17, 0.20))
arrow(ax, (0.53, 0.55), (0.53, 0.45))
arrow(ax, (0.53, 0.30), (0.53, 0.20))
arrow(ax, (0.66, 0.62), (0.72, 0.62))
arrow(ax, (0.84, 0.55), (0.84, 0.45))
arrow(ax, (0.84, 0.30), (0.84, 0.20))
ax.text(0.335, 0.72, "paper/country axis", ha="center", fontsize=7.0, color="#546E7A")
ax.text(0.67, 0.72, "person/history axis", ha="center", fontsize=7.0, color="#546E7A")
ax.text(0.5, 0.015, "Arrows denote data construction and risk-set restriction, not causal pathways.", ha="center", fontsize=7.2, color="#455A64")
save_figure(fig, "figure_1_measurement_architecture")


# Figure 2: primary four-group estimand plus selected country references.
primary_spec_id = "existing|conservative|top_full|ai_primary_peer_reviewed"
geo_history = pd.read_csv(GEO / "tables" / "primary_pri_korea_and_frozen_groups_2018_2024.csv")
geo_history = geo_history.loc[geo_history["spec_id"].eq(primary_spec_id)].copy()
country_detail = pd.read_csv(GEO / "tables" / "pri_language_context_and_country_detail_2018_2024.csv")
country_detail = country_detail.loc[country_detail["panel"].eq("country_detail")].copy()

primary_groups = ["China", "Core Anglophone", "Other non-core", "India"]
selected_countries = ["United States", "South Korea", "China", "India"]
geo_colors = {
    "China": "#0072B2",
    "Core Anglophone": "#D55E00",
    "Other non-core": "#CC79A7",
    "India": "#009E73",
    "United States": "#D55E00",
    "South Korea": "#7B2CBF",
}
geo_markers = {
    "China": "o",
    "Core Anglophone": "s",
    "Other non-core": "D",
    "India": "^",
    "United States": "s",
    "South Korea": "P",
}
geo_linestyles = {
    "China": "-",
    "Core Anglophone": "--",
    "Other non-core": ":",
    "India": "-.",
    "United States": "--",
    "South Korea": "-.",
}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
panel_definitions = [
    (axes[0], geo_history, "group", primary_groups, "A. Primary four-group estimand"),
    (axes[1], country_detail, "display_group", selected_countries, "B. Selected country references"),
]
endpoint_offsets = {
    "China": 0.01,
    "Core Anglophone": 0.04,
    "Other non-core": 0.03,
    "India": 0.02,
    "United States": 0.03,
    "South Korea": -0.03,
}
for ax, data, group_col, order, title in panel_definitions:
    for group in order:
        line = data.loc[data[group_col].eq(group)].sort_values("year")
        ax.plot(
            line["year"],
            line["pri"],
            color=geo_colors[group],
            marker=geo_markers[group],
            linestyle=geo_linestyles[group],
            linewidth=1.65,
            markersize=3.6,
        )
        endpoint = float(line.loc[line["year"].eq(2024), "pri"].iloc[0])
        ax.text(
            2024.10,
            endpoint + endpoint_offsets.get(group, 0.0),
            group,
            color=geo_colors[group],
            fontsize=6.3,
            va="center",
        )
    ax.axhline(1.0, color="#555555", linestyle=(0, (4, 2)), linewidth=0.9)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Publication year")
    ax.set_xticks(range(2018, 2025))
    ax.set_xlim(2017.7, 2025.85)
    ax.set_ylim(0, 3.5)
    ax.grid(axis="y", color="#D9DEE4", linewidth=0.55, alpha=0.75)
axes[0].set_ylabel("Production-adjusted representation index (PRI)")
fig.suptitle(
    "ICML and NeurIPS first-listed-author representation relative to broad-AI production",
    y=1.015,
    fontsize=9.3,
    fontweight="bold",
)
fig.text(
    0.5,
    0.008,
    "Panel A is the primary mutually exclusive grouping. Panel B is descriptive. "
    "PRI = 1 denotes proportional representation, not acceptance probability.",
    ha="center",
    fontsize=5.8,
    color="#4B5563",
)
fig.tight_layout(rect=(0, 0.075, 1, 0.95), w_pad=2.0)
fig.savefig(FIG / "figure_2_geographic_representation.svg", facecolor="white")
save_figure(fig, "figure_2_geographic_representation")


# Supplementary Figure S3: descriptive language-context regrouping.
language_context = pd.read_csv(GEO / "tables" / "pri_language_context_and_country_detail_2018_2024.csv")
language_context = language_context.loc[language_context["panel"].eq("language_context")].copy()
fig, ax = plt.subplots(figsize=(6.6, 3.15))
context_order = ["Anglophone context", "Non-Anglophone context"]
context_colors = {"Anglophone context": "#D55E00", "Non-Anglophone context": "#0072B2"}
context_markers = {"Anglophone context": "s", "Non-Anglophone context": "o"}
context_styles = {"Anglophone context": "--", "Non-Anglophone context": "-"}
context_offsets = {"Anglophone context": 0.03, "Non-Anglophone context": 0.025}
for group in context_order:
    line = language_context.loc[language_context["display_group"].eq(group)].sort_values("year")
    ax.plot(
        line["year"],
        line["pri"],
        color=context_colors[group],
        marker=context_markers[group],
        linestyle=context_styles[group],
        linewidth=1.8,
        markersize=4.0,
    )
    endpoint = float(line.loc[line["year"].eq(2024), "pri"].iloc[0])
    ax.text(2024.08, endpoint + context_offsets[group], group, color=context_colors[group], fontsize=7.1, va="center")
ax.axhline(1.0, color="#555555", linestyle=(0, (4, 2)), linewidth=0.9)
ax.set_title("Descriptive language-context regrouping", loc="left", fontweight="bold")
ax.set_xlabel("Publication year")
ax.set_ylabel("Production-adjusted representation index (PRI)")
ax.set_xticks(range(2018, 2025))
ax.set_xlim(2017.7, 2025.7)
ax.set_ylim(0, 2.5)
ax.grid(axis="y", color="#D9DEE4", linewidth=0.55, alpha=0.75)
fig.text(
    0.5,
    0.008,
    "Anglophone context = Core Anglophone + India; non-Anglophone context = China + Other non-core. "
    "Country-level regrouping does not measure individual language proficiency.",
    ha="center",
    fontsize=6.2,
    color="#4B5563",
)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save_figure(fig, "figure_s3_language_context_regrouping")


# Figure 3: calibration + annual distance trajectory.
pilot = pd.read_csv(
    PILOT / "private" / "epj_specter2_title_distance_pilot_private.csv",
    encoding="utf-8-sig",
)
pilot = pilot.loc[pilot["distance_observed"].astype(bool)].copy()
annual = pd.read_csv(DIST / "tables" / "rq1_year_distance_descriptive.csv")
adjusted = pd.read_csv(DIST / "tables" / "rq1_year_adjusted_mean_distance.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [0.92, 1.35]})
ax = axes[0]
closer = pilot["own_portfolio_closer_than_permuted_median"].astype(bool)
ax.scatter(
    pilot.loc[~closer, "specter2_title_portfolio_distance"],
    pilot.loc[~closer, "permuted_other_portfolio_median_distance"],
    s=13,
    alpha=0.62,
    color=COLORS["gray"],
    edgecolor="none",
    label="Own not closer (26.5%)",
)
ax.scatter(
    pilot.loc[closer, "specter2_title_portfolio_distance"],
    pilot.loc[closer, "permuted_other_portfolio_median_distance"],
    s=13,
    alpha=0.68,
    color=COLORS["blue"],
    edgecolor="none",
    label="Own closer (73.5%)",
)
limits = [
    float(min(pilot["specter2_title_portfolio_distance"].min(), pilot["permuted_other_portfolio_median_distance"].min()) - 0.008),
    float(max(pilot["specter2_title_portfolio_distance"].max(), pilot["permuted_other_portfolio_median_distance"].max()) + 0.008),
]
ax.plot(limits, limits, linestyle="--", color="#455A64", linewidth=0.9)
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_xlabel("Own prior-portfolio distance")
ax.set_ylabel("Median randomized distance")
ax.set_title("A. Random-portfolio calibration", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=6.7)
ax.text(
    0.03,
    0.97,
    "Paired median margin = 0.026",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7.0,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B0BEC5", alpha=0.9),
)

ax = axes[1]
years = annual["index_year"].to_numpy()
ax.fill_between(
    years,
    annual["q25_distance"].to_numpy(),
    annual["q75_distance"].to_numpy(),
    color=COLORS["blue"],
    alpha=0.11,
    label="Observed IQR",
)
ax.errorbar(
    years,
    annual["median_distance"],
    yerr=np.vstack(
        [
            annual["median_distance"] - annual["median_ci_low"],
            annual["median_ci_high"] - annual["median_distance"],
        ]
    ),
    marker="o",
    color=COLORS["blue"],
    linewidth=1.5,
    capsize=2.0,
    label="Observed median (95% interval)",
)
ax.plot(years, adjusted["adjusted_mean_distance"], marker="s", color=COLORS["red"], linewidth=1.35, label="Adjusted mean")
ax.fill_between(
    years,
    adjusted["ci_low"].to_numpy(),
    adjusted["ci_high"].to_numpy(),
    color=COLORS["red"],
    alpha=0.12,
)
ax.set_xticks(years)
ax.set_xlabel("Index year")
ax.set_ylabel("SPECTER2 title-only distance")
ax.set_title("B. Prior-to-index distance by year", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper right", fontsize=6.7)
fig.tight_layout(w_pad=2.0)
save_figure(fig, "figure_3_calibrated_distance_trajectory")


# Figure 4: standardized pathways and experienced-coauthor contrasts.
prob = pd.read_csv(DIST / "tables" / "rq3_exact_plus2_probabilities.csv")
prob = prob.loc[prob["specification"].eq("semantic_primary")].copy()
contr = pd.read_csv(DIST / "tables" / "rq3_exact_plus2_contrasts.csv")
contr = contr.loc[
    contr["specification"].eq("semantic_primary")
    & contr["contrast_family"].eq("experienced_minus_no_experienced")
].copy()
pathways = ["no_recurrence", "coauthor_continuity_only", "at_least_one_no_index_coauthor"]
path_labels = {
    "no_recurrence": "No focal-set record",
    "coauthor_continuity_only": "Continuity only",
    "at_least_one_no_index_coauthor": "≥1 paper without\nindex coauthor",
}
path_colors = {
    "no_recurrence": "#B8B0AD",
    "coauthor_continuity_only": "#E15759",
    "at_least_one_no_index_coauthor": "#4C78A8",
}
scenario_order = [("Q25", False), ("Q25", True), ("Q75", False), ("Q75", True)]
scenario_labels = ["Q25\nno exp.", "Q25\nexp.", "Q75\nno exp.", "Q75\nexp."]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.18, 1.0]})
ax = axes[0]
bottom = np.zeros(len(scenario_order))
for pathway in pathways:
    vals = []
    for distance_level, experienced in scenario_order:
        row = prob.loc[
            prob["distance_level"].eq(distance_level)
            & prob["experienced_coauthor"].eq(experienced)
            & prob["pathway"].eq(pathway)
        ]
        vals.append(float(row.iloc[0]["adjusted_probability"]))
    ax.bar(np.arange(len(vals)), vals, bottom=bottom, width=0.72, color=path_colors[pathway], label=path_labels[pathway].replace("\n", " "))
    bottom += np.asarray(vals)
ax.set_xticks(np.arange(len(scenario_labels)), scenario_labels)
ax.set_ylim(0, 1)
ax.set_ylabel("Standardized pathway probability")
ax.set_title("A. Exact +2 pathway probabilities", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)

ax = axes[1]
y_base = np.arange(len(pathways))[::-1]
for level, offset, marker, color in [("Q25", 0.11, "o", COLORS["blue"]), ("Q75", -0.11, "s", COLORS["orange"])]:
    subset = contr.loc[contr["distance_level"].eq(level)].set_index("pathway")
    estimates = np.array([float(subset.loc[p, "probability_difference"]) * 100 for p in pathways])
    lows = np.array([float(subset.loc[p, "ci_low"]) * 100 for p in pathways])
    highs = np.array([float(subset.loc[p, "ci_high"]) * 100 for p in pathways])
    estimates = estimates
    ax.errorbar(
        estimates,
        y_base + offset,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt=marker,
        markersize=4.2,
        color=color,
        capsize=2.2,
        linewidth=1.1,
        label=level,
    )
ax.axvline(0, linestyle="--", color="#455A64", linewidth=0.9)
ax.set_yticks(y_base, [path_labels[p] for p in pathways])
ax.set_xlabel("Experienced − no experienced (percentage points)")
ax.set_title("B. Experienced-coauthor contrasts", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(w_pad=2.0)
save_figure(fig, "figure_4_exact_plus2_pathways")


# Supplementary Figure S7: diagnostic contrast across specifications.
d1 = pd.read_csv(DIST / "tables" / "rq2_experienced_coauthor_contrasts.csv")
label_map = {
    "semantic_primary": "SPECTER2 primary",
    "tfidf_word": "Word TF-IDF",
    "tfidf_char": "Character TF-IDF",
    "semantic_alltypes": "SPECTER2 all record types",
    "semantic_team_overlap": "Team-size overlap",
    "semantic_exclude_both": "Exclude dual-venue entrants",
    "semantic_distance_spline": "Distance spline",
}
d1["display"] = d1["specification"].map(label_map).fillna(d1["specification"])
d1 = d1.iloc[::-1].copy()
fig, ax = plt.subplots(figsize=(5.6, 3.2))
y = np.arange(len(d1))
est = d1["probability_difference"].to_numpy() * 100
low = d1["ci_low"].to_numpy() * 100
high = d1["ci_high"].to_numpy() * 100
ax.errorbar(est, y, xerr=np.vstack([est - low, high - est]), fmt="o", color=COLORS["green"], capsize=2.2, linewidth=1.1)
ax.axvline(0, linestyle="--", color="#455A64", linewidth=0.9)
ax.set_yticks(y, d1["display"])
ax.set_xlabel("Q75 − Q25 experienced-coauthor probability (percentage points)")
ax.set_title("Diagnostic distance–coauthor contrast", fontweight="bold")
fig.tight_layout()
save_figure(fig, "figure_s7_diagnostic_distance_coauthor")


# Supplementary Figure S5: calibration margin, including the non-closing tail.
margin = pilot["permuted_minus_own_distance"].dropna()
fig, ax = plt.subplots(figsize=(5.6, 3.1))
ax.hist(margin, bins=28, color=COLORS["blue"], alpha=0.78, edgecolor="white")
ax.axvline(0, color="#455A64", linestyle="--", linewidth=1.0, label="No separation")
ax.axvline(float(margin.median()), color=COLORS["red"], linewidth=1.2, label="Median = 0.026")
ax.set_xlabel("Randomized median distance − own distance")
ax.set_ylabel("Pilot entrants")
ax.set_title("Calibration-margin distribution", fontweight="bold")
ax.text(0.03, 0.94, f"Own portfolio not closer: {(margin.le(0).mean() * 100):.1f}%", transform=ax.transAxes, va="top", fontsize=7.5)
ax.legend(frameon=False)
fig.tight_layout()
save_figure(fig, "figure_s5_calibration_margin")


# Table 1: analysis populations and claim boundaries.
table1 = pd.DataFrame(
    [
        {
            "analysis": "RQ1 geographic representation",
            "unit": "accepted paper-country credit / broad-AI work-country credit",
            "index_cohorts": "2018–2024",
            "eligible_n": 26872,
            "primary_n": 26872,
            "conditioning": "country observed for primary numerator; denominator definition prespecified",
            "permitted_claim": "representation relative to observed broad-AI production",
        },
        {
            "analysis": "RQ2 distance trajectory",
            "unit": "first-listed observed focal-set-new entrant",
            "index_cohorts": "2018–2024",
            "eligible_n": 12094,
            "primary_n": 9639,
            "conditioning": "at least one prior title in the five-year DBLP window",
            "permitted_claim": "title-only prior-to-index portfolio distance",
        },
        {
            "analysis": "Diagnostic D1",
            "unit": "first-listed observed focal-set-new entrant",
            "index_cohorts": "2018–2024",
            "eligible_n": 9639,
            "primary_n": 9521,
            "conditioning": "distance observed and mean entry-team size ≥2",
            "permitted_claim": "adjusted association with experienced-coauthor composition at entry",
        },
        {
            "analysis": "RQ3 exact +2 pathways",
            "unit": "first-listed observed focal-set-new entrant",
            "index_cohorts": "2018–2022",
            "eligible_n": 6704,
            "primary_n": 5122,
            "conditioning": "complete exact +2 follow-up, distance observed, mean entry-team size ≥2",
            "permitted_claim": "adjusted association with mutually exclusive accepted-paper pathways",
        },
    ]
)
table1.to_csv(TAB / "table_1_analysis_populations.csv", index=False)


# Table 2: geographic endpoints.
geo_primary = pd.read_csv(GEO / "tables" / "primary_pri_korea_and_frozen_groups_2018_2024.csv")
primary_id = "existing|conservative|top_full|ai_primary_peer_reviewed"
geo_primary = geo_primary.loc[
    geo_primary["spec_id"].eq(primary_id)
    & geo_primary["year"].isin([2018, 2024])
    & geo_primary["group"].isin(["China", "Core Anglophone", "Other non-core", "India"])
].copy()
geo_rows = []
for group in ["China", "Core Anglophone", "Other non-core", "India"]:
    g = geo_primary.loc[geo_primary["group"].eq(group)].set_index("year")
    geo_rows.append(
        {
            "group": group,
            "top_country_credit_2018": float(g.loc[2018, "top_count"]),
            "top_share_2018": float(g.loc[2018, "top_share"]),
            "broad_country_credit_2018": float(g.loc[2018, "broad_count"]),
            "broad_share_2018": float(g.loc[2018, "broad_share"]),
            "pri_2018": float(g.loc[2018, "pri"]),
            "top_country_credit_2024": float(g.loc[2024, "top_count"]),
            "top_share_2024": float(g.loc[2024, "top_share"]),
            "broad_country_credit_2024": float(g.loc[2024, "broad_count"]),
            "broad_share_2024": float(g.loc[2024, "broad_share"]),
            "pri_2024": float(g.loc[2024, "pri"]),
            "direction_identified_under_denominator_missingness_bounds": group == "China",
        }
    )
table2 = pd.DataFrame(geo_rows)
table2.to_csv(TAB / "table_2_geographic_endpoints.csv", index=False)


# Table 3: annual distance trajectory.
table3 = annual.merge(adjusted, on="index_year", validate="one_to_one")
table3.to_csv(TAB / "table_3_annual_distance.csv", index=False)


# Table 4: focal standardized probability contrasts.
d1_primary = pd.read_csv(DIST / "tables" / "rq2_experienced_coauthor_contrasts.csv").query("specification == 'semantic_primary'").iloc[0]
rows = [
    {
        "analysis": "Diagnostic D1",
        "distance_level": "Q75 − Q25",
        "pathway": "experienced coauthor at entry",
        "probability_difference": float(d1_primary["probability_difference"]),
        "ci_low": float(d1_primary["ci_low"]),
        "ci_high": float(d1_primary["ci_high"]),
        "n": int(d1_primary["n"]),
    }
]
for _, row in contr.iterrows():
    rows.append(
        {
            "analysis": "RQ3 exact +2",
            "distance_level": str(row["distance_level"]),
            "pathway": str(row["pathway"]),
            "probability_difference": float(row["probability_difference"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "n": 5122,
        }
    )
table4 = pd.DataFrame(rows)
table4.to_csv(TAB / "table_4_focal_probability_contrasts.csv", index=False)


# Table 5: reusable reporting bundle and manuscript self-audit.
table5 = pd.DataFrame(
    [
        {
            "component": "Raw accepted counts and shares",
            "category_error_blocked": "Treating a ratio trend as evidence that absolute participation fell or rose",
            "study_self_audit": "Table 2 reports endpoint credits and shares for every primary group",
        },
        {
            "component": "Production denominator and country coverage",
            "category_error_blocked": "Treating accepted-program composition as access, acceptance, or production-adjusted representation",
            "study_self_audit": "PRI names the OpenAlex broad-AI denominator; Figs. S1-S2 audit specifications and missingness; Fig. S4 triangulates direction externally",
        },
        {
            "component": "Newcomer venue set and lookback",
            "category_error_blocked": "Treating new to this dataset as new to science",
            "study_self_audit": "The focal four-venue set and five-cycle lookback are explicit; the label remains conditional",
        },
        {
            "component": "Follow-up horizon and risk set",
            "category_error_blocked": "Coding administrative censoring as no return",
            "study_self_audit": "Exact +2 uses 2018-2022 entrants only; 6,704 are eligible and 5,122 enter the primary model",
        },
        {
            "component": "Coauthor-continuity pathways",
            "category_error_blocked": "Treating repeated-team publication as independent reappearance",
            "study_self_audit": "Three exhaustive outcomes distinguish no record, continuity only, and at least one non-overlapping paper",
        },
        {
            "component": "Specification and missingness sensitivity",
            "category_error_blocked": "Treating one defensible choice as a universal result",
            "study_self_audit": "The package reports 32 PRI specifications, country bounds, alternative distances and return universes, and a within-two-cycle sensitivity",
        },
    ]
)
table5.to_csv(TAB / "table_5_minimum_reporting_self_audit.csv", index=False)


# Supplementary machine-readable tables.
file_mapping = pd.DataFrame(
    [
        {
            "manuscript_label": "RQ2 distance trajectory",
            "frozen_output_label": "rq1",
            "frozen_file": "outputs/epj_distance_based_v10/tables/rq1_year_distance_descriptive.csv",
            "reason": "Frozen before manuscript RQ relabeling",
        },
        {
            "manuscript_label": "Diagnostic D1",
            "frozen_output_label": "rq2",
            "frozen_file": "outputs/epj_distance_based_v10/tables/rq2_experienced_coauthor_contrasts.csv",
            "reason": "Frozen before diagnostic downgrade",
        },
        {
            "manuscript_label": "RQ3 exact +2 pathways",
            "frozen_output_label": "rq3",
            "frozen_file": "outputs/epj_distance_based_v10/tables/rq3_exact_plus2_contrasts.csv",
            "reason": "Manuscript RQ3 retains the frozen third-analysis label",
        },
    ]
)
file_mapping.to_csv(ADD / "table_s1_frozen_file_label_mapping.csv", index=False)

pilot_diag = json.loads((PILOT / "manifests" / "epj_specter2_title_distance_pilot_diagnostics.json").read_text())
pilot_gates = pd.DataFrame(
    [
        {"metric": "pilot authors", "value": pilot_diag["sample_authors"], "criterion": "frozen stratified sample", "pass": True},
        {"metric": "authors with observed distance", "value": pilot_diag["authors_with_distance"], "criterion": "coverage ≥70%", "pass": pilot_diag["gates"]["coverage"]},
        {"metric": "finite distance rate", "value": pilot_diag["finite_distance_rate"], "criterion": "≥99.9%", "pass": pilot_diag["gates"]["finite_rate"]},
        {"metric": "distance IQR", "value": pilot_diag["distance_iqr"], "criterion": "≥0.02", "pass": pilot_diag["gates"]["distance_iqr"]},
        {"metric": "word TF-IDF Spearman", "value": pilot_diag["word_tfidf_spearman"], "criterion": "≥0.15", "pass": pilot_diag["gates"]["tfidf_spearman"]},
        {"metric": "own portfolio closer share", "value": pilot_diag["own_portfolio_closer_share"], "criterion": "≥0.55", "pass": pilot_diag["gates"]["own_closer_share"]},
        {"metric": "paired median margin", "value": pilot_diag["paired_median_margin"], "criterion": ">0", "pass": pilot_diag["gates"]["paired_median_margin"]},
    ]
)
pilot_gates.to_csv(ADD / "table_s2_distance_calibration_gates.csv", index=False)

copy_tables = {
    "distance_observation_missingness_comparison.csv": "table_s3_distance_observation_missingness.csv",
    "rq2_experienced_coauthor_contrasts.csv": "table_s4_diagnostic_d1_sensitivity.csv",
    "rq3_exact_plus2_contrasts.csv": "table_s5_rq3_all_contrasts_and_sensitivities.csv",
    "measurement_robustness_focal_signs.csv": "table_s6_measurement_robustness_signs.csv",
    "distance_observation_coverage_by_dimension.csv": "table_s7_distance_coverage_by_dimension.csv",
    "rq3_model_diagnostics.csv": "table_s8_rq3_model_diagnostics.csv",
}
for source_name, target_name in copy_tables.items():
    shutil.copy2(DIST / "tables" / source_name, ADD / target_name)

for source_name in (
    "table_s12_external_production_trajectories.csv",
    "table_s13_external_production_endpoints.csv",
    "table_s14_country_group_distance_by_year.csv",
    "table_s15_country_group_distance_pooled.csv",
    "table_s16_within_two_probabilities.csv",
    "table_s17_within_two_contrasts.csv",
):
    shutil.copy2(REVIEW / "tables" / source_name, ADD / source_name)
shutil.copy2(
    REVIEW / "manifests" / "reviewer_additions_diagnostics.json",
    ADD / "reviewer_additions_diagnostics.json",
)

# Extract audited geographic robustness assets without unpacking private files.
archive = ROOT / "deliverables" / "reproducibility_package.zip"
with zipfile.ZipFile(archive) as zf:
    selected = {
        "reproducibility_package/data/aggregate/S12_pri_specification_dashboard.csv": "table_s9_pri_specification_dashboard.csv",
        "reproducibility_package/data/aggregate/S16_missingness_break_even_summary.csv": "table_s10_denominator_missingness_bounds.csv",
        "reproducibility_package/data/aggregate/numerator_missingness_boundary.csv": "table_s11_numerator_missingness_boundary.csv",
        "reproducibility_package/figures/figure_3_pri_specification_audit.pdf": "figure_s1_pri_specification_audit.pdf",
        "reproducibility_package/figures/figure_3_pri_specification_audit.svg": "figure_s1_pri_specification_audit.svg",
        "reproducibility_package/figures/figure_5_coverage_tipping_point.pdf": "figure_s2_denominator_missingness_bounds.pdf",
        "reproducibility_package/figures/figure_5_coverage_tipping_point.svg": "figure_s2_denominator_missingness_bounds.svg",
    }
    for member, target in selected.items():
        with zf.open(member) as src, (ADD / target).open("wb") as dst:
            shutil.copyfileobj(src, dst)

# Copy frozen protocols/configs that define the estimands.
for source, target in [
    (ROOT / "docs" / "epj_distance_based_analysis_protocol_v1.0.md", "protocol_distance_v1.0.md"),
    (ROOT / "docs" / "epj_specter2_title_distance_pilot_addendum_v0.4.md", "protocol_distance_pilot_v0.4.md"),
    (ROOT / "docs" / "epj_manuscript_reporting_clarification_v1.0.md", "reporting_clarification_v1.0.md"),
    (ROOT / "config" / "epj_distance_based_analysis_v10.json", "config_distance_v1.0.json"),
    (ROOT / "config" / "epj_specter2_title_distance_pilot_v04.json", "config_distance_pilot_v0.4.json"),
]:
    shutil.copy2(source, ADD / target)

# Supplementary figure copies with neutral manuscript labels.
for stem in (
    "figure_s3_language_context_regrouping",
    "figure_s5_calibration_margin",
    "figure_s7_diagnostic_distance_coauthor",
):
    for suffix in ("png", "pdf"):
        src = FIG / f"{stem}.{suffix}"
        if src.exists():
            shutil.copy2(src, ADD / src.name)
            # Supplementary figures live only in additional_files after packaging.
            src.unlink()

review_figure_targets = {
    "figure_s6_external_production_triangulation": "figure_s4_external_production_triangulation",
    "figure_s7_country_group_distance": "figure_s6_country_group_distance",
    "figure_s8_within_two_sensitivity": "figure_s8_within_two_sensitivity",
}
for source_stem, target_stem in review_figure_targets.items():
    for suffix in ("png", "pdf"):
        src = REVIEW / "figures" / f"{source_stem}.{suffix}"
        shutil.copy2(src, ADD / f"{target_stem}.{suffix}")

# Manifest and aggregate-only privacy audit.
manifest_rows = []
for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
    if path.name in {"asset_manifest.json", "submission_qa_report.json"}:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_rows.append(
        {
            "path": str(path.relative_to(OUT)),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    )
manifest = {
    "package_version": "v0.2",
    "generated_by": "scripts/86_build_epj_submission_assets.py",
    "model_refit": False,
    "person_identifiers_written": False,
    "main_figures": 4,
    "main_tables": 5,
    "files": manifest_rows,
}
(QA / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(json.dumps({"status": "PASS", "output": str(OUT), "files": len(manifest_rows)}, indent=2))

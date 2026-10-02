"""Plot published aggregate estimates, without fitting models or accessing people.

Figure 3B and Figure 4 are regenerated from their numerical inputs. The full
frozen Figure 3 is distributed separately: its calibration scatterplot requires
the undistributed pilot rows. Layout is refreshed; estimates are unchanged.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/distance"
OUT = ROOT / "outputs"
PATHWAYS = ("no_recurrence", "coauthor_continuity_only", "at_least_one_no_index_coauthor")
LABELS = ("No record", "Entry-coauthor papers only", "≥1 paper without entry coauthors")
COLORS = ("#B8B0AD", "#E15759", "#4C78A8")


def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    annual = pd.read_csv(DATA / "rq1_year_distance_descriptive.csv").sort_values("index_year")
    adjusted = pd.read_csv(DATA / "rq1_year_adjusted_mean_distance.csv").sort_values("index_year")
    fig, ax = plt.subplots(figsize=(6, 3.7), layout="constrained")
    ax.fill_between(annual.index_year, annual.q25_distance, annual.q75_distance,
                    color="#4C78A8", alpha=.12, label="Q1–Q3 band")
    ax.errorbar(annual.index_year, annual.median_distance,
                yerr=[annual.median_distance - annual.median_ci_low,
                      annual.median_ci_high - annual.median_distance],
                marker="o", capsize=3, color="#4C78A8", label="Observed median (95% interval)")
    ax.plot(adjusted.index_year, adjusted.adjusted_mean_distance, marker="s",
            color="#D9534F", label="Adjusted mean")
    ax.fill_between(adjusted.index_year, adjusted.ci_low, adjusted.ci_high,
                    color="#D9534F", alpha=.12)
    ax.set(xlabel="Entry year", ylabel="Title-portfolio distance",
           title="Prior-to-entry title-portfolio distance by year")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "figure3b_distance_trajectory")

    prob = pd.read_csv(DATA / "rq3_exact_plus2_probabilities.csv").query("specification == 'semantic_primary'")
    con = pd.read_csv(DATA / "rq3_exact_plus2_contrasts.csv").query(
        "specification == 'semantic_primary' and contrast_family == 'experienced_minus_no_experienced'")
    scenarios = [(q, e) for q in ("Q25", "Q75") for e in (False, True)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), gridspec_kw={"width_ratios": [1, 1.1]})
    bottom = np.zeros(4)
    for pathway, label, color in zip(PATHWAYS, LABELS, COLORS):
        vals = np.array([prob.loc[prob.distance_level.eq(q) & prob.experienced_coauthor.eq(e)
                                   & prob.pathway.eq(pathway), "adjusted_probability"].item()
                         for q, e in scenarios])
        axes[0].bar(np.arange(4), vals, bottom=bottom, color=color, label=label, width=.72)
        bottom += vals
    np.testing.assert_allclose(bottom, 1, atol=1e-10)
    axes[0].set_xticks(range(4), [f"{q}\n{'With' if e else 'No'} recent\nfocal-set\ncoauthor" for q, e in scenarios], fontsize=8)
    axes[0].set(ylim=(0, 1), ylabel="Standardized pathway probability",
                title="A. Exact +2 pathway probabilities")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.25))
    for q, shift, color, marker in [("Q25", .10, "#4C78A8", "o"), ("Q75", -.10, "#E17C05", "s")]:
        d = con.loc[con.distance_level.eq(q)].set_index("pathway").loc[list(PATHWAYS)]
        est = 100 * d.probability_difference.to_numpy()
        axes[1].errorbar(est, np.arange(3)[::-1] + shift,
                        xerr=np.vstack([est - 100*d.ci_low, 100*d.ci_high - est]),
                        fmt=marker, color=color, capsize=3, label=q)
    axes[1].axvline(0, color="#455A64", ls="--", lw=.8)
    axes[1].set_yticks(np.arange(3)[::-1], ["No record", "Entry-coauthor\npapers only", "≥1 paper without\nentry coauthors"])
    axes[1].set(xlabel="With − no recent focal-set coauthor\n(percentage points)",
                title="B. Recent-focal-set-coauthor contrasts")
    axes[1].legend(frameon=False)
    fig.subplots_adjust(left=.07, right=.98, bottom=.30, top=.90, wspace=.65)
    save(fig, "fig4")


if __name__ == "__main__":
    main()

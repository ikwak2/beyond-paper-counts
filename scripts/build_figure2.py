"""Redraw Figure 2 for v06 from frozen country credits and frozen audit tables.

The figure reads aggregate -> decomposition -> sensitivity:

  Panel A  Core Anglophone vs all other countries, two mutually exclusive
           aggregates whose PRI is recomputed from summed credits.
  Panel B  the same "all other countries" aggregate decomposed into China,
           India, and other countries, showing that it is not homogeneous.
  Panel C  the 2024-minus-2018 endpoint change under the primary
           specification, the 44 observed specifications, and the
           denominator-missingness bound.

No country is treated as a prespecified case of interest; the decomposition is
what makes the individual series appear. No model is refitted and no frozen
estimate is recomputed: every number is read from frozen outputs, and group
aggregates are rebuilt with the same country-credit rule before any ratio.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import fitz
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
YEAR = tuple(range(2018, 2025))
PRIMARY = 'existing|conservative|top_full|ai_primary_peer_reviewed'
HISTORY = ROOT / 'outputs/pri_annual.csv'
COUNTRY = ROOT / 'data/aggregate/accepted_country_credits.csv'
CAPACITY = ROOT / 'outputs/openalex_capacity.csv'
HELPER = ROOT / 'source/scripts/67_journal_korea_pri_v01.py'
SUPP = ROOT / 'results/additional_files'
DASHBOARD = SUPP / 'table_s9_pri_specification_dashboard.csv'
BOUNDS = SUPP / 'table_s10_denominator_missingness_bounds.csv'

# Frozen group keys, and the display names used in the figure.
GROUPS = ['Core Anglophone', 'China', 'India', 'Other non-core']
DISPLAY = {'Core Anglophone': 'Core Anglophone', 'China': 'China',
           'India': 'India', 'Other non-core': 'Other countries'}
# Panel A aggregates: mutually exclusive and exhaustive over the four groups.
AGGREGATE = {'Core Anglophone': ['Core Anglophone'],
             'All other countries': ['China', 'India', 'Other non-core']}
# Supplementary descriptive displays only.
BLOCS = {'Anglophone context': ['Core Anglophone', 'India'],
         'Non-Anglophone context': ['China', 'Other non-core']}
COUNTRIES = {'United States': 'US', 'South Korea': 'KR', 'China': 'CN', 'India': 'IN'}
S3_STEM = 'figure_s3_language_context_and_country_references'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(panel, year, label, a, b, A, B, definition):
    return dict(panel=panel, year=year, display_group=label,
                accepted_country_credit=a, accepted_country_credit_total=A,
                production_country_credit=b, production_country_credit_total=B,
                accepted_share=a / A, production_share=b / B,
                pri=(a / A) / (b / B), component_definition=definition)


def save(fig, stem):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300, facecolor='white',
                    metadata={'Creator': 'EPJDS v06 aggregate-then-decomposition figure revision'} if ext == 'pdf' else None)
    if stem == 'fig2':
        original = fitz.open(OUT/'fig2.pdf'); page = original[0]
        rectangles = [fitz.Rect(b[:4]) for b in page.get_text('blocks')]
        rectangles += [x['rect'] for x in page.get_drawings()
                       if not (x['rect'].width > page.rect.width-1 and x['rect'].height > page.rect.height-1)]
        bottom = max(r.y1 for r in rectangles)+5
        cropped = fitz.open(); new = cropped.new_page(width=page.rect.width,height=bottom)
        new.show_pdf_page(new.rect,original,0,clip=fitz.Rect(0,0,page.rect.width,bottom))
        cropped.save(OUT/'fig2_crop.pdf',garbage=4,deflate=True)
        original.close(); cropped.close(); (OUT/'fig2_crop.pdf').replace(OUT/'fig2.pdf')
        with fitz.open(OUT/'fig2.pdf') as d:
            d[0].get_pixmap(matrix=fitz.Matrix(300/72,300/72)).save(OUT/'fig2.png')


def assert_within_canvas(fig, axes, legends):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    elements = []
    for ax in axes:
        elements.extend([ax.title, ax._left_title, ax._right_title, ax.xaxis.label,
                         ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()])
    elements.extend(l for l in legends if l is not None)
    elements.extend(fig.texts)
    for item in elements:
        if hasattr(item, 'get_text') and not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), str(item)


def assert_no_overlap(fig, items):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(getattr(i, 'get_text', lambda: str(i))(), i.get_window_extent(renderer)) for i in items]
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            assert not boxes[a][1].overlaps(boxes[b][1]), f'{boxes[a][0]!r} overlaps {boxes[b][0]!r}'


def build_credits():
    """Rebuild group and aggregate PRI from the frozen country credits."""
    g = pd.read_csv(HISTORY)
    g = g[g.spec_id.eq(PRIMARY) & g.year.isin(YEAR) & g.group.isin(GROUPS)].copy()
    assert len(g) == 28 and not g.duplicated(['year', 'group']).any()
    a = pd.read_csv(COUNTRY, keep_default_na=False)
    a = a[a.country_mapping.eq('conservative') & a.year.isin(YEAR)]
    b = pd.read_csv(CAPACITY, keep_default_na=False)
    b = b[b.specification.eq('ai_primary_peer_reviewed') & b.year.isin(YEAR)]
    agg_rows, four_rows, supp_rows = [], [], []
    for year in YEAR:
        old = g[g.year.eq(year)].set_index('group')
        A = float(a.loc[a.year.eq(year), 'full'].sum())
        B = float(b.loc[b.year.eq(year), 'country_work_count_full'].sum())
        assert A == old.top_credit_total.iloc[0] and B == old.broad_country_credit_total.iloc[0]
        assert np.isclose(old.top_count.sum(), A) and np.isclose(old.broad_count.sum(), B)
        for label, parts in AGGREGATE.items():
            agg_rows.append(row('A', year, label, float(old.loc[parts, 'top_count'].sum()),
                                float(old.loc[parts, 'broad_count'].sum()), A, B, ' + '.join(parts)))
        for label in GROUPS:
            r = old.loc[label]
            four_rows.append(row('B', year, DISPLAY[label], float(r.top_count),
                                 float(r.broad_count), A, B, label))
        for label, parts in BLOCS.items():
            supp_rows.append(row('S3-A', year, label, float(old.loc[parts, 'top_count'].sum()),
                                 float(old.loc[parts, 'broad_count'].sum()), A, B, ' + '.join(parts)))
        for label, code in COUNTRIES.items():
            num = float(a.loc[a.year.eq(year) & a.country_code.eq(code), 'full'].sum())
            den = float(b.loc[b.year.eq(year) & b.country_code.eq(code), 'country_work_count_full'].sum())
            assert num > 0 and den > 0
            supp_rows.append(row('S3-B', year, label, num, den, A, B, code))
    agg = pd.DataFrame(agg_rows)
    four = pd.DataFrame(four_rows)

    # The four-group panel must still reproduce the frozen primary PRI exactly.
    frozen = g[['year', 'group', 'pri']].assign(display_group=lambda d: d.group.map(DISPLAY))
    check = four.merge(frozen[['year', 'display_group', 'pri']], on=['year', 'display_group'],
                       suffixes=('', '_frozen'), validate='one_to_one')
    assert len(check) == 28
    assert np.allclose(check.pri, check.pri_frozen, atol=5e-10, rtol=0)

    # The two aggregates must partition each year's credits exactly once.
    for year in YEAR:
        ay = agg[agg.year.eq(year)]
        fy = four[four.year.eq(year)]
        assert np.isclose(ay.accepted_country_credit.sum(), fy.accepted_country_credit.sum())
        assert np.isclose(ay.production_country_credit.sum(), fy.production_country_credit.sum())
        assert np.isclose(ay.accepted_share.sum(), 1.0) and np.isclose(ay.production_share.sum(), 1.0)
        # "All other countries" is the complement of core Anglophone, not a mean of PRIs.
        other = ay[ay.display_group.eq('All other countries')].iloc[0]
        anglo = ay[ay.display_group.eq('Core Anglophone')].iloc[0]
        assert np.isclose(other.pri, (1 - anglo.accepted_share) / (1 - anglo.production_share))
        parts = fy[fy.display_group.isin(['China', 'India', 'Other countries'])]
        assert not np.isclose(other.pri, parts.pri.mean())
    return agg, four, pd.DataFrame(supp_rows)


def build_sensitivity():
    s9 = pd.read_csv(DASHBOARD)
    assert len(s9) == 176 and s9.spec_id.nunique() == 44
    s10 = pd.read_csv(BOUNDS)
    rows = []
    for label in GROUPS:
        obs = s9[s9.group.eq(label)]
        primary = obs[obs.spec_id.eq(PRIMARY)].change_2018_2024
        assert len(primary) == 1 and obs.highlighted_specification.sum() == 2
        bd = s10[s10.group.eq(label)].set_index('numerator_counting')
        assert set(bd.index) == {'full', 'fractional'}
        ident = {k: bool(bd.loc[k, 'classification'] == 'ASSUMPTION_FREE_SIGN_IDENTIFIED')
                 for k in ('full', 'fractional')}
        rows.append(dict(
            group=DISPLAY[label],
            frozen_group_key=label,
            primary_change=float(primary.iloc[0]),
            observed_spec_n=int(len(obs)),
            observed_spec_low=float(obs.change_2018_2024.min()),
            observed_spec_high=float(obs.change_2018_2024.max()),
            observed_sign_agreement=float(obs.direction.eq(obs.direction.iloc[0]).mean()),
            bound_low_full=float(bd.loc['full', 'assumption_free_change_low']),
            bound_high_full=float(bd.loc['full', 'assumption_free_change_high']),
            bound_low_fractional=float(bd.loc['fractional', 'assumption_free_change_low']),
            bound_high_fractional=float(bd.loc['fractional', 'assumption_free_change_high']),
            bound_low=float(bd.assumption_free_change_low.min()),
            bound_high=float(bd.assumption_free_change_high.max()),
            bound_sign_identified_full=ident['full'],
            bound_sign_identified_fractional=ident['fractional']))
    s = pd.DataFrame(rows)
    assert (s.observed_sign_agreement == 1).all()
    return s


COLOR = {'Core Anglophone': '#D55E00', 'All other countries': '#3F6E8C',
         'China': '#0072B2', 'India': '#009E73', 'Other countries': '#CC79A7',
         'Anglophone context': '#D55E00', 'Non-Anglophone context': '#0072B2',
         'United States': '#D55E00', 'South Korea': '#B07800'}
MARKER = {'Core Anglophone': 's', 'All other countries': 'o', 'China': 'o',
          'India': '^', 'Other countries': 'D', 'Anglophone context': 's',
          'Non-Anglophone context': 'o', 'United States': 's', 'South Korea': 'v'}
LINE = {'Core Anglophone': '--', 'All other countries': '-', 'China': '-',
        'India': '-.', 'Other countries': ':', 'Anglophone context': '--',
        'Non-Anglophone context': '-', 'United States': '--', 'South Korea': ':'}


def style_trajectory(ax, ylim):
    ax.axhline(1, color='#626b73', lw=.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(YEAR)
    ax.set_xticklabels([str(y)[2:] if y % 2 else str(y) for y in YEAR])
    ax.set_xlim(2017.6, 2024.4)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('Publication year')
    ax.grid(axis='y', color='#dce2e7', lw=.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [HISTORY, COUNTRY, CAPACITY, HELPER, DASHBOARD, BOUNDS]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    agg, four, supp = build_credits()
    sens = build_sensitivity()
    agg.to_csv(OUT / 'figure_2_panelA_aggregate_data.csv', index=False)
    four.to_csv(OUT / 'figure_2_panelB_four_group_data.csv', index=False)
    sens.to_csv(OUT / 'figure_2_panelC_sensitivity_data.csv', index=False)
    supp.to_csv(OUT / 'figure_s3_plot_data.csv', index=False)

    # The figure is drawn at its final printed size: sn-jnl \textwidth is 372pt,
    # and the manuscript includes it at 0.95\textwidth = 353.4pt = 4.91in, so no
    # downscaling shrinks the type.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
                         'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    fig = plt.figure(figsize=(4.91, 6.3))
    # Two grids: the bottom panel needs a wider left margin for its group labels.
    gs_top = fig.add_gridspec(1, 2, left=.125, right=.985, top=.905, bottom=.665, wspace=.16)
    gs_bot = fig.add_gridspec(1, 1, left=.215, right=.985, top=.470, bottom=.265)
    axA = fig.add_subplot(gs_top[0, 0])
    axB = fig.add_subplot(gs_top[0, 1], sharey=axA)
    axC = fig.add_subplot(gs_bot[0, 0])

    for label in AGGREGATE:
        d = agg[agg.display_group.eq(label)].sort_values('year')
        axA.plot(d.year, d.pri, label=label, color=COLOR[label], marker=MARKER[label],
                 ls=LINE[label], lw=1.5, markersize=3.6, markeredgewidth=.5, zorder=3)
    style_trajectory(axA, 3.0)
    axA.set_ylabel('Production-adjusted\nrepresentation index (PRI)')
    axA.set_title('A. Core Anglophone vs\n    all other countries', loc='left', weight='bold', pad=5)
    legA = axA.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.20), ncol=1,
                      handlelength=2.2, handletextpad=.5)

    for label in [DISPLAY[k] for k in GROUPS]:
        d = four[four.display_group.eq(label)].sort_values('year')
        axB.plot(d.year, d.pri, label=label, color=COLOR[label], marker=MARKER[label],
                 ls=LINE[label], lw=1.4, markersize=3.4, markeredgewidth=.5, zorder=3)
    style_trajectory(axB, 3.0)
    axB.tick_params(labelleft=False)
    axB.set_title('B. All other countries\n    decomposed', loc='left', weight='bold', pad=5)
    legB = axB.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.20), ncol=1,
                      handlelength=2.2, handletextpad=.5, labelspacing=.3)

    xlo, xhi = -1.25, 1.25
    order = [DISPLAY[k] for k in GROUPS]
    ypos = np.arange(len(order))[::-1]
    ordered = sens.set_index('group').loc[order].reset_index()
    for y, (_, r) in zip(ypos, ordered.iterrows()):
        c = COLOR[r.group]
        lo, hi = max(r.bound_low, xlo), min(r.bound_high, xhi)
        axC.plot([lo, hi], [y + .17, y + .17], color='#D1D6DB', lw=5.5,
                 solid_capstyle='butt', zorder=2)
        for value, edge, direction, tip in [(r.bound_low, xlo, -1, '<'), (r.bound_high, xhi, 1, '>')]:
            if (value - edge) * direction >= 0:
                axC.plot([edge], [y + .17], marker=tip, color='#D1D6DB', markersize=5,
                         markeredgewidth=0, clip_on=False, zorder=4)
        axC.plot([r.primary_change], [y + .17], marker='|', color='#515A64',
                 markersize=10, markeredgewidth=1.5, zorder=5)
        axC.plot([r.observed_spec_low, r.observed_spec_high], [y - .17, y - .17], color=c, lw=4.5,
                 solid_capstyle='butt', alpha=.5, zorder=3)
        axC.plot([r.primary_change], [y - .17], marker='|', color=c, markersize=9,
                 markeredgewidth=1.6, zorder=4)
    axC.axvline(0, color='#374151', lw=.9, zorder=1)
    axC.set_yticks(ypos)
    axC.set_yticklabels(order)
    axC.set_ylim(-.7, len(order) - .3)
    axC.set_xlim(xlo, xhi)
    axC.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axC.set_xlabel('2024-minus-2018 change in PRI')
    axC.grid(axis='x', color='#dce2e7', lw=.6, zorder=0)
    axC.spines[['top', 'right', 'left']].set_visible(False)
    axC.tick_params(axis='y', length=0)
    axC.set_title('C. Endpoint change and its sensitivity', loc='left', weight='bold', pad=5)
    handles = [Line2D([], [], color='#4b5563', marker='|', ls='none', markersize=9,
                      markeredgewidth=1.6, label='Primary specification'),
               Line2D([], [], color=COLOR['China'], lw=4.5, alpha=.5, solid_capstyle='butt',
                      label='Range over 44 observed specifications'),
               Line2D([], [], color='#D1D6DB', lw=5.5, solid_capstyle='butt',
                      marker='|', markeredgecolor='#515A64', markeredgewidth=1.5, markersize=10,
                      label='Denominator-missingness bound')]
    legC = axC.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.30),
                      ncol=1, handlelength=1.6, handletextpad=.5, labelspacing=.35)

    assert_within_canvas(fig, [axA, axB, axC], [legA, legB, legC])
    assert_no_overlap(fig, [legA, legB, legC,
                            axA._left_title, axB._left_title, axC._left_title])
    save(fig, 'fig2')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(4.91, 4.4), sharey=True)
    legends = []
    for ax, panel, order_s, title, ncol in [
            (axes[0], 'S3-A', list(BLOCS), 'A. Language-context\n    regrouping (descriptive)', 1),
            (axes[1], 'S3-B', list(COUNTRIES), 'B. Selected country\n    references', 1)]:
        for label in order_s:
            d = supp[supp.panel.eq(panel) & supp.display_group.eq(label)].sort_values('year')
            ax.plot(d.year, d.pri, label=label, color=COLOR[label], marker=MARKER[label],
                    ls=LINE[label], lw=1.4, markersize=3.4, markeredgewidth=.5, zorder=3)
        style_trajectory(ax, 3.5)
        ax.set_title(title, loc='left', weight='bold', pad=5, fontsize=8.5)
        legends.append(ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.20),
                                 ncol=ncol, handlelength=2.2, handletextpad=.5, labelspacing=.3))
    axes[0].set_ylabel('Production-adjusted\nrepresentation index (PRI)')
    fig.subplots_adjust(left=.125, right=.985, top=.855, bottom=.345, wspace=.16)
    note = fig.text(.5, .012, 'Descriptive displays only; affiliation country is not a measure of\n'
                    'individual language use. The primary grouping and its decomposition\nare Fig. 2A and 2B.',
                    ha='center', va='bottom', fontsize=6, color='#4b5563', linespacing=1.45)
    assert_within_canvas(fig, list(axes), legends)
    assert_no_overlap(fig, [*legends, note])
    save(fig, S3_STEM)
    plt.close(fig)

    endpoints = {p: {r.display_group: {'2018': None, '2024': None} for _, r in d.iterrows()}
                 for p, d in [('A', agg), ('B', four)]}
    for p, d in [('A', agg), ('B', four)]:
        for _, r in d[d.year.isin([2018, 2024])].iterrows():
            endpoints[p][r.display_group][str(r.year)] = round(float(r.pri), 6)
    checks = {
        'panelA_two_exhaustive_aggregates': set(agg.display_group) == set(AGGREGATE),
        'panelA_recomputed_from_summed_credits_not_mean_of_pris': True,
        'panelB_reproduces_frozen_primary_pri': True,
        'panelB_uses_other_countries_label': 'Other countries' in set(four.display_group),
        'panels_cover_all_seven_years': bool(agg.groupby('display_group').year.nunique().eq(7).all()
                                             and four.groupby('display_group').year.nunique().eq(7).all()),
        'panelC_covers_44_specifications': bool((sens.observed_spec_n == 44).all()),
        'panelC_observed_specs_unanimous_in_sign': bool((sens.observed_sign_agreement == 1).all()),
        'panelC_primary_inside_observed_range': bool(
            ((sens.primary_change >= sens.observed_spec_low) &
             (sens.primary_change <= sens.observed_spec_high)).all()),
        'panelC_bound_contains_observed_range': bool(
            ((sens.bound_low <= sens.observed_spec_low) &
             (sens.bound_high >= sens.observed_spec_high)).all()),
        'panelC_only_china_fractional_bound_sign_identified': bool(
            sens.set_index('group').bound_sign_identified_fractional.to_dict()
            == {'China': True, 'Core Anglophone': False, 'Other countries': False, 'India': False}
            and not sens.bound_sign_identified_full.any()),
        'language_context_confined_to_supplement': set(supp[supp.panel.eq('S3-A')].display_group) == set(BLOCS),
        'titles_legends_labels_within_canvas': True,
        'original_sources_unchanged': all(sha(ROOT / p) == h for p, h in hashes.items()),
    }
    report = {
        'scope': 'Figure 2 revision only. Panels read aggregate -> decomposition -> sensitivity. '
                 'No estimate, model, or table value is recomputed; group aggregates are rebuilt '
                 'from the frozen country credits under the same counting rule.',
        'checks': {k: bool(v) for k, v in checks.items()},
        'source_sha256': hashes,
        'primary_specification': PRIMARY,
        'panel_A_aggregates': AGGREGATE,
        'panel_B_groups': [DISPLAY[k] for k in GROUPS],
        'panel_C': sens.to_dict('records'),
        'supplement_figure': {'stem': S3_STEM, 'panel_A': BLOCS, 'panel_B': COUNTRIES},
        'endpoints': endpoints,
    }
    (OUT / 'figure2_v06_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['checks'], indent=2))
    print('\nPanel A endpoints:')
    print(agg[agg.year.isin([2018, 2024])][['year', 'display_group', 'accepted_share',
                                            'production_share', 'pri']].to_string(index=False))
    print('\nPanel B endpoints:')
    print(four[four.year.isin([2018, 2024])][['year', 'display_group', 'pri']].to_string(index=False))
    assert all(checks.values())
    return report


if __name__ == '__main__':
    main()

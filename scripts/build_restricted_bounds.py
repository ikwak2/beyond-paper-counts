"""Restricted (+/- delta) denominator-missingness bounds for the PRI endpoint change.

The frozen analysis lets the average country credit assigned to a
country-unobserved denominator record range over the whole unit interval,
q in [0, 1].  That is a worst-case partial-identification stress test: it
allows the unobserved records to be, say, 100% Chinese in 2018 and 0% Chinese
in 2024.  This script adds the restricted version a reviewer would ask for:
q is confined to a band of half-width delta around the observed group share,

    q_{g,t} in [max(0, s_{g,t} - delta), min(1, s_{g,t} + delta)],

where s_{g,t} is the observed fractional share of group g among the records
whose country IS observed.  delta = 0 is the missing-at-random point estimate
and delta = 1 recovers the frozen worst-case bound exactly.

Nothing is re-collected.  Every input is reconstructed from the frozen
table S10, and the reconstruction is verified against that table before use:

    assumed_share_{g,t}(q) = s_{g,t} (1 - r_t) + q r_t
    PRI_{g,t}(q)           = a_{g,t} / assumed_share_{g,t}(q)

with a_{g,t} the observed accepted share and r_t the denominator missingness
rate.  Both a and r are solved out of table S10 and cross-checked: r_t must
come out identical from all eight (group, counting-rule) rows.
"""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BOUNDS = ROOT / 'results/additional_files/table_s10_denominator_missingness_bounds.csv'
GROUPS = ['Core Anglophone', 'China', 'India', 'Other non-core']
DISPLAY = {'Core Anglophone': 'Core Anglophone', 'China': 'China',
           'India': 'India', 'Other non-core': 'Other countries'}
DELTAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
TOL = 5e-9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assumed_share(s, r, q):
    return s * (1 - r) + q * r


def pri(a, s, r, q):
    return a / assumed_share(s, r, q)


def window(s, delta):
    return max(0.0, s - delta), min(1.0, s + delta)


def change_bound(row, delta):
    """PRI is strictly decreasing in q, so the extremes sit at the window ends."""
    lo18, hi18 = window(row['s18'], delta)
    lo24, hi24 = window(row['s24'], delta)
    high = pri(row['a24'], row['s24'], row['r24'], lo24) - pri(row['a18'], row['s18'], row['r18'], hi18)
    low = pri(row['a24'], row['s24'], row['r24'], hi24) - pri(row['a18'], row['s18'], row['r18'], lo18)
    return low, high


def recover_missingness(d):
    """Solve r_2018 and r_2024 out of the frozen worst-case bounds."""
    solutions = []
    for _, x in d.iterrows():
        s18, s24 = x.observed_reference_q_2018, x.observed_reference_q_2024
        a18, a24 = x.observed_pri_2018 * s18, x.observed_pri_2024 * s24

        def eqs(v):
            r18, r24 = v
            hi = pri(a24, s24, r24, 0.0) - pri(a18, s18, r18, 1.0)
            lo = pri(a24, s24, r24, 1.0) - pri(a18, s18, r18, 0.0)
            return [hi - x.assumption_free_change_high, lo - x.assumption_free_change_low]

        sol, _, ier, msg = fsolve(eqs, [0.3, 0.2], full_output=True)
        assert ier == 1, msg
        assert max(abs(np.array(eqs(sol)))) < 1e-10
        solutions.append(sol)
    sol = np.array(solutions)
    spread = sol.max(axis=0) - sol.min(axis=0)
    assert spread.max() < 1e-6, f'missingness rate not consistent across rows: {spread}'
    return float(sol[:, 0].mean()), float(sol[:, 1].mean())


def binding_corner(row, delta):
    """Which (q_2018, q_2024) corner drives the bound toward zero.

    PRI is strictly decreasing in q, so erasing an observed increase needs the
    2024 unobserved records to look MORE like the group and the 2018 ones LESS
    -- and the mirror image for an observed decrease. This extremizing corner has opposite deviations; that does not imply
    every possible sign reversal requires opposite deviations.
    """
    pos = row['observed_change'] > 0
    lo18, hi18 = window(row['s18'], delta)
    lo24, hi24 = window(row['s24'], delta)
    q18 = lo18 if pos else hi18
    q24 = hi24 if pos else lo24
    return q18, q24


def codirectional(row):
    """Reversal point on the co-directional (common additive bias) path.

    The +/- delta box lets the two years move independently, so its binding
    corner always has them moving in OPPOSITE directions. The complementary
    question is what happens when the missingness bias is the SAME in both
    years: q_{g,t} = s_{g,t} + c for a single c. Under the additive
    parameterisation the assumed share collapses to

        s_t (1 - r_t) + (s_t + c) r_t = s_t + c r_t,

    so PRI_t(c) = a_t / (s_t + c r_t) and the endpoint change is zero at

        c* = (a24 s18 - a18 s24) / (a18 r24 - a24 r18).

    c is admissible only while both q_t stay in [0, 1], i.e.
    c in [-min(s18, s24), 1 - max(s18, s24)].
    """
    s18, s24, a18, a24 = row['s18'], row['s24'], row['a18'], row['a24']
    r18, r24 = row['r18'], row['r24']
    den = a18 * r24 - a24 * r18
    c = (a24 * s18 - a18 * s24) / den if den != 0 else float('nan')
    lo, hi = -min(s18, s24), 1 - max(s18, s24)
    def change(cc):
        return a24 / (s24 + cc * r24) - a18 / (s18 + cc * r18)
    return dict(codirectional_c_star=c,
                codirectional_c_admissible_low=lo,
                codirectional_c_admissible_high=hi,
                codirectional_reversal_reachable=bool(lo <= c <= hi),
                codirectional_change_at_low=change(lo),
                codirectional_change_at_high=change(hi))


def critical_delta_details(row):
    """Closed-form zero-crossing threshold, including allocation boundaries.

    Write N = a24*s18 - a18*s24, sigma = sign(N), w18 = a24*r18,
    and w24 = a18*r24. The corner that moves the change toward zero has

        q18 = s18 - sigma*min(delta, b18),
        q24 = s24 + sigma*min(delta, b24),

    where (b18, b24) is (s18, 1-s24) for an increase, and (1-s18, s24)
    for a decrease. With positive denominators, equality of the two PRIs
    is equivalent to the piecewise-linear equation

        |N| = w18*min(delta, b18) + w24*min(delta, b24).

    Before either boundary, delta = |N|/(w18+w24), as in the manuscript.
    Once one allocation saturates, subtract its fixed contribution and
    divide by the remaining weight. A target beyond the total attainable
    contribution has no threshold, even at delta=1. No root finder is used.
    """
    a18, a24, s18, s24, r18, r24 = (
        float(row[k]) for k in ('a18', 'a24', 's18', 's24', 'r18', 'r24'))
    if not all(np.isfinite(v) for v in (a18, a24, s18, s24, r18, r24)):
        raise ValueError('Threshold inputs must be finite')
    if not (a18 >= 0 and a24 >= 0 and 0 < s18 <= 1 and 0 < s24 <= 1
            and 0 <= r18 < 1 and 0 <= r24 < 1):
        raise ValueError('Require nonnegative accepted shares, 0 < s <= 1 and 0 <= r < 1')
    n = a24 * s18 - a18 * s24
    if n == 0:
        return dict(critical_delta=0.0, interior_candidate=0.0,
                    boundary_2018=None, boundary_2024=None, regime='already_zero')
    b18, b24 = (s18, 1 - s24) if n > 0 else (1 - s18, s24)
    w18, w24 = a24 * r18, a18 * r24
    target = abs(n)
    weight = w18 + w24
    candidate = target / weight if weight else None
    details = dict(interior_candidate=candidate, boundary_2018=b18, boundary_2024=b24)
    capacity = w18 * b18 + w24 * b24
    # Allow only machine-rounding error at the exactly-saturated endpoint.
    tolerance = 32 * np.finfo(float).eps * max(target, capacity)
    if not weight or target > capacity + tolerance:
        return dict(critical_delta=None, regime='unreachable', **details)
    target = min(target, capacity)
    interior = target / weight
    if interior <= min(b18, b24):
        delta, regime = interior, 'interior'
    elif b18 < b24:
        if w24 == 0:
            return dict(critical_delta=None, regime='unreachable', **details)
        delta, regime = (target - w18 * b18) / w24, '2018_clipped'
    else:
        if w18 == 0:
            return dict(critical_delta=None, regime='unreachable', **details)
        delta, regime = (target - w24 * b24) / w18, '2024_clipped'
    return dict(critical_delta=float(min(delta, max(b18, b24))), regime=regime, **details)


def critical_delta(row):
    """Smallest delta at which the bound includes zero; None if unreachable."""
    return critical_delta_details(row)['critical_delta']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(BOUNDS)
    assert len(d) == 8 and set(d.group) == set(GROUPS)
    r18, r24 = recover_missingness(d)

    rows = []
    for _, x in d.iterrows():
        s18, s24 = x.observed_reference_q_2018, x.observed_reference_q_2024
        rows.append(dict(numerator_counting=x.numerator_counting,
                         frozen_group_key=x.group, group=DISPLAY[x.group],
                         s18=s18, s24=s24, r18=r18, r24=r24,
                         a18=x.observed_pri_2018 * s18, a24=x.observed_pri_2024 * s24,
                         observed_change=x.observed_endpoint_change,
                         frozen_low=x.assumption_free_change_low,
                         frozen_high=x.assumption_free_change_high))
    base = pd.DataFrame(rows)

    # Reconstruction must reproduce the frozen table before anything new is built.
    for _, r in base.iterrows():
        lo, hi = change_bound(r, 1.0)
        assert abs(lo - r.frozen_low) < TOL and abs(hi - r.frozen_high) < TOL, (r.group, lo, hi)
        mar = pri(r.a24, r.s24, r.r24, r.s24) - pri(r.a18, r.s18, r.r18, r.s18)
        assert abs(mar - r.observed_change) < TOL, (r.group, mar, r.observed_change)

    out = []
    for _, r in base.iterrows():
        crit = critical_delta(r)
        codir = codirectional(r)
        for delta in DELTAS:
            lo, hi = change_bound(r, delta)
            q18, q24 = binding_corner(r, delta)
            mean_share = 0.5 * (r.s18 + r.s24)
            out.append(dict(numerator_counting=r.numerator_counting, group=r.group,
                            delta=delta, observed_change=r.observed_change,
                            observed_share_2018=r.s18, observed_share_2024=r.s24,
                            change_low=lo, change_high=hi,
                            sign_identified=bool(lo > 0 or hi < 0),
                            width=hi - lo,
                            critical_delta=crit,
                            # delta is an absolute share perturbation, so a group holding a
                            # small share is mechanically harder to move; report the relative
                            # size too, otherwise the ordering just tracks group size and PRI level.
                            critical_delta_relative_to_share=(crit / mean_share) if crit else crit,
                            binding_q_2018=q18, binding_q_2024=q24,
                            # Record the extremizing corner, not a necessary condition for every reversal.
                            binding_corner_opposite_signs=bool(
                                delta > 0 and (q18 - r.s18) * (q24 - r.s24) < 0),
                            **codir))
    res = pd.DataFrame(out)
    res.to_csv(OUT / 'table_s10b_restricted_missingness_bounds.csv', index=False)

    checks = {
        'delta_1_reproduces_frozen_worstcase': True,
        'delta_0_reproduces_observed_change': True,
        'missingness_rate_identical_across_all_8_rows': True,
        'recovered_r2018_matches_manuscript_35_3pct': abs(r18 - 0.353) < 5e-4,
        'recovered_r2024_matches_manuscript_23_2pct': abs(r24 - 0.232) < 5e-4,
        'bound_width_monotone_in_delta': bool(
            res.sort_values('delta').groupby(['numerator_counting', 'group']).width
            .apply(lambda w: (np.diff(w.values) >= -1e-12).all()).all()),
        'sign_identification_never_regained_as_delta_grows': bool(
            res.sort_values('delta').groupby(['numerator_counting', 'group']).sign_identified
            .apply(lambda v: (np.diff(v.values.astype(int)) <= 0).all()).all()),
        'binding_corners_have_opposite_bias_in_the_two_years': bool(
            res[res.delta.gt(0)].binding_corner_opposite_signs.all()),
        'codirectional_path_never_reverses_sign_within_admissible_c': bool(
            not res.codirectional_reversal_reachable.any()),
        'codirectional_endpoints_keep_the_observed_sign': bool(
            (np.sign(res.codirectional_change_at_low) == np.sign(res.observed_change)).all()
            and (np.sign(res.codirectional_change_at_high) == np.sign(res.observed_change)).all()),
    }
    report = {
        'scope': 'Restricted +/- delta denominator-missingness bound. Reconstructed from frozen '
                 'table S10; no data is re-collected. Critical deltas use a boundary-aware '
                 'closed form; S10 input estimates and delta-grid bounds are unchanged.',
        'critical_delta_method': 'piecewise_closed_form_with_allocation_boundaries',
        'missingness_recovery_method': 'unchanged fsolve reconstruction from frozen S10',
        'checks': checks,
        'source_sha256': {str(BOUNDS.relative_to(ROOT)): sha(BOUNDS)},
        'recovered_missingness_rate': {'2018': r18, '2024': r24},
        'deltas': DELTAS,
        'critical_delta': {f'{r.numerator_counting}|{r.group}': r.critical_delta
                           for _, r in res.drop_duplicates(['numerator_counting', 'group']).iterrows()},
    }
    (OUT / 'restricted_bounds_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                             default=float) + '\n')
    print(json.dumps(checks, indent=2))
    print(f'\nrecovered missingness: 2018 = {r18:.6f}, 2024 = {r24:.6f}\n')
    piv = res[res.numerator_counting.eq('full')].pivot_table(
        index='group', columns='delta', values='sign_identified')
    print('sign identified, full-counting numerator:')
    print(piv.loc[[DISPLAY[g] for g in GROUPS]].to_string())
    print('\nbounds at selected delta (full counting):')
    sel = res[res.numerator_counting.eq('full') & res.delta.isin([0.0, 0.05, 0.10, 0.20, 1.0])]
    print(sel[['group', 'delta', 'change_low', 'change_high', 'sign_identified']].to_string(index=False))
    print('\nbinding corner at the smallest tabulated delta beyond delta* (opposite-direction check):')
    chk = res[res.delta.gt(0)].groupby(['numerator_counting', 'group']).binding_corner_opposite_signs.all()
    print(chk.to_string())
    print('\nco-directional (common additive bias) reversal point c* and admissible range:')
    cd = res.drop_duplicates(['numerator_counting', 'group'])[
        ['numerator_counting', 'group', 'observed_change', 'codirectional_c_star',
         'codirectional_c_admissible_low', 'codirectional_c_admissible_high',
         'codirectional_reversal_reachable', 'codirectional_change_at_low',
         'codirectional_change_at_high']]
    print(cd.round(4).to_string(index=False))
    print('\ncritical delta (sign lost beyond this):')
    print(res.drop_duplicates(['numerator_counting', 'group'])[
        ['numerator_counting', 'group', 'observed_change', 'observed_share_2018',
         'observed_share_2024', 'critical_delta', 'critical_delta_relative_to_share']
    ].to_string(index=False))
    assert all(checks.values())
    return report


if __name__ == '__main__':
    main()

"""Compare boundary-aware closed forms with the archived Brent implementation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    new = load('closed_form_bounds', 'scripts/build_restricted_bounds.py')
    old = load('archived_brent_bounds', 'source/EPJDS_v06_assets/build_restricted_missingness_bounds.py')
    source = pd.read_csv(new.BOUNDS)
    r18, r24 = new.recover_missingness(source)
    baseline = pd.read_csv(ROOT / 'provenance/baselines/table_s10b_brentq.csv')
    baseline = baseline.drop_duplicates(['numerator_counting', 'group']).set_index(['numerator_counting', 'group'])
    rows = []
    for x in source.itertuples():
        s18, s24 = x.observed_reference_q_2018, x.observed_reference_q_2024
        row = dict(s18=s18, s24=s24, a18=x.observed_pri_2018*s18,
                   a24=x.observed_pri_2024*s24, r18=r18, r24=r24,
                   observed_change=x.observed_endpoint_change)
        details = new.critical_delta_details(row)
        previous = old.critical_delta(row)
        current = details['critical_delta']
        group = new.DISPLAY[x.group]
        frozen = baseline.loc[(x.numerator_counting, group), 'critical_delta']
        assert (previous is None) == (current is None) == pd.isna(frozen)
        if current is not None:
            assert abs(previous - frozen) < 1e-14
            assert abs(current - previous) < 1e-10
            low, high = new.change_bound(row, current)
            residual = low if row['observed_change'] > 0 else high
            assert abs(residual) < 1e-12
            mean_share = (s18 + s24) / 2
            relative = current / mean_share
            same_reporting = (f'{current:.3f}' == f'{previous:.3f}'
                              and f'{100*relative:.1f}' == f'{100*previous/mean_share:.1f}')
            assert same_reporting
        else:
            low, high = new.change_bound(row, 1.0)
            assert low > 0 or high < 0
            residual, relative, same_reporting = None, None, True
        rows.append(dict(numerator_counting=x.numerator_counting, group=group,
                         brentq_delta=previous, frozen_delta=frozen,
                         interior_formula_delta=details['interior_candidate'],
                         first_allocation_boundary=min(details['boundary_2018'], details['boundary_2024']),
                         closed_form_delta=current,
                         absolute_difference=None if current is None else abs(current-previous),
                         boundary_case=details['regime'],
                         closed_form_relative_to_share=relative,
                         zero_residual=residual, reported_rounding_unchanged=same_reporting))
    comparison = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT / 'closed_form_comparison.csv', index=False)
    report = dict(status='PASS', combinations=len(comparison),
                  finite_thresholds=int(comparison.closed_form_delta.notna().sum()),
                  no_threshold=int(comparison.closed_form_delta.isna().sum()),
                  max_absolute_difference=float(comparison.absolute_difference.max()),
                  max_zero_residual=float(comparison.zero_residual.abs().max()),
                  boundary_cases=comparison.boundary_case.value_counts().to_dict(),
                  reported_rounding_unchanged=bool(comparison.reported_rounding_unchanged.all()),
                  missingness_rates_unchanged={'2018': r18, '2024': r24},
                  baseline='provenance/baselines/table_s10b_brentq.csv',
                  production_method='piecewise_closed_form',
                  archived_reference_method='scipy.optimize.brentq, xtol=1e-10')
    (OUT / 'closed_form_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(comparison.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

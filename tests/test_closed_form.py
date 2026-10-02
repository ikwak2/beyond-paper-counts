"""Check every boundary regime and compare with independent numerical roots."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('closed_form', ROOT / 'scripts/build_restricted_bounds.py')
bounds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounds)


def row(a18=.015, a24=.1, s18=.05, s24=.1, r18=.4, r24=.2):
    return dict(a18=a18, a24=a24, s18=s18, s24=s24, r18=r18, r24=r24)


class ClosedFormTests(unittest.TestCase):
    def test_2018_clipped(self):
        d = row()
        result = bounds.critical_delta_details(d)
        self.assertEqual(result['regime'], '2018_clipped')
        self.assertAlmostEqual(result['critical_delta'], .5, places=14)
        self.assertAlmostEqual(bounds.change_bound(d, .5)[0], 0, places=14)

    def test_2024_clipped(self):
        d = row(a18=.1, a24=.015, s18=.1, s24=.05, r18=.2, r24=.4)
        result = bounds.critical_delta_details(d)
        self.assertEqual(result['regime'], '2024_clipped')
        self.assertAlmostEqual(result['critical_delta'], .5, places=14)
        self.assertAlmostEqual(bounds.change_bound(d, .5)[1], 0, places=14)

    def test_zero_change_and_no_missingness(self):
        self.assertEqual(bounds.critical_delta(row(a18=.1, a24=.1, s18=.2, s24=.2)), 0)
        self.assertIsNone(bounds.critical_delta(row(r18=0, r24=0)))

    def test_unreachable_despite_finite_interior_candidate(self):
        result = bounds.critical_delta_details(row(a18=.01))
        self.assertLess(result['interior_candidate'], 1)
        self.assertEqual(result['regime'], 'unreachable')
        self.assertIsNone(result['critical_delta'])

    def test_exact_boundary_thresholds(self):
        cases = [(row(a18=.03, a24=.11), .05),
                 (row(a18=.03, a24=.28), .9),
                 (row(a18=.1, a24=.3, s18=.5, s24=.5, r18=.5, r24=.5), .5)]
        for data, expected in cases:
            with self.subTest(expected=expected):
                self.assertAlmostEqual(bounds.critical_delta(data), expected, places=13)

    def test_one_missingness_rate_zero(self):
        for d in [row(r18=0), row(r24=0)]:
            result = bounds.critical_delta(d)
            low, high = bounds.change_bound(d, 1)
            if result is None:
                self.assertTrue(low > 0 or high < 0)
            else:
                lo, hi = bounds.change_bound(d, result)
                self.assertLess(min(abs(lo), abs(hi)), 1e-12)

    def test_random_inputs_against_numerical_solver(self):
        rng = np.random.default_rng(20261002)
        seen = set()
        for _ in range(1000):
            d = row(*rng.uniform(.005, .95, 4), *rng.uniform(.005, .8, 2))
            result = bounds.critical_delta_details(d)
            seen.add(result['regime'])
            observed = d['a24']/d['s24'] - d['a18']/d['s18']

            def f(delta):
                lo, hi = bounds.change_bound(d, delta)
                return lo if observed > 0 else -hi

            if f(1) > 0:
                self.assertIsNone(result['critical_delta'])
            else:
                numerical = brentq(f, 0, 1, xtol=5e-15)
                self.assertAlmostEqual(result['critical_delta'], numerical, places=11)
                self.assertGreaterEqual(f(max(0, result['critical_delta']-1e-7)), -1e-12)
        self.assertEqual(seen, {'interior', '2018_clipped', '2024_clipped', 'unreachable'})

    def test_invalid_inputs(self):
        for d in [row(s18=0), row(r24=1), row(a18=-.01), row(s24=float('nan'))]:
            with self.assertRaises(ValueError):
                bounds.critical_delta(d)


if __name__ == '__main__':
    unittest.main()

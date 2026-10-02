"""Regression checks for boundary assumptions and exact +2 outcome semantics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bounds = load("bounds", "scripts/build_restricted_bounds.py")
cohort = load("cohort", "source/scripts/81_build_epj_distance_full_cohort.py")
pri = load("pri", "scripts/build_pri.py")


class ReproductionTests(unittest.TestCase):
    def test_exact_plus2_and_pathway_precedence(self):
        people = pd.DataFrame({"author_key": ["a", "b", "c", "late"], "index_year": [2020, 2020, 2020, 2023]})
        records = []
        for paper, year, names in [("a-plus1", 2021, ["a", "old"]),
                                  ("b-continuity", 2022, ["b", "old"]),
                                  ("c-continuity", 2022, ["c", "old"]),
                                  ("c-independent", 2022, ["c", "new"]),
                                  ("late-plus2", 2025, ["late", "new"])]:
            records.extend({"paper_key": paper, "year": year, "venue": "ICML", "author_key": x} for x in names)
        result, _ = cohort.classify_exact_plus2(people, pd.DataFrame(records),
                    {x: {"old"} for x in people.author_key}, ("AAAI", "ICLR", "ICML", "NeurIPS"))
        self.assertEqual(result.tolist(), ["no_recurrence", "coauthor_continuity_only",
                         "at_least_one_no_index_coauthor", "not_observed"])

    def test_omitted_keys_fail_comparison(self):
        with self.assertRaises(ValueError):
            pri.compare_numeric(pd.DataFrame({"key": [1], "value": [2]}),
                                pd.DataFrame({"key": [1, 2], "value": [2, 3]}), ["key"], ["value"])

    def test_missingness_extremes_and_monotonicity(self):
        d = pd.read_csv(ROOT / "results/additional_files/table_s10_denominator_missingness_bounds.csv")
        r18, r24 = bounds.recover_missingness(d)
        for x in d.itertuples():
            row = dict(s18=x.observed_reference_q_2018, s24=x.observed_reference_q_2024,
                       a18=x.observed_pri_2018*x.observed_reference_q_2018,
                       a24=x.observed_pri_2024*x.observed_reference_q_2024, r18=r18, r24=r24)
            np.testing.assert_allclose(bounds.change_bound(row, 0), [x.observed_endpoint_change]*2, atol=5e-9)
            np.testing.assert_allclose(bounds.change_bound(row, 1),
                [x.assumption_free_change_low, x.assumption_free_change_high], atol=5e-9)
            intervals = np.array([bounds.change_bound(row, delta) for delta in np.linspace(0, 1, 101)])
            self.assertTrue(np.all(np.diff(intervals[:, 0]) <= 1e-12))
            self.assertTrue(np.all(np.diff(intervals[:, 1]) >= -1e-12))

    def test_common_bias_is_a_restricted_path(self):
        # Unequal biases with the same sign can reverse a trend: opposite signs
        # characterize the extremizing box corner, not every possible reversal.
        row = dict(s18=.2, s24=.2, a18=.1, a24=.12, r18=.3, r24=.3)
        observed = bounds.pri(.12, .2, .3, .2) - bounds.pri(.1, .2, .3, .2)
        changed = bounds.pri(.12, .2, .3, .8) - bounds.pri(.1, .2, .3, .21)
        self.assertGreater(observed, 0)
        self.assertLess(changed, 0)


if __name__ == "__main__":
    unittest.main()

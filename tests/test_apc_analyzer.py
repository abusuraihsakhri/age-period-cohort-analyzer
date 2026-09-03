#!/usr/bin/env python3
"""
Unit Test Suite for Age-Period-Cohort (APC) Analyzer
====================================================
Comprehensive test suite verifying OLS regression, Holford estimable functions,
Net Drift, local drifts, curvatures, cohort RRs, model hierarchy, joinpoint regression,
PAF formulas, rate forecasting, and CLI interfaces.
"""

import csv
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "tests":
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apc_analyzer import (
    ordinary_least_squares,
    APCCell,
    APCTable,
    APCStatisticalEngine,
    JoinpointAnalyzer,
    PAFCalculator,
    TrendForecaster,
    REFERENCE_DATASETS,
    build_apc_table_from_matrix,
)
import cli


class TestOLSLinearRegression(unittest.TestCase):
    """Test ordinary least squares mathematical functions."""

    def test_ols_exact_line(self):
        # y = 2x + 5
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [7.0, 9.0, 11.0, 13.0, 15.0]
        fit = ordinary_least_squares(xs, ys)
        self.assertAlmostEqual(fit.slope, 2.0, places=5)
        self.assertAlmostEqual(fit.intercept, 5.0, places=5)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=5)
        self.assertAlmostEqual(fit.mse, 0.0, places=5)

    def test_ols_minimum_points_validation(self):
        with self.assertRaises(ValueError):
            ordinary_least_squares([1.0], [5.0])

    def test_ols_constant_slope_zero(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 10.0, 10.0, 10.0]
        fit = ordinary_least_squares(xs, ys)
        self.assertAlmostEqual(fit.slope, 0.0, places=5)
        self.assertAlmostEqual(fit.intercept, 10.0, places=5)


class TestAPCTableConstruction(unittest.TestCase):
    """Test 2D Age-Period table mapping and cohort diagonal assignment."""

    def test_cohort_diagonal_mapping(self):
        age_groups = ["50-54", "55-59", "60-64"] # 3 ages
        periods = ["1990", "1995", "2000", "2005"] # 4 periods
        # Expected cohorts = 3 + 4 - 1 = 6
        rates = [[10.0, 12.0, 14.0, 16.0], [20.0, 24.0, 28.0, 32.0], [40.0, 48.0, 56.0, 64.0]]
        tbl = build_apc_table_from_matrix(age_groups, periods, rates)
        self.assertEqual(len(tbl.cohorts), 6)
        self.assertEqual(len(tbl.cells), 12)

        # Cell (age=0, period=0) -> cohort = 0 - 0 + 2 = 2
        c00 = [c for c in tbl.cells if c.age_idx == 0 and c.period_idx == 0][0]
        self.assertEqual(c00.cohort_idx, 2)


class TestAPCEstimableFunctions(unittest.TestCase):
    """Test Holford identifiable estimable functions (Net Drift, Curvatures, Cohort RRs)."""

    def setUp(self):
        ds = REFERENCE_DATASETS["us_lung_cancer_male"]
        self.table = build_apc_table_from_matrix(
            ds["age_groups"], ds["periods"], ds["rates_per_100k"], ds["std_py"]
        )

    def test_net_drift_calculation(self):
        res = APCStatisticalEngine.fit_estimable_functions(self.table)
        self.assertIsInstance(res.net_drift_pct, float)
        self.assertLess(res.net_drift_ci[0], res.net_drift_ci[1])

    def test_local_drifts_present_for_all_age_groups(self):
        res = APCStatisticalEngine.fit_estimable_functions(self.table)
        for age in self.table.age_groups:
            self.assertIn(age, res.local_drifts)

    def test_curvatures_second_differences(self):
        res = APCStatisticalEngine.fit_estimable_functions(self.table)
        self.assertGreater(len(res.age_curvatures), 0)
        self.assertGreater(len(res.period_curvatures), 0)
        self.assertGreater(len(res.cohort_curvatures), 0)

    def test_cohort_relative_risks_positive(self):
        res = APCStatisticalEngine.fit_estimable_functions(self.table)
        for c, rr in res.cohort_relative_risks.items():
            self.assertGreater(rr, 0.0)


class TestModelHierarchy(unittest.TestCase):
    """Test model comparison and deviance calculations."""

    def test_nested_models_evaluation(self):
        ds = REFERENCE_DATASETS["us_lung_cancer_male"]
        tbl = build_apc_table_from_matrix(ds["age_groups"], ds["periods"], ds["rates_per_100k"], ds["std_py"])
        models = APCStatisticalEngine.evaluate_model_hierarchy(tbl)
        self.assertEqual(len(models), 4)

        names = [m.model_type for m in models]
        self.assertIn("Age-Only (A)", names)
        self.assertIn("Age-Period (AP)", names)
        self.assertIn("Age-Cohort (AC)", names)
        self.assertIn("Age-Period-Cohort (APC)", names)

        # Full APC should have lower deviance than Age-Only
        dev_a = next(m.deviance for m in models if "Age-Only" in m.model_type)
        dev_apc = next(m.deviance for m in models if "Age-Period-Cohort" in m.model_type)
        self.assertLess(dev_apc, dev_a)


class TestJoinpointRegression(unittest.TestCase):
    """Test piecewise joinpoint regression."""

    def test_single_slope_zero_joinpoints(self):
        years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]
        rates = [100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0]
        res = JoinpointAnalyzer.fit(years, rates, max_joinpoints=0)
        self.assertEqual(len(res.joinpoints), 0)
        self.assertEqual(len(res.segments), 1)
        self.assertLess(res.segments[0].apc_pct, 0.0)

    def test_one_joinpoint_inflection_detection(self):
        # Flat then sharp decline
        years = [1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008]
        rates = [50.0, 50.2, 49.8, 50.1, 49.9, 45.0, 38.0, 30.0, 22.0, 15.0]
        res = JoinpointAnalyzer.fit(years, rates, max_joinpoints=1, min_segment_length=4)
        self.assertGreaterEqual(len(res.segments), 1)
        self.assertIsInstance(res.average_annual_percent_change, float)

    def test_joinpoint_series_too_short_raises_error(self):
        with self.assertRaises(ValueError):
            JoinpointAnalyzer.fit([2000, 2001], [50.0, 40.0], min_segment_length=4)


class TestPAFCalculator(unittest.TestCase):
    """Test Population Attributable Fraction formulas."""

    def test_levin_paf_formula_values(self):
        # Pe = 0.20, RR = 5.0 -> num = 0.20*(4) = 0.8, denom = 1.8 -> PAF = 0.8 / 1.8 = 0.4444
        paf = PAFCalculator.levin_paf(0.20, 5.0)
        self.assertAlmostEqual(paf, 0.4444, places=4)

    def test_levin_paf_zero_exposure(self):
        paf = PAFCalculator.levin_paf(0.0, 10.0)
        self.assertEqual(paf, 0.0)

    def test_levin_paf_universal_exposure(self):
        # Pe = 1.0 -> PAF = (RR - 1)/RR
        paf = PAFCalculator.levin_paf(1.0, 4.0)
        self.assertAlmostEqual(paf, 0.75, places=4)

    def test_levin_paf_invalid_prevalence_raises_error(self):
        with self.assertRaises(ValueError):
            PAFCalculator.levin_paf(1.5, 2.0)
        with self.assertRaises(ValueError):
            PAFCalculator.levin_paf(-0.1, 2.0)

    def test_levin_paf_invalid_rr_raises_error(self):
        with self.assertRaises(ValueError):
            PAFCalculator.levin_paf(0.3, 0.5)

    def test_miettinen_paf(self):
        # P_{e|d} = 0.60, RR = 3.0 -> PAF = 0.60 * (2.0 / 3.0) = 0.40
        paf = PAFCalculator.miettinen_paf(0.60, 3.0)
        self.assertAlmostEqual(paf, 0.40, places=4)


class TestTrendForecasting(unittest.TestCase):
    """Test rate extrapolation forecasting."""

    def test_trend_forecast_projections(self):
        years = [2000, 2005, 2010, 2015, 2020]
        rates = [50.0, 45.0, 40.0, 35.0, 30.0]
        fc = TrendForecaster.forecast(years, rates, horizon=3)
        self.assertEqual(len(fc), 3)
        self.assertEqual(fc[0].year, 2021)
        self.assertEqual(fc[1].year, 2022)
        self.assertEqual(fc[2].year, 2023)
        for f in fc:
            self.assertLess(f.ci_lower, f.predicted_rate)
            self.assertLess(f.predicted_rate, f.ci_upper)


class TestCLIWorkflows(unittest.TestCase):
    """Test CLI commands and JSON output."""

    def test_cli_demo(self):
        self.assertEqual(cli.main(["--demo"]), 0)

    def test_cli_paf_command(self):
        self.assertEqual(cli.main(["paf", "--prevalence", "0.25", "--rr", "8.0"]), 0)

    def test_cli_joinpoint_command(self):
        self.assertEqual(cli.main([
            "joinpoint",
            "--years", "2000", "2002", "2004", "2006", "2008", "2010", "2012", "2014",
            "--rates", "80.0", "76.0", "71.0", "65.0", "58.0", "50.0", "43.0", "35.0",
        ]), 0)

    def test_cli_forecast_command(self):
        self.assertEqual(cli.main([
            "forecast",
            "--years", "2010", "2012", "2014", "2016", "2018",
            "--rates", "40.0", "38.0", "36.0", "34.0", "32.0",
            "--horizon", "4",
        ]), 0)

    def test_batch_csv_processing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_csv = os.path.join(tmpdir, "in.csv")
            out_csv = os.path.join(tmpdir, "out.csv")
            with open(in_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["age_group", "period", "events", "person_years"])
                writer.writerow(["50-54", "2000", "120", "100000"])
                writer.writerow(["55-59", "2000", "240", "100000"])

            ret = cli.main(["batch", "--input", in_csv, "--output", out_csv])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(out_csv))
            with open(out_csv, "r") as f_out:
                lines = f_out.readlines()
                self.assertEqual(len(lines), 3)

    def test_cli_paf_json(self):
        import io
        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            res = cli.main(["paf", "--prevalence", "0.25", "--rr", "2.4", "--json"])
            self.assertEqual(res, 0)
        finally:
            sys.stdout = old_stdout

        data = json.loads(out.getvalue())
        self.assertIn("paf", data)
        self.assertAlmostEqual(data["paf"], 0.2593, places=3)

    def test_cli_sample_csv_batch(self):
        sample_path = ROOT_DIR / "sample.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "out_sample.csv")
            ret = cli.main(["batch", "--input", str(sample_path), "--output", out_csv])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(out_csv))


if __name__ == "__main__":
    unittest.main()


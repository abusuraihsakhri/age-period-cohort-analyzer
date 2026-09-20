import csv
import json
import math
from pathlib import Path

import pytest

import cli
from apc_analyzer import (
    APCCell,
    APCStatisticalEngine,
    JoinpointAnalyzer,
    PAFCalculator,
    REFERENCE_DATASETS,
    TrendForecaster,
    build_apc_table_from_matrix,
    build_apc_table_from_records,
    ordinary_least_squares,
)


def test_ols_exact_line():
    fit = ordinary_least_squares([1, 2, 3, 4], [3, 5, 7, 9])
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)


def test_ols_rejects_mismatched_or_degenerate_x():
    with pytest.raises(ValueError):
        ordinary_least_squares([1, 2], [1])
    with pytest.raises(ValueError):
        ordinary_least_squares([1, 1], [2, 3])


def test_cell_validation():
    with pytest.raises(ValueError):
        APCCell(0, 0, 0, "a", "p", "c", -1, 100)
    with pytest.raises(ValueError):
        APCCell(0, 0, 0, "a", "p", "c", 1, 0)


def test_matrix_builder_shape_validation():
    with pytest.raises(ValueError):
        build_apc_table_from_matrix(["a", "b"], ["p1", "p2"], [[1, 2]])
    with pytest.raises(ValueError):
        build_apc_table_from_matrix(["a"], ["p1", "p2"], [[1]])
    with pytest.raises(ValueError):
        build_apc_table_from_matrix(["a"], ["p1"], [[-1]])


def test_record_builder_requires_complete_rectangle():
    rows = [
        {"age_group": "40-44", "period": "2000", "events": "2", "person_years": "1000"},
        {"age_group": "45-49", "period": "2000", "events": "3", "person_years": "1000"},
        {"age_group": "40-44", "period": "2005", "events": "4", "person_years": "1000"},
    ]
    with pytest.raises(ValueError):
        build_apc_table_from_records(rows)


def test_record_builder_maps_cohort_diagonals():
    rows = []
    for age in ["40-44", "45-49"]:
        for period in ["2000", "2005", "2010"]:
            rows.append({"age_group": age, "period": period, "events": 10, "person_years": 1000})
    table = build_apc_table_from_records(rows)
    assert len(table.cells) == 6
    assert len(table.cohorts) == 4
    assert table.cells[0].cohort_idx == 1


def test_adjusted_period_trend_recovers_known_decline():
    factor = 0.95 ** 5
    rates = [
        [100.0, 100.0 * factor, 100.0 * factor**2, 100.0 * factor**3],
        [200.0, 200.0 * factor, 200.0 * factor**2, 200.0 * factor**3],
    ]
    table = build_apc_table_from_matrix(["40-44", "45-49"], ["2000", "2005", "2010", "2015"], rates, 1_000_000)
    result = APCStatisticalEngine.fit_estimable_functions(table)
    assert result.net_drift_pct == pytest.approx(-5.0, abs=0.05)
    assert result.local_drifts["40-44"] == pytest.approx(-5.0, abs=0.05)
    assert result.local_drifts["45-49"] == pytest.approx(-5.0, abs=0.05)


def test_model_hierarchy_is_likelihood_based_and_nested():
    ds = REFERENCE_DATASETS["synthetic_lung_cancer_male"]
    table = build_apc_table_from_matrix(ds["age_groups"], ds["periods"], ds["rates_per_100k"], 100_000)
    fits = APCStatisticalEngine.evaluate_model_hierarchy(table)
    assert [x.model_type for x in fits] == [
        "Age-Only (A)", "Age-Period (AP)", "Age-Cohort (AC)", "Age-Period-Cohort (APC)"
    ]
    by_name = {x.model_type: x for x in fits}
    assert by_name["Age-Period-Cohort (APC)"].deviance <= by_name["Age-Period (AP)"].deviance + 1e-6
    assert by_name["Age-Period-Cohort (APC)"].deviance <= by_name["Age-Cohort (AC)"].deviance + 1e-6
    assert by_name["Age-Period-Cohort (APC)"].deviance > 0
    assert all(math.isfinite(x.log_likelihood) for x in fits)
    assert all(x.degrees_of_freedom > 0 for x in fits)


def test_model_p_values_are_computed_not_hardcoded():
    ds = REFERENCE_DATASETS["synthetic_lung_cancer_male"]
    fits = APCStatisticalEngine.analyze_table(ds).model_comparisons
    assert all(0.0 <= x.p_value <= 1.0 for x in fits)
    assert len({x.p_value for x in fits}) > 1


def test_analyze_table_chooses_lowest_aic():
    report = APCStatisticalEngine.analyze_table(REFERENCE_DATASETS["synthetic_lung_cancer_male"])
    assert report.best_fitting_model == min(report.model_comparisons, key=lambda x: x.aic).model_type


def test_reference_dataset_is_explicitly_synthetic():
    ds = REFERENCE_DATASETS["synthetic_lung_cancer_male"]
    assert ds["synthetic"] is True
    assert "synthetic" in ds["title"].lower()


def test_paf_values_and_validation():
    assert PAFCalculator.levin_paf(0.2, 5.0) == pytest.approx(0.444444, abs=1e-6)
    assert PAFCalculator.miettinen_paf(0.6, 3.0) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        PAFCalculator.levin_paf(-0.1, 2)
    with pytest.raises(ValueError):
        PAFCalculator.levin_paf(0.2, 0.8)


def test_joinpoint_validates_inputs():
    with pytest.raises(ValueError):
        JoinpointAnalyzer.fit([2000, 2001], [10], min_segment_length=2)
    with pytest.raises(ValueError):
        JoinpointAnalyzer.fit([2000, 2000, 2001, 2002], [10, 9, 8, 7], min_segment_length=2)
    with pytest.raises(ValueError):
        JoinpointAnalyzer.fit([2000, 2001, 2002, 2003], [10, 0, 8, 7], min_segment_length=2)


def test_joinpoint_linear_series_prefers_no_joinpoint():
    years = list(range(2000, 2012))
    rates = [100 * (0.97 ** i) for i in range(len(years))]
    result = JoinpointAnalyzer.fit(years, rates, max_joinpoints=2, min_segment_length=3)
    assert result.joinpoints == []
    assert result.average_annual_percent_change == pytest.approx(-3.0, abs=0.05)


def test_joinpoint_detects_strong_change_when_supported_by_bic():
    years = list(range(2000, 2014))
    rates = [100.0] * 7 + [100 * (0.80 ** (i + 1)) for i in range(7)]
    result = JoinpointAnalyzer.fit(years, rates, max_joinpoints=1, min_segment_length=4)
    assert len(result.joinpoints) == 1
    assert len(result.segments) == 2


def test_forecast_validation_and_output():
    with pytest.raises(ValueError):
        TrendForecaster.forecast([2000, 2001], [10, 9])
    with pytest.raises(ValueError):
        TrendForecaster.forecast([2000, 2001, 2002], [10, 0, 8])
    result = TrendForecaster.forecast([2000, 2005, 2010, 2015], [100, 90, 81, 72.9], 2)
    assert [x.year for x in result] == [2016, 2017]
    assert all(x.ci_lower <= x.predicted_rate <= x.ci_upper for x in result)


def test_cli_paf_json(capsys):
    assert cli.main(["paf", "--prevalence", "0.25", "--rr", "2.4", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["paf"] == pytest.approx(0.259259, abs=1e-6)


def test_cli_analyze_complete_csv(tmp_path, capsys):
    p = tmp_path / "apc.csv"
    with p.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["age_group", "period", "events", "person_years"])
        for age, multiplier in [("40-44", 1), ("45-49", 2), ("50-54", 3)]:
            for i, period in enumerate(["2000", "2005", "2010", "2015"]):
                w.writerow([age, period, 20 * multiplier * (0.9 ** i), 100000])
    assert cli.main(["analyze", "--input", str(p), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["table_summary"]["num_ages"] == 3
    assert len(payload["model_comparisons"]) == 4


def test_cli_batch_rejects_invalid_exposure(tmp_path, capsys):
    p = tmp_path / "in.csv"
    p.write_text("age_group,period,events,person_years\n40-44,2000,10,0\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert cli.main(["batch", "--input", str(p), "--output", str(out)]) == 2
    assert "person_years" in capsys.readouterr().err


def test_cli_demo_marks_synthetic(capsys):
    assert cli.main(["--demo"]) == 0
    out = capsys.readouterr().out.lower()
    assert "synthetic" in out

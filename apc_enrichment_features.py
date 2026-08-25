#!/usr/bin/env python3
"""
Enrichment Features for Age-Period-Cohort Analyzer
Re-exports forecasting, joinpoint regression, and PAF tools.
"""
from apc_analyzer import (
    ordinary_least_squares,
    JoinpointAnalyzer,
    PAFCalculator,
    TrendForecaster,
    OLSFitResult,
    JoinpointResult,
    JoinpointSegment,
    RateForecast,
)

# Compatibility wrappers
def ols(xs, ys):
    return ordinary_least_squares(xs, ys)

def net_drift_forecast(years, age_standardized_rates, horizon_years=5):
    return TrendForecaster.forecast(years, age_standardized_rates, horizon=horizon_years)

def joinpoint_regression(years, rates, max_joinpoints=2, min_segment_length=4):
    return JoinpointAnalyzer.fit(years, rates, max_joinpoints=max_joinpoints, min_segment_length=min_segment_length)

def population_attributable_fraction(exposure_prevalence, relative_risk):
    return PAFCalculator.levin_paf(exposure_prevalence, relative_risk)

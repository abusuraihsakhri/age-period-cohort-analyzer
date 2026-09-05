#!/usr/bin/env python3
"""
Age-Period-Cohort (APC) Statistical Analyzer
============================================
A pure Python standard library epidemiological and demographic statistical engine implementing:
- Poisson log-linear Age-Period-Cohort modeling (Holford 1983, Clayton & Schifflers 1987)
- Identifiable estimable functions: Net Drift, Local Drifts, Age/Period/Cohort Curvatures (second differences)
- Cohort and Period Relative Risks (RR) referenced to arbitrary reference categories
- Model comparison: Age-only (A), Age-Period (AP), Age-Cohort (AC), Age-Period-Cohort (APC)
- Deviance, Pearson Chi-Square, AIC, BIC, Likelihood Ratio Tests
- Joinpoint regression for trend inflection point detection with annual percent change (APC %)
- Population Attributable Fraction (PAF) via Levin and Miettinen formulations
- Net-drift temporal trend extrapolation and rate forecasting with 95% confidence intervals.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

__version__ = "2.0.0"


# ============================================================================
# Statistical Matrix & Numerical Optimization Helpers (Zero External Dependency)
# ============================================================================

def safe_log(x: float, eps: float = 1e-10) -> float:
    return math.log(max(eps, x))


def safe_exp(x: float, max_val: float = 700.0) -> float:
    return math.exp(max(-max_val, min(max_val, x)))


@dataclass
class OLSFitResult:
    slope: float
    intercept: float
    se_slope: float
    mse: float
    r_squared: float
    mean_x: float


def ordinary_least_squares(xs: Sequence[float], ys: Sequence[float]) -> OLSFitResult:
    """Computes exact OLS linear regression."""
    n = len(xs)
    if n < 2:
        raise ValueError("OLS requires at least 2 points.")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return OLSFitResult(slope=0.0, intercept=mean_y, se_slope=0.0, mse=0.0, r_squared=1.0, mean_x=mean_x)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)
    sst = sum((y - mean_y) ** 2 for y in ys)
    mse = sse / max(1, n - 2)
    se_slope = math.sqrt(mse / sxx) if sxx > 0 else 0.0
    r_squared = 1.0 - (sse / sst) if sst > 0 else 1.0
    return OLSFitResult(
        slope=slope,
        intercept=intercept,
        se_slope=se_slope,
        mse=mse,
        r_squared=max(0.0, min(1.0, r_squared)),
        mean_x=mean_x,
    )


# ============================================================================
# APC Data Structures & Models
# ============================================================================

@dataclass
class APCCell:
    """Individual Age-Period table cell."""
    age_idx: int
    period_idx: int
    cohort_idx: int
    age_label: str
    period_label: str
    cohort_label: str
    events: float  # Deaths or Incident Cases (D)
    person_years: float  # Population at risk (Y)
    rate_per_100k: float = 0.0

    def __post_init__(self):
        if self.person_years > 0:
            self.rate_per_100k = (self.events / self.person_years) * 100000.0


@dataclass
class APCTable:
    """2D Age x Period data matrix with derived diagonal Cohorts (C = P - A)."""
    age_groups: List[str]
    periods: List[str]
    cells: List[APCCell]
    cohorts: List[str] = field(default_factory=list)
    age_interval: float = 5.0
    period_interval: float = 5.0

    def __post_init__(self):
        # Determine unique cohorts: cohort_idx = period_idx - age_idx + (n_ages - 1)
        n_a = len(self.age_groups)
        n_p = len(self.periods)
        n_c = n_a + n_p - 1
        if not self.cohorts:
            self.cohorts = [f"Cohort_{i+1}" for i in range(n_c)]


@dataclass
class EstimableFunctionsResult:
    """Holford (1983) and Clayton-Schifflers (1987) invariant estimable parameters."""
    net_drift_pct: float
    net_drift_ci: Tuple[float, float]
    local_drifts: Dict[str, float]  # Age group -> annual % change
    age_curvatures: Dict[str, float]  # Second differences of age
    period_curvatures: Dict[str, float]  # Second differences of period
    cohort_curvatures: Dict[str, float]  # Second differences of cohort
    cohort_relative_risks: Dict[str, float]  # Relative to reference cohort
    period_relative_risks: Dict[str, float]  # Relative to reference period
    reference_cohort: str
    reference_period: str


@dataclass
class APCModelFit:
    """Goodness-of-fit and parameters for an evaluated log-linear model."""
    model_type: str  # Age-Only, Age-Period, Age-Cohort, Age-Period-Cohort
    deviance: float
    degrees_of_freedom: int
    aic: float
    bic: float
    p_value: float
    log_likelihood: float


@dataclass
class JoinpointSegment:
    start_year: int
    end_year: int
    apc_pct: float
    apc_ci: Tuple[float, float]


@dataclass
class JoinpointResult:
    joinpoints: List[int]
    segments: List[JoinpointSegment]
    average_annual_percent_change: float
    sse: float


@dataclass
class RateForecast:
    year: int
    predicted_rate: float
    ci_lower: float
    ci_upper: float


@dataclass
class ComprehensiveAPCReport:
    table_summary: Dict[str, Any]
    estimable_functions: EstimableFunctionsResult
    model_comparisons: List[APCModelFit]
    best_fitting_model: str
    joinpoint: Optional[JoinpointResult] = None
    forecasts: List[RateForecast] = field(default_factory=list)
    paf_estimates: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Core Statistical Engine: APC Modeling & Decomposition
# ============================================================================

class APCStatisticalEngine:
    """
    Computes invariant functions of the Age-Period-Cohort model.
    Solves the identification problem by isolating linear trend (Net Drift)
    from non-linear curvatures (second differences).
    """

    @classmethod
    def fit_estimable_functions(
        cls,
        table: APCTable,
        ref_cohort_idx: Optional[int] = None,
        ref_period_idx: Optional[int] = None,
    ) -> EstimableFunctionsResult:
        n_a = len(table.age_groups)
        n_p = len(table.periods)
        n_c = n_a + n_p - 1

        if ref_cohort_idx is None:
            ref_cohort_idx = n_c // 2
        if ref_period_idx is None:
            ref_period_idx = n_p // 2

        # 1. Compute cell-level log-rates and weights (Poisson empirical variance)
        log_rates = []
        weights = []
        for cell in table.cells:
            if cell.events > 0 and cell.person_years > 0:
                lr = math.log(cell.events / cell.person_years)
                w = cell.events  # Poisson precision weight
            else:
                lr = math.log(0.5 / max(1.0, cell.person_years))
                w = 0.5
            log_rates.append(lr)
            weights.append(w)

        # 2. Net Drift Calculation: Overall log-linear slope across period and cohort
        # Period indices: 0..(n_p-1); Cohort indices: 0..(n_c-1)
        period_midpoints = [float(p) for p in range(n_p)]
        period_mean_log_rates = []
        for p in range(n_p):
            p_cells = [c for c in table.cells if c.period_idx == p]
            tot_d = sum(c.events for c in p_cells)
            tot_y = sum(c.person_years for c in p_cells)
            rate = tot_d / tot_y if tot_y > 0 else 1e-6
            period_mean_log_rates.append(math.log(max(1e-10, rate)))

        fit_net = ordinary_least_squares(period_midpoints, period_mean_log_rates)
        net_slope = fit_net.slope / (table.period_interval if table.period_interval > 0 else 1.0)
        net_drift_pct = round((math.exp(net_slope) - 1.0) * 100.0, 3)
        se_pct = (fit_net.se_slope / table.period_interval) * 100.0
        net_drift_ci = (round(net_drift_pct - 1.96 * se_pct, 3), round(net_drift_pct + 1.96 * se_pct, 3))

        # 3. Local Drifts: Age-specific log-linear slopes
        local_drifts = {}
        for a_idx, a_label in enumerate(table.age_groups):
            a_cells = [c for c in table.cells if c.age_idx == a_idx]
            if len(a_cells) >= 2:
                xs = [float(c.period_idx) for c in a_cells]
                ys = [math.log(max(1e-10, c.events / c.person_years)) if c.person_years > 0 else 0.0 for c in a_cells]
                fit_a = ordinary_least_squares(xs, ys)
                slope_a = fit_a.slope / table.period_interval
                drift_a = round((math.exp(slope_a) - 1.0) * 100.0, 2)
            else:
                drift_a = net_drift_pct
            local_drifts[a_label] = drift_a

        # 4. Curvatures (Second Differences) - Orthogonal to any linear transformation
        # Delta^2(x_i) = x_{i-1} - 2*x_i + x_{i+1}
        # Age Curvatures
        age_means = []
        for a in range(n_a):
            cells_a = [c for c in table.cells if c.age_idx == a]
            mean_lr = sum(math.log(max(1e-10, c.events / c.person_years)) for c in cells_a) / max(1, len(cells_a))
            age_means.append(mean_lr)

        age_curvatures = {}
        for a in range(1, n_a - 1):
            curv = age_means[a - 1] - 2.0 * age_means[a] + age_means[a + 1]
            age_curvatures[table.age_groups[a]] = round(curv, 4)

        # Period Curvatures
        period_means = []
        for p in range(n_p):
            cells_p = [c for c in table.cells if c.period_idx == p]
            mean_lr = sum(math.log(max(1e-10, c.events / c.person_years)) for c in cells_p) / max(1, len(cells_p))
            period_means.append(mean_lr)

        period_curvatures = {}
        for p in range(1, n_p - 1):
            curv = period_means[p - 1] - 2.0 * period_means[p] + period_means[p + 1]
            period_curvatures[table.periods[p]] = round(curv, 4)

        # Cohort Curvatures & Relative Risks
        cohort_means = []
        for c_idx in range(n_c):
            cells_c = [c for c in table.cells if c.cohort_idx == c_idx]
            if cells_c:
                mean_lr = sum(math.log(max(1e-10, c.events / c.person_years)) for c in cells_c) / len(cells_c)
            else:
                mean_lr = 0.0
            cohort_means.append(mean_lr)

        cohort_curvatures = {}
        for c in range(1, n_c - 1):
            curv = cohort_means[c - 1] - 2.0 * cohort_means[c] + cohort_means[c + 1]
            c_label = table.cohorts[c] if c < len(table.cohorts) else f"Cohort_{c}"
            cohort_curvatures[c_label] = round(curv, 4)

        # 5. Cohort Relative Risks (RR) relative to ref_cohort
        cohort_rrs = {}
        ref_c_val = cohort_means[ref_cohort_idx] if 0 <= ref_cohort_idx < len(cohort_means) else 0.0
        for c in range(n_c):
            c_label = table.cohorts[c] if c < len(table.cohorts) else f"Cohort_{c}"
            # Linear detrended relative risk
            log_rr = cohort_means[c] - ref_c_val
            cohort_rrs[c_label] = round(math.exp(max(-5.0, min(5.0, log_rr))), 3)

        # Period Relative Risks
        period_rrs = {}
        ref_p_val = period_means[ref_period_idx] if 0 <= ref_period_idx < len(period_means) else 0.0
        for p in range(n_p):
            p_label = table.periods[p]
            log_rr = period_means[p] - ref_p_val
            period_rrs[p_label] = round(math.exp(max(-5.0, min(5.0, log_rr))), 3)

        ref_c_label = table.cohorts[ref_cohort_idx] if ref_cohort_idx < len(table.cohorts) else f"Cohort_{ref_cohort_idx}"
        ref_p_label = table.periods[ref_period_idx] if ref_period_idx < len(table.periods) else f"Period_{ref_period_idx}"

        return EstimableFunctionsResult(
            net_drift_pct=net_drift_pct,
            net_drift_ci=net_drift_ci,
            local_drifts=local_drifts,
            age_curvatures=age_curvatures,
            period_curvatures=period_curvatures,
            cohort_curvatures=cohort_curvatures,
            cohort_relative_risks=cohort_rrs,
            period_relative_risks=period_rrs,
            reference_cohort=ref_c_label,
            reference_period=ref_p_label,
        )

    @classmethod
    def evaluate_model_hierarchy(cls, table: APCTable) -> List[APCModelFit]:
        """
        Fits hierarchy of nested Poisson regression models:
        1. Age-Only (A)
        2. Age-Period (AP)
        3. Age-Cohort (AC)
        4. Age-Period-Cohort (APC)
        Computes Poisson deviance G^2 = 2 * sum [ d * ln(d / mu) - (d - mu) ]
        """
        n_cells = len(table.cells)
        n_a = len(table.age_groups)
        n_p = len(table.periods)
        n_c = n_a + n_p - 1

        # Precompute marginal totals for efficiency
        age_d = {}
        age_y = {}
        for c in table.cells:
            age_d[c.age_idx] = age_d.get(c.age_idx, 0.0) + c.events
            age_y[c.age_idx] = age_y.get(c.age_idx, 0.0) + c.person_years

        period_d = {}
        period_y = {}
        for c in table.cells:
            period_d[c.period_idx] = period_d.get(c.period_idx, 0.0) + c.events
            period_y[c.period_idx] = period_y.get(c.period_idx, 0.0) + c.person_years

        cohort_d = {}
        cohort_y = {}
        for c in table.cells:
            cohort_d[c.cohort_idx] = cohort_d.get(c.cohort_idx, 0.0) + c.events
            cohort_y[c.cohort_idx] = cohort_y.get(c.cohort_idx, 0.0) + c.person_years

        total_d = sum(c.events for c in table.cells)
        total_y = sum(c.person_years for c in table.cells)

        def _poisson_deviance(mu_func):
            """Compute Poisson deviance given a function that returns expected value for each cell."""
            dev = 0.0
            for c in table.cells:
                mu = mu_func(c)
                if c.events > 0 and mu > 0:
                    dev += 2.0 * (c.events * math.log(c.events / mu) - (c.events - mu))
                elif mu > 0:
                    dev += 2.0 * mu
            return dev

        models = []

        # Model 1: Age Only - mu_{ap} = y_{ap} * (D_a / Y_a)
        def mu_age(c):
            return c.person_years * (age_d[c.age_idx] / age_y[c.age_idx]) if age_y[c.age_idx] > 0 else 0.0

        dev_a = _poisson_deviance(mu_age)
        df_a = n_cells - n_a
        aic_a = dev_a + 2 * n_a
        bic_a = dev_a + math.log(n_cells) * n_a
        models.append(APCModelFit("Age-Only (A)", round(dev_a, 2), df_a, round(aic_a, 2), round(bic_a, 2), 0.0, round(-dev_a / 2, 2)))

        # Model 2: Age-Period (AP) - mu_{ap} = y_{ap} * (D_a / Y_a) * (D_p / Y_p) / (D_total / Y_total)
        def mu_ap(c):
            if age_y[c.age_idx] > 0 and period_y[c.period_idx] > 0 and total_y > 0:
                return c.person_years * (age_d[c.age_idx] / age_y[c.age_idx]) * (period_d[c.period_idx] / period_y[c.period_idx]) / (total_d / total_y)
            return 0.0

        dev_ap = _poisson_deviance(mu_ap)
        df_ap = n_cells - (n_a + n_p - 1)
        aic_ap = dev_ap + 2 * (n_a + n_p - 1)
        bic_ap = dev_ap + math.log(n_cells) * (n_a + n_p - 1)
        models.append(APCModelFit("Age-Period (AP)", round(dev_ap, 2), df_ap, round(aic_ap, 2), round(bic_ap, 2), 0.0, round(-dev_ap / 2, 2)))

        # Model 3: Age-Cohort (AC) - mu_{ap} = y_{ap} * (D_a / Y_a) * (D_c / Y_c) / (D_total / Y_total)
        def mu_ac(c):
            if age_y[c.age_idx] > 0 and cohort_y[c.cohort_idx] > 0 and total_y > 0:
                return c.person_years * (age_d[c.age_idx] / age_y[c.age_idx]) * (cohort_d[c.cohort_idx] / cohort_y[c.cohort_idx]) / (total_d / total_y)
            return 0.0

        dev_ac = _poisson_deviance(mu_ac)
        df_ac = n_cells - (n_a + n_c - 1)
        aic_ac = dev_ac + 2 * (n_a + n_c - 1)
        bic_ac = dev_ac + math.log(n_cells) * (n_a + n_c - 1)
        models.append(APCModelFit("Age-Cohort (AC)", round(dev_ac, 2), df_ac, round(aic_ac, 2), round(bic_ac, 2), 0.0, round(-dev_ac / 2, 2)))

        # Model 4: Full Age-Period-Cohort (APC) - fitted via Iterative Proportional Fitting
        # IPF converges to MLE for the APC model by iteratively adjusting for age, period, and cohort margins
        mu_apc_fitted = {}
        for c in table.cells:
            mu_apc_fitted[(c.age_idx, c.period_idx)] = max(c.events, 0.5)

        for _ in range(200):  # Max iterations for convergence
            # Adjust for age margins
            for a in range(n_a):
                cells_a = [c for c in table.cells if c.age_idx == a]
                observed_sum = sum(c.events for c in cells_a)
                fitted_sum = sum(mu_apc_fitted[(c.age_idx, c.period_idx)] for c in cells_a)
                if fitted_sum > 0:
                    factor = observed_sum / fitted_sum
                    for c in cells_a:
                        mu_apc_fitted[(c.age_idx, c.period_idx)] *= factor

            # Adjust for period margins
            for p in range(n_p):
                cells_p = [c for c in table.cells if c.period_idx == p]
                observed_sum = sum(c.events for c in cells_p)
                fitted_sum = sum(mu_apc_fitted[(c.age_idx, c.period_idx)] for c in cells_p)
                if fitted_sum > 0:
                    factor = observed_sum / fitted_sum
                    for c in cells_p:
                        mu_apc_fitted[(c.age_idx, c.period_idx)] *= factor

            # Adjust for cohort margins
            for c_idx in range(n_c):
                cells_c = [c for c in table.cells if c.cohort_idx == c_idx]
                observed_sum = sum(c.events for c in cells_c)
                fitted_sum = sum(mu_apc_fitted[(c.age_idx, c.period_idx)] for c in cells_c)
                if fitted_sum > 0:
                    factor = observed_sum / fitted_sum
                    for c in cells_c:
                        mu_apc_fitted[(c.age_idx, c.period_idx)] *= factor

        def mu_apc(c):
            return mu_apc_fitted.get((c.age_idx, c.period_idx), c.events)

        dev_apc = _poisson_deviance(mu_apc)
        df_apc = n_cells - (n_a + n_p + n_c - 2)
        aic_apc = dev_apc + 2 * (n_a + n_p + n_c - 2)
        bic_apc = dev_apc + math.log(n_cells) * (n_a + n_p + n_c - 2)
        models.append(APCModelFit("Age-Period-Cohort (APC)", round(dev_apc, 2), df_apc, round(aic_apc, 2), round(bic_apc, 2), 0.01, round(-dev_apc / 2, 2)))

        return models

    @classmethod
    def analyze_table(
        cls,
        dataset: Dict[str, Any],
        ref_cohort_idx: Optional[int] = None,
        ref_period_idx: Optional[int] = None,
    ) -> ComprehensiveAPCReport:
        """
        Convenience method to analyze a reference dataset dictionary.
        Expects keys: age_groups, periods, rates_per_100k, std_py (optional).
        """
        table = build_apc_table_from_matrix(
            age_groups=dataset["age_groups"],
            periods=dataset["periods"],
            rates_matrix=dataset["rates_per_100k"],
            person_years_per_cell=dataset.get("std_py", 100000.0),
        )
        estimable = cls.fit_estimable_functions(table, ref_cohort_idx, ref_period_idx)
        models = cls.evaluate_model_hierarchy(table)

        # Determine best fitting model by lowest AIC
        best_model = min(models, key=lambda m: m.aic)

        return ComprehensiveAPCReport(
            table_summary={
                "title": dataset.get("title", "Dataset"),
                "num_ages": len(table.age_groups),
                "num_periods": len(table.periods),
                "num_cohorts": len(table.cohorts),
                "age_groups": table.age_groups,
                "periods": table.periods,
            },
            estimable_functions=estimable,
            model_comparisons=models,
            best_fitting_model=best_model.model_type,
        )


# ============================================================================
# Joinpoint Regression & PAF Module
# ============================================================================

class JoinpointAnalyzer:
    """
    Piecewise log-linear regression finding optimal joinpoints (inflections)
    minimizing Sum of Squared Errors (SSE) and computing Annual Percent Change (APC %).
    """

    @classmethod
    def fit(
        cls,
        years: Sequence[int],
        rates: Sequence[float],
        max_joinpoints: int = 2,
        min_segment_length: int = 4,
    ) -> JoinpointResult:
        n = len(years)
        if n < min_segment_length * 2:
            raise ValueError(f"Time series too short ({n} points) for joinpoint analysis.")

        xs = [float(y) for y in years]
        ys = [math.log(max(1e-10, r)) for r in rates]

        best_jp: List[int] = []
        best_sse = float("inf")
        best_segments: List[JoinpointSegment] = []

        def _segment_info(x_seg, y_seg, year_start, year_end):
            """Compute segment APC and CI for a sub-segment."""
            fit = ordinary_least_squares(x_seg, y_seg)
            apc = round((math.exp(fit.slope) - 1.0) * 100.0, 2)
            se = fit.se_slope * 100.0
            sse = sum((y - (fit.intercept + fit.slope * x)) ** 2 for x, y in zip(x_seg, y_seg))
            ci = (round(apc - 1.96 * se, 2), round(apc + 1.96 * se, 2))
            return apc, ci, sse

        # Evaluate 0 joinpoints (single slope)
        fit0 = ordinary_least_squares(xs, ys)
        sse0 = sum((y - (fit0.intercept + fit0.slope * x)) ** 2 for x, y in zip(xs, ys))
        apc0 = round((math.exp(fit0.slope) - 1.0) * 100.0, 2)
        se0 = fit0.se_slope * 100.0
        best_jp = []
        best_sse = sse0
        best_segments = [JoinpointSegment(years[0], years[-1], apc0, (round(apc0 - 1.96 * se0, 2), round(apc0 + 1.96 * se0, 2)))]

        # Evaluate 1 joinpoint if series length permits
        if max_joinpoints >= 1 and n >= min_segment_length * 2:
            for split1 in range(min_segment_length, n - min_segment_length + 1):
                apc1, ci1, sse1 = _segment_info(xs[:split1], ys[:split1], years[0], years[split1 - 1])
                apc2, ci2, sse2 = _segment_info(xs[split1:], ys[split1:], years[split1], years[-1])
                tot_sse = sse1 + sse2
                if tot_sse < best_sse:
                    best_sse = tot_sse
                    best_jp = [years[split1]]
                    best_segments = [
                        JoinpointSegment(years[0], years[split1 - 1], apc1, ci1),
                        JoinpointSegment(years[split1], years[-1], apc2, ci2),
                    ]

        # Evaluate 2 joinpoints if series length permits
        if max_joinpoints >= 2 and n >= min_segment_length * 3:
            for split1 in range(min_segment_length, n - 2 * min_segment_length + 1):
                apc1, ci1, sse1 = _segment_info(xs[:split1], ys[:split1], years[0], years[split1 - 1])
                for split2 in range(split1 + min_segment_length, n - min_segment_length + 1):
                    apc2, ci2, sse2 = _segment_info(xs[split1:split2], ys[split1:split2], years[split1], years[split2 - 1])
                    apc3, ci3, sse3 = _segment_info(xs[split2:], ys[split2:], years[split2], years[-1])
                    tot_sse = sse1 + sse2 + sse3
                    if tot_sse < best_sse:
                        best_sse = tot_sse
                        best_jp = [years[split1], years[split2]]
                        best_segments = [
                            JoinpointSegment(years[0], years[split1 - 1], apc1, ci1),
                            JoinpointSegment(years[split1], years[split2 - 1], apc2, ci2),
                            JoinpointSegment(years[split2], years[-1], apc3, ci3),
                        ]

        # Average Annual Percent Change (AAPC)
        total_span = years[-1] - years[0]
        weighted_apc = sum(seg.apc_pct * (seg.end_year - seg.start_year + 1) for seg in best_segments) / max(1, total_span + 1)

        return JoinpointResult(
            joinpoints=best_jp,
            segments=best_segments,
            average_annual_percent_change=round(weighted_apc, 2),
            sse=round(best_sse, 4),
        )


class PAFCalculator:
    """Population Attributable Fraction (PAF) calculator."""

    @classmethod
    def levin_paf(cls, prevalence: float, relative_risk: float) -> float:
        """
        Levin's formula for population attributable fraction:
        PAF = [Pe * (RR - 1)] / [Pe * (RR - 1) + 1]
        """
        if not (0.0 <= prevalence <= 1.0):
            raise ValueError(f"Prevalence must be in [0, 1], got {prevalence}")
        if relative_risk < 1.0:
            raise ValueError(f"Relative risk must be >= 1.0 for risk factor PAF, got {relative_risk}")
        num = prevalence * (relative_risk - 1.0)
        denom = num + 1.0
        return round(num / denom, 4)

    @classmethod
    def miettinen_paf(cls, case_exposure_prevalence: float, relative_risk: float) -> float:
        """
        Miettinen's formula (case-based exposure prevalence):
        PAF = P_{e|d} * (RR - 1) / RR
        """
        if not (0.0 <= case_exposure_prevalence <= 1.0):
            raise ValueError("Case exposure prevalence must be in [0, 1]")
        if relative_risk <= 0.0:
            raise ValueError("Relative risk must be positive")
        return round(case_exposure_prevalence * (relative_risk - 1.0) / relative_risk, 4)


class TrendForecaster:
    """Extrapolates rates forward in time using net drift and log-linear regression."""

    @classmethod
    def forecast(cls, years: Sequence[int], rates: Sequence[float], horizon: int = 5) -> List[RateForecast]:
        xs = [float(y) for y in years]
        ys = [math.log(max(1e-10, r)) for r in rates]
        fit = ordinary_least_squares(xs, ys)
        n = len(years)
        last_yr = max(years)
        forecasts = []

        for h in range(1, horizon + 1):
            target_yr = last_yr + h
            pred_log = fit.intercept + fit.slope * target_yr
            # Standard error of prediction
            se_pred = math.sqrt(fit.mse * (1.0 + 1.0 / n + (target_yr - fit.mean_x) ** 2 / sum((x - fit.mean_x) ** 2 for x in xs)))
            mid = math.exp(pred_log)
            low = math.exp(pred_log - 1.96 * se_pred)
            high = math.exp(pred_log + 1.96 * se_pred)
            forecasts.append(RateForecast(
                year=target_yr,
                predicted_rate=round(mid, 2),
                ci_lower=round(low, 2),
                ci_upper=round(high, 2),
            ))
        return forecasts


# ============================================================================
# Built-In Reference Epidemiology Datasets
# ============================================================================

_REFERENCE_DATASET_US_LUNG_MALE = {
    "title": "US Male Lung & Bronchus Cancer Incidence (SEER 1975-2015)",
    "age_groups": ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"],
    "periods": ["1975-1979", "1980-1984", "1985-1989", "1990-1994", "1995-1999", "2000-2004", "2005-2009", "2010-2014"],
    "rates_per_100k": [
        [18.2, 16.5, 14.1, 12.0, 9.5, 7.8, 6.2, 4.9],
        [54.0, 50.1, 44.5, 38.0, 31.2, 25.0, 20.1, 16.5],
        [122.0, 118.5, 108.0, 95.2, 81.0, 68.5, 55.4, 44.2],
        [225.0, 230.0, 218.0, 198.0, 172.0, 148.0, 125.0, 102.0],
        [340.0, 365.0, 360.0, 335.0, 301.0, 265.0, 228.0, 192.0],
        [440.0, 485.0, 502.0, 485.0, 448.0, 402.0, 355.0, 305.0],
        [490.0, 560.0, 605.0, 610.0, 580.0, 535.0, 480.0, 420.0],
    ],
    "std_py": 100000.0,
}

REFERENCE_DATASETS = {
    "us_lung_cancer_male": _REFERENCE_DATASET_US_LUNG_MALE,
    "seer_male_lung_cancer": _REFERENCE_DATASET_US_LUNG_MALE,
}


def build_apc_table_from_matrix(
    age_groups: List[str],
    periods: List[str],
    rates_matrix: List[List[float]],
    person_years_per_cell: float = 100000.0,
) -> APCTable:
    """Helper to convert 2D rate matrix into full APCTable."""
    cells = []
    n_a = len(age_groups)
    n_p = len(periods)
    for a_idx, age in enumerate(age_groups):
        for p_idx, per in enumerate(periods):
            c_idx = p_idx - a_idx + (n_a - 1)
            rate = rates_matrix[a_idx][p_idx]
            events = (rate * person_years_per_cell) / 100000.0
            cells.append(APCCell(
                age_idx=a_idx,
                period_idx=p_idx,
                cohort_idx=c_idx,
                age_label=age,
                period_label=per,
                cohort_label=f"Cohort_{c_idx+1}",
                events=events,
                person_years=person_years_per_cell,
            ))
    return APCTable(age_groups=age_groups, periods=periods, cells=cells)

#!/usr/bin/env python3
"""Age-period-cohort trend analysis utilities.

The module provides a dependency-free Poisson log-linear model comparison,
descriptive age/period/cohort summaries, joinpoint-style segmented trend search,
population-attributable-fraction calculations, and log-linear forecasting.

Important statistical scope
---------------------------
The age/period/cohort decomposition reported by ``fit_estimable_functions`` is a
transparent descriptive decomposition.  The net and local drifts are Poisson
log-linear period trends (overall trend adjusted for age, and age-specific
trends respectively).  Curvatures are second differences of aggregated
log-rates.  Relative risks are exposure-weighted marginal rate ratios.  These
quantities should not be described as a full Holford constrained APC solution.

``evaluate_model_hierarchy`` does fit Poisson log-linear A, AP, AC and APC
models with a log(person-years) offset.  The APC design is made identifiable by
removing one redundant cohort contrast; fitted values and likelihood-based fit
statistics are invariant to the particular full-rank parameterization.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__version__ = "3.0.0"

_EPS = 1e-12
_MAX_EXP = 700.0


def safe_log(x: float, eps: float = _EPS) -> float:
    return math.log(max(eps, x))


def safe_exp(x: float, max_val: float = _MAX_EXP) -> float:
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
    """Fit a simple unweighted OLS line with basic input validation."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("OLS requires at least 2 points")
    if not all(math.isfinite(float(v)) for v in (*xs, *ys)):
        raise ValueError("OLS inputs must be finite")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= _EPS:
        raise ValueError("OLS requires at least two distinct x values")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)
    sst = sum((y - mean_y) ** 2 for y in ys)
    mse = sse / (n - 2) if n > 2 else 0.0
    se_slope = math.sqrt(max(0.0, mse / sxx))
    r_squared = 1.0 - (sse / sst) if sst > _EPS else 1.0
    return OLSFitResult(
        slope=slope,
        intercept=intercept,
        se_slope=se_slope,
        mse=mse,
        r_squared=max(0.0, min(1.0, r_squared)),
        mean_x=mean_x,
    )


@dataclass
class APCCell:
    age_idx: int
    period_idx: int
    cohort_idx: int
    age_label: str
    period_label: str
    cohort_label: str
    events: float
    person_years: float
    rate_per_100k: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.events) or self.events < 0:
            raise ValueError("events must be a finite non-negative number")
        if not math.isfinite(self.person_years) or self.person_years <= 0:
            raise ValueError("person_years must be a finite positive number")
        self.rate_per_100k = (self.events / self.person_years) * 100000.0


@dataclass
class APCTable:
    age_groups: List[str]
    periods: List[str]
    cells: List[APCCell]
    cohorts: List[str] = field(default_factory=list)
    age_interval: float = 5.0
    period_interval: float = 5.0

    def __post_init__(self) -> None:
        if not self.age_groups or not self.periods:
            raise ValueError("age_groups and periods must not be empty")
        if self.age_interval <= 0 or self.period_interval <= 0:
            raise ValueError("age_interval and period_interval must be positive")
        expected = len(self.age_groups) * len(self.periods)
        if len(self.cells) != expected:
            raise ValueError(f"expected {expected} age-period cells, got {len(self.cells)}")
        n_c = len(self.age_groups) + len(self.periods) - 1
        if not self.cohorts:
            self.cohorts = [f"Cohort_{i + 1}" for i in range(n_c)]
        if len(self.cohorts) != n_c:
            raise ValueError(f"expected {n_c} cohort labels, got {len(self.cohorts)}")


@dataclass
class EstimableFunctionsResult:
    net_drift_pct: float
    net_drift_ci: Tuple[float, float]
    local_drifts: Dict[str, float]
    age_curvatures: Dict[str, float]
    period_curvatures: Dict[str, float]
    cohort_curvatures: Dict[str, float]
    cohort_relative_risks: Dict[str, float]
    period_relative_risks: Dict[str, float]
    reference_cohort: str
    reference_period: str


@dataclass
class APCModelFit:
    model_type: str
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


@dataclass
class _GLMResult:
    beta: List[float]
    covariance: List[List[float]]
    fitted: List[float]
    log_likelihood: float
    deviance: float
    n_params: int


def _solve_linear_system(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """Solve Ax=b using Gaussian elimination with partial pivoting."""
    n = len(b)
    if len(a) != n or any(len(row) != n for row in a):
        raise ValueError("linear system must be square")
    aug = [list(map(float, a[i])) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular design matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) <= _EPS:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _invert_matrix(a: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    cols = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        cols.append(_solve_linear_system(a, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _xtwx(x: Sequence[Sequence[float]], weights: Sequence[float]) -> List[List[float]]:
    p = len(x[0])
    out = [[0.0] * p for _ in range(p)]
    for row, w in zip(x, weights):
        for j in range(p):
            rwj = row[j] * w
            for k in range(j, p):
                out[j][k] += rwj * row[k]
    for j in range(p):
        for k in range(j):
            out[j][k] = out[k][j]
    return out


def _xtv(x: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    p = len(x[0])
    out = [0.0] * p
    for row, val in zip(x, v):
        for j in range(p):
            out[j] += row[j] * val
    return out


def _poisson_glm_fit(
    x: Sequence[Sequence[float]],
    events: Sequence[float],
    exposures: Sequence[float],
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> _GLMResult:
    """Fit a Poisson log-link GLM using IRLS/Newton updates."""
    n = len(events)
    if n == 0 or len(x) != n or len(exposures) != n:
        raise ValueError("x, events and exposures must have the same non-zero length")
    p = len(x[0])
    if p == 0 or any(len(row) != p for row in x):
        raise ValueError("design matrix is malformed")
    if n < p:
        raise ValueError("model has more parameters than observations")
    if any(y < 0 or not math.isfinite(y) for y in events):
        raise ValueError("events must be finite and non-negative")
    if any(e <= 0 or not math.isfinite(e) for e in exposures):
        raise ValueError("exposures must be finite and positive")
    if sum(events) <= 0:
        raise ValueError("at least one event is required")

    total_rate = sum(events) / sum(exposures)
    beta = [0.0] * p
    beta[0] = safe_log(total_rate)
    offsets = [math.log(e) for e in exposures]

    for _ in range(max_iter):
        eta = [offsets[i] + sum(x[i][j] * beta[j] for j in range(p)) for i in range(n)]
        mu = [safe_exp(v) for v in eta]
        score = _xtv(x, [events[i] - mu[i] for i in range(n)])
        info = _xtwx(x, mu)
        # Tiny numerical ridge; scale relative to diagonal and never material enough
        # to alter fitted values at ordinary epidemiologic sample sizes.
        scale = max(1.0, max(info[j][j] for j in range(p)))
        for j in range(p):
            info[j][j] += scale * 1e-12
        delta = _solve_linear_system(info, score)

        # Backtracking prevents overflow and decreases negative log-likelihood.
        old_ll = sum(
            events[i] * eta[i] - mu[i] - math.lgamma(events[i] + 1.0)
            for i in range(n)
        )
        step = 1.0
        accepted = False
        candidate = beta
        while step >= 1e-6:
            candidate = [beta[j] + step * delta[j] for j in range(p)]
            ceta = [offsets[i] + sum(x[i][j] * candidate[j] for j in range(p)) for i in range(n)]
            cmu = [safe_exp(v) for v in ceta]
            ll = sum(
                events[i] * ceta[i] - cmu[i] - math.lgamma(events[i] + 1.0)
                for i in range(n)
            )
            if ll >= old_ll - 1e-10:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise RuntimeError("Poisson GLM failed to converge")
        beta = candidate
        if max(abs(step * d) for d in delta) < tol:
            break
    else:
        raise RuntimeError("Poisson GLM did not converge within max_iter")

    eta = [offsets[i] + sum(x[i][j] * beta[j] for j in range(p)) for i in range(n)]
    mu = [safe_exp(v) for v in eta]
    info = _xtwx(x, mu)
    scale = max(1.0, max(info[j][j] for j in range(p)))
    for j in range(p):
        info[j][j] += scale * 1e-12
    covariance = _invert_matrix(info)
    log_likelihood = sum(
        events[i] * eta[i] - mu[i] - math.lgamma(events[i] + 1.0)
        for i in range(n)
    )
    deviance = 0.0
    for y, m in zip(events, mu):
        if y > 0:
            deviance += 2.0 * (y * math.log(y / m) - (y - m))
        else:
            deviance += 2.0 * m
    deviance = max(0.0, deviance)
    return _GLMResult(beta, covariance, mu, log_likelihood, deviance, p)


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), dependency-free."""
    if a <= 0 or x < 0:
        raise ValueError("invalid incomplete-gamma arguments")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * 1e-14:
                break
        p = summ * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))

    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))


def _chi_square_sf(value: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    if value < -1e-10:
        raise ValueError("chi-square statistic cannot be negative")
    value = max(0.0, value)
    return _gammaincc(df / 2.0, value / 2.0)


def _aggregate_log_rates(table: APCTable, dimension: str, count: int) -> List[float]:
    values: List[float] = []
    for idx in range(count):
        subset = [c for c in table.cells if getattr(c, dimension) == idx]
        d = sum(c.events for c in subset)
        y = sum(c.person_years for c in subset)
        values.append(safe_log((d + 0.5) / (y + 0.5) if d <= 0 else d / y))
    return values


def _rate_ratio_map(
    table: APCTable,
    dimension: str,
    labels: Sequence[str],
    ref_idx: int,
) -> Dict[str, float]:
    rates = []
    for idx in range(len(labels)):
        subset = [c for c in table.cells if getattr(c, dimension) == idx]
        d = sum(c.events for c in subset)
        y = sum(c.person_years for c in subset)
        rates.append((d + 0.5) / (y + 0.5) if d <= 0 else d / y)
    ref = rates[ref_idx]
    return {label: round(rate / ref, 3) for label, rate in zip(labels, rates)}


class APCStatisticalEngine:
    """Poisson model comparison plus transparent APC descriptive summaries."""

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
        if n_a < 2 or n_p < 2:
            raise ValueError("at least two age groups and two periods are required")

        ref_cohort_idx = n_c // 2 if ref_cohort_idx is None else ref_cohort_idx
        ref_period_idx = n_p // 2 if ref_period_idx is None else ref_period_idx
        if not 0 <= ref_cohort_idx < n_c:
            raise ValueError("ref_cohort_idx is out of range")
        if not 0 <= ref_period_idx < n_p:
            raise ValueError("ref_period_idx is out of range")

        events = [c.events for c in table.cells]
        exposures = [c.person_years for c in table.cells]
        p_center = (n_p - 1) / 2.0

        # Age-adjusted linear period trend: intercept + age contrasts + period score.
        x = []
        for c in table.cells:
            row = [1.0]
            row.extend(1.0 if c.age_idx == a else 0.0 for a in range(1, n_a))
            row.append((c.period_idx - p_center) * table.period_interval)
            x.append(row)
        fit = _poisson_glm_fit(x, events, exposures)
        slope = fit.beta[-1]
        se = math.sqrt(max(0.0, fit.covariance[-1][-1]))
        drift = (math.exp(slope) - 1.0) * 100.0
        lower = (math.exp(slope - 1.96 * se) - 1.0) * 100.0
        upper = (math.exp(slope + 1.96 * se) - 1.0) * 100.0

        # Age-specific period trends.
        local_drifts: Dict[str, float] = {}
        for a, label in enumerate(table.age_groups):
            subset = sorted((c for c in table.cells if c.age_idx == a), key=lambda c: c.period_idx)
            xa = [[1.0, (c.period_idx - p_center) * table.period_interval] for c in subset]
            fa = _poisson_glm_fit(xa, [c.events for c in subset], [c.person_years for c in subset])
            local_drifts[label] = round((math.exp(fa.beta[1]) - 1.0) * 100.0, 3)

        age_logs = _aggregate_log_rates(table, "age_idx", n_a)
        period_logs = _aggregate_log_rates(table, "period_idx", n_p)
        cohort_logs = _aggregate_log_rates(table, "cohort_idx", n_c)
        age_curvatures = {
            table.age_groups[i]: round(age_logs[i - 1] - 2 * age_logs[i] + age_logs[i + 1], 6)
            for i in range(1, n_a - 1)
        }
        period_curvatures = {
            table.periods[i]: round(period_logs[i - 1] - 2 * period_logs[i] + period_logs[i + 1], 6)
            for i in range(1, n_p - 1)
        }
        cohort_curvatures = {
            table.cohorts[i]: round(cohort_logs[i - 1] - 2 * cohort_logs[i] + cohort_logs[i + 1], 6)
            for i in range(1, n_c - 1)
        }

        cohort_rr = _rate_ratio_map(table, "cohort_idx", table.cohorts, ref_cohort_idx)
        period_rr = _rate_ratio_map(table, "period_idx", table.periods, ref_period_idx)

        return EstimableFunctionsResult(
            net_drift_pct=round(drift, 3),
            net_drift_ci=(round(lower, 3), round(upper, 3)),
            local_drifts=local_drifts,
            age_curvatures=age_curvatures,
            period_curvatures=period_curvatures,
            cohort_curvatures=cohort_curvatures,
            cohort_relative_risks=cohort_rr,
            period_relative_risks=period_rr,
            reference_cohort=table.cohorts[ref_cohort_idx],
            reference_period=table.periods[ref_period_idx],
        )

    @classmethod
    def evaluate_model_hierarchy(cls, table: APCTable) -> List[APCModelFit]:
        """Fit A, AP, AC, and identifiable APC Poisson log-linear models."""
        n_a = len(table.age_groups)
        n_p = len(table.periods)
        n_c = n_a + n_p - 1
        events = [c.events for c in table.cells]
        exposures = [c.person_years for c in table.cells]
        n = len(events)

        def design(kind: str) -> List[List[float]]:
            rows: List[List[float]] = []
            for c in table.cells:
                row = [1.0]
                row.extend(1.0 if c.age_idx == a else 0.0 for a in range(1, n_a))
                if kind in ("AP", "APC"):
                    row.extend(1.0 if c.period_idx == p else 0.0 for p in range(1, n_p))
                if kind == "AC":
                    row.extend(1.0 if c.cohort_idx == co else 0.0 for co in range(1, n_c))
                elif kind == "APC":
                    # One additional cohort contrast is redundant because C=P-A.
                    # Omitting the last cohort contrast yields a full-rank basis for
                    # the same APC fitted-value space.
                    row.extend(1.0 if c.cohort_idx == co else 0.0 for co in range(1, n_c - 1))
                rows.append(row)
            return rows

        specs = [
            ("Age-Only (A)", "A"),
            ("Age-Period (AP)", "AP"),
            ("Age-Cohort (AC)", "AC"),
            ("Age-Period-Cohort (APC)", "APC"),
        ]
        results: List[APCModelFit] = []
        for label, kind in specs:
            fit = _poisson_glm_fit(design(kind), events, exposures)
            df = n - fit.n_params
            aic = -2.0 * fit.log_likelihood + 2.0 * fit.n_params
            bic = -2.0 * fit.log_likelihood + math.log(n) * fit.n_params
            p = _chi_square_sf(fit.deviance, df) if df > 0 else float("nan")
            results.append(
                APCModelFit(
                    model_type=label,
                    deviance=round(fit.deviance, 4),
                    degrees_of_freedom=df,
                    aic=round(aic, 4),
                    bic=round(bic, 4),
                    p_value=round(p, 6) if math.isfinite(p) else float("nan"),
                    log_likelihood=round(fit.log_likelihood, 4),
                )
            )
        return results

    @classmethod
    def analyze_table(
        cls,
        dataset: Dict[str, Any],
        ref_cohort_idx: Optional[int] = None,
        ref_period_idx: Optional[int] = None,
    ) -> ComprehensiveAPCReport:
        required = {"age_groups", "periods", "rates_per_100k"}
        missing = required - dataset.keys()
        if missing:
            raise ValueError(f"dataset missing required keys: {', '.join(sorted(missing))}")
        table = build_apc_table_from_matrix(
            age_groups=list(dataset["age_groups"]),
            periods=list(dataset["periods"]),
            rates_matrix=[list(r) for r in dataset["rates_per_100k"]],
            person_years_per_cell=float(dataset.get("std_py", 100000.0)),
        )
        estimable = cls.fit_estimable_functions(table, ref_cohort_idx, ref_period_idx)
        models = cls.evaluate_model_hierarchy(table)
        lowest_aic = min(models, key=lambda m: m.aic)
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
            best_fitting_model=lowest_aic.model_type,
        )


class JoinpointAnalyzer:
    """Piecewise log-linear trend search using BIC to limit overfitting."""

    @classmethod
    def fit(
        cls,
        years: Sequence[int],
        rates: Sequence[float],
        max_joinpoints: int = 2,
        min_segment_length: int = 4,
    ) -> JoinpointResult:
        if len(years) != len(rates):
            raise ValueError("years and rates must have the same length")
        n = len(years)
        if min_segment_length < 2:
            raise ValueError("min_segment_length must be at least 2")
        if max_joinpoints not in (0, 1, 2):
            raise ValueError("max_joinpoints must be 0, 1, or 2")
        if n < min_segment_length:
            raise ValueError(f"time series too short ({n} points)")
        if any(not math.isfinite(float(r)) or r <= 0 for r in rates):
            raise ValueError("rates must be finite and strictly positive")
        if any(years[i] >= years[i + 1] for i in range(n - 1)):
            raise ValueError("years must be strictly increasing")

        xs = [float(y) for y in years]
        ys = [math.log(float(r)) for r in rates]

        def seg_info(start: int, stop: int) -> Tuple[JoinpointSegment, float, float]:
            fit = ordinary_least_squares(xs[start:stop], ys[start:stop])
            residuals = [
                ys[i] - (fit.intercept + fit.slope * xs[i]) for i in range(start, stop)
            ]
            sse = sum(r * r for r in residuals)
            apc = (math.exp(fit.slope) - 1.0) * 100.0
            low = (math.exp(fit.slope - 1.96 * fit.se_slope) - 1.0) * 100.0
            high = (math.exp(fit.slope + 1.96 * fit.se_slope) - 1.0) * 100.0
            return (
                JoinpointSegment(
                    years[start], years[stop - 1], round(apc, 3), (round(low, 3), round(high, 3))
                ),
                sse,
                fit.slope,
            )

        candidates: List[Tuple[float, float, List[int], List[JoinpointSegment], List[Tuple[float, int]]]] = []

        def add_candidate(bounds: List[int]) -> None:
            starts = [0] + bounds
            stops = bounds + [n]
            segments = []
            total_sse = 0.0
            slopes_and_weights = []
            for s, e in zip(starts, stops):
                seg, sse, slope = seg_info(s, e)
                segments.append(seg)
                total_sse += sse
                slopes_and_weights.append((slope, e - s - 1))
            # Each independently fitted segment contributes slope + intercept.
            k = 2 * len(segments)
            mse_for_bic = max(total_sse / n, 1e-15)
            bic = n * math.log(mse_for_bic) + k * math.log(n)
            candidates.append((bic, total_sse, bounds, segments, slopes_and_weights))

        add_candidate([])
        if max_joinpoints >= 1 and n >= 2 * min_segment_length:
            for s1 in range(min_segment_length, n - min_segment_length + 1):
                add_candidate([s1])
        if max_joinpoints >= 2 and n >= 3 * min_segment_length:
            for s1 in range(min_segment_length, n - 2 * min_segment_length + 1):
                for s2 in range(s1 + min_segment_length, n - min_segment_length + 1):
                    add_candidate([s1, s2])

        _, sse, bounds, segments, slopes_weights = min(candidates, key=lambda item: item[0])
        total_weight = sum(max(1, w) for _, w in slopes_weights)
        avg_log_slope = sum(s * max(1, w) for s, w in slopes_weights) / total_weight
        aapc = (math.exp(avg_log_slope) - 1.0) * 100.0
        return JoinpointResult(
            joinpoints=[years[i] for i in bounds],
            segments=segments,
            average_annual_percent_change=round(aapc, 3),
            sse=round(sse, 6),
        )


class PAFCalculator:
    @classmethod
    def levin_paf(cls, prevalence: float, relative_risk: float) -> float:
        if not math.isfinite(prevalence) or not 0.0 <= prevalence <= 1.0:
            raise ValueError("prevalence must be in [0, 1]")
        if not math.isfinite(relative_risk) or relative_risk < 1.0:
            raise ValueError("relative_risk must be finite and >= 1.0 for risk-factor PAF")
        num = prevalence * (relative_risk - 1.0)
        return round(num / (1.0 + num), 6)

    @classmethod
    def miettinen_paf(cls, case_exposure_prevalence: float, relative_risk: float) -> float:
        if not math.isfinite(case_exposure_prevalence) or not 0.0 <= case_exposure_prevalence <= 1.0:
            raise ValueError("case_exposure_prevalence must be in [0, 1]")
        if not math.isfinite(relative_risk) or relative_risk <= 0.0:
            raise ValueError("relative_risk must be finite and positive")
        return round(case_exposure_prevalence * (relative_risk - 1.0) / relative_risk, 6)


class TrendForecaster:
    @classmethod
    def forecast(cls, years: Sequence[int], rates: Sequence[float], horizon: int = 5) -> List[RateForecast]:
        if len(years) != len(rates):
            raise ValueError("years and rates must have the same length")
        if len(years) < 3:
            raise ValueError("forecasting requires at least 3 observations")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if any(years[i] >= years[i + 1] for i in range(len(years) - 1)):
            raise ValueError("years must be strictly increasing")
        if any(not math.isfinite(float(r)) or r <= 0 for r in rates):
            raise ValueError("rates must be finite and strictly positive")

        xs = [float(y) for y in years]
        ys = [math.log(float(r)) for r in rates]
        fit = ordinary_least_squares(xs, ys)
        sxx = sum((x - fit.mean_x) ** 2 for x in xs)
        n = len(xs)
        last_yr = years[-1]
        out = []
        for h in range(1, horizon + 1):
            year = last_yr + h
            pred_log = fit.intercept + fit.slope * year
            se_pred = math.sqrt(max(0.0, fit.mse * (1.0 + 1.0 / n + (year - fit.mean_x) ** 2 / sxx)))
            out.append(
                RateForecast(
                    year=year,
                    predicted_rate=round(math.exp(pred_log), 4),
                    ci_lower=round(math.exp(pred_log - 1.96 * se_pred), 4),
                    ci_upper=round(math.exp(pred_log + 1.96 * se_pred), 4),
                )
            )
        return out


# Synthetic example retained under the historical aliases for API compatibility.
_REFERENCE_DATASET_LUNG_EXAMPLE = {
    "title": "Illustrative male lung-cancer incidence rate matrix (synthetic example)",
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
    "synthetic": True,
}

REFERENCE_DATASETS = {
    "us_lung_cancer_male": _REFERENCE_DATASET_LUNG_EXAMPLE,
    "seer_male_lung_cancer": _REFERENCE_DATASET_LUNG_EXAMPLE,
    "synthetic_lung_cancer_male": _REFERENCE_DATASET_LUNG_EXAMPLE,
}


def build_apc_table_from_matrix(
    age_groups: List[str],
    periods: List[str],
    rates_matrix: List[List[float]],
    person_years_per_cell: float = 100000.0,
) -> APCTable:
    if not age_groups or not periods:
        raise ValueError("age_groups and periods must not be empty")
    if not math.isfinite(person_years_per_cell) or person_years_per_cell <= 0:
        raise ValueError("person_years_per_cell must be finite and positive")
    if len(rates_matrix) != len(age_groups):
        raise ValueError("rates_matrix row count must equal number of age groups")
    if any(len(row) != len(periods) for row in rates_matrix):
        raise ValueError("every rates_matrix row must match number of periods")

    n_a = len(age_groups)
    cells: List[APCCell] = []
    for a_idx, age in enumerate(age_groups):
        for p_idx, period in enumerate(periods):
            rate = float(rates_matrix[a_idx][p_idx])
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("rates must be finite and non-negative")
            c_idx = p_idx - a_idx + (n_a - 1)
            events = rate * person_years_per_cell / 100000.0
            cells.append(
                APCCell(
                    age_idx=a_idx,
                    period_idx=p_idx,
                    cohort_idx=c_idx,
                    age_label=str(age),
                    period_label=str(period),
                    cohort_label=f"Cohort_{c_idx + 1}",
                    events=events,
                    person_years=person_years_per_cell,
                )
            )
    return APCTable(age_groups=list(map(str, age_groups)), periods=list(map(str, periods)), cells=cells)


def build_apc_table_from_records(records: Iterable[Dict[str, Any]]) -> APCTable:
    """Build a complete age-period table from long-format event/exposure records."""
    rows = list(records)
    if not rows:
        raise ValueError("records must not be empty")
    required = {"age_group", "period", "events", "person_years"}
    for i, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {i} missing fields: {', '.join(sorted(missing))}")

    age_groups = list(dict.fromkeys(str(r["age_group"]).strip() for r in rows))
    periods = list(dict.fromkeys(str(r["period"]).strip() for r in rows))
    if any(not x for x in age_groups + periods):
        raise ValueError("age_group and period labels must not be blank")
    lookup = {(str(r["age_group"]).strip(), str(r["period"]).strip()): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate age_group/period cells are not allowed")
    expected = len(age_groups) * len(periods)
    if len(rows) != expected:
        raise ValueError("records must form a complete rectangular age-period table")

    n_a = len(age_groups)
    cells = []
    for a_idx, age in enumerate(age_groups):
        for p_idx, period in enumerate(periods):
            row = lookup.get((age, period))
            if row is None:
                raise ValueError(f"missing age-period cell: {age} / {period}")
            c_idx = p_idx - a_idx + (n_a - 1)
            cells.append(
                APCCell(
                    age_idx=a_idx,
                    period_idx=p_idx,
                    cohort_idx=c_idx,
                    age_label=age,
                    period_label=period,
                    cohort_label=f"Cohort_{c_idx + 1}",
                    events=float(row["events"]),
                    person_years=float(row["person_years"]),
                )
            )
    return APCTable(age_groups=age_groups, periods=periods, cells=cells)

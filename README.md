# Age-Period-Cohort (APC) Epidemiological Analyzer

A pure Python epidemiological and demographic statistical engine implementing:
- Poisson log-linear Age-Period-Cohort modeling (Holford 1983, Clayton & Schifflers 1987).
- Invariant estimable functions resolving the fundamental APC identification problem ($C = P - A$): Net Drift, Local Drifts, and second-order Curvatures.
- Cohort and Period Relative Risks (RR) referenced to arbitrary reference cohorts/periods.
- Hierarchical model comparison: Age-only (A), Age-Period (AP), Age-Cohort (AC), and Age-Period-Cohort (APC).
- Goodness-of-fit assessment: Deviance ($G^2$), Degrees of Freedom, AIC, and BIC.
- Joinpoint segmented regression for inflection detection and Annual Percent Change (APC %).
- Population Attributable Fraction (PAF) calculations via Levin and Miettinen formulations.
- Trend forecasting and rate extrapolation with 95% confidence intervals.

Requires Python standard library only (zero external runtime dependencies).

---

## Features

- **Identification Problem Resolution:** Solves the linear dependency ($P - A - C = 0$) using Holford's orthogonal decomposition into invariant curvature components and estimable linear drifts.
- **Net Drift & Local Drifts:** Calculates overall log-linear drift across time and cohort along with age-specific local drifts.
- **Model Hierarchy & Fit Selection:** Automatically fits nested sub-models (A, AP, AC, APC) and selects best fit by AIC.
- **Piecewise Joinpoint Regression:** Detects trend inflections (grid search over valid split points) and reports Segment Annual Percent Change (APC) and Average Annual Percent Change (AAPC). Supports up to 2 joinpoints.
- **Attributable Risk Metrics:** Levin's formula for population exposure and Miettinen's formula for case-exposure cohorts.
- **Batch CSV Processing:** High-throughput processing and rate computation for multi-age and multi-period surveillance tables with input validation.

---

## Installation & Requirements

- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- Zero external runtime dependencies. `pytest` is optional for running tests.

```bash
git clone https://github.com/abusuraihsakhri/age-period-cohort-analyzer.git
cd age-period-cohort-analyzer
```

---

## CLI Usage

### 1. Population Attributable Fraction (PAF)
Calculate Levin's PAF from prevalence and relative risk:
```bash
python cli.py paf --prevalence 0.25 --rr 2.4
```
Output as JSON:
```bash
python cli.py paf --prevalence 0.25 --rr 2.4 --json
```

### 2. Piecewise Joinpoint Regression
Detect trend inflections in rates:
```bash
python cli.py joinpoint --years 2000 2002 2004 2006 2008 2010 2012 2014 --rates 80.0 76.0 71.0 65.0 58.0 50.0 43.0 35.0
```
Output as JSON:
```bash
python cli.py joinpoint --years 2000 2002 2004 2006 2008 2010 2012 2014 --rates 80.0 76.0 71.0 65.0 58.0 50.0 43.0 35.0 --json
```

### 3. Trend Rate Forecasting
Forecast future incidence rates:
```bash
python cli.py forecast --years 2010 2012 2014 2016 2018 --rates 40.0 38.0 36.0 34.0 32.0 --horizon 4 --json
```

### 4. Run SEER Benchmark Demo
```bash
python cli.py --demo
```
Or output as JSON:
```bash
python cli.py --demo --json
```

### 5. Batch CSV Processing
Process epidemiological cohort tables with path validation:
```bash
python cli.py batch --input sample.csv --output results.csv
```

---

## Python API Quickstart

```python
from apc_analyzer import (
    REFERENCE_DATASETS,
    APCStatisticalEngine,
    PAFCalculator,
    JoinpointAnalyzer,
)

# 1. Evaluate SEER Lung Cancer benchmark dataset
seer_data = REFERENCE_DATASETS["seer_male_lung_cancer"]
report = APCStatisticalEngine.analyze_table(seer_data)
print(f"Net Drift: {report.estimable_functions.net_drift_pct:+.2f}% / year")
print(f"Best Fitting Model: {report.best_fitting_model}")

# 2. Population Attributable Fraction (PAF)
paf = PAFCalculator.levin_paf(prevalence=0.20, relative_risk=5.0)
print(f"PAF: {paf * 100:.1f}%")

# 3. Joinpoint analysis (supports up to 2 joinpoints)
jp = JoinpointAnalyzer.fit(
    years=[2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014],
    rates=[80.0, 76.0, 71.0, 65.0, 58.0, 50.0, 43.0, 35.0],
    max_joinpoints=2,
)
print(f"AAPC: {jp.average_annual_percent_change:+.2f}% / year | Joinpoints: {jp.joinpoints}")
```

---

## Running Tests

Run the test suite using standard `unittest` or `pytest`:

```bash
# Run root-level test suite
python test_apc_analyzer.py
# or
pytest test_apc_analyzer.py -v

# Run tests in tests/ directory (includes additional JSON and sample CSV tests)
pytest tests/test_apc_analyzer.py -v

# Run all tests
pytest -v
```

---

## Project Structure

```
age-period-cohort-analyzer/
├── apc_analyzer.py          # Core statistical engine (OLS, APC models, Joinpoint, PAF, Forecasting)
├── cli.py                   # Command-line interface
├── apc_enrichment_features.py  # Compatibility wrappers and re-exports
├── test_apc_analyzer.py     # Root-level unit tests (24 tests)
├── tests/
│   ├── __init__.py
│   └── test_apc_analyzer.py # Extended test suite (26 tests, includes JSON/CSV tests)
├── sample.csv               # Sample batch input data
├── benchmark_dataset.json   # Golden benchmark test cases
├── LICENSE                  # MIT License
└── README.md                # This file
```

---

## Security Features

- **Path validation:** Batch command validates input/output paths and rejects hidden directory traversal
- **Input sanitization:** All numeric inputs validated for range and type
- **Zero external dependencies:** Eliminates supply chain attack surface

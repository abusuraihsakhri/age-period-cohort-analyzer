# Age-Period-Cohort (APC) Statistical Analyzer

A mathematically rigorous, zero-dependency Python statistical engine for **Age-Period-Cohort (APC) analysis** in epidemiology, cancer surveillance, and public health demography.

---

## Methodological Background

Because calendar period ($P$), birth cohort ($C$), and chronological age ($A$) satisfy the exact linear dependency:
$$C = P - A$$
a full log-linear Poisson rate model $\ln(\lambda_{ap}) = \mu + \alpha_a + \beta_p + \gamma_c$ is non-identifiable without additional mathematical constraints.

This package implements the canonical identifiable parameterizations established by **Holford (1983)**, **Clayton & Schifflers (1987)**, and the **NCI APC Tool (Rosenberg et al. 2014)**:

1. **Net Drift ($\beta_L + \gamma_L$)**:
   The overall annual percentage change in disease incidence or mortality rates over calendar time and birth cohort:
   $$\text{Net Drift (\%/year)} = 100 \times (\exp(\text{slope}) - 1)$$

2. **Local Drifts ($\text{Drift}_a$)**:
   Age-specific annual percentage change, capturing variations in secular trends across demographic life stages.

3. **Age, Period, and Cohort Curvatures (Second Differences)**:
   Invariants orthogonal to any linear transformation, isolating acceleration and deceleration points:
   $$\ddot{\alpha}_a = \alpha_{a-1} - 2\alpha_a + \alpha_{a+1}$$
   $$\ddot{\beta}_p = \beta_{p-1} - 2\beta_p + \beta_{p+1}$$
   $$\ddot{\gamma}_c = \gamma_{c-1} - 2\gamma_c + \gamma_{c+1}$$

4. **Cohort & Period Relative Risks (RR)**:
   Estimated rate ratios relative to reference birth cohorts and calendar periods.

5. **Hierarchical Nested Model Comparison**:
   Fits Age-Only (A), Age-Period (AP), Age-Cohort (AC), and full Age-Period-Cohort (APC) models, reporting Poisson deviance $G^2$, degrees of freedom, AIC, and BIC.

6. **Joinpoint Regression**:
   Piecewise log-linear regression identifying trend transition years and segment-specific Annual Percent Changes (APC %).

7. **Population Attributable Fraction (PAF)**:
   Levin's formula for population risk factor burden:
   $$\text{PAF} = \frac{P_e (\text{RR} - 1)}{P_e (\text{RR} - 1) + 1}$$

---

## CLI Usage

### 1. Interactive Studio
```bash
python cli.py
# or
python cli.py --interactive
```

### 2. Run SEER US Lung Cancer Benchmark Demo
```bash
python cli.py --demo
```

### 3. Joinpoint Segment Regression
```bash
python cli.py joinpoint --years 2000 2002 2004 2006 2008 2010 2012 2014 \
  --rates 80.0 76.0 71.0 65.0 58.0 50.0 43.0 35.0
```

### 4. Calculate Population Attributable Fraction (PAF)
```bash
python cli.py paf --prevalence 0.22 --rr 12.0
```

### 5. Extrapolate Rate Forecast
```bash
python cli.py forecast --years 2010 2012 2014 2016 2018 \
  --rates 40.0 38.0 36.0 34.0 32.0 --horizon 5
```

### 6. Batch Process CSV Table
```bash
python cli.py batch --input rates_matrix.csv --output apc_results.csv
```

---

## Python API Example

```python
from apc_analyzer import (
    APCStatisticalEngine,
    build_apc_table_from_matrix,
    JoinpointAnalyzer,
    PAFCalculator,
    REFERENCE_DATASETS,
)

# Load reference dataset
ds = REFERENCE_DATASETS["us_lung_cancer_male"]
table = build_apc_table_from_matrix(
    age_groups=ds["age_groups"],
    periods=ds["periods"],
    rates_matrix=ds["rates_per_100k"],
)

# Fit estimable functions
estimable = APCStatisticalEngine.fit_estimable_functions(table)
print(f"Net Drift: {estimable.net_drift_pct:+.2f}% / year")
for age, drift in estimable.local_drifts.items():
    print(f"  Age {age}: {drift:+.2f}%/yr")

# Compute PAF for smoking
paf = PAFCalculator.levin_paf(prevalence=0.22, relative_risk=12.0)
print(f"Smoking PAF: {paf * 100:.2f}%")
```

---

## Unit Testing

Run the test suite:

```bash
python -m unittest test_apc_analyzer.py
```

---

## License

MIT License. Designed for cancer epidemiology, registry biostatistics, and demographic surveillance.

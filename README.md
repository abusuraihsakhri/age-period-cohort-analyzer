# Age Period Cohort Analyzer

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

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

Enrichment Features for Age-Period-Cohort Analyzer
Re-exports forecasting, joinpoint regression, and PAF tools.

---

## ⚙️ Key Capabilities & Algorithmic Modules

### 🔬 Core Algorithmic & Evaluation Engines

- **`OLSFitResult`** — dedicated module for o l s fit result evaluation and state verification.
- **`APCCell`**: Individual Age-Period table cell.
- **`APCTable`**: 2D Age x Period data matrix with derived diagonal Cohorts (C = P - A).
- **`EstimableFunctionsResult`**: Holford (1983) and Clayton-Schifflers (1987) invariant estimable parameters.
- **`APCModelFit`**: Goodness-of-fit and parameters for an evaluated log-linear model.
- **`JoinpointSegment`** — dedicated module for joinpoint segment evaluation and state verification.

---

## 📐 Mathematical Formulation & Logic

```text
  - Population Attributable Fraction (PAF) via Levin and Miettinen formulations
  Levin's formula for population attributable fraction:
  Miettinen's formula (case-based exposure prevalence):
```

---

## 💻 CLI Quickstart & Usage

### 1. Guided Interactive Mode
```bash
python cli.py
```

### 2. Direct Parameterized Evaluation
```bash
python cli.py --interactive <value> --demo <value> --json <value> --years <value>
```

### Parameter Reference
- `--interactive`: Specifies input measurement or parameter value.
- `--demo`: Specifies input measurement or parameter value.
- `--json`: Specifies input measurement or parameter value.
- `--years`: Specifies input measurement or parameter value.
- `--rates`: Specifies input measurement or parameter value.
- `--max-joinpoints`: Specifies input measurement or parameter value.
- `--prevalence`: Specifies input measurement or parameter value.
- `--rr`: Specifies input measurement or parameter value.
- `--horizon`: Specifies input measurement or parameter value.
- `--input`: Specifies input measurement or parameter value.

### Input Data Schema

| Field | Description | Requirement |
|:------|:------------|:------------|
| `id` | Parameter / observation metric | Required |
| `value` | Parameter / observation metric | Required |
| `qty` | Parameter / observation metric | Required |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py --tasks 1000 --concurrency 8
```

---

## 🐳 Container Deployment

```bash
docker build -t age-period-cohort-analyzer .
docker run -p 8000:8000 age-period-cohort-analyzer
```

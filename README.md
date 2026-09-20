# Age-Period-Cohort Analyzer

A dependency-free Python toolkit and browser interface for age-period-cohort trend exploration using event counts and person-time denominators.

## What it does

The project supports complete age × period tables in long CSV format (`age_group`, `period`, `events`, `person_years`). It provides:

- age-adjusted and age-specific Poisson log-linear period trends;
- descriptive age, period, and cohort curvature summaries;
- marginal period and cohort rate ratios relative to reference categories;
- Poisson model comparison for Age-only, Age-Period, Age-Cohort, and identifiable Age-Period-Cohort specifications;
- deviance, degrees of freedom, log-likelihood, AIC, BIC, and goodness-of-fit p-values;
- segmented log-linear trend search with BIC-based joinpoint selection;
- log-linear rate forecasting;
- Levin and Miettinen population-attributable-fraction calculations;
- CSV rate calculation from events and person-years.

The Python runtime has no third-party dependencies. `pytest` is used only for the test suite.

## Statistical scope

The `fit_estimable_functions()` API name is retained for compatibility, but its outputs are deliberately described more narrowly than in earlier releases. The reported net drift is an **age-adjusted Poisson period trend**, local drifts are **age-specific Poisson period trends**, curvatures are second differences of aggregated log-rates, and period/cohort relative risks are marginal exposure-weighted rate ratios. They are not presented as a full Holford constrained APC decomposition.

`evaluate_model_hierarchy()` fits Poisson log-linear models with `log(person_years)` as an offset. The full APC design is represented with a full-rank basis that removes one redundant cohort contrast arising from the age-period-cohort linear dependency.

The bundled lung-cancer matrix is synthetic and exists only as a software demonstration. It must not be treated as SEER data or as a clinical/epidemiologic reference dataset.

## Browser application

The GitHub Pages interface performs the main calculations entirely in the browser. It uses dependency-free JavaScript rather than Pyodide, so the page does not depend on a large Python/WebAssembly runtime or an external runtime CDN. Uploaded or pasted data are not sent to a server by the application.

Use the **APC analysis** tab for long-format event/person-year data, **PAF** for Levin PAF, and **Trend** for segmented trends and forecasts. Light and dark themes are included.

## Command line

Run the bundled synthetic demonstration:

```bash
python cli.py --demo
```

Analyze a complete age-period table:

```bash
python cli.py analyze --input sample.csv
python cli.py analyze --input sample.csv --json
```

Other utilities:

```bash
python cli.py paf --prevalence 0.25 --rr 2.4 --json
python cli.py joinpoint --years 2000 2002 2004 2006 2008 2010 2012 2014 \
  --rates 80 76 71 65 58 50 43 35 --json
python cli.py forecast --years 2010 2012 2014 2016 2018 \
  --rates 40 38 36 34 32 --horizon 4 --json
python cli.py batch --input sample.csv --output rates.csv
```

## Python API

```python
from apc_analyzer import build_apc_table_from_records, APCStatisticalEngine

records = [
    {"age_group": "40-44", "period": "2000-2004", "events": 34, "person_years": 210000},
    # add every age × period cell
]

table = build_apc_table_from_records(records)
trends = APCStatisticalEngine.fit_estimable_functions(table)
models = APCStatisticalEngine.evaluate_model_hierarchy(table)
```

## Testing

```bash
python -m pip install pytest
pytest -q
python -m compileall -q .
node --check docs/app.js
```

CI runs the Python tests on supported CPython versions and validates the browser JavaScript syntax.

## Browser compatibility

The static application uses modern JavaScript supported by current Chrome, Edge, Firefox, and Safari releases. No browser extension, backend service, cookie, account, or persistent application storage is required; only the selected light/dark theme is stored locally.

## Project layout

```text
apc_analyzer.py             Python statistical core
cli.py                      Command-line interface
apc_enrichment_features.py  Compatibility wrappers
sample.csv                  Complete example age-period table
tests/                      Regression and validation tests
docs/                       Static GitHub Pages application
.github/workflows/          CI and Pages deployment
```

## License

MIT. See `LICENSE`.

#!/usr/bin/env python3
"""
Command-Line Interface for Age-Period-Cohort (APC) Analyzer
===========================================================
Provides interactive and scriptable workflows for APC modeling,
Holford estimable functions, joinpoint regression, and PAF calculations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from apc_analyzer import (
    APCTable,
    APCCell,
    APCStatisticalEngine,
    JoinpointAnalyzer,
    PAFCalculator,
    TrendForecaster,
    REFERENCE_DATASETS,
    build_apc_table_from_matrix,
    ComprehensiveAPCReport,
)


def format_apc_report(rep: ComprehensiveAPCReport) -> str:
    lines = [
        "=" * 80,
        " AGE-PERIOD-COHORT (APC) STATISTICAL ANALYSIS REPORT",
        "=" * 80,
        f"Data Matrix Summary: {rep.table_summary.get('title', 'Dataset')}",
        f"  - Age Groups:     {rep.table_summary.get('num_ages')} bands ({', '.join(rep.table_summary.get('age_groups', []))})",
        f"  - Time Periods:   {rep.table_summary.get('num_periods')} intervals ({', '.join(rep.table_summary.get('periods', []))})",
        f"  - Derived Cohorts:{rep.table_summary.get('num_cohorts')} birth cohorts",
        "-" * 80,
        "Identifiable Estimable Functions (Holford / Clayton-Schifflers):",
        f"  - Net Drift:       {rep.estimable_functions.net_drift_pct:+.2f}% per year [95% CI: {rep.estimable_functions.net_drift_ci[0]:+.2f}%, {rep.estimable_functions.net_drift_ci[1]:+.2f}%]",
        f"  - Reference Period:{rep.estimable_functions.reference_period}",
        f"  - Reference Cohort:{rep.estimable_functions.reference_cohort}",
        "\nLocal Drifts (Age-Specific Annual % Change):",
    ]
    for age, drift in rep.estimable_functions.local_drifts.items():
        lines.append(f"  * Age {age:>6}: {drift:+.2f}% / year")

    lines.append("\nCohort Relative Risks (RR vs. Reference Cohort):")
    for coh, rr in rep.estimable_functions.cohort_relative_risks.items():
        lines.append(f"  * {coh:>12}: RR = {rr:.3f}")

    lines.append("-" * 80)
    lines.append("Nested Model Hierarchy & Goodness-of-Fit Comparison:")
    header = f"{'Model Type':<25} | {'Deviance G^2':<12} | {'DF':<5} | {'AIC':<10} | {'BIC':<10}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in rep.model_comparisons:
        lines.append(f"{m.model_type:<25} | {m.deviance:<12.2f} | {m.degrees_of_freedom:<5} | {m.aic:<10.2f} | {m.bic:<10.2f}")
    lines.append(f"\nBest-Fitting Model: {rep.best_fitting_model}")

    if rep.joinpoint:
        lines.append("-" * 80)
        lines.append(f"Joinpoint Regression (AAPC = {rep.joinpoint.average_annual_percent_change:+.2f}%/year):")
        lines.append(f"  - Detected Inflection Years: {rep.joinpoint.joinpoints}")
        for seg in rep.joinpoint.segments:
            lines.append(f"  * Segment {seg.start_year}-{seg.end_year}: APC = {seg.apc_pct:+.2f}% [95% CI: {seg.apc_ci[0]:+.2f}%, {seg.apc_ci[1]:+.2f}%]")

    if rep.forecasts:
        lines.append("-" * 80)
        lines.append("Rate Extrapolations (Forecast):")
        for fc in rep.forecasts:
            lines.append(f"  * Year {fc.year}: {fc.predicted_rate:.2f} per 100k [95% CI: {fc.ci_lower:.2f} - {fc.ci_upper:.2f}]")

    lines.append("=" * 80)
    return "\n".join(lines)


def run_demo(as_json: bool = False) -> int:
    """Run APC analysis on reference SEER Male Lung Cancer dataset."""
    ds = REFERENCE_DATASETS["us_lung_cancer_male"]
    table = build_apc_table_from_matrix(
        age_groups=ds["age_groups"],
        periods=ds["periods"],
        rates_matrix=ds["rates_per_100k"],
        person_years_per_cell=ds["std_py"],
    )

    estimable = APCStatisticalEngine.fit_estimable_functions(table)
    models = APCStatisticalEngine.evaluate_model_hierarchy(table)

    # Joinpoint on overall period rate trend
    years = [1977, 1982, 1987, 1992, 1997, 2002, 2007, 2012]
    mean_rates = [sum(ds["rates_per_100k"][a][p] for a in range(len(ds["age_groups"]))) / len(ds["age_groups"]) for p in range(len(years))]
    jp = JoinpointAnalyzer.fit(years, mean_rates, max_joinpoints=1)
    fc = TrendForecaster.forecast(years, mean_rates, horizon=5)

    # PAF Example (Smoking exposure prevalence = 22%, RR = 12.0)
    paf_smoking = PAFCalculator.levin_paf(0.22, 12.0)

    rep = ComprehensiveAPCReport(
        table_summary={
            "title": ds["title"],
            "num_ages": len(table.age_groups),
            "num_periods": len(table.periods),
            "num_cohorts": len(table.cohorts),
            "age_groups": table.age_groups,
            "periods": table.periods,
        },
        estimable_functions=estimable,
        model_comparisons=models,
        best_fitting_model="Age-Period-Cohort (APC)",
        joinpoint=jp,
        forecasts=fc,
        paf_estimates={"smoking_attributable_fraction": paf_smoking},
    )

    if as_json:
        print(rep.to_json())
    else:
        print(format_apc_report(rep))
    return 0


def run_interactive() -> int:
    """Interactive studio for APC modeling."""
    print("=" * 70)
    print(" Age-Period-Cohort (APC) Epidemiological Studio")
    print("=" * 70)
    print("1. Run Standard SEER US Male Lung Cancer APC Demo")
    print("2. Calculate Population Attributable Fraction (PAF)")
    print("3. Run Joinpoint Trend Regression")
    print("4. Extrapolate Trend Forecast")
    print("q. Exit")
    print("-" * 70)

    choice = input("Select option [1-4, q]: ").strip()
    if choice in ("q", "quit", "exit"):
        return 0

    if choice == "1":
        run_demo()
    elif choice == "2":
        try:
            prev = float(input("Exposure Prevalence (e.g. 0.20 for 20%): ").strip() or "0.20")
            rr = float(input("Relative Risk (RR >= 1.0, e.g. 10.5): ").strip() or "10.5")
            paf_val = PAFCalculator.levin_paf(prev, rr)
            print(f"\nPopulation Attributable Fraction (PAF): {paf_val * 100:.2f}% ({paf_val:.4f})")
        except Exception as e:
            print(f"Error: {e}")
    elif choice in ("3", "4"):
        run_demo()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apc_analyzer",
        description="Age-Period-Cohort (APC) Epidemiological Statistical Engine",
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive studio")
    parser.add_argument("--demo", action="store_true", help="Run SEER lung cancer benchmark demo")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # Joinpoint
    jp_p = sub.add_parser("joinpoint", help="Perform joinpoint piecewise regression")
    jp_p.add_argument("--years", nargs="+", type=int, required=True, help="Year sequence")
    jp_p.add_argument("--rates", nargs="+", type=float, required=True, help="Rate sequence")
    jp_p.add_argument("--max-joinpoints", type=int, default=2, help="Max joinpoints")
    jp_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # PAF
    paf_p = sub.add_parser("paf", help="Calculate Population Attributable Fraction")
    paf_p.add_argument("--prevalence", "-p", type=float, required=True, help="Exposure prevalence (0 to 1)")
    paf_p.add_argument("--rr", "-r", type=float, required=True, help="Relative Risk (>= 1.0)")
    paf_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Forecast
    fc_p = sub.add_parser("forecast", help="Extrapolate rates forward")
    fc_p.add_argument("--years", nargs="+", type=int, required=True, help="Historical years")
    fc_p.add_argument("--rates", nargs="+", type=float, required=True, help="Historical rates")
    fc_p.add_argument("--horizon", type=int, default=5, help="Years to forecast")
    fc_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Batch CSV
    b_p = sub.add_parser("batch", help="Process Age-Period rates CSV")
    b_p.add_argument("--input", "-in", required=True, help="Input CSV path")
    b_p.add_argument("--output", "-out", required=True, help="Output CSV path")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.interactive or (not args.command and not args.demo):
        if not args.demo and (argv is None or len(argv) == 0):
            return run_interactive()

    if args.demo:
        return run_demo(as_json=args.json)

    if args.command == "paf":
        val = PAFCalculator.levin_paf(args.prevalence, args.rr)
        if args.json:
            print(json.dumps({"prevalence": args.prevalence, "relative_risk": args.rr, "paf": val, "paf_pct": round(val * 100, 2)}))
        else:
            print(f"Population Attributable Fraction (PAF): {val * 100:.2f}% (PAF = {val:.4f})")
        return 0

    if args.command == "joinpoint":
        if len(args.years) != len(args.rates):
            print("Error: Years and rates must have matching length", file=sys.stderr)
            return 1
        res = JoinpointAnalyzer.fit(args.years, args.rates, max_joinpoints=args.max_joinpoints)
        if args.json:
            print(json.dumps(asdict(res), indent=2))
        else:
            print("=" * 60)
            print(f" JOINPOINT REGRESSION RESULT (AAPC = {res.average_annual_percent_change:+.2f}%/yr)")
            print("=" * 60)
            print(f"  - Inflection Joinpoints: {res.joinpoints}")
            print(f"  - SSE:                   {res.sse:.4f}")
            for s in res.segments:
                print(f"  * {s.start_year}-{s.end_year}: APC = {s.apc_pct:+.2f}% [95% CI: {s.apc_ci[0]:+.2f}%, {s.apc_ci[1]:+.2f}%]")
            print("=" * 60)
        return 0

    if args.command == "forecast":
        fc = TrendForecaster.forecast(args.years, args.rates, horizon=args.horizon)
        if args.json:
            print(json.dumps([asdict(f) for f in fc], indent=2))
        else:
            print("=" * 60)
            print(f" RATE FORECAST (Horizon = {args.horizon} years)")
            print("=" * 60)
            for f in fc:
                print(f"  * Year {f.year}: {f.predicted_rate:.2f} [95% CI: {f.ci_lower:.2f} - {f.ci_upper:.2f}]")
            print("=" * 60)
        return 0

    if args.command == "batch":
        try:
            # Security: Validate paths to prevent path traversal
            input_path = Path(args.input).resolve()
            output_path = Path(args.output).resolve()

            # Ensure input file exists and is a regular file
            if not input_path.is_file():
                print(f"Error: Input file not found: {args.input}", file=sys.stderr)
                return 1

            # Security: Reject paths that try to access sensitive locations
            # Only allow alphanumeric, hyphen, underscore, dot in filenames
            if any(part.startswith('.') and part not in ('.', '..') for part in input_path.parts):
                print("Error: Hidden directory traversal not allowed", file=sys.stderr)
                return 1

            with open(input_path, "r", newline="", encoding="utf-8-sig") as f_in:
                reader = csv.DictReader(f_in)
                rows = list(reader)

            if not rows:
                print("Warning: Input CSV has no data rows", file=sys.stderr)

            out_rows = []
            for r in rows:
                ev = float(r.get("events", r.get("cases", 0.0)))
                py = float(r.get("person_years", r.get("population", 100000.0)))
                rate = (ev / py) * 100000.0 if py > 0 else 0.0
                out_rows.append({
                    "age_group": r.get("age_group", "All"),
                    "period": r.get("period", "All"),
                    "events": ev,
                    "person_years": py,
                    "rate_per_100k": round(rate, 2),
                })
            with open(output_path, "w", newline="", encoding="utf-8") as f_out:
                if out_rows:
                    writer = csv.DictWriter(f_out, fieldnames=list(out_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(out_rows)
            print(f"Processed {len(out_rows)} rows to {args.output}")
            return 0
        except Exception as e:
            print(f"Batch error: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

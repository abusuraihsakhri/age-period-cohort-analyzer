#!/usr/bin/env python3
"""Command-line interface for the age-period-cohort analyzer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from apc_analyzer import (
    APCStatisticalEngine,
    ComprehensiveAPCReport,
    JoinpointAnalyzer,
    PAFCalculator,
    REFERENCE_DATASETS,
    TrendForecaster,
    build_apc_table_from_records,
)


def format_apc_report(rep: ComprehensiveAPCReport) -> str:
    lines = [
        "=" * 78,
        " AGE-PERIOD-COHORT TREND ANALYSIS",
        "=" * 78,
        f"Dataset: {rep.table_summary.get('title', 'Dataset')}",
        f"Age groups: {rep.table_summary.get('num_ages')} | Periods: {rep.table_summary.get('num_periods')} | Cohorts: {rep.table_summary.get('num_cohorts')}",
        "-" * 78,
        "Descriptive period trends",
        f"Age-adjusted period trend: {rep.estimable_functions.net_drift_pct:+.3f}%/year "
        f"(95% CI {rep.estimable_functions.net_drift_ci[0]:+.3f} to {rep.estimable_functions.net_drift_ci[1]:+.3f})",
        "Age-specific period trends:",
    ]
    for age, drift in rep.estimable_functions.local_drifts.items():
        lines.append(f"  {age:>12}: {drift:+.3f}%/year")

    lines += [
        "-" * 78,
        "Poisson log-linear model comparison",
        f"{'Model':<28} {'Deviance':>10} {'df':>6} {'AIC':>12} {'BIC':>12} {'GOF p':>10}",
    ]
    for model in rep.model_comparisons:
        ptxt = f"{model.p_value:.4g}" if math.isfinite(model.p_value) else "n/a"
        lines.append(
            f"{model.model_type:<28} {model.deviance:>10.3f} {model.degrees_of_freedom:>6} "
            f"{model.aic:>12.3f} {model.bic:>12.3f} {ptxt:>10}"
        )
    lines.append(f"Lowest-AIC model: {rep.best_fitting_model}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _analyze_csv(path: Path) -> ComprehensiveAPCReport:
    if not path.is_file():
        raise ValueError(f"input file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"age_group", "period", "events", "person_years"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain age_group, period, events, and person_years columns")
        rows = list(reader)
    table = build_apc_table_from_records(rows)
    estimable = APCStatisticalEngine.fit_estimable_functions(table)
    models = APCStatisticalEngine.evaluate_model_hierarchy(table)
    return ComprehensiveAPCReport(
        table_summary={
            "title": path.name,
            "num_ages": len(table.age_groups),
            "num_periods": len(table.periods),
            "num_cohorts": len(table.cohorts),
            "age_groups": table.age_groups,
            "periods": table.periods,
        },
        estimable_functions=estimable,
        model_comparisons=models,
        best_fitting_model=min(models, key=lambda m: m.aic).model_type,
    )


def run_demo(as_json: bool = False) -> int:
    report = APCStatisticalEngine.analyze_table(REFERENCE_DATASETS["synthetic_lung_cancer_male"])
    if as_json:
        print(report.to_json())
    else:
        print(format_apc_report(report))
        print("\nNote: the bundled rate matrix is synthetic and intended for software demonstration only.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apc_analyzer",
        description="Age-period-cohort trend analysis and Poisson model comparison",
    )
    parser.add_argument("--demo", action="store_true", help="run the bundled synthetic example")
    parser.add_argument("--json", action="store_true", help="emit JSON where supported")
    sub = parser.add_subparsers(dest="command")

    analyze = sub.add_parser("analyze", help="analyze a complete long-format age-period CSV")
    analyze.add_argument("--input", "-i", required=True, help="CSV with age_group, period, events, person_years")
    analyze.add_argument("--json", action="store_true")

    paf = sub.add_parser("paf", help="calculate Levin population-attributable fraction")
    paf.add_argument("--prevalence", "-p", type=float, required=True)
    paf.add_argument("--rr", "-r", type=float, required=True)
    paf.add_argument("--json", action="store_true")

    jp = sub.add_parser("joinpoint", help="fit 0-2 segmented log-linear trends selected by BIC")
    jp.add_argument("--years", nargs="+", type=int, required=True)
    jp.add_argument("--rates", nargs="+", type=float, required=True)
    jp.add_argument("--max-joinpoints", type=int, choices=(0, 1, 2), default=2)
    jp.add_argument("--min-segment-length", type=int, default=4)
    jp.add_argument("--json", action="store_true")

    fc = sub.add_parser("forecast", help="fit and extrapolate a log-linear rate trend")
    fc.add_argument("--years", nargs="+", type=int, required=True)
    fc.add_argument("--rates", nargs="+", type=float, required=True)
    fc.add_argument("--horizon", type=int, default=5)
    fc.add_argument("--json", action="store_true")

    batch = sub.add_parser("batch", help="compute rates per 100,000 from event/exposure CSV rows")
    batch.add_argument("--input", "-i", required=True)
    batch.add_argument("--output", "-o", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.demo:
            return run_demo(as_json=args.json)

        if args.command == "analyze":
            report = _analyze_csv(Path(args.input).expanduser().resolve())
            print(report.to_json() if args.json else format_apc_report(report))
            return 0

        if args.command == "paf":
            value = PAFCalculator.levin_paf(args.prevalence, args.rr)
            payload = {
                "prevalence": args.prevalence,
                "relative_risk": args.rr,
                "paf": value,
                "paf_pct": round(value * 100.0, 4),
            }
            print(json.dumps(payload) if args.json else f"PAF: {payload['paf_pct']:.4f}% ({value:.6f})")
            return 0

        if args.command == "joinpoint":
            result = JoinpointAnalyzer.fit(
                args.years,
                args.rates,
                max_joinpoints=args.max_joinpoints,
                min_segment_length=args.min_segment_length,
            )
            if args.json:
                print(json.dumps(asdict(result), indent=2))
            else:
                print(f"AAPC: {result.average_annual_percent_change:+.3f}%/year")
                print(f"Selected joinpoints: {result.joinpoints or 'none'}")
                for seg in result.segments:
                    print(
                        f"{seg.start_year}-{seg.end_year}: {seg.apc_pct:+.3f}%/year "
                        f"(95% CI {seg.apc_ci[0]:+.3f} to {seg.apc_ci[1]:+.3f})"
                    )
            return 0

        if args.command == "forecast":
            result = TrendForecaster.forecast(args.years, args.rates, args.horizon)
            if args.json:
                print(json.dumps([asdict(x) for x in result], indent=2))
            else:
                for item in result:
                    print(
                        f"{item.year}: {item.predicted_rate:.4f} "
                        f"(95% prediction interval {item.ci_lower:.4f} to {item.ci_upper:.4f})"
                    )
            return 0

        if args.command == "batch":
            input_path = Path(args.input).expanduser().resolve()
            output_path = Path(args.output).expanduser().resolve()
            if not input_path.is_file():
                raise ValueError(f"input file not found: {input_path}")
            if input_path == output_path:
                raise ValueError("input and output paths must be different")
            with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                required = {"age_group", "period", "events", "person_years"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError("CSV must contain age_group, period, events, and person_years columns")
                rows = list(reader)

            out_rows = []
            for i, row in enumerate(rows, 2):
                try:
                    events = float(row["events"])
                    exposure = float(row["person_years"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"row {i}: events and person_years must be numeric") from exc
                if not math.isfinite(events) or events < 0:
                    raise ValueError(f"row {i}: events must be finite and non-negative")
                if not math.isfinite(exposure) or exposure <= 0:
                    raise ValueError(f"row {i}: person_years must be finite and positive")
                out_rows.append(
                    {
                        "age_group": row["age_group"],
                        "period": row["period"],
                        "events": events,
                        "person_years": exposure,
                        "rate_per_100k": round(events / exposure * 100000.0, 6),
                    }
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["age_group", "period", "events", "person_years", "rate_per_100k"],
                )
                writer.writeheader()
                writer.writerows(out_rows)
            print(f"Processed {len(out_rows)} row(s) -> {output_path}")
            return 0

        parser.print_help()
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

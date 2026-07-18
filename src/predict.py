"""Offline, one-command prediction entry point required by the hackathon contract."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from .constants import DEFAULT_HORIZONS, DEFAULT_SIMULATIONS, DEFAULT_TARGET_ROAS, OUTPUT_COLUMNS
from .decision import build_decision_summary
from .forecast import forecast_portfolio
from .ingest import load_budget_plan, load_datasets
from .model import adapt_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--target-roas", type=float, default=DEFAULT_TARGET_ROAS)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    with model_path.open("rb") as handle:
        base_artifact = pickle.load(handle)

    data, quality = load_datasets(args.data_dir)
    artifact = adapt_model_artifact(base_artifact, data, quality)
    budget_plan = load_budget_plan(args.data_dir)
    predictions, diagnostics = forecast_portfolio(
        data,
        artifact,
        horizons=horizons,
        target_roas=float(args.target_roas),
        simulations=int(args.simulations),
        budget_plan=budget_plan,
    )
    output = predictions[OUTPUT_COLUMNS].copy()
    numeric_columns = [
        "budget",
        "revenue_p10",
        "revenue_p50",
        "revenue_p90",
        "roas_p10",
        "roas_p50",
        "roas_p90",
        "target_roas",
        "probability_target",
    ]
    output[numeric_columns] = output[numeric_columns].round(4)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    quality.to_csv(output_path.with_name("data_quality.csv"), index=False)
    decision_summary = build_decision_summary(predictions, diagnostics)
    output_path.with_name("decision_summary.json").write_text(
        json.dumps(decision_summary, indent=2), encoding="utf-8"
    )
    adaptation = artifact.get("runtime_adaptation", {})
    print(f"Loaded artifact: {artifact.get('model_kind', 'unknown')}")
    print(
        "Runtime calibration: "
        f"{adaptation.get('runtime_rows', 0):,} rows / "
        f"{adaptation.get('runtime_campaigns', 0)} campaigns; "
        f"global prior={adaptation.get('global_prior_source', 'unknown')}"
    )
    print(f"Rows: {len(data):,}; active campaigns: {diagnostics['active_campaigns']}")
    print(f"Wrote {len(output):,} forecast rows to {output_path}")


if __name__ == "__main__":
    main()

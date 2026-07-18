"""Rolling-origin evaluation with actual future spend used as the planned budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .boosting import fit_boosting_bundle
from .forecast import active_campaigns, forecast_portfolio
from .evaluate import run_detailed_backtest
from .ingest import build_quality_report, load_datasets
from .model import build_model_artifact


def _origins(df: pd.DataFrame, folds: int, max_horizon: int) -> list[pd.Timestamp]:
    minimum = df["date"].min() + pd.Timedelta(days=365)
    maximum = df["date"].max() - pd.Timedelta(days=max_horizon)
    if maximum <= minimum:
        raise ValueError("Not enough history for rolling-origin evaluation")
    values = pd.date_range(minimum, maximum, periods=folds)
    return [pd.Timestamp(value).normalize() for value in values]


def run_backtest(
    df: pd.DataFrame,
    quality: pd.DataFrame,
    horizons: tuple[int, ...] = (30, 60, 90),
    folds: int = 5,
    simulations: int = 300,
) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    for fold, origin in enumerate(_origins(df, folds, max(horizons)), start=1):
        history = df[df["date"] <= origin].copy()
        artifact = build_model_artifact(history, build_quality_report(history))
        boosting_bundle, boosting_training = fit_boosting_bundle(history)
        artifact["boosting_training"] = boosting_training
        if boosting_bundle is not None:
            artifact["boosting_bundle"] = boosting_bundle
        active = active_campaigns(history)[["channel", "campaign_id", "campaign_name"]].drop_duplicates()
        active_keys = set(zip(active["channel"], active["campaign_id"].astype(str), strict=False))
        for horizon in horizons:
            future = df[(df["date"] > origin) & (df["date"] <= origin + pd.Timedelta(days=horizon))].copy()
            future = future[
                future.apply(
                    lambda row: (row["channel"], str(row["campaign_id"])) in active_keys,
                    axis=1,
                )
            ]
            if future.empty:
                continue
            plan = (
                future.groupby(["channel", "campaign_id", "campaign_name"], as_index=False)["spend"]
                .sum()
                .rename(columns={"spend": "budget"})
            )
            plan["horizon_days"] = int(horizon)
            predicted, _ = forecast_portfolio(
                history,
                artifact,
                horizons=(int(horizon),),
                simulations=simulations,
                budget_plan=plan,
                seed=20260717 + fold * 100,
            )
            overall = predicted[predicted["forecast_level"] == "overall"].iloc[0]
            actual_revenue = float(future["revenue"].sum())
            actual_spend = float(future["spend"].sum())
            recent = history[history["date"] > origin - pd.Timedelta(days=horizon)]
            recent_roas = float(recent["revenue"].sum() / recent["spend"].sum()) if recent["spend"].sum() > 0 else 0.0
            naive = recent_roas * actual_spend
            records.append(
                {
                    "fold": fold,
                    "origin": origin.date().isoformat(),
                    "horizon_days": int(horizon),
                    "actual_revenue": actual_revenue,
                    "actual_spend": actual_spend,
                    "revenue_p10": float(overall["revenue_p10"]),
                    "revenue_p50": float(overall["revenue_p50"]),
                    "revenue_p90": float(overall["revenue_p90"]),
                    "naive_revenue": naive,
                    "covered": bool(
                        float(overall["revenue_p10"])
                        <= actual_revenue
                        <= float(overall["revenue_p90"])
                    ),
                }
            )
    results = pd.DataFrame(records)
    summaries = []
    for horizon, group in results.groupby("horizon_days"):
        model_abs = (group["revenue_p50"] - group["actual_revenue"]).abs()
        naive_abs = (group["naive_revenue"] - group["actual_revenue"]).abs()
        summaries.append(
            {
                "horizon_days": int(horizon),
                "folds": int(len(group)),
                "model_wape": float(model_abs.sum() / group["actual_revenue"].sum()),
                "naive_wape": float(naive_abs.sum() / group["actual_revenue"].sum()),
                "interval_coverage": float(group["covered"].mean()),
                "median_interval_width": float(
                    ((group["revenue_p90"] - group["revenue_p10"]) / group["revenue_p50"].clip(lower=1)).median()
                ),
            }
        )
    return results, {"summary": summaries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--output-dir", default="./output/backtest")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, quality = load_datasets(args.data_dir)
    origins = _origins(data, int(args.folds), 90)
    results, summary = run_detailed_backtest(
        data,
        quality,
        origins=origins,
        simulations=int(args.simulations),

    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "folds.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

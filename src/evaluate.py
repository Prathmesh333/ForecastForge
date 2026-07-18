"""Detailed rolling-origin evaluation at every forecast hierarchy level."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .boosting import fit_boosting_bundle
from .forecast import active_campaigns, forecast_portfolio
from .ingest import build_quality_report
from .model import build_model_artifact


def _entity_key(channel: object, campaign_id: object) -> str:
    return f"{channel}||{campaign_id}"


def _prediction_keys(row: pd.Series, active: pd.DataFrame) -> list[str]:
    selected = active
    if row["forecast_level"] == "campaign":
        selected = selected[
            (selected["channel"] == row["channel"])
            & (selected["campaign_id"].astype(str) == str(row["campaign_id"]))
        ]
    elif row["forecast_level"] == "campaign_type":
        selected = selected[
            (selected["channel"] == row["channel"])
            & (selected["campaign_family"] == row["campaign_type"])
        ]
    elif row["forecast_level"] == "channel":
        selected = selected[selected["channel"] == row["channel"]]
    return selected["entity_key"].tolist()


def _metric_record(horizon: int, level: str, group: pd.DataFrame) -> dict:
    evaluated = group[group["actual_spend"] > 0].copy()
    actual_total = float(evaluated["actual_revenue"].sum())
    model_error = (evaluated["revenue_p50"] - evaluated["actual_revenue"]).abs()
    naive_error = (evaluated["naive_revenue"] - evaluated["actual_revenue"]).abs()
    alpha = 0.20
    below = (evaluated["revenue_p10"] - evaluated["actual_revenue"]).clip(lower=0)
    above = (evaluated["actual_revenue"] - evaluated["revenue_p90"]).clip(lower=0)
    interval_score = (
        evaluated["revenue_p90"]
        - evaluated["revenue_p10"]
        + (2.0 / alpha) * below
        + (2.0 / alpha) * above
    )
    return {
        "horizon_days": int(horizon),
        "forecast_level": str(level),
        "folds": int(evaluated["fold"].nunique()),
        "evaluated_entities": int(len(evaluated)),
        "model_wape": float(model_error.sum() / actual_total) if actual_total > 0 else None,
        "naive_wape": float(naive_error.sum() / actual_total) if actual_total > 0 else None,
        "interval_coverage": float(evaluated["covered"].mean()) if len(evaluated) else None,
        "normalized_interval_score": (
            float(interval_score.sum() / actual_total) if actual_total > 0 else None
        ),
        "median_interval_width": (
            float(
                (
                    (evaluated["revenue_p90"] - evaluated["revenue_p10"])
                    / evaluated["revenue_p50"].clip(lower=1.0)
                ).median()
            )
            if len(evaluated)
            else None
        ),
    }


def _campaign_slice_records(results: pd.DataFrame) -> list[dict]:
    campaigns = results[
        (results["forecast_level"] == "campaign") & (results["actual_spend"] > 0)
    ]
    records: list[dict] = []
    for dimension in ("support_status", "lifecycle_state", "channel"):
        for (horizon, value), group in campaigns.groupby(["horizon_days", dimension]):
            actual_total = float(group["actual_revenue"].sum())
            absolute_error = (group["revenue_p50"] - group["actual_revenue"]).abs().sum()
            records.append(
                {
                    "horizon_days": int(horizon),
                    "slice_type": dimension,
                    "slice_value": str(value),
                    "entities": int(len(group)),
                    "wape": float(absolute_error / actual_total) if actual_total > 0 else None,
                    "interval_coverage": float(group["covered"].mean()),
                }
            )
    return records


def run_detailed_backtest(
    df: pd.DataFrame,
    quality: pd.DataFrame,
    origins: list[pd.Timestamp],
    horizons: tuple[int, ...] = (30, 60, 90),
    simulations: int = 300,
) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    for fold, origin in enumerate(origins, start=1):
        history = df[df["date"] <= origin].copy()
        history_quality = build_quality_report(history)
        artifact = build_model_artifact(history, history_quality)
        boosting_bundle, boosting_training = fit_boosting_bundle(history)
        artifact["boosting_training"] = boosting_training
        if boosting_bundle is not None:
            artifact["boosting_bundle"] = boosting_bundle
        active = active_campaigns(history)[
            ["channel", "campaign_id", "campaign_name", "campaign_family"]
        ].drop_duplicates()
        active["entity_key"] = [
            _entity_key(channel, campaign_id)
            for channel, campaign_id in zip(
                active["channel"], active["campaign_id"], strict=False
            )
        ]
        active_keys = set(active["entity_key"])
        for horizon in horizons:
            future = df[
                (df["date"] > origin)
                & (df["date"] <= origin + pd.Timedelta(days=int(horizon)))
            ].copy()
            future["entity_key"] = [
                _entity_key(channel, campaign_id)
                for channel, campaign_id in zip(
                    future["channel"], future["campaign_id"], strict=False
                )
            ]
            future = future[future["entity_key"].isin(active_keys)]

            future_totals = future.groupby("entity_key")[["spend", "revenue"]].sum()
            future_spend = future_totals["spend"].to_dict()
            future_revenue = future_totals["revenue"].to_dict()
            plan = active[["channel", "campaign_id", "campaign_name"]].copy()
            plan["budget"] = plan.apply(
                lambda row: float(
                    future_spend.get(_entity_key(row["channel"], row["campaign_id"]), 0.0)
                ),
                axis=1,
            )
            plan["horizon_days"] = int(horizon)

            recent = history[history["date"] > origin - pd.Timedelta(days=int(horizon))].copy()
            recent["entity_key"] = [
                _entity_key(channel, campaign_id)
                for channel, campaign_id in zip(
                    recent["channel"], recent["campaign_id"], strict=False
                )
            ]
            recent = recent[recent["entity_key"].isin(active_keys)]
            recent_totals = recent.groupby("entity_key")[["spend", "revenue"]].sum()
            naive_by_key: dict[str, float] = {}
            for key in active_keys:
                spend = float(recent_totals.at[key, "spend"]) if key in recent_totals.index else 0.0
                revenue = float(recent_totals.at[key, "revenue"]) if key in recent_totals.index else 0.0
                roas = revenue / spend if spend > 0 else 0.0
                naive_by_key[key] = roas * float(future_spend.get(key, 0.0))

            predicted, _ = forecast_portfolio(
                history,
                artifact,
                horizons=(int(horizon),),
                simulations=int(simulations),
                budget_plan=plan,
                seed=20260717 + fold * 100,
            )
            for _, row in predicted.iterrows():
                keys = _prediction_keys(row, active)
                actual_revenue = float(sum(future_revenue.get(key, 0.0) for key in keys))
                actual_spend = float(sum(future_spend.get(key, 0.0) for key in keys))
                naive_revenue = float(sum(naive_by_key.get(key, 0.0) for key in keys))
                records.append(
                    {
                        "fold": int(fold),
                        "origin": origin.date().isoformat(),
                        "horizon_days": int(horizon),
                        "forecast_level": str(row["forecast_level"]),
                        "channel": str(row["channel"]),
                        "campaign_type": str(row["campaign_type"]),
                        "campaign_id": str(row["campaign_id"]),
                        "campaign_name": str(row["campaign_name"]),
                        "support_status": str(row["support_status"]),
                        "lifecycle_state": str(row["lifecycle_state"]),
                        "actual_revenue": actual_revenue,
                        "actual_spend": actual_spend,
                        "revenue_p10": float(row["revenue_p10"]),
                        "revenue_p50": float(row["revenue_p50"]),
                        "revenue_p90": float(row["revenue_p90"]),
                        "naive_revenue": naive_revenue,
                        "covered": bool(
                            float(row["revenue_p10"])
                            <= actual_revenue
                            <= float(row["revenue_p90"])
                        ),
                    }
                )

    results = pd.DataFrame(records)
    summaries = [
        _metric_record(int(horizon), str(level), group)
        for (horizon, level), group in results.groupby(["horizon_days", "forecast_level"])
    ]
    report = {
        "protocol": {
            "future_budget": "actual future spend for every cutoff-active campaign, including zeros",
            "baseline": "matched campaign recent ROAS multiplied by that campaign's future spend",
            "evaluation_filter": "positive-actual-spend entities",
            "intended_interval": "P10-P90 (80%)",
        },
        "summary": summaries,
        "campaign_slices": _campaign_slice_records(results),
    }
    return results, report

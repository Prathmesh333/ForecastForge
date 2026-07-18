"""Serializable empirical-Bayes priors for campaign lifecycle forecasting."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import MODEL_KIND, SCHEMA_VERSION


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _sample_values(values: Iterable[float], maximum: int = 21) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return [1.0]
    low, high = np.quantile(array, [0.01, 0.99]) if array.size >= 20 else (array.min(), array.max())
    array = np.clip(array, low, high)
    point_count = min(maximum, 21 if array.size >= 20 else max(1, min(7, array.size)))
    quantiles = np.linspace(0, 1, point_count)
    array = np.quantile(array, quantiles)
    return [float(value) for value in array]


def _distribution_stats(group: pd.DataFrame) -> dict:
    valid_spend = group.loc[group["spend"] > 0, "spend"].astype(float)
    positive = group[(group["spend"] > 0) & (group["revenue"] > 0)].copy()
    positive_roas = (positive["revenue"] / positive["spend"]).replace([np.inf, -np.inf], np.nan).dropna()
    spend_active = group[group["spend"] > 0]
    nonzero = int((spend_active["revenue"] > 0).sum())
    rows = int(len(group))
    total_spend = float(group["spend"].sum())
    total_revenue = float(group["revenue"].sum())
    return {
        "rows": rows,
        "campaigns": int(group[["channel", "campaign_id"]].drop_duplicates().shape[0]),
        "spend_active_rows": int(len(spend_active)),
        "p_nonzero": float((nonzero + 1.0) / (len(spend_active) + 2.0)) if len(spend_active) else 0.5,
        "observed_roas": float(total_revenue / total_spend) if total_spend > 0 else 0.0,
        "positive_roas": _sample_values(positive_roas),
        "median_positive_roas": float(positive_roas.median()) if len(positive_roas) else 1.0,
        "spend_q10": float(valid_spend.quantile(0.10)) if len(valid_spend) else 0.0,
        "spend_median": float(valid_spend.median()) if len(valid_spend) else 0.0,
        "spend_q90": float(valid_spend.quantile(0.90)) if len(valid_spend) else 0.0,
        "spend_q05": float(valid_spend.quantile(0.05)) if len(valid_spend) else 0.0,
        "spend_q95": float(valid_spend.quantile(0.95)) if len(valid_spend) else 0.0,
    }


def _group_stats(df: pd.DataFrame, columns: list[str], min_rows: int = 1) -> dict[str, dict]:
    result: dict[str, dict] = {}
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for group_values, group in df.groupby(grouper, dropna=False):
        if len(group) < min_rows:
            continue
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        result[_key(*values)] = _distribution_stats(group)
    return result


def _seasonality(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    working = df[(df["spend"] > 0) & (df["revenue"] >= 0)].copy()
    working["month"] = working["date"].dt.month
    result: dict[str, dict[int, float]] = {}
    for (channel, family), group in working.groupby(["channel", "campaign_family"]):
        monthly = group.groupby("month")[["revenue", "spend"]].sum()
        monthly["roas"] = monthly["revenue"] / monthly["spend"].replace(0, np.nan)
        anchor = float(monthly["roas"].median())
        if not np.isfinite(anchor) or anchor <= 0:
            continue
        factors = (monthly["roas"] / anchor).clip(0.5, 1.8).fillna(1.0)
        # Sparse month estimates are shrunk toward no seasonal effect.
        counts = group.groupby("month").size()
        factors = 1.0 + (factors - 1.0) * (counts / (counts + 30.0))
        result[_key(channel, family)] = {int(month): float(value) for month, value in factors.items()}
    return result


def _channel_shocks(df: pd.DataFrame) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for channel, group in df.groupby("channel"):
        daily = group.groupby("date")[["revenue", "spend"]].sum().sort_index()
        daily["roas"] = daily["revenue"] / daily["spend"].replace(0, np.nan)
        baseline = daily["roas"].rolling(28, min_periods=7).median()
        factors = (daily["roas"] / baseline).replace([np.inf, -np.inf], np.nan).dropna()
        factors = factors.clip(0.35, 2.5)
        result[str(channel)] = _sample_values(factors, maximum=21)
    return result


def _lifecycle(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    campaigns = (
        df.groupby(["channel", "campaign_id", "campaign_family"])["date"]
        .agg(["min", "max"])
        .reset_index()
    )
    campaigns["lifetime"] = (campaigns["max"] - campaigns["min"]).dt.days + 1
    ages = (7, 14, 30, 60, 90, 180, 365)
    result: dict[str, dict[int, float]] = {}
    for family, group in campaigns.groupby("campaign_family"):
        result[str(family)] = {
            age: float((group["lifetime"] >= age).mean()) for age in ages
        }
    return result


def _launch_outcomes(df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Estimate persistent launch success after excluding right-censored cohorts."""
    records: list[dict] = []
    channel_ends = df.groupby("channel")["date"].max().to_dict()
    for (channel, campaign_id, family), group in df.groupby(
        ["channel", "campaign_id", "campaign_family"]
    ):
        start = group["date"].min()
        channel_end = channel_ends[channel]
        for horizon in (30, 60, 90):
            end = start + pd.Timedelta(days=horizon - 1)
            if end > channel_end:
                continue
            launch = group[(group["date"] >= start) & (group["date"] <= end)]
            if float(launch["spend"].sum()) <= 0:
                continue
            records.append(
                {
                    "channel": str(channel),
                    "campaign_id": str(campaign_id),
                    "campaign_family": str(family),
                    "horizon_days": int(horizon),
                    "success": int(float(launch["revenue"].sum()) > 0),
                }
            )

    columns = ["channel", "campaign_id", "campaign_family", "horizon_days", "success"]
    cohorts = pd.DataFrame(records, columns=columns)

    def summarize(columns: list[str]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        if cohorts.empty:
            return result
        grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
        for values, group in cohorts.groupby(grouper, dropna=False):
            parts = values if isinstance(values, tuple) else (values,)
            campaigns = int(len(group))
            successes = int(group["success"].sum())
            result[_key(*parts)] = {
                "campaigns": campaigns,
                "successes": successes,
                "p_success": float((successes + 1.0) / (campaigns + 2.0)),
            }
        return result

    return {
        "family": summarize(["channel", "campaign_family", "horizon_days"]),
        "channel": summarize(["channel", "horizon_days"]),
        "global": summarize(["horizon_days"]),
    }


def build_model_artifact(df: pd.DataFrame, quality_report: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Cannot train an artifact from an empty dataset")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_min_date": df["date"].min().date().isoformat(),
        "training_max_date": df["date"].max().date().isoformat(),
        "training_rows": int(len(df)),
        "training_campaigns": int(df[["channel", "campaign_id"]].drop_duplicates().shape[0]),
        "global": _distribution_stats(df),
        "groups": {
            "analog": _group_stats(df, ["analog_key"], min_rows=5),
            "family_funnel": _group_stats(
                df, ["channel", "campaign_family", "funnel_signal"], min_rows=5
            ),
            "family": _group_stats(df, ["channel", "campaign_family"], min_rows=5),
            "channel": _group_stats(df, ["channel"], min_rows=5),
        },
        "seasonality": _seasonality(df),
        "channel_shocks": _channel_shocks(df),
        "lifecycle_survival": _lifecycle(df),
        "launch_success": _launch_outcomes(df),
        "quality_snapshot": quality_report.to_dict(orient="records"),
        "assumptions": {
            "meta_conversion_field": "treated as attributed conversion value/revenue",
            "attribution": "provided channel attribution is treated as source of truth",
            "budget_response": "conditional planning scenario, not causal incrementality",
        },
    }
    return artifact


def adapt_model_artifact(
    base_artifact: dict,
    runtime_df: pd.DataFrame,
    quality_report: pd.DataFrame,
    *,
    minimum_runtime_rows: int = 30,
) -> dict:
    """Re-estimate priors from scoring-time history without changing the saved artifact.

    The committed artifact is a portable fallback. When the evaluator supplies a larger or
    differently distributed history, the transient artifact learns its peer groups, seasonality,
    sparsity, and channel shocks from that history. Sparse or missing runtime groups retain the
    corresponding committed fallback.
    """
    if runtime_df.empty:
        raise ValueError("Cannot adapt an artifact from an empty runtime dataset")

    runtime = build_model_artifact(runtime_df, quality_report)
    adapted = deepcopy(base_artifact)
    runtime_rows = int(runtime["training_rows"])
    runtime_campaigns = int(runtime["training_campaigns"])
    adapted.setdefault("groups", {})
    for base_groups in adapted["groups"].values():
        for stats in base_groups.values():
            stats.setdefault("prior_layer", "committed fallback")
    adapted.setdefault("global", {}).setdefault("prior_layer", "committed fallback")

    runtime_global = runtime["global"]
    use_runtime_global = (
        runtime_rows >= int(minimum_runtime_rows)
        and int(runtime_global.get("spend_active_rows", 0)) >= 14
        and runtime_campaigns >= 2
    )

    group_overrides = 0
    for group_name, runtime_groups in runtime.get("groups", {}).items():
        merged_groups = dict(adapted["groups"].get(group_name, {}))
        for key, runtime_stats in runtime_groups.items():
            reliable = (
                int(runtime_stats.get("rows", 0)) >= 14
                and int(runtime_stats.get("spend_active_rows", 0)) >= 7
                and int(runtime_stats.get("campaigns", 0)) >= 2
            )
            # Keep sparse runtime-only groups so an explicit planned launch can borrow from
            # one real peer. Historical campaigns still require two campaigns in choose_peer.
            if reliable or key not in merged_groups:
                selected = deepcopy(runtime_stats)
                selected["prior_layer"] = "runtime" if reliable else "runtime sparse"
                merged_groups[key] = selected
                group_overrides += int(reliable)
        adapted["groups"][group_name] = merged_groups

    if use_runtime_global:
        adapted["global"] = deepcopy(runtime_global)
        adapted["global"]["prior_layer"] = "runtime"

    # Merge seasonality at month granularity. A sparse runtime month can fill a missing value,
    # while replacing a committed month requires evidence from at least two campaigns.
    merged_seasonality = deepcopy(adapted.get("seasonality", {}))
    seasonality_overrides = 0
    runtime_months = runtime_df.copy()
    runtime_months["month"] = runtime_months["date"].dt.month
    for key, factors in runtime.get("seasonality", {}).items():
        channel, family = key.split("||", 1)
        support = runtime_months[
            (runtime_months["channel"].astype(str) == channel)
            & (runtime_months["campaign_family"].astype(str) == family)
        ]
        destination = dict(merged_seasonality.get(key, {}))
        for month, value in factors.items():
            month_support = support[support["month"] == int(month)]
            reliable = (
                len(month_support) >= 14
                and month_support[["channel", "campaign_id"]].drop_duplicates().shape[0] >= 2
            )
            if reliable or int(month) not in destination:
                destination[int(month)] = float(value)
                seasonality_overrides += 1
        merged_seasonality[key] = destination
    adapted["seasonality"] = merged_seasonality

    # A shock distribution needs at least 28 dates and 14 valid rolling comparisons.
    merged_shocks = dict(adapted.get("channel_shocks", {}))
    shock_overrides = 0
    for channel, values in runtime.get("channel_shocks", {}).items():
        channel_frame = runtime_df[runtime_df["channel"].astype(str) == str(channel)]
        daily = channel_frame.groupby("date")[["revenue", "spend"]].sum().sort_index()
        daily_roas = daily["revenue"] / daily["spend"].replace(0, np.nan)
        baseline = daily_roas.rolling(28, min_periods=7).median()
        valid_shocks = (daily_roas / baseline).replace([np.inf, -np.inf], np.nan).dropna()
        if int(daily.index.nunique()) >= 28 and len(valid_shocks) >= 14:
            merged_shocks[str(channel)] = values
            shock_overrides += 1
    adapted["channel_shocks"] = merged_shocks

    merged_lifecycle = dict(adapted.get("lifecycle_survival", {}))
    for family, values in runtime.get("lifecycle_survival", {}).items():
        family_campaigns = runtime_df.loc[
            runtime_df["campaign_family"].astype(str) == str(family),
            ["channel", "campaign_id"],
        ].drop_duplicates()
        if len(family_campaigns) >= 5:
            merged_lifecycle[str(family)] = values
    adapted["lifecycle_survival"] = merged_lifecycle
    merged_launch = deepcopy(adapted.get("launch_success", {}))
    launch_overrides = 0
    for level, runtime_groups in runtime.get("launch_success", {}).items():
        destination = dict(merged_launch.get(level, {}))
        minimum_campaigns = 3 if level == "family" else 5
        for key, stats in runtime_groups.items():
            reliable = int(stats.get("campaigns", 0)) >= minimum_campaigns
            if reliable or key not in destination:
                selected = deepcopy(stats)
                selected["prior_layer"] = "runtime" if reliable else "runtime sparse"
                destination[key] = selected
                launch_overrides += int(reliable)
        merged_launch[level] = destination
    adapted["launch_success"] = merged_launch


    runtime_group_count = sum(
        stats.get("prior_layer") == "runtime"
        for groups in adapted["groups"].values()
        for stats in groups.values()
    )
    fallback_group_count = sum(
        stats.get("prior_layer") == "committed fallback"
        for groups in adapted["groups"].values()
        for stats in groups.values()
    )

    adapted["training_min_date"] = runtime["training_min_date"]
    adapted["training_max_date"] = runtime["training_max_date"]
    adapted["training_rows"] = runtime_rows
    adapted["training_campaigns"] = runtime_campaigns
    adapted["quality_snapshot"] = runtime["quality_snapshot"]
    adapted["runtime_adaptation"] = {
        "enabled": True,
        "runtime_rows": runtime_rows,
        "runtime_campaigns": runtime_campaigns,
        "runtime_min_date": runtime["training_min_date"],
        "runtime_max_date": runtime["training_max_date"],
        "global_prior_source": "runtime" if use_runtime_global else "committed fallback",
        "runtime_group_overrides": int(runtime_group_count),
        "new_or_replaced_groups": int(group_overrides),
        "committed_fallback_groups": int(fallback_group_count),
        "seasonality_month_overrides": int(seasonality_overrides),
        "channel_shock_overrides": int(shock_overrides),
        "launch_success_overrides": int(launch_overrides),
        "fallback_model_kind": base_artifact.get("model_kind", "unknown"),
    }
    adapted.setdefault("assumptions", {})["runtime_adaptation"] = (
        "peer priors and calibration are rebuilt transiently from all supplied scoring history; "
        "the committed artifact is used only for missing or sparse groups"
    )
    return adapted
def choose_peer(artifact: dict, row: pd.Series) -> tuple[dict, str]:
    candidates = [
        ("analog", _key(row["analog_key"]), "cross-platform analog"),
        (
            "family_funnel",
            _key(row["channel"], row["campaign_family"], row["funnel_signal"]),
            "same channel/family/funnel",
        ),
        (
            "family",
            _key(row["channel"], row["campaign_family"]),
            "same channel/family",
        ),
        ("channel", _key(row["channel"]), "channel prior"),
    ]
    minimum_peer_campaigns = 1 if bool(row.get("is_planned_new", False)) else 2
    for group_name, key, label in candidates:
        stats = artifact["groups"].get(group_name, {}).get(key)
        if (
            stats
            and int(stats.get("rows", 0)) >= 14
            and int(stats.get("spend_active_rows", 0)) >= 7
            and int(stats.get("campaigns", 0)) >= minimum_peer_campaigns
        ):
            layer = str(stats.get("prior_layer", "artifact"))
            return stats, f"{label} [{layer}]"
    global_stats = artifact["global"]
    layer = str(global_stats.get("prior_layer", "artifact"))
    return global_stats, f"global prior [{layer}]"


def choose_launch_success(
    artifact: dict, row: pd.Series, horizon_days: int
) -> tuple[float, str]:
    groups = artifact.get("launch_success", {})
    candidates = [
        (
            "family",
            _key(row["channel"], row["campaign_family"], int(horizon_days)),
            3,
            "channel/family launch cohort",
        ),
        (
            "channel",
            _key(row["channel"], int(horizon_days)),
            5,
            "channel launch cohort",
        ),
        ("global", _key(int(horizon_days)), 5, "global launch cohort"),
    ]
    sparse_fallback: tuple[dict, str] | None = None
    for level, key, minimum, label in candidates:
        stats = groups.get(level, {}).get(key)
        if not stats:
            continue
        source = f"{label} [{stats.get('prior_layer', 'artifact')}]"
        if int(stats.get("campaigns", 0)) >= minimum:
            return float(np.clip(stats.get("p_success", 0.65), 0.02, 0.98)), source
        if sparse_fallback is None:
            sparse_fallback = (stats, source)
    if sparse_fallback is not None:
        stats, source = sparse_fallback
        return float(np.clip(stats.get("p_success", 0.65), 0.05, 0.95)), source
    return 0.65, "conservative default launch prior"

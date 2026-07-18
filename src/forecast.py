"""Bottom-up probabilistic forecasts with lifecycle and evidence-support controls."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np
import pandas as pd

from .boosting import predict_boosting_bundle
from .constants import (
    ACTIVE_WINDOW_DAYS,
    DEFAULT_SIMULATIONS,
    DEFAULT_TARGET_ROAS,
    RANDOM_SEED,
    RECENT_WINDOW_DAYS,
)
from .ingest import campaign_analog_key, infer_campaign_family, infer_funnel_signal
from .model import choose_launch_success, choose_peer


AGGREGATE_INSUFFICIENT_SHARE = 0.25
AGGREGATE_UNSUPPORTED_SHARE = 0.10


def _campaign_key(channel: object, campaign_id: object) -> str:
    return f"{channel}||{campaign_id}"


def _daily_campaign(group: pd.DataFrame, as_of: pd.Timestamp, days: int = 56) -> pd.DataFrame:
    start = max(group["date"].min(), as_of - pd.Timedelta(days=days - 1))
    index = pd.date_range(start, as_of, freq="D")
    daily = group.groupby("date")[["spend", "revenue"]].sum().reindex(index, fill_value=0.0)
    daily.index.name = "date"
    return daily


def _lifecycle_profile(group: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    first = group["date"].min()
    last = group["date"].max()
    age = int((as_of - first).days + 1)
    recency = int((as_of - last).days)
    daily = _daily_campaign(group, as_of)
    recent = daily.tail(14)
    previous = daily.iloc[-28:-14]
    recent_spend = float(recent["spend"].sum())
    previous_spend = float(previous["spend"].sum())
    recent_roas = float(recent["revenue"].sum() / recent_spend) if recent_spend > 0 else 0.0
    previous_roas = (
        float(previous["revenue"].sum() / previous_spend) if previous_spend > 0 else 0.0
    )
    if recency > ACTIVE_WINDOW_DAYS:
        state = "Inactive"
    elif age <= 14:
        state = "Launch"
    elif age <= 30:
        state = "Ramp"
    elif previous_spend > 0 and recent_spend < previous_spend * 0.55:
        state = "Declining"
    else:
        state = "Mature"
    if recent_roas > 0 and previous_roas > 0:
        # Trend is intentionally damped: short-term ROAS ratios are noisy and a
        # full-strength extrapolation badly overreacts around sale periods.
        trend = float((recent_roas / previous_roas) ** 0.25)
    else:
        trend = 1.0
    trend = float(np.clip(trend, 0.85, 1.15))
    window_days = min(RECENT_WINDOW_DAYS, max(age, 1))
    baseline_daily_spend = float(daily.tail(window_days)["spend"].sum() / window_days)
    return {
        "first_date": first,
        "last_date": last,
        "age_days": age,
        "recency_days": recency,
        "state": state,
        "trend_factor": trend,
        "baseline_daily_spend": baseline_daily_spend,
        "daily": daily,
    }


def active_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    channel_ends = df.groupby("channel")["date"].max().to_dict()
    last_seen = df.groupby(["channel", "campaign_id"])["date"].max().reset_index()
    last_seen["days_from_channel_end"] = last_seen.apply(
        lambda row: int((channel_ends[row["channel"]] - row["date"]).days), axis=1
    )
    active = last_seen[last_seen["days_from_channel_end"] <= ACTIVE_WINDOW_DAYS]
    keys = set(
        _campaign_key(row.channel, row.campaign_id)
        for row in active.itertuples(index=False)
    )
    result = df[
        df.apply(lambda row: _campaign_key(row["channel"], row["campaign_id"]) in keys, axis=1)
    ].copy()
    return result


def default_campaign_budgets(df: pd.DataFrame, horizon_days: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for (channel, campaign_id), group in active_campaigns(df).groupby(["channel", "campaign_id"]):
        as_of = df.loc[df["channel"] == channel, "date"].max()
        profile = _lifecycle_profile(group, as_of)
        result[_campaign_key(channel, campaign_id)] = max(
            0.0, profile["baseline_daily_spend"] * horizon_days
        )
    return result


RESOLVED_PLAN_COLUMNS = [
    "channel",
    "campaign_id",
    "campaign_name",
    "native_campaign_type",
    "campaign_family",
    "funnel_signal",
    "analog_key",
    "horizon_days",
    "budget",
    "is_new_campaign",
    "plan_row",
]


def _clean_plan_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _plan_flag(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(
        "is_new_campaign must be true/false, yes/no, or 1/0; "
        f"received {value!r}"
    )


def _analog_key(value: object) -> str:
    return campaign_analog_key(value)


def _latest_campaign_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "channel",
        "campaign_id",
        "campaign_name",
        "native_campaign_type",
        "campaign_family",
        "funnel_signal",
        "analog_key",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    dimensions = (
        df.sort_values("date")
        .drop_duplicates(["channel", "campaign_id"], keep="last")[columns]
        .copy()
    )
    dimensions["campaign_id"] = dimensions["campaign_id"].astype(str)
    return dimensions.reset_index(drop=True)


def _canonical_plan_channel(value: object, available: list[str], row_number: int) -> str:
    text = _clean_plan_text(value)
    if text is None:
        raise ValueError(f"Budget plan row {row_number} has a blank channel")
    direct = {channel.casefold(): channel for channel in available}
    aliases = {
        "google": "Google Ads",
        "google ads": "Google Ads",
        "meta": "Meta Ads",
        "meta ads": "Meta Ads",
        "facebook": "Meta Ads",
        "microsoft": "Microsoft Ads",
        "microsoft ads": "Microsoft Ads",
        "bing": "Microsoft Ads",
    }
    canonical = direct.get(text.casefold())
    if canonical is None:
        alias = aliases.get(text.casefold())
        canonical = alias if alias in available else None
    if canonical is None:
        raise ValueError(
            f"Budget plan row {row_number} uses unknown channel {text!r}; "
            f"known channels are {available}"
        )
    return canonical


def _resolve_budget_plan(
    df: pd.DataFrame,
    active: pd.DataFrame,
    plan: pd.DataFrame | None,
) -> pd.DataFrame:
    if plan is None or plan.empty:
        return pd.DataFrame(columns=RESOLVED_PLAN_COLUMNS)

    required = {"channel", "horizon_days", "budget"}
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(f"Budget plan is missing columns: {sorted(missing)}")
    if "campaign_id" not in plan and "campaign_name" not in plan:
        raise ValueError("Budget plan needs campaign_id or campaign_name")

    history_dimensions = _latest_campaign_dimensions(df)
    active_dimensions = _latest_campaign_dimensions(active)
    active_keys = {
        _campaign_key(row.channel, row.campaign_id)
        for row in active_dimensions.itertuples(index=False)
    }
    available_channels = sorted(history_dimensions["channel"].unique().tolist())
    records: list[dict] = []

    for index, row in plan.reset_index(drop=True).iterrows():
        row_number = int(index) + 2
        channel = _canonical_plan_channel(row.get("channel"), available_channels, row_number)
        campaign_id = _clean_plan_text(row.get("campaign_id"))
        campaign_name = _clean_plan_text(row.get("campaign_name"))
        is_new = _plan_flag(row.get("is_new_campaign", False))
        try:
            horizon_raw = float(row.get("horizon_days"))
            budget = float(row.get("budget"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Budget plan row {row_number} has a non-numeric horizon or budget"
            ) from error
        if not np.isfinite(horizon_raw) or horizon_raw <= 0 or horizon_raw % 1:
            raise ValueError(
                f"Budget plan row {row_number} horizon_days must be a positive whole number"
            )
        if not np.isfinite(budget) or budget < 0:
            raise ValueError(
                f"Budget plan row {row_number} budget must be finite and non-negative"
            )
        horizon_days = int(horizon_raw)

        channel_history = history_dimensions[
            history_dimensions["channel"] == channel
        ].copy()
        if is_new:
            campaign_type = _clean_plan_text(row.get("campaign_type"))
            missing_new = [
                field
                for field, value in (
                    ("campaign_id", campaign_id),
                    ("campaign_name", campaign_name),
                    ("campaign_type", campaign_type),
                )
                if value is None
            ]
            if missing_new:
                raise ValueError(
                    f"Budget plan row {row_number} declares a new campaign but is missing "
                    f"{missing_new}"
                )
            if budget <= 0:
                raise ValueError(
                    f"Budget plan row {row_number} declares a new campaign with zero budget"
                )
            id_collision = channel_history["campaign_id"].astype(str) == str(campaign_id)
            name_collision = (
                channel_history["campaign_name"].astype(str).str.casefold()
                == str(campaign_name).casefold()
            )
            if (id_collision | name_collision).any():
                raise ValueError(
                    f"Budget plan row {row_number} declares {campaign_name!r} as new, "
                    "but its ID or name already exists in that channel"
                )
            native_type = str(campaign_type)
            family = infer_campaign_family(campaign_name, native_type)
            funnel = _clean_plan_text(row.get("funnel_signal")) or infer_funnel_signal(
                campaign_name
            )
            record = {
                "channel": channel,
                "campaign_id": str(campaign_id),
                "campaign_name": str(campaign_name),
                "native_campaign_type": native_type,
                "campaign_family": family,
                "funnel_signal": funnel,
                "analog_key": _analog_key(campaign_name),
            }
        else:
            if campaign_id is None and campaign_name is None:
                raise ValueError(
                    f"Budget plan row {row_number} needs campaign_id or campaign_name"
                )
            matches = channel_history
            if campaign_id is not None:
                matches = matches[
                    matches["campaign_id"].astype(str) == str(campaign_id)
                ]
            if campaign_name is not None:
                matches = matches[
                    matches["campaign_name"].astype(str).str.casefold()
                    == str(campaign_name).casefold()
                ]
            if len(matches) == 0:
                identity = campaign_id or campaign_name
                raise ValueError(
                    f"Budget plan row {row_number} does not match a campaign in "
                    f"{channel}: {identity!r}. Set is_new_campaign=true only for a "
                    "genuinely new campaign with a new ID."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Budget plan row {row_number} is ambiguous; use campaign_id "
                    f"to identify {campaign_name!r}"
                )
            candidate = matches.iloc[0]
            key = _campaign_key(candidate["channel"], candidate["campaign_id"])
            if key not in active_keys:
                raise ValueError(
                    f"Budget plan row {row_number} targets inactive campaign "
                    f"{candidate['campaign_name']!r}. Relaunch it with a new campaign ID "
                    "and is_new_campaign=true."
                )
            record = {
                column: candidate[column]
                for column in (
                    "channel",
                    "campaign_id",
                    "campaign_name",
                    "native_campaign_type",
                    "campaign_family",
                    "funnel_signal",
                    "analog_key",
                )
            }

        record.update(
            {
                "horizon_days": horizon_days,
                "budget": budget,
                "is_new_campaign": bool(is_new),
                "plan_row": row_number,
            }
        )
        records.append(record)

    resolved = pd.DataFrame(records, columns=RESOLVED_PLAN_COLUMNS)
    duplicates = resolved.duplicated(
        ["channel", "campaign_id", "horizon_days"], keep=False
    )
    if duplicates.any():
        rows = sorted(resolved.loc[duplicates, "plan_row"].astype(int).tolist())
        raise ValueError(
            "Budget plan has duplicate rows resolving to the same campaign and horizon: "
            f"CSV rows {rows}"
        )

    new_rows = resolved[resolved["is_new_campaign"]]
    for (channel, campaign_id), group in new_rows.groupby(["channel", "campaign_id"]):
        metadata = [
            "campaign_name",
            "native_campaign_type",
            "campaign_family",
            "funnel_signal",
            "analog_key",
        ]
        if any(group[column].nunique(dropna=False) > 1 for column in metadata):
            rows = sorted(group["plan_row"].astype(int).tolist())
            raise ValueError(
                f"New campaign {channel}/{campaign_id} has conflicting metadata "
                f"across CSV rows {rows}"
            )
    return resolved


def _planned_new_history(
    resolved_plan: pd.DataFrame,
    df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    selected = resolved_plan[
        (resolved_plan["horizon_days"] == horizon_days)
        & resolved_plan["is_new_campaign"]
    ].drop_duplicates(["channel", "campaign_id"])
    rows: list[dict] = []
    for row in selected.itertuples(index=False):
        as_of = df.loc[df["channel"] == row.channel, "date"].max()
        rows.append(
            {
                "date": as_of,
                "channel": row.channel,
                "campaign_id": str(row.campaign_id),
                "campaign_name": row.campaign_name,
                "native_campaign_type": row.native_campaign_type,
                "campaign_family": row.campaign_family,
                "funnel_signal": row.funnel_signal,
                "analog_key": row.analog_key,
                "spend": 0.0,
                "revenue": 0.0,
                "conversions": np.nan,
                "clicks": np.nan,
                "impressions": np.nan,
                "daily_budget": np.nan,
                "source_file": "future_budgets.csv (planned new)",
                "source_revenue_field": "not observed",
                "is_planned_new": True,
            }
        )
    return pd.DataFrame(rows)


def _budget_overrides(
    resolved_plan: pd.DataFrame,
    horizon_days: int,
) -> dict[str, float]:
    selected = resolved_plan[resolved_plan["horizon_days"] == horizon_days]
    return {
        _campaign_key(row.channel, row.campaign_id): float(row.budget)
        for row in selected.itertuples(index=False)
    }


def _seasonality_factors(
    artifact: dict, channel: str, family: str, start: pd.Timestamp, horizon_days: int
) -> np.ndarray:
    mapping = artifact.get("seasonality", {}).get(f"{channel}||{family}", {})
    dates = pd.date_range(start + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    return np.asarray([float(mapping.get(int(date.month), 1.0)) for date in dates], dtype=float)


def _support_status(profile: dict, peer: dict, daily_budget: float) -> str:
    if profile["age_days"] < 14 or peer.get("rows", 0) < 14 or peer.get("p_nonzero", 0) < 0.03:
        return "Insufficient Evidence"
    low = float(peer.get("spend_q05", 0.0))
    high = float(peer.get("spend_q95", 0.0))
    if high > 0 and (daily_budget < low * 0.5 or daily_budget > high * 1.25):
        return "Extrapolating"
    return "Supported"


def _recommendation(probability: float, support_status: str, budget: float) -> str:
    if budget <= 0:
        return "REVIEW"
    if support_status != "Supported":
        return "TEST"
    if probability >= 0.80:
        return "ACT"
    if probability >= 0.55:
        return "HOLD"
    return "REVIEW"


def _summarize_samples(
    samples: np.ndarray,
    budget: float,
    target_roas: float,
    support_status: str,
) -> dict:
    p10, p50, p90 = np.quantile(samples, [0.10, 0.50, 0.90])
    if budget > 0:
        probability = float(np.mean(samples / budget >= target_roas))
        roas = (p10 / budget, p50 / budget, p90 / budget)
    else:
        probability = 0.0
        roas = (0.0, 0.0, 0.0)
    return {
        "revenue_p10": float(max(p10, 0)),
        "revenue_p50": float(max(p50, 0)),
        "revenue_p90": float(max(p90, 0)),
        "roas_p10": float(max(roas[0], 0)),
        "roas_p50": float(max(roas[1], 0)),
        "roas_p90": float(max(roas[2], 0)),
        "probability_target": probability,
        "recommendation": _recommendation(probability, support_status, budget),
    }


def _campaign_samples(
    group: pd.DataFrame,
    artifact: dict,
    horizon_days: int,
    budget: float,
    target_roas: float,
    simulations: int,
    rng: np.random.Generator,
    channel_shock: np.ndarray,
    as_of: pd.Timestamp,
    challenger_roas: float | None = None,
) -> tuple[dict, np.ndarray]:
    representative = group.iloc[-1]
    planned_new = bool(representative.get("is_planned_new", False))
    profile = _lifecycle_profile(group, as_of)
    peer, peer_source = choose_peer(artifact, representative)
    daily = profile["daily"]
    recent = daily.tail(min(56, len(daily)))
    recent_positive = recent[(recent["spend"] > 0) & (recent["revenue"] > 0)]
    own_roas = (recent_positive["revenue"] / recent_positive["spend"]).to_numpy(dtype=float)
    peer_roas = np.asarray(peer.get("positive_roas", [1.0]), dtype=float)
    peer_roas = peer_roas[np.isfinite(peer_roas) & (peer_roas >= 0)]
    if peer_roas.size == 0:
        peer_roas = np.asarray([1.0])

    spend_active_recent = recent[recent["spend"] > 0]
    observed_days = 0 if planned_new else max(len(spend_active_recent), 1)
    own_nonzero = int((spend_active_recent["revenue"] > 0).sum())
    prior_strength = 14.0 if profile["age_days"] <= 30 else 7.0
    p_nonzero = (own_nonzero + prior_strength * float(peer["p_nonzero"])) / (
        observed_days + prior_strength
    )
    p_nonzero = float(np.clip(p_nonzero, 0.01, 0.995))

    # Calibrate the forecast centre to realized, spend-weighted ROAS. The peer
    # prior is most influential for launches and naturally fades as evidence
    # accumulates. This keeps mature campaigns competitive with a strong recent
    # baseline without giving up cross-platform cold-start transfer.
    anchor_days = min(max(int(horizon_days), 28), 56)
    anchor = daily.tail(min(anchor_days, len(daily)))
    anchor_spend = float(anchor["spend"].sum())
    own_expected_roas = (
        float(anchor["revenue"].sum() / anchor_spend) if anchor_spend > 0 else 0.0
    )
    peer_expected_roas = max(float(peer.get("observed_roas", 0.0)), 0.0)
    evidence_days = int(((anchor["spend"] > 0) | (anchor["revenue"] > 0)).sum())
    centre_own_weight = min(0.92, evidence_days / (evidence_days + 10.0))
    centre_own_weight *= min(1.0, profile["age_days"] / 30.0)
    if own_expected_roas > 0 and peer_expected_roas > 0:
        # Prevent a broad peer group from overwhelming direct campaign evidence.
        bounded_peer = float(
            np.clip(peer_expected_roas, own_expected_roas * 0.55, own_expected_roas * 1.55)
        )
        target_expected_roas = (
            centre_own_weight * own_expected_roas
            + (1.0 - centre_own_weight) * bounded_peer
        )
    else:
        target_expected_roas = own_expected_roas or peer_expected_roas
    target_expected_roas = float(max(target_expected_roas, 0.05))
    empirical_roas_center = target_expected_roas
    boosting_weight = 0.0
    boosting_roas = np.nan
    bounded_boosting_roas = np.nan
    if (
        not planned_new
        and challenger_roas is not None
        and np.isfinite(challenger_roas)
        and challenger_roas >= 0
    ):
        boosting_roas = float(challenger_roas)
        boosting_weight = 0.30
        boosting_weight *= min(1.0, evidence_days / 14.0)
        boosting_weight *= min(1.0, profile["age_days"] / 30.0)
        bounded_boosting_roas = float(
            np.clip(boosting_roas, target_expected_roas * 0.50, target_expected_roas * 1.75)
        )
        target_expected_roas = (
            (1.0 - boosting_weight) * target_expected_roas
            + boosting_weight * bounded_boosting_roas
        )
    hybrid_roas_center = target_expected_roas

    own_weight = min(0.85, profile["age_days"] / (profile["age_days"] + 30.0))
    own_weight *= min(1.0, len(own_roas) / 14.0)
    choose_own = rng.random((simulations, horizon_days)) < own_weight
    peer_draws = rng.choice(peer_roas, size=(simulations, horizon_days), replace=True)
    if own_roas.size:
        own_low, own_high = np.quantile(own_roas, [0.02, 0.98]) if len(own_roas) >= 10 else (own_roas.min(), own_roas.max())
        own_pool = np.clip(own_roas, own_low, own_high)
        own_draws = rng.choice(own_pool, size=(simulations, horizon_days), replace=True)
        roas_draws = np.where(choose_own, own_draws, peer_draws)
    else:
        roas_draws = peer_draws

    nonzero_draws = rng.random((simulations, horizon_days)) < p_nonzero
    base_draws = nonzero_draws * roas_draws
    simulated_centre = float(base_draws.mean())
    if simulated_centre > 0:
        roas_draws = roas_draws * (target_expected_roas / simulated_centre)
    daily_budget = budget / horizon_days if horizon_days else 0.0
    if planned_new:
        baseline_daily_spend = max(float(peer.get("spend_median", 0.0)), 0.01)
    else:
        baseline_daily_spend = max(float(profile["baseline_daily_spend"]), 0.01)
    budget_ratio = daily_budget / baseline_daily_spend if daily_budget > 0 else 0.0
    # A lightweight saturating response curve: above the observed operating
    # range, marginal ROAS falls with spend. Holiday peak plans receive a much
    # softer penalty because elevated budgets are structurally normal then.
    future_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    holiday_share = float(np.mean(future_dates.month.isin([11, 12]))) if len(future_dates) else 0.0
    elasticity = 0.20 - 0.16 * holiday_share
    response_factor = float(max(budget_ratio, 1.0) ** (-elasticity))
    response_factor = float(np.clip(response_factor, 0.60, 1.0))
    seasonal = _seasonality_factors(
        artifact,
        str(representative["channel"]),
        str(representative["campaign_family"]),
        as_of,
        horizon_days,
    )
    revenue = (
        nonzero_draws
        * roas_draws
        * daily_budget
        * seasonal.reshape(1, -1)
        * profile["trend_factor"]
        * response_factor
    )
    total_samples = revenue.sum(axis=1) * channel_shock
    support_status = (
        "Insufficient Evidence" if planned_new else _support_status(profile, peer, daily_budget)
    )
    peer_source = f"planned new -> {peer_source}" if planned_new else peer_source
    calibrated_median = float(np.median(total_samples))
    persistent_sigma = {
        "Supported": 0.38,
        "Extrapolating": 1.10,
        "Insufficient Evidence": 1.20,
    }[support_status]
    persistent_sigma += 0.30 * (1.0 - p_nonzero)
    persistent_sigma += 0.10 if profile["state"] == "Launch" else 0.05 if profile["state"] == "Ramp" else 0.0
    persistent_effect = rng.lognormal(mean=0.0, sigma=persistent_sigma, size=simulations)
    total_samples = total_samples * persistent_effect
    sparsity = max(0.0, (0.30 - p_nonzero) / 0.30)
    dark_period_probability = float(np.clip(0.25 * sparsity, 0.0, 0.25))
    dark_period = rng.random(simulations) < dark_period_probability
    total_samples = np.where(dark_period, 0.0, total_samples)

    adjusted_median = float(np.median(total_samples))
    if adjusted_median > 0:
        total_samples = total_samples * (calibrated_median / adjusted_median)
    launch_success_probability = 1.0
    launch_prior_source = "not applicable"
    if planned_new or profile["state"] == "Launch":
        launch_success_probability, launch_prior_source = choose_launch_success(
            artifact, representative, horizon_days
        )
        if not planned_new and float(group["revenue"].sum()) > 0:
            launch_success_probability = 1.0
            launch_prior_source = "own observed launch revenue"
        persistent_launch_failure = (
            rng.random(simulations) >= launch_success_probability
        )
        total_samples = np.where(persistent_launch_failure, 0.0, total_samples)
    summary = _summarize_samples(
        total_samples, budget, target_roas, support_status
    )
    summary.update(
        {
            "forecast_level": "campaign",
            "channel": str(representative["channel"]),
            "campaign_type": str(representative["campaign_family"]),
            "campaign_id": str(representative["campaign_id"]),
            "campaign_name": str(representative["campaign_name"]),
            "horizon_days": int(horizon_days),
            "budget": float(budget),
            "target_roas": float(target_roas),
            "support_status": support_status,
            "lifecycle_state": "Launch" if planned_new else profile["state"],
            "peer_source": peer_source,
            "data_points": 0 if planned_new else int(len(group)),
            "planned_new_campaign": planned_new,
            "budget_response_factor": response_factor,
            "budget_to_recent_ratio": float(budget_ratio),
            "boosting_roas_p50": float(boosting_roas),
            "bounded_boosting_roas_p50": float(bounded_boosting_roas),
            "boosting_weight": float(boosting_weight),
            "empirical_roas_center": float(empirical_roas_center),
            "hybrid_roas_center": float(hybrid_roas_center),
            "supported_budget": float(budget if support_status == "Supported" else 0.0),
            "dark_period_probability": dark_period_probability,
            "launch_success_probability": float(launch_success_probability),
            "launch_prior_source": launch_prior_source,
            "extrapolating_budget": float(budget if support_status == "Extrapolating" else 0.0),
            "insufficient_budget": float(budget if support_status == "Insufficient Evidence" else 0.0),
            "evidence_coverage": float(1.0 if support_status == "Supported" else 0.0),
        }
    )
    return summary, total_samples


def _aggregate_support(rows: list[dict], budget: float) -> tuple[str, dict[str, float]]:
    supported = float(sum(row.get("supported_budget", 0.0) for row in rows))
    extrapolating = float(sum(row.get("extrapolating_budget", 0.0) for row in rows))
    insufficient = float(sum(row.get("insufficient_budget", 0.0) for row in rows))
    if budget <= 0:
        status = "Insufficient Evidence"
        coverage = 0.0
    else:
        coverage = float(np.clip(supported / budget, 0.0, 1.0))
        insufficient_share = insufficient / budget
        unsupported_share = (extrapolating + insufficient) / budget
        if insufficient_share >= AGGREGATE_INSUFFICIENT_SHARE:
            status = "Insufficient Evidence"
        elif unsupported_share >= AGGREGATE_UNSUPPORTED_SHARE:
            status = "Extrapolating"
        else:
            status = "Supported"
    return status, {
        "supported_budget": supported,
        "extrapolating_budget": extrapolating,
        "insufficient_budget": insufficient,
        "evidence_coverage": coverage,
    }


def _aggregate_row(
    level: str,
    rows: list[dict],
    samples: list[np.ndarray],
    target_roas: float,
    **labels: str,
) -> tuple[dict, np.ndarray]:
    combined = np.sum(np.vstack(samples), axis=0)
    # Quantiles are not additive. Preserve the correlated sample distribution, but calibrate
    # its centre so every aggregate P50 exactly reconciles to its child P50 forecasts.
    child_median_total = float(sum(row["revenue_p50"] for row in rows))
    combined_median = float(np.median(combined))
    if child_median_total > 0 and combined_median > 0:
        combined = combined * (child_median_total / combined_median)
    budget = float(sum(row["budget"] for row in rows))
    support, evidence = _aggregate_support(rows, budget)
    summary = _summarize_samples(combined, budget, target_roas, support)
    summary.update(
        {
            "forecast_level": level,
            "channel": labels.get("channel", "ALL"),
            "campaign_type": labels.get("campaign_type", "ALL"),
            "campaign_id": "ALL",
            "campaign_name": "ALL",
            "horizon_days": int(rows[0]["horizon_days"]),
            "budget": budget,
            "target_roas": float(target_roas),
            "support_status": support,
            "lifecycle_state": "Portfolio",
            "peer_source": "bottom-up reconciled samples",
            "data_points": int(sum(row["data_points"] for row in rows)),
            "budget_response_factor": float(
                np.average(
                    [row.get("budget_response_factor", 1.0) for row in rows],
                    weights=[max(row["budget"], 1e-9) for row in rows],
                )
            ),
            "budget_to_recent_ratio": float(
                np.average(
                    [row.get("budget_to_recent_ratio", 1.0) for row in rows],
                    weights=[max(row["budget"], 1e-9) for row in rows],
                )
            ),
            **evidence,
        }
    )
    return summary, combined


def forecast_portfolio(
    df: pd.DataFrame,
    artifact: dict,
    horizons: tuple[int, ...] = (30, 60, 90),
    target_roas: float = DEFAULT_TARGET_ROAS,
    simulations: int = DEFAULT_SIMULATIONS,
    budget_plan: pd.DataFrame | None = None,
    channel_multipliers: Mapping[str, float] | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        raise ValueError("Cannot forecast an empty dataset")
    historical_active = active_campaigns(df).copy()
    historical_active["is_planned_new"] = False
    historical_dimensions = _latest_campaign_dimensions(historical_active)
    resolved_plan = _resolve_budget_plan(df, historical_active, budget_plan)
    all_rows: list[dict] = []
    forecast_campaign_keys: set[str] = set()
    forecast_channels: set[str] = set()
    planned_new_keys: set[str] = set()
    boosting_diagnostics: list[dict] = []

    for horizon in horizons:
        rng = np.random.default_rng(seed + int(horizon))
        horizon_active = historical_active.copy()
        new_history = _planned_new_history(resolved_plan, df, int(horizon))
        if not new_history.empty:
            horizon_active = pd.concat([horizon_active, new_history], ignore_index=True)
        campaign_dimensions = _latest_campaign_dimensions(horizon_active)
        horizon_keys = {
            _campaign_key(row.channel, row.campaign_id)
            for row in campaign_dimensions.itertuples(index=False)
        }
        forecast_campaign_keys.update(horizon_keys)
        forecast_channels.update(campaign_dimensions["channel"].astype(str))
        selected_new = resolved_plan[
            (resolved_plan["horizon_days"] == int(horizon))
            & resolved_plan["is_new_campaign"]
        ]
        planned_new_keys.update(
            _campaign_key(row.channel, row.campaign_id)
            for row in selected_new.itertuples(index=False)
        )
        defaults = default_campaign_budgets(df, int(horizon))
        overrides = _budget_overrides(resolved_plan, int(horizon))
        planned_budgets: dict[str, float] = {}
        boosting_requests: list[dict] = []
        for dimension in campaign_dimensions.itertuples(index=False):
            key = _campaign_key(dimension.channel, dimension.campaign_id)
            multiplier = float(
                (channel_multipliers or {}).get(str(dimension.channel), 1.0)
            )
            budget = float(overrides.get(key, defaults.get(key, 0.0))) * multiplier
            planned_budgets[key] = budget
            boosting_requests.append(
                {
                    "channel": str(dimension.channel),
                    "campaign_id": str(dimension.campaign_id),
                    "horizon_days": int(horizon),
                    "budget": budget,
                }
            )
        boosting_predictions, boosting_status = predict_boosting_bundle(
            df,
            pd.DataFrame(boosting_requests),
            artifact.get("boosting_bundle"),
        )
        boosting_diagnostics.append(boosting_status)
        boosting_by_key = {
            _campaign_key(row.channel, row.campaign_id): float(row.boosting_roas_p50)
            for row in boosting_predictions.itertuples(index=False)
        }
        campaign_rows: list[dict] = []
        campaign_samples: dict[str, np.ndarray] = {}
        channel_shocks: dict[str, np.ndarray] = {}
        for channel in horizon_active["channel"].unique():
            pool = np.asarray(artifact.get("channel_shocks", {}).get(str(channel), [1.0]), dtype=float)
            pool = pool[np.isfinite(pool) & (pool > 0)]
            if pool.size == 0:
                pool = np.asarray([1.0])
            drawn = rng.choice(pool, size=simulations, replace=True)
            empirical = np.exp(0.25 * np.log(np.clip(drawn, 0.35, 2.5)))
            empirical = empirical / max(float(np.median(empirical)), 1e-6)
            # This correlated shock does not disappear when campaign samples are
            # aggregated, so portfolio intervals still reflect macro uncertainty.
            # Longer horizons carry more unresolved market risk. These values
            # are deliberately material because campaign noise otherwise
            # diversifies away and produces overconfident portfolio intervals.
            horizon_sigma = 0.225 + 0.0015 * int(horizon)
            macro = rng.lognormal(
                # Preserve the scenario median at one; this shock represents
                # uncertainty around the calibrated centre, not expected uplift.
                mean=0.0,
                sigma=horizon_sigma,
                size=simulations,
            )
            channel_shocks[str(channel)] = empirical * macro

        for (channel, campaign_id), group in horizon_active.groupby(["channel", "campaign_id"]):
            key = _campaign_key(channel, campaign_id)
            budget = planned_budgets.get(key, 0.0)
            as_of = df.loc[df["channel"] == channel, "date"].max()
            row, samples = _campaign_samples(
                group,
                artifact,
                int(horizon),
                budget,
                float(target_roas),
                int(simulations),
                rng,
                channel_shocks[str(channel)],
                as_of,
                boosting_by_key.get(key),
            )
            campaign_rows.append(row)
            campaign_samples[key] = samples

        all_rows.extend(campaign_rows)

        type_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        channel_groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(campaign_rows):
            type_groups[(row["channel"], row["campaign_type"])].append(index)
            channel_groups[row["channel"]].append(index)

        for (channel, campaign_type), indices in type_groups.items():
            rows = [campaign_rows[index] for index in indices]
            arrays = [campaign_samples[_campaign_key(row["channel"], row["campaign_id"])] for row in rows]
            aggregate, _ = _aggregate_row(
                "campaign_type",
                rows,
                arrays,
                float(target_roas),
                channel=channel,
                campaign_type=campaign_type,
            )
            all_rows.append(aggregate)

        channel_aggregate_rows: list[dict] = []
        channel_aggregate_samples: list[np.ndarray] = []
        for channel, indices in channel_groups.items():
            rows = [campaign_rows[index] for index in indices]
            arrays = [campaign_samples[_campaign_key(row["channel"], row["campaign_id"])] for row in rows]
            aggregate, samples = _aggregate_row(
                "channel", rows, arrays, float(target_roas), channel=channel
            )
            all_rows.append(aggregate)
            channel_aggregate_rows.append(aggregate)
            channel_aggregate_samples.append(samples)

        overall, _ = _aggregate_row(
            "overall",
            channel_aggregate_rows,
            channel_aggregate_samples,
            float(target_roas),
        )
        all_rows.append(overall)

    predictions = pd.DataFrame(all_rows)
    adaptation = artifact.get("runtime_adaptation", {})
    diagnostics = {
        "active_campaigns": int(historical_dimensions.shape[0]),
        "active_channels": int(historical_dimensions["channel"].nunique()),
        "planned_new_campaigns": int(len(planned_new_keys)),
        "forecast_campaigns": int(len(forecast_campaign_keys)),
        "forecast_channels": int(len(forecast_channels)),
        "horizons": [int(value) for value in horizons],
        "simulations": int(simulations),
        "target_roas": float(target_roas),
        "as_of_date": df["date"].max().date().isoformat(),
        "runtime_adaptation": bool(adaptation.get("enabled", False)),
        "runtime_context_rows": int(adaptation.get("runtime_rows", 0)),
        "runtime_context_campaigns": int(adaptation.get("runtime_campaigns", 0)),
        "global_prior_source": str(adaptation.get("global_prior_source", "artifact")),
        "boosting_challenger_used": any(
            bool(status.get("used", False)) for status in boosting_diagnostics
        ),
        "boosting_prediction_rows": int(
            sum(int(status.get("predicted_rows", 0)) for status in boosting_diagnostics)
        ),
        "boosting_status": sorted(
            {str(status.get("status", "unknown")) for status in boosting_diagnostics}
        ),
    }
    return predictions, diagnostics

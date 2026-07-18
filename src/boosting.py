"""Leakage-safe, pretrained XGBoost challenger for campaign forecasts.

The module imports XGBoost lazily and keeps fitting separate from scoring.
Prediction returns an empty frame plus structured diagnostics whenever the
committed bundle, dependency, or sufficient request evidence is absent.

Training observations are campaign/origin/horizon records.  A record is only
created when its complete target window is present in ``history``.  The model
therefore never uses a label that extends beyond the supplied history cutoff.
Raw campaign identifiers and names are retained solely for row matching; the
feature matrix contains stable categorical fields, whitelisted semantic name
tokens, calendar values, planned spend, and lagged performance summaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import ACTIVE_WINDOW_DAYS, RANDOM_SEED


WINDOW_DAYS = (7, 14, 28, 56, 84)
SEMANTIC_TOKENS = (
    "performance_max",
    "search",
    "shopping",
    "display",
    "video",
    "demand_gen",
    "dynamic_product",
    "brand",
    "prospecting",
    "remarketing",
    "generic",
    "trademark",
    "non_trademark",
    "advantage_plus",
)

CATEGORICAL_FEATURES = (
    "channel_feature",
    "native_type_feature",
    "family_feature",
    "funnel_feature",
)

NUMERIC_FEATURES = (
    "log_budget",
    "log_planned_daily_spend",
    "log_latest_daily_budget",
    "log_age_days",
    "recency_days",
    "horizon_days_feature",
    "budget_ratio_28",
    "daily_budget_ratio",
    "roas_trend",
    "spend_trend",
    "origin_month_sin",
    "origin_month_cos",
    "end_month_sin",
    "end_month_cos",
    *tuple(f"log_spend_{days}" for days in WINDOW_DAYS),
    *tuple(f"roas_{days}" for days in WINDOW_DAYS),
    *tuple(f"active_share_{days}" for days in WINDOW_DAYS),
    *tuple(f"zero_revenue_share_{days}" for days in WINDOW_DAYS),
    *tuple(f"token_{token}" for token in SEMANTIC_TOKENS),
)

PREDICTION_COLUMNS = [
    "channel",
    "campaign_id",
    "horizon_days",
    "budget",
    "boosting_roas_p50",
    "boosting_revenue_p50",
]

BUNDLE_KIND = "xgboost_quantile_campaign_roas_bundle"
BUNDLE_VERSION = 1
BOOSTER_FORMAT = "ubj"


@dataclass(frozen=True)
class BoostingConfig:
    """Runtime and evidence controls for the optional challenger."""

    enabled: bool = True
    origin_step_days: int = 14
    max_origins: int = 24
    min_history_days: int = 56
    min_training_rows: int = 120
    min_training_campaigns: int = 6
    min_training_origins: int = 4
    max_categories_per_field: int = 32
    min_category_count: int = 2
    boost_rounds: int = 180
    seed: int = RANDOM_SEED


@dataclass(frozen=True)
class _CampaignSeries:
    channel: str
    campaign_id: str
    days: np.ndarray
    spend: np.ndarray
    revenue: np.ndarray
    active: np.ndarray
    zero_revenue: np.ndarray
    cumulative_spend: np.ndarray
    cumulative_revenue: np.ndarray
    cumulative_active: np.ndarray
    cumulative_zero_revenue: np.ndarray
    campaign_name: np.ndarray
    native_type: np.ndarray
    campaign_family: np.ndarray
    funnel_signal: np.ndarray
    daily_budget: np.ndarray


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _base_diagnostics(reason: str, *, status: str = "disabled") -> dict[str, Any]:
    return {
        "status": status,
        "used": False,
        "reason": reason,
        "model_kind": "xgboost_quantile_campaign_roas",
        "training_rows": 0,
        "training_campaigns": 0,
        "training_origins": 0,
        "predicted_rows": 0,
        "skipped_requests": 0,
    }


def _clean_category(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def _semantic_token_flags(*values: object) -> dict[str, float]:
    """Return whitelisted semantic flags; sequence numbers are ignored."""

    text = " ".join("" if pd.isna(value) else str(value) for value in values).lower()
    normalized = re.sub(r"[^a-z0-9+]+", " ", text)
    words = set(normalized.split())

    non_trademark = bool(
        "ntm" in words
        or "nontrademark" in words
        or ("non" in words and "trademark" in words)
    )
    trademark = bool(("tm" in words or "trademark" in words) and not non_trademark)
    flags = {
        "performance_max": bool(
            "pmax" in words or "performancemax" in words or {"performance", "max"} <= words
        ),
        "search": "search" in words,
        "shopping": "shopping" in words,
        "display": "display" in words,
        "video": "video" in words,
        "demand_gen": bool("demandgen" in words or {"demand", "gen"} <= words),
        "dynamic_product": bool(
            "dpa" in words
            or "dynamicproduct" in words
            or {"dynamic", "product"} <= words
        ),
        "brand": "brand" in words,
        "prospecting": "prospecting" in words,
        "remarketing": bool("remarketing" in words or "retargeting" in words),
        "generic": "generic" in words,
        "trademark": trademark,
        "non_trademark": non_trademark,
        "advantage_plus": bool(
            "advantage+" in words
            or "advantageplus" in words
            or ("advantage" in words and "plus" in words)
        ),
    }
    return {f"token_{token}": float(flags[token]) for token in SEMANTIC_TOKENS}


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "channel", "campaign_id", "spend", "revenue"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")

    result = history.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date", "channel", "campaign_id"]).copy()
    result["channel"] = result["channel"].astype(str).str.strip()
    result["campaign_id"] = result["campaign_id"].astype(str).str.strip()
    for column in ("spend", "revenue"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        result[column] = result[column].clip(lower=0.0)
    optional_defaults = {
        "campaign_name": "Unknown campaign",
        "native_campaign_type": "Unknown",
        "campaign_family": "Unknown",
        "funnel_signal": "Unknown",
        "daily_budget": np.nan,
    }
    for column, default in optional_defaults.items():
        if column not in result:
            result[column] = default
    result["daily_budget"] = pd.to_numeric(result["daily_budget"], errors="coerce")
    result = result[(result["channel"] != "") & (result["campaign_id"] != "")]
    return result.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


def _build_series(history: pd.DataFrame) -> dict[tuple[str, str], _CampaignSeries]:
    result: dict[tuple[str, str], _CampaignSeries] = {}
    dimension_columns = [
        "campaign_name",
        "native_campaign_type",
        "campaign_family",
        "funnel_signal",
    ]
    for (channel, campaign_id), group in history.groupby(
        ["channel", "campaign_id"], sort=True
    ):
        aggregations: dict[str, str] = {"spend": "sum", "revenue": "sum"}
        aggregations.update({column: "last" for column in dimension_columns})
        aggregations["daily_budget"] = "last"
        daily = group.groupby("date", sort=True, as_index=False).agg(aggregations)
        days = daily["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
        spend = daily["spend"].to_numpy(dtype=float)
        revenue = daily["revenue"].to_numpy(dtype=float)
        active = (spend > 0).astype(float)
        zero_revenue = ((spend > 0) & (revenue <= 0)).astype(float)
        result[(str(channel), str(campaign_id))] = _CampaignSeries(
            channel=str(channel),
            campaign_id=str(campaign_id),
            days=days,
            spend=spend,
            revenue=revenue,
            active=active,
            zero_revenue=zero_revenue,
            cumulative_spend=_prefix(spend),
            cumulative_revenue=_prefix(revenue),
            cumulative_active=_prefix(active),
            cumulative_zero_revenue=_prefix(zero_revenue),
            campaign_name=daily["campaign_name"].astype(str).to_numpy(),
            native_type=daily["native_campaign_type"].astype(str).to_numpy(),
            campaign_family=daily["campaign_family"].astype(str).to_numpy(),
            funnel_signal=daily["funnel_signal"].astype(str).to_numpy(),
            daily_budget=daily["daily_budget"].to_numpy(dtype=float),
        )
    return result


def _sum(cumulative: np.ndarray, left: int, right: int) -> float:
    return float(cumulative[right] - cumulative[left])


def _month_pair(timestamp: pd.Timestamp) -> tuple[float, float]:
    angle = 2.0 * np.pi * (int(timestamp.month) - 1) / 12.0
    return float(np.sin(angle)), float(np.cos(angle))


def _feature_at(
    series: _CampaignSeries,
    origin: pd.Timestamp,
    horizon_days: int,
    budget: float,
) -> dict[str, Any] | None:
    origin = pd.Timestamp(origin).normalize()
    origin_day = int(origin.to_datetime64().astype("datetime64[D]").astype(np.int64))
    right = int(np.searchsorted(series.days, origin_day, side="right"))
    if right == 0:
        return None

    latest_index = right - 1
    age_days = max(1, origin_day - int(series.days[0]) + 1)
    positive_spend_indices = np.flatnonzero(series.spend[:right] > 0)
    recency_days = (
        max(0, origin_day - int(series.days[int(positive_spend_indices[-1])]))
        if positive_spend_indices.size
        else age_days
    )
    daily_budget = float(series.daily_budget[latest_index])
    if not np.isfinite(daily_budget) or daily_budget < 0:
        daily_budget = 0.0
    planned_daily_spend = float(budget) / max(int(horizon_days), 1)
    end = origin + pd.Timedelta(days=int(horizon_days))
    origin_sin, origin_cos = _month_pair(origin)
    end_sin, end_cos = _month_pair(end)

    row: dict[str, Any] = {
        "channel": series.channel,
        "campaign_id": series.campaign_id,
        "horizon_days": int(horizon_days),
        "budget": float(budget),
        "origin": origin,
        "target_end": end,
        "channel_feature": _clean_category(series.channel),
        "native_type_feature": _clean_category(series.native_type[latest_index]),
        "family_feature": _clean_category(series.campaign_family[latest_index]),
        "funnel_feature": _clean_category(series.funnel_signal[latest_index]),
        "log_budget": float(np.log1p(max(float(budget), 0.0))),
        "log_planned_daily_spend": float(np.log1p(max(planned_daily_spend, 0.0))),
        "log_latest_daily_budget": float(np.log1p(daily_budget)),
        "log_age_days": float(np.log1p(age_days)),
        "recency_days": float(min(recency_days, 365)),
        "horizon_days_feature": float(horizon_days),
        "daily_budget_ratio": float(
            np.clip(planned_daily_spend / max(daily_budget, 1e-3), 0.0, 10.0)
            if daily_budget > 0
            else 1.0
        ),
        "origin_month_sin": origin_sin,
        "origin_month_cos": origin_cos,
        "end_month_sin": end_sin,
        "end_month_cos": end_cos,
    }

    for window in WINDOW_DAYS:
        left_day = origin_day - int(window)
        left = int(np.searchsorted(series.days, left_day, side="right"))
        spend = _sum(series.cumulative_spend, left, right)
        revenue = _sum(series.cumulative_revenue, left, right)
        active = _sum(series.cumulative_active, left, right)
        zero = _sum(series.cumulative_zero_revenue, left, right)
        row[f"log_spend_{window}"] = float(np.log1p(max(spend, 0.0)))
        row[f"roas_{window}"] = float(np.clip(revenue / spend, 0.0, 30.0)) if spend else 0.0
        row[f"active_share_{window}"] = float(np.clip(active / window, 0.0, 1.0))
        row[f"zero_revenue_share_{window}"] = float(zero / active) if active else 1.0

    previous_spend = max(
        np.expm1(row["log_spend_28"]) - np.expm1(row["log_spend_14"]), 0.0
    )
    previous_revenue = max(
        row["roas_28"] * np.expm1(row["log_spend_28"])
        - row["roas_14"] * np.expm1(row["log_spend_14"]),
        0.0,
    )
    previous_roas = previous_revenue / previous_spend if previous_spend > 0 else 0.0
    row["roas_trend"] = float(
        np.clip((row["roas_14"] + 0.25) / (previous_roas + 0.25), 0.25, 4.0)
    )
    recent_spend = float(np.expm1(row["log_spend_14"]))
    row["spend_trend"] = float(
        np.clip((recent_spend + 1.0) / (previous_spend + 1.0), 0.10, 10.0)
    )
    recent_daily_spend = float(np.expm1(row["log_spend_28"])) / 28.0
    row["budget_ratio_28"] = float(
        np.clip(planned_daily_spend / max(recent_daily_spend, 1e-3), 0.0, 10.0)
    )
    row.update(
        _semantic_token_flags(
            series.campaign_name[latest_index],
            series.native_type[latest_index],
            series.campaign_family[latest_index],
            series.funnel_signal[latest_index],
        )
    )
    return row


def _future_totals(
    series: _CampaignSeries, origin: pd.Timestamp, horizon_days: int
) -> tuple[float, float]:
    origin_day = int(
        pd.Timestamp(origin).normalize().to_datetime64().astype("datetime64[D]").astype(np.int64)
    )
    end_day = origin_day + int(horizon_days)
    left = int(np.searchsorted(series.days, origin_day, side="right"))
    right = int(np.searchsorted(series.days, end_day, side="right"))
    return (
        _sum(series.cumulative_spend, left, right),
        _sum(series.cumulative_revenue, left, right),
    )


def _channel_days(history: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(channel): np.sort(
            group["date"].drop_duplicates().to_numpy(dtype="datetime64[D]").astype(np.int64)
        )
        for channel, group in history.groupby("channel")
    }


def _latest_channel_day(days: np.ndarray, origin_day: int) -> int | None:
    index = int(np.searchsorted(days, origin_day, side="right")) - 1
    return int(days[index]) if index >= 0 else None


def _select_origins(
    history: pd.DataFrame,
    horizons: tuple[int, ...],
    config: BoostingConfig,
) -> list[pd.Timestamp]:
    minimum = history["date"].min() + pd.Timedelta(days=int(config.min_history_days))
    maximum = history["date"].max() - pd.Timedelta(days=min(horizons))
    if maximum < minimum:
        return []
    origins = list(
        pd.date_range(minimum, maximum, freq=f"{int(config.origin_step_days)}D")
    )
    if not origins or origins[-1] != maximum:
        origins.append(pd.Timestamp(maximum).normalize())
    if len(origins) > int(config.max_origins):
        indices = np.linspace(0, len(origins) - 1, int(config.max_origins), dtype=int)
        origins = [origins[index] for index in sorted(set(indices.tolist()))]
    return [pd.Timestamp(origin).normalize() for origin in origins]


def _build_training_examples(
    history: pd.DataFrame,
    horizons: Iterable[int],
    config: BoostingConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = _prepare_history(history)
    horizon_values = tuple(sorted({int(value) for value in horizons if int(value) > 0}))
    if prepared.empty or not horizon_values:
        return pd.DataFrame(), {"as_of": None, "origins": []}

    as_of = pd.Timestamp(prepared["date"].max()).normalize()
    origins = _select_origins(prepared, horizon_values, config)
    series_by_key = _build_series(prepared)
    channel_dates = _channel_days(prepared)
    records: list[dict[str, Any]] = []

    for origin in origins:
        origin_day = int(origin.to_datetime64().astype("datetime64[D]").astype(np.int64))
        channel_ends = {
            channel: _latest_channel_day(days, origin_day)
            for channel, days in channel_dates.items()
        }
        for series in series_by_key.values():
            history_right = int(np.searchsorted(series.days, origin_day, side="right"))
            channel_end = channel_ends.get(series.channel)
            if history_right == 0 or channel_end is None:
                continue
            last_seen = int(series.days[history_right - 1])
            if channel_end - last_seen > int(ACTIVE_WINDOW_DAYS):
                continue
            for horizon in horizon_values:
                target_end = origin + pd.Timedelta(days=int(horizon))
                if target_end > as_of:
                    continue
                future_spend, future_revenue = _future_totals(series, origin, horizon)
                if future_spend <= 0:
                    continue
                latest_daily_budget = float(series.daily_budget[history_right - 1])
                if np.isfinite(latest_daily_budget) and latest_daily_budget > 0:
                    planned_budget = latest_daily_budget * int(horizon)
                else:
                    origin_day_minus_28 = origin_day - 28
                    history_left = int(
                        np.searchsorted(series.days, origin_day_minus_28, side="right")
                    )
                    prior_spend = _sum(
                        series.cumulative_spend, history_left, history_right
                    )
                    planned_budget = prior_spend * int(horizon) / 28.0
                if planned_budget <= 0:
                    continue
                row = _feature_at(series, origin, horizon, planned_budget)
                if row is None:
                    continue
                row["future_spend"] = float(future_spend)
                row["future_revenue"] = float(future_revenue)
                row["target_roas"] = float(future_revenue / future_spend)
                records.append(row)

    return pd.DataFrame(records), {"as_of": as_of, "origins": origins}


def _prepare_requests(requests: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = {"channel", "campaign_id", "horizon_days", "budget"}
    missing = required.difference(requests.columns)
    if missing:
        raise ValueError(f"requests are missing required columns: {sorted(missing)}")
    result = requests.copy().reset_index(drop=True)
    result["_request_order"] = np.arange(len(result), dtype=int)
    result["channel"] = result["channel"].astype(str).str.strip()
    result["campaign_id"] = result["campaign_id"].astype(str).str.strip()
    result["horizon_days"] = pd.to_numeric(result["horizon_days"], errors="coerce")
    result["budget"] = pd.to_numeric(result["budget"], errors="coerce")
    valid = (
        result["channel"].ne("")
        & result["campaign_id"].ne("")
        & result["horizon_days"].notna()
        & result["budget"].notna()
        & np.isfinite(result["horizon_days"])
        & np.isfinite(result["budget"])
        & result["horizon_days"].gt(0)
        & result["horizon_days"].mod(1).eq(0)
        & result["budget"].ge(0)
    )
    invalid_count = int((~valid).sum())
    result = result[valid].copy()
    result["horizon_days"] = result["horizon_days"].astype(int)
    result = result.drop_duplicates(
        ["channel", "campaign_id", "horizon_days"], keep="last"
    )
    return result, invalid_count


def _build_request_examples(
    history: pd.DataFrame, requests: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    prepared = _prepare_history(history)
    clean_requests, invalid_count = _prepare_requests(requests)
    if prepared.empty or clean_requests.empty:
        return pd.DataFrame(), invalid_count + len(clean_requests)

    series_by_key = _build_series(prepared)
    channel_end = prepared.groupby("channel")["date"].max().to_dict()
    records: list[dict[str, Any]] = []
    skipped = int(invalid_count)
    for row in clean_requests.to_dict(orient="records"):
        key = (str(row["channel"]), str(row["campaign_id"]))
        series = series_by_key.get(key)
        origin = channel_end.get(str(row["channel"]))
        if series is None or origin is None:
            skipped += 1
            continue
        feature = _feature_at(
            series,
            pd.Timestamp(origin),
            int(row["horizon_days"]),
            float(row["budget"]),
        )
        if feature is None:
            skipped += 1
            continue
        feature["_request_order"] = int(row["_request_order"])
        records.append(feature)
    return pd.DataFrame(records), skipped


class _FeatureEncoder:
    def __init__(self, config: BoostingConfig):
        self.config = config
        self.numeric_medians: dict[str, float] = {}
        self.categories: dict[str, tuple[str, ...]] = {}
        self.feature_names: list[str] = []

    def _refresh_feature_names(self) -> None:
        self.feature_names = list(NUMERIC_FEATURES)
        for column in CATEGORICAL_FEATURES:
            self.feature_names.extend(
                f"{column}={category}"
                for category in self.categories.get(column, ())
            )
            self.feature_names.append(f"{column}=__other__")

    def fit(self, frame: pd.DataFrame) -> "_FeatureEncoder":
        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(frame.get(column), errors="coerce")
            finite = values[np.isfinite(values)]
            self.numeric_medians[column] = float(finite.median()) if len(finite) else 0.0

        for column in CATEGORICAL_FEATURES:
            values = frame.get(column, pd.Series("unknown", index=frame.index)).map(
                _clean_category
            )
            counts = values.value_counts()
            ranked = sorted(
                (
                    (str(category), int(count))
                    for category, count in counts.items()
                    if int(count) >= int(self.config.min_category_count)
                ),
                key=lambda item: (-item[1], item[0]),
            )
            selected = tuple(
                category
                for category, _ in ranked[: int(self.config.max_categories_per_field)]
            )
            self.categories[column] = selected

        self._refresh_feature_names()
        return self

    def to_metadata(self) -> dict[str, Any]:
        """Return plain, pickle/JSON-friendly fitted encoder state."""

        return {
            "encoder_kind": "capped_one_hot_with_other",
            "numeric_medians": dict(self.numeric_medians),
            "categories": {
                column: list(self.categories.get(column, ()))
                for column in CATEGORICAL_FEATURES
            },
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, Any], config: BoostingConfig
    ) -> "_FeatureEncoder":
        """Reconstruct a fitted encoder without seeing labels or fitting."""

        if metadata.get("encoder_kind") != "capped_one_hot_with_other":
            raise ValueError("unsupported encoder metadata")
        medians = metadata.get("numeric_medians")
        categories = metadata.get("categories")
        if not isinstance(medians, Mapping) or not isinstance(categories, Mapping):
            raise ValueError("incomplete encoder metadata")
        encoder = cls(config)
        encoder.numeric_medians = {
            column: float(medians[column]) for column in NUMERIC_FEATURES
        }
        encoder.categories = {
            column: tuple(str(value) for value in categories[column])
            for column in CATEGORICAL_FEATURES
        }
        encoder._refresh_feature_names()
        if list(metadata.get("feature_names", ())) != encoder.feature_names:
            raise ValueError("encoder feature contract mismatch")
        return encoder

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for column in NUMERIC_FEATURES:
            values = pd.to_numeric(frame.get(column), errors="coerce").to_numpy(
                dtype=float, copy=True
            )
            values[~np.isfinite(values)] = self.numeric_medians[column]
            columns.append(values)

        for column in CATEGORICAL_FEATURES:
            values = frame.get(column, pd.Series("unknown", index=frame.index)).map(
                _clean_category
            )
            selected = self.categories[column]
            for category in selected:
                columns.append(values.eq(category).to_numpy(dtype=float))
            columns.append((~values.isin(selected)).to_numpy(dtype=float))

        if not columns:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def _load_xgboost() -> tuple[Any | None, str | None]:
    try:
        import xgboost as xgb  # type: ignore[import-not-found]
    except (ImportError, OSError) as error:
        return None, type(error).__name__
    return xgb, None


def _settings_from_metadata(metadata: object) -> BoostingConfig:
    if not isinstance(metadata, Mapping):
        raise ValueError("missing bundle configuration")
    allowed = {field.name for field in fields(BoostingConfig)}
    values = {key: metadata[key] for key in allowed if key in metadata}
    return BoostingConfig(**values)


def _xgboost_parameters(settings: BoostingConfig) -> dict[str, Any]:
    return {
        "objective": "reg:quantileerror",
        "quantile_alpha": 0.5,
        "eval_metric": "quantile",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.04,
        "min_child_weight": 8.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 8.0,
        "alpha": 0.05,
        "max_bin": 128,
        "seed": int(settings.seed),
        "nthread": 1,
        "verbosity": 0,
    }


def fit_boosting_bundle(
    history: pd.DataFrame,
    horizons: Iterable[int] = (30, 60, 90),
    *,
    config: BoostingConfig | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Fit and serialize the optional challenger during explicit training only.

    The returned dictionary contains only plain metadata and stable UBJ booster
    bytes.  It can be nested in the primary pickle artifact; no live estimator,
    campaign identifier, or raw campaign name is retained.
    """

    settings = config or BoostingConfig()
    if not settings.enabled:
        return None, _base_diagnostics("disabled_by_config")
    if (
        settings.origin_step_days <= 0
        or settings.max_origins <= 0
        or settings.boost_rounds <= 0
    ):
        return None, _base_diagnostics("invalid_training_config")

    horizon_values = tuple(sorted({int(value) for value in horizons if int(value) > 0}))
    if not horizon_values:
        return None, _base_diagnostics("no_valid_horizons")
    try:
        training, training_meta = _build_training_examples(
            history, horizon_values, settings
        )
    except (KeyError, TypeError, ValueError) as error:
        return None, _base_diagnostics(f"invalid_input:{type(error).__name__}")

    training_campaigns = (
        int(training[["channel", "campaign_id"]].drop_duplicates().shape[0])
        if not training.empty
        else 0
    )
    training_origins = int(training["origin"].nunique()) if not training.empty else 0
    evidence = {
        "training_rows": int(len(training)),
        "training_campaigns": training_campaigns,
        "training_origins": training_origins,
    }
    if len(training) < int(settings.min_training_rows):
        diagnostics = _base_diagnostics("insufficient_training_rows")
        diagnostics.update(evidence)
        return None, diagnostics
    if training_campaigns < int(settings.min_training_campaigns):
        diagnostics = _base_diagnostics("insufficient_training_campaigns")
        diagnostics.update(evidence)
        return None, diagnostics
    if training_origins < int(settings.min_training_origins):
        diagnostics = _base_diagnostics("insufficient_training_origins")
        diagnostics.update(evidence)
        return None, diagnostics

    xgb, import_error = _load_xgboost()
    if xgb is None:
        diagnostics = _base_diagnostics(f"xgboost_unavailable:{import_error}")
        diagnostics.update(evidence)
        return None, diagnostics

    try:
        encoder = _FeatureEncoder(settings).fit(training)
        train_matrix = encoder.transform(training)
        targets = training["target_roas"].to_numpy(dtype=float)
        targets = np.clip(
            np.nan_to_num(targets, nan=0.0, posinf=30.0, neginf=0.0), 0.0, 30.0
        )
        budgets = training["future_spend"].to_numpy(dtype=float)
        positive_budgets = budgets[budgets > 0]
        median_budget = float(np.median(positive_budgets))
        weights = np.sqrt(
            np.clip(budgets / max(median_budget, 1e-6), 0.0625, 16.0)
        )
        train_data = xgb.DMatrix(
            train_matrix,
            label=targets,
            weight=weights,
            feature_names=encoder.feature_names,
        )
        booster = xgb.train(
            _xgboost_parameters(settings),
            train_data,
            num_boost_round=int(settings.boost_rounds),
        )
        booster_bytes = bytes(booster.save_raw(raw_format=BOOSTER_FORMAT))
    except Exception as error:  # Challenger failure must never break training.
        diagnostics = _base_diagnostics(f"xgboost_failed:{type(error).__name__}")
        diagnostics.update(evidence)
        return None, diagnostics

    as_of = training_meta.get("as_of")
    latest_target = pd.Timestamp(training["target_end"].max())
    training_metadata = {
        **evidence,
        "horizons": list(horizon_values),
        "history_as_of": as_of.date().isoformat() if as_of is not None else None,
        "latest_training_target": latest_target.date().isoformat(),
        "feature_count": int(len(encoder.feature_names)),
        "xgboost_version": str(getattr(xgb, "__version__", "unknown")),
        "boost_rounds": int(settings.boost_rounds),
        "seed": int(settings.seed),
    }
    bundle = {
        "bundle_kind": BUNDLE_KIND,
        "bundle_version": BUNDLE_VERSION,
        "booster_format": BOOSTER_FORMAT,
        "booster_bytes": booster_bytes,
        "encoder": encoder.to_metadata(),
        "config": asdict(settings),
        "training": training_metadata,
    }
    diagnostics = {
        "status": "fitted",
        "used": False,
        "trained": True,
        "reason": None,
        "model_kind": "xgboost_quantile_campaign_roas",
        **training_metadata,
        "predicted_rows": 0,
        "skipped_requests": 0,
        "bundle_bytes": int(len(booster_bytes)),
    }
    return bundle, diagnostics


def _restore_bundle(
    bundle: Mapping[str, Any],
) -> tuple[bytes, _FeatureEncoder, BoostingConfig, Mapping[str, Any]]:
    if bundle.get("bundle_kind") != BUNDLE_KIND:
        raise ValueError("unsupported boosting bundle kind")
    if int(bundle.get("bundle_version", -1)) != BUNDLE_VERSION:
        raise ValueError("unsupported boosting bundle version")
    if bundle.get("booster_format") != BOOSTER_FORMAT:
        raise ValueError("unsupported booster serialization format")
    booster_bytes = bundle.get("booster_bytes")
    if not isinstance(booster_bytes, (bytes, bytearray)) or not booster_bytes:
        raise ValueError("missing serialized booster")
    settings = _settings_from_metadata(bundle.get("config"))
    encoder_metadata = bundle.get("encoder")
    if not isinstance(encoder_metadata, Mapping):
        raise ValueError("missing fitted encoder")
    encoder = _FeatureEncoder.from_metadata(encoder_metadata, settings)
    if any(
        feature_name.lower().split("=", 1)[0] in {"campaign_id", "campaign_name"}
        for feature_name in encoder.feature_names
    ):
        raise ValueError("raw campaign identity found in feature contract")
    training = bundle.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("missing bundle training metadata")
    return bytes(booster_bytes), encoder, settings, training


def predict_boosting_bundle(
    history: pd.DataFrame,
    requests: pd.DataFrame,
    bundle: Mapping[str, Any] | None,
    *,
    config: BoostingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score requests from a committed bundle without fitting at runtime."""

    runtime_settings = config or BoostingConfig()
    if not runtime_settings.enabled:
        return _empty_predictions(), _base_diagnostics("disabled_by_config")
    if bundle is None:
        return _empty_predictions(), _base_diagnostics("bundle_missing")

    try:
        booster_bytes, encoder, _, training = _restore_bundle(bundle)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _empty_predictions(), _base_diagnostics(
            f"invalid_bundle:{type(error).__name__}"
        )

    try:
        clean_requests, invalid_requests = _prepare_requests(requests)
        if clean_requests.empty:
            diagnostics = _base_diagnostics("no_valid_requests")
            diagnostics["skipped_requests"] = int(invalid_requests)
            return _empty_predictions(), diagnostics
        request_features, skipped_requests = _build_request_examples(
            history, clean_requests
        )
    except (KeyError, TypeError, ValueError) as error:
        return _empty_predictions(), _base_diagnostics(
            f"invalid_input:{type(error).__name__}"
        )

    evidence = {
        "training_rows": int(training.get("training_rows", 0)),
        "training_campaigns": int(training.get("training_campaigns", 0)),
        "training_origins": int(training.get("training_origins", 0)),
        "skipped_requests": int(skipped_requests + invalid_requests),
    }
    if request_features.empty:
        diagnostics = _base_diagnostics("no_supported_requests")
        diagnostics.update(evidence)
        return _empty_predictions(), diagnostics

    xgb, import_error = _load_xgboost()
    if xgb is None:
        diagnostics = _base_diagnostics(f"xgboost_unavailable:{import_error}")
        diagnostics.update(evidence)
        return _empty_predictions(), diagnostics

    try:
        request_matrix = encoder.transform(request_features)
        request_data = xgb.DMatrix(
            request_matrix,
            feature_names=encoder.feature_names,
        )
        booster = xgb.Booster()
        booster.load_model(bytearray(booster_bytes))
        booster.set_param({"nthread": 1})
        roas = np.clip(booster.predict(request_data), 0.0, 30.0)
    except Exception as error:  # Challenger failure must never break primary scoring.
        diagnostics = _base_diagnostics(f"xgboost_failed:{type(error).__name__}")
        diagnostics.update(evidence)
        return _empty_predictions(), diagnostics

    predictions = request_features[
        ["channel", "campaign_id", "horizon_days", "budget", "_request_order"]
    ].copy()
    predictions["boosting_roas_p50"] = roas.astype(float)
    predictions["boosting_revenue_p50"] = (
        predictions["boosting_roas_p50"] * predictions["budget"]
    )
    predictions = predictions.sort_values("_request_order").drop(
        columns="_request_order"
    )
    predictions = predictions[PREDICTION_COLUMNS].reset_index(drop=True)

    runtime_as_of = pd.to_datetime(history.get("date"), errors="coerce").max()
    diagnostics = {
        "status": "used",
        "used": True,
        "reason": None,
        "model_kind": "xgboost_quantile_campaign_roas",
        **evidence,
        "predicted_rows": int(len(predictions)),
        "feature_count": int(len(encoder.feature_names)),
        "xgboost_version": str(getattr(xgb, "__version__", "unknown")),
        "artifact_history_as_of": training.get("history_as_of"),
        "latest_training_target": training.get("latest_training_target"),
        "runtime_history_as_of": (
            pd.Timestamp(runtime_as_of).date().isoformat()
            if pd.notna(runtime_as_of)
            else None
        ),
        "boost_rounds": int(training.get("boost_rounds", 0)),
        "seed": int(training.get("seed", 0)),
    }
    return predictions, diagnostics


def run_boosting_challenger(
    history: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    bundle: Mapping[str, Any] | None = None,
    config: BoostingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compatibility wrapper for scoring; this function never fits a model."""

    return predict_boosting_bundle(
        history,
        requests,
        bundle,
        config=config,
    )

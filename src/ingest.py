"""Schema normalization and auditable campaign taxonomy construction."""

from __future__ import annotations
import hashlib

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "date",
    "channel",
    "campaign_id",
    "campaign_name",
    "native_campaign_type",
    "campaign_family",
    "funnel_signal",
    "analog_key",
    "spend",
    "revenue",
    "conversions",
    "clicks",
    "impressions",
    "daily_budget",
    "source_file",
    "source_revenue_field",
]


def _series(df: pd.DataFrame, column: str | None, default: object = np.nan) -> pd.Series:
    if column and column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)

def _numeric_series(values: pd.Series) -> pd.Series:
    """Parse numeric exports while preserving invalid values for quality checks."""
    text = values.astype("string").str.strip()
    accounting_negative = text.str.match(r"^\(.*\)$", na=False)
    cleaned = text.str.replace(r"^\((.*)\)$", r"\1", regex=True).str.replace(
        r"[^0-9.eE+\-]", "", regex=True
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric.loc[accounting_negative] = -numeric.loc[accounting_negative].abs()
    return numeric.astype(float)


def _canonical_channel(value: object, fallback: str = "Unknown Channel") -> str:
    if pd.isna(value) or not str(value).strip():
        return fallback
    text = str(value).strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if "google" in normalized or normalized in {"google ads", "adwords"}:
        return "Google Ads"
    if any(token in normalized for token in ("meta", "facebook", "instagram")):
        return "Meta Ads"
    if any(token in normalized for token in ("microsoft", "bing")):
        return "Microsoft Ads"
    return text



def _detect_channel(filename: str, columns: Iterable[str]) -> str:
    name = filename.lower()
    lowered = {str(column).lower() for column in columns}
    if "google" in name or "metrics_cost_micros" in lowered:
        return "Google Ads"
    if "meta" in name or "facebook" in name or "date_start" in lowered:
        return "Meta Ads"
    if "bing" in name or "microsoft" in name or "timeperiod" in lowered:
        return "Microsoft Ads"
    return "Unknown Channel"


def campaign_analog_key(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    normalized = text.strip("_") or "unknown"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"analog_{digest}"


def infer_funnel_signal(campaign_name: object) -> str:
    name = str(campaign_name or "").upper()
    if "REMARKETING" in name:
        return "Remarketing"
    if "PROSPECTING" in name:
        return "Prospecting"
    if "_TM_" in name:
        return "Trademark"
    if "_NTM_" in name:
        return "Non-trademark"
    if "GENERIC" in name:
        return "Generic"
    return "Unclassified"


def infer_campaign_family(campaign_name: object, native_type: object) -> str:
    name = str(campaign_name or "").upper()
    native = str(native_type or "").upper().replace(" ", "_")
    if "PMAX" in name or native in {"PERFORMANCE_MAX", "PERFORMANCEMAX"}:
        return "Performance Max"
    if "DEMAND GEN" in name or native == "DEMAND_GEN":
        return "Demand Gen"
    if "SEARCH" in name or native == "SEARCH":
        return "Search"
    if "SHOPPING" in name or native == "SHOPPING":
        return "Shopping"
    if "VIDEO" in name or native == "VIDEO":
        return "Video"
    if "DISPLAY" in name or native == "DISPLAY":
        return "Display"
    if "DPA" in name:
        return "Dynamic Product Ads"
    if "ADV_PLUS" in name or "ADVANTAGE" in name:
        return "Advantage+"
    if "BRAND" in name:
        return "Brand Creative"
    if "GENERIC" in name:
        return "Generic Social"
    if native == "AUDIENCE":
        return "Audience"
    return "Other"


def _from_google(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": _series(df, "segments_date"),
            "channel": "Google Ads",
            "campaign_id": _series(df, "campaign_id"),
            "campaign_name": _series(df, "campaign_name", "Unknown Campaign"),
            "native_campaign_type": _series(
                df, "campaign_advertising_channel_type", "Unknown"
            ),
            "spend": _numeric_series(_series(df, "metrics_cost_micros", 0))
            / 1_000_000.0,
            "revenue": _series(df, "metrics_conversions_value", 0),
            "conversions": _series(df, "metrics_conversions"),
            "clicks": _series(df, "metrics_clicks"),
            "impressions": _series(df, "metrics_impressions"),
            "daily_budget": _series(df, "campaign_budget_amount"),
            "source_file": path.name,
            "source_revenue_field": "metrics_conversions_value",
        }
    )


def _from_microsoft(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": _series(df, "TimePeriod"),
            "channel": "Microsoft Ads",
            "campaign_id": _series(df, "CampaignId"),
            "campaign_name": _series(df, "CampaignName", "Unknown Campaign"),
            "native_campaign_type": _series(df, "CampaignType", "Unknown"),
            "spend": _series(df, "Spend", 0),
            "revenue": _series(df, "Revenue", 0),
            "conversions": _series(df, "Conversions"),
            "clicks": _series(df, "Clicks"),
            "impressions": _series(df, "Impressions"),
            "daily_budget": _series(df, "DailyBudget"),
            "source_file": path.name,
            "source_revenue_field": "Revenue",
        }
    )


def _from_meta(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    # The supplied `conversion` values can exceed clicks and include decimals, so they
    # behave like attributed conversion value/revenue rather than a conversion count.
    return pd.DataFrame(
        {
            "date": _series(df, "date_start"),
            "channel": "Meta Ads",
            "campaign_id": _series(df, "campaign_id"),
            "campaign_name": _series(df, "campaign_name", "Unknown Campaign"),
            "native_campaign_type": "Inferred from campaign name",
            "spend": _series(df, "spend", 0),
            "revenue": _series(df, "conversion", 0),
            "conversions": np.nan,
            "clicks": _series(df, "clicks"),
            "impressions": _series(df, "impressions"),
            "daily_budget": _series(df, "daily_budget"),
            "source_file": path.name,
            "source_revenue_field": "conversion (assumed conversion value)",
        }
    )


def _from_generic(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    aliases = {
        "channel": ["channel", "platform", "source"],
        "date": ["date", "day", "time_period", "timeperiod"],
        "campaign_id": ["campaign_id", "campaignid", "id"],
        "campaign_name": ["campaign_name", "campaignname", "campaign"],
        "native_campaign_type": ["campaign_type", "campaigntype", "type"],
        "spend": ["spend", "cost", "amount_spent"],
        "revenue": ["revenue", "conversion_value", "sales"],
        "conversions": ["conversions", "purchases", "transactions"],
        "clicks": ["clicks"],
        "impressions": ["impressions"],
        "daily_budget": ["daily_budget", "budget"],
    }
    lower_to_original = {str(column).strip().lower(): column for column in df.columns}

    def find(field: str) -> str | None:
        return next(
            (lower_to_original[name] for name in aliases[field] if name in lower_to_original),
            None,
        )

    required = {field: find(field) for field in ("date", "campaign_name", "spend", "revenue")}
    missing = [field for field, column in required.items() if column is None]
    if missing:
        raise ValueError(f"Unsupported CSV schema in {path.name}; missing {missing}")
    detected_channel = _detect_channel(path.name, df.columns)
    channel_column = find("channel")
    channel = (
        _series(df, channel_column).map(lambda value: _canonical_channel(value, detected_channel))
        if channel_column
        else detected_channel
    )
    return pd.DataFrame(
        {
            "date": _series(df, required["date"]),
            "channel": channel,
            "campaign_id": _series(df, find("campaign_id"), "unknown"),
            "campaign_name": _series(df, required["campaign_name"], "Unknown Campaign"),
            "native_campaign_type": _series(df, find("native_campaign_type"), "Unknown"),
            "spend": _series(df, required["spend"], 0),
            "revenue": _series(df, required["revenue"], 0),
            "conversions": _series(df, find("conversions")),
            "clicks": _series(df, find("clicks")),
            "impressions": _series(df, find("impressions")),
            "daily_budget": _series(df, find("daily_budget")),
            "source_file": path.name,
            "source_revenue_field": str(required["revenue"]),
        }
    )


def normalize_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype="string")
    raw.columns = [str(column).strip() for column in raw.columns]
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    columns = set(raw.columns)
    if {"segments_date", "metrics_cost_micros"}.issubset(columns):
        normalized = _from_google(raw, path)
    elif {"TimePeriod", "Revenue", "Spend"}.issubset(columns):
        normalized = _from_microsoft(raw, path)
    elif {"date_start", "spend", "conversion"}.issubset(columns):
        normalized = _from_meta(raw, path)
    else:
        normalized = _from_generic(raw, path)

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["campaign_id"] = normalized["campaign_id"].astype("string").fillna("unknown")
    normalized["campaign_name"] = (
        normalized["campaign_name"].astype("string").fillna("Unknown Campaign")
    )
    normalized["native_campaign_type"] = (
        normalized["native_campaign_type"].astype("string").fillna("Unknown")
    )
    numeric = [
        "spend",
        "revenue",
        "conversions",
        "clicks",
        "impressions",
        "daily_budget",
    ]
    for column in numeric:
        normalized[column] = _numeric_series(normalized[column])
    normalized["campaign_family"] = [
        infer_campaign_family(name, native)
        for name, native in zip(
            normalized["campaign_name"], normalized["native_campaign_type"], strict=False
        )
    ]
    normalized["funnel_signal"] = normalized["campaign_name"].map(infer_funnel_signal)
    normalized["analog_key"] = normalized["campaign_name"].map(campaign_analog_key)
    return normalized[STANDARD_COLUMNS]


def build_quality_report(
    df: pd.DataFrame,
    invalid_dates: int = 0,
    duplicate_rows_removed: int = 0,
) -> pd.DataFrame:
    checks = [
        ("invalid_dates", invalid_dates, "error" if invalid_dates else "ok"),
        (
            "exact_duplicate_rows_removed",
            int(duplicate_rows_removed),
            "info",
        ),
        (
            "duplicate_channel_campaign_dates",
            int(df.duplicated(["channel", "campaign_id", "date"], keep=False).sum()),
            "warning",
        ),
        ("negative_spend", int((df["spend"] < 0).sum()), "error"),
        ("negative_revenue", int((df["revenue"] < 0).sum()), "error"),
        ("missing_spend", int(df["spend"].isna().sum()), "error"),
        ("missing_revenue", int(df["revenue"].isna().sum()), "warning"),
        (
            "positive_revenue_with_zero_spend",
            int(((df["revenue"] > 0) & (df["spend"] <= 0)).sum()),
            "warning",
        ),
        (
            "inferred_meta_campaign_type_rows",
            int((df["channel"] == "Meta Ads").sum()),
            "info",
        ),
    ]
    report = pd.DataFrame(checks, columns=["check", "count", "severity"])
    report.loc[report["count"] == 0, "severity"] = "ok"
    return report


def load_datasets(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    files = sorted(
        path
        for path in root.rglob("*.csv")
        if path.name.lower() not in {"future_budgets.csv", "example_budget_plan.csv"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {root}")

    frames = [normalize_file(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    invalid_dates = int(combined["date"].isna().sum())
    combined = combined.dropna(subset=["date"]).copy()
    dedup_columns = [column for column in STANDARD_COLUMNS if column != "source_file"]
    before_deduplication = len(combined)
    combined = combined.drop_duplicates(subset=dedup_columns, keep="first").copy()
    quality = build_quality_report(
        combined,
        invalid_dates,
        duplicate_rows_removed=before_deduplication - len(combined),
    )

    for column in ("spend", "revenue"):
        combined[column] = combined[column].fillna(0).clip(lower=0)
    combined = combined.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)
    return combined, quality


def load_budget_plan(data_dir: str | Path) -> pd.DataFrame | None:
    path = Path(data_dir) / "future_budgets.csv"
    if not path.exists():
        return None
    plan = pd.read_csv(path, dtype="string")
    plan = plan.loc[:, ~plan.columns.astype(str).str.startswith("Unnamed")].copy()
    plan.columns = [str(column).strip() for column in plan.columns]
    required = {"channel", "horizon_days", "budget"}
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(f"future_budgets.csv is missing columns: {sorted(missing)}")
    if "campaign_id" not in plan and "campaign_name" not in plan:
        raise ValueError("future_budgets.csv needs campaign_id or campaign_name")

    for column in (
        "channel",
        "campaign_id",
        "campaign_name",
        "campaign_type",
        "funnel_signal",
    ):
        if column in plan:
            plan[column] = plan[column].astype("string").str.strip().replace("", pd.NA)
    if plan["channel"].isna().any():
        rows = (plan.index[plan["channel"].isna()] + 2).tolist()
        raise ValueError(f"future_budgets.csv has blank channel values on rows {rows}")

    id_missing = (
        plan["campaign_id"].isna()
        if "campaign_id" in plan
        else pd.Series(True, index=plan.index)
    )
    name_missing = (
        plan["campaign_name"].isna()
        if "campaign_name" in plan
        else pd.Series(True, index=plan.index)
    )
    missing_identity = id_missing & name_missing
    if missing_identity.any():
        rows = (plan.index[missing_identity] + 2).tolist()
        raise ValueError(
            "future_budgets.csv needs campaign_id or campaign_name on every row; "
            f"missing on rows {rows}"
        )

    if "is_new_campaign" not in plan:
        plan["is_new_campaign"] = False
    else:
        truthy = {"1", "true", "yes", "y"}
        falsy = {"0", "false", "no", "n", ""}

        def parse_flag(value: object) -> bool:
            if pd.isna(value):
                return False
            normalized = str(value).strip().lower()
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
            raise ValueError(
                "is_new_campaign must be true/false, yes/no, or 1/0; "
                f"received {value!r}"
            )

        plan["is_new_campaign"] = plan["is_new_campaign"].map(parse_flag)

    new_without_name = plan["is_new_campaign"] & name_missing
    if new_without_name.any():
        rows = (plan.index[new_without_name] + 2).tolist()
        raise ValueError(
            "New campaigns require campaign_name so a peer family can be inferred; "
            f"missing on rows {rows}"
        )

    horizon_numeric = pd.to_numeric(plan["horizon_days"], errors="raise")
    if ((horizon_numeric % 1) != 0).any() or (horizon_numeric <= 0).any():
        raise ValueError("future_budgets.csv horizon_days must contain positive whole numbers")
    plan["horizon_days"] = horizon_numeric.astype(int)

    budget_numeric = pd.to_numeric(plan["budget"], errors="raise")
    if (~np.isfinite(budget_numeric)).any() or (budget_numeric < 0).any():
        raise ValueError("future_budgets.csv budget must contain finite non-negative values")
    plan["budget"] = budget_numeric.astype(float)
    return plan

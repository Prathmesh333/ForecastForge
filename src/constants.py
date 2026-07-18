"""Shared constants for the offline and interactive forecast paths."""

SCHEMA_VERSION = "0.2.0"
MODEL_KIND = "hybrid_empirical_bayes_xgboost_lifecycle"
DEFAULT_HORIZONS = (30, 60, 90)
DEFAULT_TARGET_ROAS = 3.0
DEFAULT_SIMULATIONS = 1000
ACTIVE_WINDOW_DAYS = 7
RECENT_WINDOW_DAYS = 14
RANDOM_SEED = 20260717

OUTPUT_COLUMNS = [
    "forecast_level",
    "channel",
    "campaign_type",
    "campaign_id",
    "campaign_name",
    "horizon_days",
    "budget",
    "revenue_p10",
    "revenue_p50",
    "revenue_p90",
    "roas_p10",
    "roas_p50",
    "roas_p90",
    "target_roas",
    "probability_target",
    "support_status",
    "lifecycle_state",
    "recommendation",
    "peer_source",
    "data_points",
]

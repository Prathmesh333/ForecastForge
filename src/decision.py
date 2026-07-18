"""Decision-risk and value-of-information experiment recommendations."""

from __future__ import annotations

import pandas as pd


def _experiment_for_horizon(campaigns: pd.DataFrame, horizon_days: int) -> dict:
    working = campaigns[campaigns["budget"] > 0].copy()
    if working.empty:
        return {
            "horizon_days": int(horizon_days),
            "campaign": "No active spend",
            "channel": "Portfolio",
            "campaign_type": "Paused plan",
            "lifecycle_state": "Paused",
            "support_status": "No spend",
            "value_of_information_score": 0.0,
            "score_components": {
                "decision_sensitivity": 0.0,
                "downside_at_p10": 0.0,
                "relative_interval_width": 0.0,
            },
            "hypothesis": "No learning claim is appropriate while every planned budget is zero.",
            "test_design": (
                "Do not run a media experiment. Confirm that the zero-budget plan is intentional, "
                "then define a bounded launch budget before requesting a forecast."
            ),
            "randomization_unit": "Not applicable",
            "holdout_share": 0.0,
            "budget_exposed_during_test": 0.0,
            "holdout_budget": 0.0,
            "suggested_test_budget": 0.0,
            "primary_metric": "Plan validation",
            "guardrail": "Keep spend at zero until the planner confirms a launch budget.",
            "decision_rule": "No scale decision is made for a fully paused plan.",
            "uncertainty_addressed": "Whether the zero-budget plan is intentional.",
        }
    width = (working["revenue_p90"] - working["revenue_p10"]).clip(lower=0)
    relative_width = width / working["revenue_p50"].clip(lower=1.0)
    target_revenue = working["target_roas"] * working["budget"]
    downside_at_p10 = (target_revenue - working["revenue_p10"]).clip(lower=0)
    decision_sensitivity = (4.0 * working["probability_target"] * (1.0 - working["probability_target"])).clip(lower=0.10)
    support_penalty = working["support_status"].map(
        {"Supported": 1.0, "Extrapolating": 1.5, "Insufficient Evidence": 1.9}
    ).fillna(1.4)
    lifecycle_penalty = working["lifecycle_state"].map(
        {"Launch": 1.7, "Ramp": 1.4, "Declining": 1.2, "Mature": 1.0}
    ).fillna(1.1)
    working["decision_sensitivity"] = decision_sensitivity
    working["downside_at_p10"] = downside_at_p10
    working["relative_width"] = relative_width
    working["voi_score"] = (
        (downside_at_p10 + 0.25 * width)
        * (0.35 + 0.65 * decision_sensitivity)
        * support_penalty
        * lifecycle_penalty
    )
    selected = working.sort_values("voi_score", ascending=False).iloc[0]
    daily_budget = float(selected["budget"] / horizon_days) if horizon_days else 0.0
    budget_exposed = daily_budget * 14.0
    holdout_budget = budget_exposed * 0.10
    target = float(selected["target_roas"])
    return {
        "horizon_days": int(horizon_days),
        "campaign": str(selected["campaign_name"]),
        "channel": str(selected["channel"]),
        "campaign_type": str(selected["campaign_type"]),
        "lifecycle_state": str(selected["lifecycle_state"]),
        "support_status": str(selected["support_status"]),
        "value_of_information_score": round(float(selected["voi_score"]), 2),
        "score_components": {
            "decision_sensitivity": round(float(selected["decision_sensitivity"]), 4),
            "downside_at_p10": round(float(selected["downside_at_p10"]), 2),
            "relative_interval_width": round(float(selected["relative_width"]), 4),
        },
        "hypothesis": (
            f"Test whether {selected['campaign_name']} can sustain the proposed spend while "
            f"meeting the {target:.2f} ROAS threshold."
        ),
        "test_design": (
            "Run a 14-day randomized 90/10 treatment-holdout split within the campaign's "
            "eligible audience; use a geo split when audience randomization is unavailable."
        ),
        "randomization_unit": "Eligible audience member, or matched geography",
        "holdout_share": 0.10,
        "budget_exposed_during_test": round(float(budget_exposed), 2),
        "holdout_budget": round(float(holdout_budget), 2),
        "suggested_test_budget": round(float(holdout_budget), 2),
        "primary_metric": "Holdout-adjusted attributed revenue and ROAS",
        "guardrail": f"Pause or review the plan if observed ROAS falls below the {target:.2f} business target.",
        "decision_rule": (
            f"Scale only if observed ROAS is at least {target:.2f}, the treatment beats holdout, "
            "and the result remains consistent with the forecast interval."
        ),
        "uncertainty_addressed": (
            f"{selected['lifecycle_state']} lifecycle evidence and {selected['support_status'].lower()} "
            "budget-response support."
        ),
    }


def build_decision_summary(predictions: pd.DataFrame, diagnostics: dict) -> dict:
    decisions: list[dict] = []
    for horizon in sorted(predictions["horizon_days"].unique()):
        overall = predictions[
            (predictions["horizon_days"] == horizon)
            & (predictions["forecast_level"] == "overall")
        ].iloc[0]
        campaigns = predictions[
            (predictions["horizon_days"] == horizon)
            & (predictions["forecast_level"] == "campaign")
        ]
        experiment = _experiment_for_horizon(campaigns, int(horizon))
        decisions.append(
            {
                "horizon_days": int(horizon),
                "recommendation": str(overall["recommendation"]),
                "support_status": str(overall["support_status"]),
                "probability_target": round(float(overall["probability_target"]), 4),
                "target_roas": round(float(overall["target_roas"]), 4),
                "revenue_p10": round(float(overall["revenue_p10"]), 2),
                "revenue_p50": round(float(overall["revenue_p50"]), 2),
                "revenue_p90": round(float(overall["revenue_p90"]), 2),
                "roas_p50": round(float(overall["roas_p50"]), 4),
                "experiment": experiment,
            }
        )
    return {"diagnostics": diagnostics, "decisions": decisions}

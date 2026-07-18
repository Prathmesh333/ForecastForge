from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision import build_decision_summary
from src.explain import deterministic_explanation
from src.forecast import forecast_portfolio
from src.ingest import (
    _numeric_series,
    build_quality_report,
    infer_campaign_family,
    load_budget_plan,
    load_datasets,
)
from src.model import adapt_model_artifact, build_model_artifact


class IngestionTests(unittest.TestCase):
    def test_currency_parser_is_unicode_safe(self) -> None:
        values = pd.Series(
            [
                f"{chr(0x20AC)}1,234.50",
                f"{chr(0x20B9)}99.25",
                f"({chr(0x00A3)}5.00)",
            ]
        )
        parsed = _numeric_series(values)
        np.testing.assert_allclose(parsed, [1234.50, 99.25, -5.00])

    def test_campaign_family_inference(self) -> None:
        self.assertEqual(infer_campaign_family("US_NTM_Search", "SEARCH"), "Search")
        self.assertEqual(infer_campaign_family("PMax_Prospecting", ""), "Performance Max")
        self.assertEqual(infer_campaign_family("DPA_Remarketing", ""), "Dynamic Product Ads")

    def test_google_micros_are_normalized_to_currency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "google_ads.csv"
            pd.DataFrame(
                {
                    "segments_date": ["2026-01-01"],
                    "campaign_id": [123],
                    "campaign_name": ["Search_Generic"],
                    "campaign_advertising_channel_type": ["SEARCH"],
                    "metrics_cost_micros": [2_500_000],
                    "metrics_conversions_value": [12.0],
                    "metrics_conversions": [1],
                    "metrics_clicks": [3],
                    "metrics_impressions": [100],
                }
            ).to_csv(path, index=False)
            result, quality = load_datasets(directory)
        self.assertAlmostEqual(float(result.iloc[0]["spend"]), 2.5)
        self.assertEqual(result.iloc[0]["channel"], "Google Ads")
        self.assertEqual(int(quality.loc[quality["check"] == "invalid_dates", "count"].iloc[0]), 0)

    def test_generic_schema_preserves_ids_currency_channel_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame(
                {
                    " channel ": ["Meta"],
                    " date ": ["2026-01-02"],
                    " campaign_id ": ["00123"],
                    " campaign_name ": ["Prospecting_Generic"],
                    " campaign_type ": ["SOCIAL"],
                    " spend ": ["$1,234.50"],
                    " revenue ": ["$2,500.25"],
                    " daily_budget ": ["500"],
                }
            )
            frame.to_csv(Path(directory) / "export_part_1.csv", index=False)
            frame.to_csv(Path(directory) / "export_part_2.csv", index=False)

            result, quality = load_datasets(directory)

        self.assertEqual(len(result), 1)
        self.assertEqual(str(result.iloc[0]["campaign_id"]), "00123")
        self.assertEqual(result.iloc[0]["channel"], "Meta Ads")
        self.assertAlmostEqual(float(result.iloc[0]["spend"]), 1234.50)
        self.assertAlmostEqual(float(result.iloc[0]["revenue"]), 2500.25)
        removed = quality.loc[
            quality["check"] == "exact_duplicate_rows_removed", "count"
        ].iloc[0]
        self.assertEqual(int(removed), 1)



    def test_budget_plan_preserves_ids_and_rejects_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future_budgets.csv"
            pd.DataFrame(
                {
                    "channel": ["Google Ads"],
                    "campaign_id": ["00123"],
                    "campaign_name": ["Fictional_Search_Launch"],
                    "campaign_type": ["SEARCH"],
                    "horizon_days": ["30"],
                    "budget": ["1200.50"],
                    "is_new_campaign": ["yes"],
                }
            ).to_csv(path, index=False)
            plan = load_budget_plan(directory)
            self.assertIsNotNone(plan)
            self.assertEqual(str(plan.iloc[0]["campaign_id"]), "00123")
            self.assertTrue(bool(plan.iloc[0]["is_new_campaign"]))
            self.assertAlmostEqual(float(plan.iloc[0]["budget"]), 1200.5)

            pd.DataFrame(
                {
                    "channel": ["Google Ads"],
                    "campaign_id": ["00123"],
                    "horizon_days": [30],
                    "budget": [-1],
                    "is_new_campaign": ["maybe"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "is_new_campaign"):
                load_budget_plan(directory)



class ForecastTests(unittest.TestCase):
    @staticmethod
    def synthetic_data() -> pd.DataFrame:
        rows: list[dict] = []
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "channel": "Google Ads",
                    "campaign_id": "search-1",
                    "campaign_name": "US_NTM_Search",
                    "native_campaign_type": "SEARCH",
                    "campaign_family": "Search",
                    "funnel_signal": "Non-trademark",
                    "analog_key": "us_ntm_search",
                    "spend": 100.0 + index % 5,
                    "revenue": 0.0 if index % 9 == 0 else 460.0 + index % 13,
                    "conversions": 3.0,
                    "clicks": 20.0,
                    "impressions": 500.0,
                    "daily_budget": 120.0,
                    "source_file": "synthetic.csv",
                    "source_revenue_field": "revenue",
                }
            )
        for index, date in enumerate(dates[-12:]):
            rows.append(
                {
                    "date": date,
                    "channel": "Microsoft Ads",
                    "campaign_id": "search-2",
                    "campaign_name": "US_NTM_Search",
                    "native_campaign_type": "SEARCH",
                    "campaign_family": "Search",
                    "funnel_signal": "Non-trademark",
                    "analog_key": "us_ntm_search",
                    "spend": 45.0,
                    "revenue": 0.0 if index % 3 == 0 else 150.0,
                    "conversions": 1.0,
                    "clicks": 8.0,
                    "impressions": 220.0,
                    "daily_budget": 50.0,
                    "source_file": "synthetic.csv",
                    "source_revenue_field": "revenue",
                }
            )
        return pd.DataFrame(rows)

    def test_runtime_adaptation_uses_full_history_without_mutating_fallback(self) -> None:
        base_data = self.synthetic_data()
        base_data["revenue"] = base_data["revenue"] * 0.25
        base = build_model_artifact(base_data, build_quality_report(base_data))
        before = pickle.dumps(base)

        runtime = self.synthetic_data()
        runtime["revenue"] = runtime["revenue"] * 1.75
        adapted = adapt_model_artifact(base, runtime, build_quality_report(runtime))

        self.assertGreater(
            float(adapted["global"]["observed_roas"]),
            float(base["global"]["observed_roas"]),
        )
        self.assertEqual(adapted["global"]["prior_layer"], "runtime")
        self.assertEqual(adapted["runtime_adaptation"]["runtime_rows"], len(runtime))
        self.assertGreater(adapted["runtime_adaptation"]["runtime_group_overrides"], 0)
        self.assertEqual(before, pickle.dumps(base))

        sparse = runtime.iloc[:10].copy()
        sparse_adapted = adapt_model_artifact(
            base, sparse, build_quality_report(sparse)
        )
        self.assertEqual(
            sparse_adapted["runtime_adaptation"]["global_prior_source"],
            "committed fallback",
        )
        self.assertEqual(
            sparse_adapted["global"]["observed_roas"],
            base["global"]["observed_roas"],
        )

    def test_forecast_is_reconciled_ordered_and_repeatable(self) -> None:
        data = self.synthetic_data()
        artifact = build_model_artifact(data, build_quality_report(data))
        first, diagnostics = forecast_portfolio(
            data, artifact, horizons=(30,), simulations=250, seed=42
        )
        second, _ = forecast_portfolio(
            data, artifact, horizons=(30,), simulations=250, seed=42
        )
        self.assertEqual(diagnostics["active_campaigns"], 2)
        self.assertTrue({"campaign", "campaign_type", "channel", "overall"}.issubset(set(first["forecast_level"])))
        self.assertTrue(np.all(first["revenue_p10"] <= first["revenue_p50"]))
        self.assertTrue(np.all(first["revenue_p50"] <= first["revenue_p90"]))
        campaign_budget = first.loc[first["forecast_level"] == "campaign", "budget"].sum()
        overall_budget = first.loc[first["forecast_level"] == "overall", "budget"].iloc[0]
        self.assertAlmostEqual(float(campaign_budget), float(overall_budget), places=6)
        campaign_p50 = first.loc[
            first["forecast_level"] == "campaign", "revenue_p50"
        ].sum()
        overall_p50 = first.loc[
            first["forecast_level"] == "overall", "revenue_p50"
        ].iloc[0]
        self.assertAlmostEqual(float(campaign_p50), float(overall_p50), places=6)
        np.testing.assert_allclose(first["revenue_p50"], second["revenue_p50"])
        self.assertTrue(first["probability_target"].between(0, 1).all())
        self.assertTrue(first["budget_response_factor"].between(0.60, 1.0).all())
        campaign_diagnostics = first[first["forecast_level"] == "campaign"]
        self.assertTrue(
            {
                "empirical_roas_center",
                "hybrid_roas_center",
                "bounded_boosting_roas_p50",
            }.issubset(campaign_diagnostics.columns)
        )
        np.testing.assert_allclose(
            campaign_diagnostics["empirical_roas_center"],
            campaign_diagnostics["hybrid_roas_center"],
        )


    def test_explicit_new_campaign_uses_peer_prior_without_fake_history(self) -> None:
        data = self.synthetic_data()
        artifact = build_model_artifact(data, build_quality_report(data))
        peer = artifact["groups"]["family_funnel"][
            "Google Ads||Search||Non-trademark"
        ]
        launch_budget = float(peer["spend_median"]) * 30.0
        plan = pd.DataFrame(
            [
                {
                    "channel": "Google Ads",
                    "campaign_id": "new-search-003",
                    "campaign_name": "Fresh_NTM_Search_Launch",
                    "campaign_type": "SEARCH",
                    "horizon_days": 30,
                    "budget": launch_budget,
                    "is_new_campaign": True,
                }
            ]
        )

        predictions, diagnostics = forecast_portfolio(
            data,
            artifact,
            horizons=(30, 60),
            simulations=200,
            budget_plan=plan,
            seed=17,
        )
        launch = predictions[
            (predictions["forecast_level"] == "campaign")
            & (predictions["campaign_id"] == "new-search-003")
        ]
        self.assertEqual(launch["horizon_days"].tolist(), [30])
        row = launch.iloc[0]
        self.assertEqual(row["lifecycle_state"], "Launch")
        self.assertEqual(row["support_status"], "Insufficient Evidence")
        self.assertEqual(row["recommendation"], "TEST")
        self.assertEqual(int(row["data_points"]), 0)
        self.assertIn("planned new ->", row["peer_source"])
        self.assertAlmostEqual(float(row["budget_to_recent_ratio"]), 1.0, places=6)
        self.assertAlmostEqual(float(row["budget_response_factor"]), 1.0, places=6)
        self.assertEqual(diagnostics["active_campaigns"], 2)
        self.assertEqual(diagnostics["planned_new_campaigns"], 1)
        self.assertEqual(diagnostics["forecast_campaigns"], 3)

        for horizon in (30, 60):
            campaign_budget = predictions.loc[
                (predictions["forecast_level"] == "campaign")
                & (predictions["horizon_days"] == horizon),
                "budget",
            ].sum()
            overall_budget = predictions.loc[
                (predictions["forecast_level"] == "overall")
                & (predictions["horizon_days"] == horizon),
                "budget",
            ].iloc[0]
            self.assertAlmostEqual(float(campaign_budget), float(overall_budget), places=6)

    def test_budget_plan_rejects_unmatched_duplicate_and_colliding_rows(self) -> None:
        data = self.synthetic_data()
        artifact = build_model_artifact(data, build_quality_report(data))
        unmatched = pd.DataFrame(
            [
                {
                    "channel": "Google Ads",
                    "campaign_id": "typo-id",
                    "horizon_days": 30,
                    "budget": 100.0,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            forecast_portfolio(
                data, artifact, horizons=(30,), simulations=40, budget_plan=unmatched
            )

        duplicate = pd.DataFrame(
            [
                {
                    "channel": "Google Ads",
                    "campaign_id": "search-1",
                    "horizon_days": 30,
                    "budget": 100.0,
                },
                {
                    "channel": "Google Ads",
                    "campaign_name": "US_NTM_Search",
                    "horizon_days": 30,
                    "budget": 120.0,
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate rows"):
            forecast_portfolio(
                data, artifact, horizons=(30,), simulations=40, budget_plan=duplicate
            )

        collision = pd.DataFrame(
            [
                {
                    "channel": "Google Ads",
                    "campaign_id": "search-1",
                    "campaign_name": "Fresh_NTM_Search_Launch",
                    "campaign_type": "SEARCH",
                    "horizon_days": 30,
                    "budget": 100.0,
                    "is_new_campaign": True,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            forecast_portfolio(
                data, artifact, horizons=(30,), simulations=40, budget_plan=collision
            )



class DecisionTests(unittest.TestCase):
    def test_fully_paused_plan_returns_safe_no_test_decision(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "horizon_days": 30,
                    "forecast_level": "overall",
                    "recommendation": "REVIEW",
                    "support_status": "Insufficient Evidence",
                    "probability_target": 0.0,
                    "target_roas": 3.0,
                    "revenue_p10": 0.0,
                    "revenue_p50": 0.0,
                    "revenue_p90": 0.0,
                    "roas_p50": 0.0,
                    "budget": 0.0,
                },
                {
                    "horizon_days": 30,
                    "forecast_level": "campaign",
                    "budget": 0.0,
                },
            ]
        )
        summary = build_decision_summary(predictions, {"active_campaigns": 1})
        decision = summary["decisions"][0]
        self.assertEqual(decision["experiment"]["campaign"], "No active spend")
        self.assertEqual(decision["experiment"]["holdout_budget"], 0.0)
        explanation = deterministic_explanation(decision)
        self.assertIn("recommended_test", explanation)
        self.assertIn("zero", explanation["recommended_test"].lower())



if __name__ == "__main__":
    unittest.main()

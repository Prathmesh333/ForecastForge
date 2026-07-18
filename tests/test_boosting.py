from __future__ import annotations

import importlib.util
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.boosting import (
    BUNDLE_KIND,
    NUMERIC_FEATURES,
    BoostingConfig,
    _build_training_examples,
    _semantic_token_flags,
    fit_boosting_bundle,
    predict_boosting_bundle,
    run_boosting_challenger,
)


def _synthetic_history(days: int = 240, campaigns: int = 6) -> pd.DataFrame:
    rows: list[dict] = []
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    for campaign in range(campaigns):
        channel = "Google Ads" if campaign % 2 == 0 else "Microsoft Ads"
        family = "Search" if campaign % 3 else "Performance Max"
        native = "SEARCH" if family == "Search" else "PERFORMANCE_MAX"
        funnel = "Non-trademark" if campaign % 2 == 0 else "Trademark"
        marker = "NTM" if funnel == "Non-trademark" else "TM"
        for index, date in enumerate(dates):
            spend = 35.0 + campaign * 4.0 + index % 7
            roas = 2.2 + campaign * 0.22 + 0.12 * np.sin(index / 13.0)
            rows.append(
                {
                    "date": date,
                    "channel": channel,
                    "campaign_id": f"campaign-{campaign:03d}",
                    "campaign_name": f"{family}_{marker}_Campaign_{campaign:03d}",
                    "native_campaign_type": native,
                    "campaign_family": family,
                    "funnel_signal": funnel,
                    "spend": spend,
                    "revenue": spend * roas,
                    "daily_budget": spend * 1.15,
                }
            )
    return pd.DataFrame(rows)


def _requests(history: pd.DataFrame) -> pd.DataFrame:
    latest = history.sort_values("date").drop_duplicates(
        ["channel", "campaign_id"], keep="last"
    )
    records: list[dict] = []
    for row in latest.itertuples(index=False):
        for horizon in (30, 60):
            records.append(
                {
                    "channel": row.channel,
                    "campaign_id": row.campaign_id,
                    "horizon_days": horizon,
                    "budget": float(row.daily_budget) * horizon,
                }
            )
    return pd.DataFrame(records)


class BoostingSafetyTests(unittest.TestCase):
    def test_disabled_config_is_a_safe_noop(self) -> None:
        predictions, diagnostics = run_boosting_challenger(
            pd.DataFrame(),
            pd.DataFrame(),
            config=BoostingConfig(enabled=False),
        )
        self.assertTrue(predictions.empty)
        self.assertFalse(diagnostics["used"])
        self.assertEqual(diagnostics["reason"], "disabled_by_config")

    def test_scoring_requires_bundle_and_never_builds_targets(self) -> None:
        history = _synthetic_history()
        with patch(
            "src.boosting._build_training_examples",
            side_effect=AssertionError("scoring must not construct labels"),
        ):
            predictions, diagnostics = run_boosting_challenger(
                history,
                _requests(history),
            )
        self.assertTrue(predictions.empty)
        self.assertEqual(diagnostics["reason"], "bundle_missing")

    def test_training_targets_never_extend_beyond_history(self) -> None:
        history = _synthetic_history()
        config = BoostingConfig(
            max_origins=10,
            min_training_rows=1,
            min_training_campaigns=1,
            min_training_origins=1,
        )
        training, metadata = _build_training_examples(history, (30, 60, 90), config)
        self.assertFalse(training.empty)
        self.assertLessEqual(training["target_end"].max(), metadata["as_of"])
        self.assertTrue((training["origin"] < training["target_end"]).all())
        self.assertTrue(
            (np.abs(training["budget"] - training["future_spend"]) > 1e-6).any(),
            "pseudo-origin budget must be origin-known, not realized future spend",
        )

    def test_raw_names_and_ids_are_not_model_features(self) -> None:
        first = _semantic_token_flags(
            "Search_NTM_Campaign_001", "SEARCH", "Search", "Non-trademark"
        )
        second = _semantic_token_flags(
            "Search_NTM_Campaign_999999", "SEARCH", "Search", "Non-trademark"
        )
        self.assertEqual(first, second)
        self.assertEqual(first["token_search"], 1.0)
        self.assertEqual(first["token_non_trademark"], 1.0)
        joined = " ".join(NUMERIC_FEATURES).lower()
        self.assertNotIn("campaign_id", joined)
        self.assertNotIn("campaign_name", joined)

    def test_missing_xgboost_returns_primary_safe_fallback(self) -> None:
        history = _synthetic_history()
        config = BoostingConfig(
            max_origins=8,
            min_training_rows=10,
            min_training_campaigns=2,
            min_training_origins=2,
            boost_rounds=5,
        )
        with patch("src.boosting._load_xgboost", return_value=(None, "ImportError")):
            bundle, diagnostics = fit_boosting_bundle(
                history, (30, 60), config=config
            )
        self.assertIsNone(bundle)
        self.assertFalse(diagnostics["used"])
        self.assertIn("xgboost_unavailable", diagnostics["reason"])
        self.assertGreater(diagnostics["training_rows"], 0)

    def test_invalid_bundle_returns_primary_safe_fallback(self) -> None:
        history = _synthetic_history()
        predictions, diagnostics = predict_boosting_bundle(
            history,
            _requests(history),
            {"bundle_kind": "not-ours"},
        )
        self.assertTrue(predictions.empty)
        self.assertIn("invalid_bundle", diagnostics["reason"])


@unittest.skipUnless(
    importlib.util.find_spec("xgboost") is not None,
    "XGBoost is intentionally optional",
)
class BoostingRuntimeTests(unittest.TestCase):
    def _fit(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        history = _synthetic_history()
        requests = _requests(history)
        config = BoostingConfig(
            max_origins=8,
            min_training_rows=10,
            min_training_campaigns=2,
            min_training_origins=2,
            boost_rounds=24,
            seed=47,
        )
        bundle, diagnostics = fit_boosting_bundle(
            history,
            (30, 60),
            config=config,
        )
        self.assertIsNotNone(bundle, diagnostics)
        assert bundle is not None
        self.assertEqual(bundle["bundle_kind"], BUNDLE_KIND)
        self.assertIsInstance(bundle["booster_bytes"], bytes)
        self.assertTrue(diagnostics["trained"])
        return history, requests, bundle

    def test_bundle_is_compact_serializable_and_identity_free(self) -> None:
        history, _, bundle = self._fit()
        payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertLess(len(payload), 2_000_000)
        self.assertNotIn(b"campaign-000", payload)
        self.assertNotIn(b"Campaign_000", payload)
        restored = pickle.loads(payload)
        self.assertEqual(restored["booster_bytes"], bundle["booster_bytes"])
        self.assertEqual(
            restored["encoder"]["feature_names"],
            bundle["encoder"]["feature_names"],
        )
        raw_columns = " ".join(restored["encoder"]["feature_names"]).lower()
        self.assertNotIn("campaign_id", raw_columns)
        self.assertNotIn("campaign_name", raw_columns)
        self.assertGreater(len(history), 0)

    def test_scoring_is_fit_free_deterministic_finite_and_reconciled(self) -> None:
        history, requests, bundle = self._fit()
        with (
            patch(
                "src.boosting._build_training_examples",
                side_effect=AssertionError("scoring must not construct labels"),
            ),
            patch(
                "xgboost.train",
                side_effect=AssertionError("scoring must not fit XGBoost"),
            ),
        ):
            first, first_diagnostics = predict_boosting_bundle(
                history, requests, bundle
            )
            second, second_diagnostics = run_boosting_challenger(
                history, requests, bundle=bundle
            )

        self.assertTrue(first_diagnostics["used"])
        self.assertTrue(second_diagnostics["used"])
        self.assertEqual(len(first), len(requests))
        self.assertTrue(np.isfinite(first["boosting_roas_p50"]).all())
        self.assertTrue(np.isfinite(first["boosting_revenue_p50"]).all())
        self.assertTrue((first["boosting_roas_p50"] >= 0).all())
        np.testing.assert_allclose(
            first["boosting_revenue_p50"],
            first["boosting_roas_p50"] * first["budget"],
        )
        np.testing.assert_allclose(
            first["boosting_revenue_p50"], second["boosting_revenue_p50"]
        )
        self.assertLessEqual(
            pd.Timestamp(first_diagnostics["latest_training_target"]),
            pd.Timestamp(first_diagnostics["artifact_history_as_of"]),
        )

    def test_unseen_runtime_categories_route_to_other_bucket(self) -> None:
        history, requests, bundle = self._fit()
        novel = history.copy()
        latest = novel["date"].max()
        mask = novel["date"].eq(latest)
        novel.loc[mask, "native_campaign_type"] = "NEVER_SEEN_NATIVE_TYPE"
        novel.loc[mask, "campaign_family"] = "Never Seen Family"
        novel.loc[mask, "funnel_signal"] = "Never Seen Funnel"
        novel.loc[mask, "campaign_name"] = "Unknown_New_Format_999999"

        with patch(
            "xgboost.train",
            side_effect=AssertionError("scoring must not fit XGBoost"),
        ):
            predictions, diagnostics = predict_boosting_bundle(
                novel, requests, bundle
            )
        self.assertTrue(diagnostics["used"])
        self.assertEqual(len(predictions), len(requests))
        self.assertTrue(np.isfinite(predictions["boosting_roas_p50"]).all())


if __name__ == "__main__":
    unittest.main()

# Technical Documentation

## Scope

ForecastForge forecasts attributed ecommerce revenue and ROAS for 30, 60, and 90-day aggregate planning periods. Outputs are produced at campaign, campaign-type, channel, and overall portfolio levels for Google Ads, Meta Ads, and Microsoft Ads. The system accepts an optional future media budget plan and treats the supplied channel attribution as the source of truth.

## Data preprocessing

The ingestion layer reads every usable CSV in the supplied data directory instead of relying on a fixed row count or private filename. Platform adapters normalize dates, campaign identity, campaign type, spend, budget, and attributed conversion value into one schema.

Important controls include:

- string-first CSV loading to preserve leading-zero campaign IDs;
- whitespace-tolerant column matching and generic channel/platform/source support;
- Unicode-safe currency parsing and Google micros conversion;
- exact-row deduplication across overlapping exports;
- reporting, rather than silently removing, conflicting campaign-date records;
- non-negative and finite value checks;
- explicit validation for unmatched, duplicate, inactive, or colliding budget-plan rows;
- deterministic, auditable campaign-family and funnel inference.

## Forecasting methodology

The production forecast is a guarded hybrid:

1. An empirical-Bayes center combines direct recent campaign performance with the most specific credible peer prior. New and sparse campaigns borrow more; mature campaigns rely mainly on their own evidence.
2. A pretrained XGBoost challenger uses planned budget, calendar terms, stable categories, semantic campaign flags, and 7/14/28/56/84-day lag summaries.
3. XGBoost influence is capped at 30%, shrunk for sparse or young histories, bounded relative to the empirical estimate, and disabled for plan-only launches.
4. A hurdle process models zero-revenue versus positive-revenue days. Horizon-specific launch-success priors separately represent campaigns that remain structurally at zero.
5. Month-of-year factors, lifecycle trends, budget saturation, channel shocks, and shared horizon uncertainty modify the simulated distribution.
6. Campaign simulations aggregate bottom-up. Aggregate samples retain correlated uncertainty and are scaled so every parent P50 exactly equals the sum of its child P50 values.

The result is P10/P50/P90 revenue and ROAS, probability of a target ROAS, evidence status, lifecycle state, and an ACT/HOLD/TEST/REVIEW recommendation.

## Model selection

All challengers used leakage-safe pseudo-origins with the same future-budget information and reusable features. Raw campaign ID and raw campaign name were excluded.

| Guarded blend | 30-day WAPE | 60-day WAPE | 90-day WAPE |
|---|---:|---:|---:|
| Empirical + XGBoost | 41.50% | 32.14% | 32.82% |
| Empirical + CatBoost categories | 41.40% | 32.15% | 34.03% |
| Empirical + CatBoost sanitized text | 40.82% | 32.20% | 33.69% |

CatBoost text was the measured 30-day champion. XGBoost was selected for the committed production artifact because it was nearly tied at 30/60 days, stronger at 90 days, trained about 4.5 times faster in the rolling comparison, and avoids shipping a second large model dependency. The dashboard discloses the horizon champions and shows live empirical, XGBoost, and hybrid scenario views. It does not fabricate a live CatBoost prediction because no CatBoost artifact is shipped.

## Future budget handling

`future_budgets.csv` may specify campaign budgets by horizon. Existing campaigns are resolved by ID or an unambiguous name. A new campaign must explicitly set `is_new_campaign=true`; it receives peer and launch priors but no fabricated outcome history.

If no plan is provided, the application extends the most recent 14-day mean campaign spend. This is a convenience scenario and not an unconditional budget forecast. Backtests condition on actual future spend so they evaluate outcome forecasting rather than budget guessing.

## Uncertainty and evidence

- P10-P90 ranges come from deterministic seeded simulations, not symmetric error bars.
- Shared channel and market shocks prevent portfolio uncertainty from disappearing through naive aggregation.
- Support is labeled `Supported`, `Extrapolating`, or `Insufficient Evidence` using direct and peer history plus the proposed spend range.
- Unsupported scenarios receive a `TEST` recommendation even when their point estimate is attractive.
- The recommended experiment includes a bounded budget, holdout, target guardrail, and scale/stop rule.

## AI integration strategy

The numeric path is deterministic and offline. The interactive application provides two isolated explanation paths:

1. The highlighted standalone `Ask ForecastForge AI` workspace accepts a session-only Gemini API key and supports direct questions about the current generated scenario. Only an allowlisted forecast snapshot is transmitted after explicit consent; raw history, campaign IDs, local paths, environment variables, and the API key itself are excluded from the grounding JSON.
2. A server-configured OpenAI-compatible adapter can generate the fixed executive summary through `RANGE_LLM_ENDPOINT`, `RANGE_LLM_API_KEY`, and `RANGE_LLM_MODEL`.

The Gemini analyst uses a fixed Google Generative Language endpoint, an allowlisted stable model ID, a high-priority system instruction, bounded conversation history, and a JSON response schema. The prompt requires source exclusivity, numeric fidelity, internal evidence tags, non-causal language, refusal when evidence is missing, and resistance to instructions embedded in campaign data. The application validates the returned shape and exposes safe errors without API response bodies or credentials.

Both paths are available only in the interactive prototype and are never imported or invoked by `run.sh`.

## Assumptions and limitations

- Forecasts estimate attributed revenue under a budget scenario, not incremental causal lift.
- Meta's `conversion` field is provisionally treated as attributed conversion value and should be confirmed with the organizer.
- Campaign-name ontology can miss unfamiliar naming conventions.
- Promotions, inventory, pricing, competitor activity, and tracking changes are not supplied features.
- Microsoft Ads and launch cohorts are sparse and zero-inflated; their intervals and evidence labels should be interpreted cautiously.
- The 30-day integrated model is slightly behind the strong recent-ROAS baseline; 60/90-day results are stronger.
- The exact organizer output template was not included in the supplied PDFs and still requires confirmation.

## Reproducibility

- Python 3.12.13 is the verified runtime.
- Every dependency is exactly pinned.
- `pickle/model.pkl` contains the pretrained booster and fitted encoder.
- `run.sh` accepts `DATA_DIR`, `MODEL_PATH`, and `OUTPUT_PATH`, supports defaults, makes no network calls, prompts for no input, and writes the output fresh.
- The submission validator checks artifact privacy, runner permissions, output hierarchy, determinism, and byte-identical protected inputs before and after scoring.

See [MODEL_CARD.md](MODEL_CARD.md) for the full backtest table and [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries.

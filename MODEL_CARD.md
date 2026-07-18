# Model Card

## Intended use

ForecastForge supports short-term ecommerce media planning across 30, 60, and 90-day horizons. It is designed for portfolios containing launches, sparse campaigns, and heterogeneous platform exports. It is decision support, not an autonomous budget allocator.

## Training and judge-time contract

The committed artifact was trained only on organizer-provided history. It contains compact aggregate priors, hashed analog keys, an identity-free fitted encoder, and a serialized XGBoost booster. Raw rows, raw campaign IDs as features, and readable private campaign names are excluded.

At judge time the scorer loads this artifact, computes lag and descriptive context features from the supplied history, and predicts. It does not fit a booster, mutate the artifact, use future outcomes, download data, or make a network request. Runtime peer/context estimates must meet support thresholds; otherwise committed priors remain in force.

## Model design

- Empirical-Bayes center combining recent campaign ROAS with the most specific credible peer prior.
- Pretrained XGBoost challenger using planned budget, 7/14/28/56/84-day lags, calendar terms, stable categories, and whitelisted semantic flags.
- XGBoost influence capped at 30%, reduced for sparse/young histories, disabled for plan-only launches, and range-bounded relative to the empirical center.
- Lifecycle-aware weighting for Launch, Ramp, Mature, Declining, and Inactive campaigns.
- Hurdle simulation for zero versus positive attributed-revenue days.
- Horizon-specific persistent launch-success priors, excluding right-censored launch cohorts.
- Shrunk month factors, conservative budget saturation, channel shocks, and horizon-level market risk.
- Bottom-up aggregation with aggregate P50 exactly reconciled to the sum of child P50s.

## XGBoost versus CatBoost

Both models were evaluated with leakage-safe pseudo-origins and the same non-identity feature information.

| Rolling benchmark | 30-day WAPE | 60-day WAPE | 90-day WAPE |
|---|---:|---:|---:|
| Empirical center | 41.51% | 33.58% | 33.72% |
| XGBoost only | 46.47% | 31.97% | 36.41% |
| CatBoost native categories | 46.51% | 34.63% | 41.09% |
| Empirical + XGBoost blend | 41.50% | 32.14% | 32.82% |
| Empirical + CatBoost blend | 41.40% | 32.15% | 34.03% |

Across eight rolling fits, XGBoost training took 7.33 seconds versus 32.79 seconds for CatBoost. A CatBoost model with sanitized text improved on plain CatBoost and reached 43.70%, 32.42%, and 39.26% model-only WAPE, but its blend remained weaker at 90 days. CatBoost text is a credible future challenger; XGBoost was selected for the submission because it was faster, smaller operationally, and stronger overall.

## Latest integrated rolling-origin result

Protocol:

- eight equally spaced historical cutoffs after at least one year of history;
- actual future spend for every cutoff-active campaign, including zeros, supplied as the scenario budget;
- targets restricted to complete future windows;
- P50 compared with actual attributed revenue for positive-actual-spend entities;
- baseline equals matched campaign recent ROAS multiplied by that campaign's future spend;
- P10–P90 coverage evaluated against an intended 80% interval;
- 200 deterministic simulations per entity in this regression run.

| Level | Horizon | Entities | Model WAPE | Baseline WAPE | Coverage |
|---|---:|---:|---:|---:|---:|
| Campaign | 30 | 309 | 42.69% | 41.33% | 82.5% |
| Campaign | 60 | 309 | 31.89% | 34.47% | 85.8% |
| Campaign | 90 | 309 | 31.51% | 40.21% | 87.1% |
| Campaign type | 30 | 55 | 34.75% | 30.53% | 80.0% |
| Campaign type | 60 | 55 | 24.09% | 25.51% | 83.6% |
| Campaign type | 90 | 55 | 22.85% | 29.92% | 87.3% |
| Channel | 30 | 24 | 21.94% | 19.83% | 83.3% |
| Channel | 60 | 24 | 16.05% | 16.85% | 95.8% |
| Channel | 90 | 24 | 16.84% | 24.67% | 91.7% |
| Overall | 30 | 8 | 18.74% | 15.26% | 87.5% |
| Overall | 60 | 8 | 11.56% | 13.49% | 100.0% |
| Overall | 90 | 8 | 10.78% | 19.81% | 100.0% |

This is a regression benchmark on the available organizer data, not an estimate of the hidden score. It demonstrates the intended behavior: strong 60/90-day improvement, reasonable interval coverage, and coherent aggregation. The 30-day model is slightly worse than the strong recent-ROAS baseline at every level and is intentionally disclosed.

Launch campaign WAPE was 43.2%, 35.0%, and 22.5% at 30/60/90 days, with nine evaluated entities per horizon. Microsoft Ads remains the weakest slice (campaign WAPE 63.7%, 69.5%, and 52.3%; coverage 61.1%, 55.6%, and 58.3%) because its history is sparse and heavily zero-inflated. ForecastForge reflects that uncertainty in support status and launch priors instead of making a broad accuracy claim.

## Generalization controls

- Targets for pseudo-origin training end on or before the artifact's history cutoff.
- Features at each origin use history available at or before that origin only.
- Raw campaign ID and raw name are excluded; sanitized semantic flags represent only reusable concepts.
- Categories are capped and unknown values map to `__other__`.
- The booster is guarded by an empirical prior rather than trusted for unrestricted extrapolation.
- Persistent launch failure is modeled separately from ordinary zero-revenue days.
- Runtime context overrides require minimum rows, active days, and campaign diversity.
- The offline validator hashes all protected inputs before and after scoring.

## Decision policy

- `ACT`: supported evidence and at least 80% probability of meeting target.
- `HOLD`: supported evidence and 55–80% target probability.
- `REVIEW`: supported evidence below 55%, or zero budget.
- `TEST`: extrapolating or insufficient evidence, regardless of point forecast.

## Known limitations

- The dataset is observational; saturation is a planning heuristic, not a causal lift curve.
- Backtests condition on known future spend. The no-plan 14-day spend extension is only a convenience assumption and performs materially worse as an unconditional budget forecast.
- Campaign-name ontology is deterministic and auditable but may miss unfamiliar conventions.
- Month effects have only about two annual cycles in the supplied history.
- Promotions, inventory, pricing, competitor actions, and tracking changes are unavailable.
- Meta revenue semantics and the exact hidden output schema need organizer confirmation.
- Microsoft and launch slices have limited sample sizes.
- The optional LLM explains structured evidence; it does not improve numeric accuracy or establish causality.

## Safe use

Use P50 together with evidence status and interval width. For `Extrapolating` or `Insufficient Evidence` scenarios, run the recommended bounded experiment before scaling. Re-estimate the offline artifact after material attribution, catalog, pricing, or campaign-structure changes.

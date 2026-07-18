# Five-Minute Demo Runbook

## Before recording

1. Activate the pinned Python 3.12.13 environment.
2. Run python -m src.validate_submission.
3. Start streamlit run app.py.
4. Keep the optional LLM toggle off unless a tested endpoint is configured.
5. Open the app at 100% browser zoom and select the 30-day horizon.

## 0:00-0:35 — The decision problem

Say: "A point forecast is not a budget decision. A planner needs to know the range of outcomes,
whether the proposed spend is supported by evidence, and what to test when it is not."

Show the portfolio P10/P50/P90 range, chance of target, recommendation, and evidence-supported
budget share.

## 0:35-1:30 — A real budget decision

Keep total budget fixed. Increase one channel and reduce another. Show:

- unchanged total budget;
- requested versus actually applied channel allocation;
- baseline diamonds against scenario forecast bars;
- P50, P10, ROAS, and target-probability changes.

Say: "The baseline and scenario use the same random numbers, so the displayed difference reflects
the budget decision rather than simulation noise."

## 1:30-2:25 — Evidence, not false certainty

Open the decision queue and point to lifecycle, evidence status, peer source, interval, and
saturation columns. Explain that aggregate support is budget-weighted: one small sparse campaign
does not poison the whole portfolio, but unsupported spend remains visible.

## 2:25-3:25 — The cold-start differentiator

Copy data/example_budget_plan.csv to data/future_budgets.csv, then refresh the app. The example
contains one existing campaign and one explicitly declared new Meta launch.

Show that the launch:

- appears only in the planned horizon;
- has Launch lifecycle and zero direct data points;
- borrows the most specific credible peer prior and a horizon-specific launch-success probability;
- is labeled Insufficient Evidence;
- receives TEST, not an overconfident scale recommendation.

Say: "Typos are never auto-treated as launches. Unmatched, duplicate, inactive, or colliding rows
fail loudly."

Remove data/future_budgets.csv after the recording if it should not be part of the local demo state.

## 3:25-4:15 — Turn uncertainty into learning

Show the highest-value learning action. Explain the score inputs: downside to target, interval
width, decision sensitivity, lifecycle, and support. Point out the randomized 90/10 treatment-
holdout design, spend withheld, business-target guardrail, and scale rule.

## 4:15-4:40 — Grounded AI analyst

Use the highlighted standalone `Ask ForecastForge AI` workspace above the forecast tabs, paste a temporary Gemini API key, confirm data transmission, and ask:
"Why is the portfolio marked TEST?" Show that the response cites the generated portfolio/campaign
evidence, states limitations, and recommends the supplied bounded experiment. Expand the context
preview to prove that raw daily history, campaign IDs, and the API key are not sent.

## 4:40-5:00 — Trust and proof

Open the data-quality expander and finish with:

- all four forecast levels reconcile bottom-up, including exact P50 reconciliation;
- the intended 80% campaign interval achieved 82.5%, 85.8%, and 87.1% coverage at 30/60/90 days;
- the model beat the matched campaign baseline at 60 and 90 days, while the 30-day result was
  slightly worse and is disclosed;
- a committed XGBoost challenger was chosen only after a fair CatBoost comparison and never trains
  inside the judge runner;
- the scoring runner is deterministic and offline;
- the public model is compact and identifier-sanitized.

Never show the API key in the recording. Clear the conversation and remove the key after the demo.

## Backup if the UI fails

~~~powershell
.\run.ps1 .\data .\pickle\model.pkl ".\output\demo predictions.csv"
python -m src.validate_submission
~~~

Open output/demo predictions.csv, output/data_quality.csv, and output/decision_summary.json. The
numeric submission does not depend on Streamlit or an LLM.

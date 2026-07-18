# Submission Readiness

This repository is structured as an offline, reproducible hackathon submission. The working name is
temporary; the product claim is the forecast-to-evidence-to-experiment decision loop.

## Deadline

The supplied submission guide sets the deadline at **July 19, 2026, 10:00 PM IST** and instructs
teams to email the public repository URL, exact run command, team members, and college name to the
organizer address in the guide. This attached guide is more specific than the older July 15 date
still shown on the public event page.



## Ready now

- One-command Linux/macOS runner: run.sh DATA_DIR MODEL_PATH OUTPUT_PATH
- Matching Windows runner: run.ps1 DATA_DIR MODEL_PATH OUTPUT_PATH
- Committed, compact model artifact with hashed campaign analog identifiers
- Pretrained XGBoost booster and fitted identity-free encoder inside the artifact
- Fictional three-platform public sample
- Dynamic schema normalization for Google, Meta, Microsoft, and generic exports
- Strict budget-plan validation, including an explicit new-campaign path
- Reconciled campaign, campaign-type, channel, and portfolio forecasts
- Detailed rolling-origin evaluation and automated tests
- Interactive Streamlit demo
- Session-only, consent-gated Gemini analyst grounded in generated forecast JSON
- First-class technical documentation, architecture overview, and demo workflow
- Fail-fast privacy, packaging, determinism, and output validator

## Organizer confirmation still required

Two items cannot be inferred safely from the supplied materials:

1. the exact hidden-scoring CSV filename, column names, order, and shape;
2. whether Meta's conversion field is officially attributed value/revenue.

The current predictions.csv contract is provisional. Once the exact scoring template is received,
change only the final projection in src/predict.py and the matching validator contract. Do not
change the forecast engine to guess a schema.

## Official command

~~~bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
~~~

The command does not install, fit a model, prompt, or make a network call. It computes lag/context
features, loads the committed booster, overwrites the designated output, and writes a data-quality
report and decision summary beside it. The validator confirms scoring leaves the artifact unchanged.

## Final packaging checklist

- [ ] Add team name, member names, college/company, contact, and repository URL.
- [ ] Replace the provisional output projection with the organizer's exact template.
- [ ] Confirm Meta revenue semantics in writing.
- [x] Run python -m unittest discover -s tests -v.
- [x] Run python -m src.validate_submission.
- [x] Run the official command from a staged clean export and paths containing spaces.
- [x] Confirm git ls-files --stage run.sh reports mode 100755.
- [x] Confirm no PDF, private dataset, internal idea note, output, secret, or log is tracked.
- [ ] Record the five-minute demo using DEMO_RUNBOOK.md.
- [x] Keep the optional LLM disabled for the offline scoring path.

## Confidentiality controls

The public repository intentionally excludes organizer CSVs, PDFs, and internal planning notes.
The sample files are fictional. The model artifact uses hashed analog keys and compact aggregate
quantiles (at most 21 support points), rather than readable campaign names or near-row-level
empirical arrays. src.validate_submission checks these properties and scans the Git index for
common secret and local-path leaks.

## Originality claim

No public search can prove that no participant has built something similar, and private submissions
are not observable. The defensible differentiation is the integrated workflow:

1. lifecycle-aware cross-platform cold-start transfer;
2. probabilistic, bottom-up reconciled forecasts;
3. exposure-weighted evidence coverage;
4. baseline-versus-scenario comparison under common random numbers;
5. strict plan-only launch handling without fabricated history;
6. value-of-information experiment routing when evidence is weak;
7. a leakage-safe, guarded XGBoost challenger selected after a measured CatBoost comparison.

Present that combination and the working trust controls as the novelty. Avoid claiming that a
specific regression or dashboard component is unique.

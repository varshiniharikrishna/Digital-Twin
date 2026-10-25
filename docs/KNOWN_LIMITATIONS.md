# Known limitations

## Data and target validity

- The bundled CSVs contain synthetic records. The benchmark checks the implementation against its generated data distribution and is not external or clinical validation.
- Recovery percentage may use an observed final field, but in the bundled source it is generally a constructed strength/ROM proxy. It is not a validated patient-reported outcome measure.
- Recovery weeks uses observed duration. It does not fall back to the estimated clinical duration feature; missing target values are excluded from fitting and scoring.
- No real, governed clinical dataset was supplied, and there is no prospective validation.

## Model evaluation

- Current R² values are split-specific. Repeated grouped CV uses a development subset of training-year patients, while a patient-disjoint subset calibrates the conformal interval. The fitted production pipeline is not refit on calibration patients, preserving the calibration/model separation at the cost of using fewer fit examples.
- Split-conformal finite-sample coverage relies on exchangeability. A chronological or hospital/domain shift can violate that assumption. The 2025 coverage values are measurements on this bundled holdout, not a guarantee for future cases.
- LightGBM was not installed in the runtime used for the latest benchmark; it is not in the current candidate results. XGBoost and ensemble hyperparameters are fixed in this implementation rather than searched. Candidate results and model-selection uncertainty should be interpreted accordingly.
- No statistical threshold establishes clinical acceptability. Recovery duration remains less predictable than recovery percentage on the held-out data, with MAE around five weeks.

## Digital twin, adaptive loop, and pathways

- Trajectory parameters and simulated treatment response are engineering assumptions. The held-out trajectory report predates the latest shared `twin_state` engine and needs a fresh evaluation against persistence, cohort-mean, and old-curve baselines.
- The adaptive replay measures forecast error, not causal treatment response or patient benefit. Existing replay results showed worse forecast error for adaptive arms than static on generated data and have not been rerun after the latest shared-engine change.
- There is no patient-level pathway assignment history linked to outcomes. Pathway rankings are based on historical similarity/compatibility and cohort summaries; they do not estimate treatment effects or prove one protocol is superior.
- Deviation thresholds are engineering candidates. Without adjudicated deterioration labels, sensitivity, specificity, alert burden, and detection delay cannot be clinically calibrated.

## Explainability and deployment

- SHAP was unavailable during the latest run. Tree-specific explanations for both targets, aggregation of encoded categories, and bootstrap stability of top features remain incomplete.
- The app is a prototype, not a regulated medical device or autonomous decision maker. Privacy/security and regulatory requirements must be assessed for the actual deployment jurisdiction, data controller, and intended use before real patient data are used.
- The Streamlit app remains a large monolithic module; report freshness checks are not universal. Run `make all` before presenting results so ignored local report/artifact outputs match the current data and source.

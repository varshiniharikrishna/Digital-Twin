# Five-minute teacher demo

## Before the demo

From the `Digital twin` project folder, make sure the reports and selected model artifacts exist. `make all` regenerates the model benchmark and analysis reports; run it ahead of the presentation because grouped cross-validation can take several minutes. Then start the Streamlit dashboard with:

```bash
DT_DATA_SOURCE=csv python3 -m streamlit run app.py
```

The bundled CSV records are synthetic demo data. This demo shows the software and its measured behavior on that dataset, not a validated clinical decision-support tool.

## Walkthrough

### 0:00–0:45 — Digital Twin Overview

Open **Digital Twin Overview** and select a patient from the 2025 test cohort. Say: “The dashboard separates the baseline record from the model estimate. Recovery percentage is an observed value when present or a formula-based strength/ROM proxy; weeks uses recorded recovery duration. These are generated demo records, not clinical outcome validation.”

Point out the model names and the holdout R² interval. The app uses separate persisted models for recovery percentage and recovery weeks. Do not call the confidence interval a probability that this patient's outcome is correct.

### 0:45–1:45 — Model Benchmark

Open **Model Benchmark**. Show both target tables. Say: “We select with repeated patient-grouped cross-validation on the development portion of the 2023–24 cohort. A patient-disjoint subset calibrates the prediction interval. The chronological 2025 cohort is held out for one final measurement.”

Numbers from the latest generated reports:

- Recovery percentage: VotingRegressor; CV R² **0.8829 ± 0.0477**, grouped OOF bootstrap 95% CI **0.8491–0.9094**. Holdout R² **0.9227** (patient-bootstrap 95% CI **0.8971–0.9431**), MAE **1.6102 points**, RMSE **2.2685 points**, n=224.
- Recovery weeks: VotingRegressor; CV R² **0.7397 ± 0.0408**, grouped OOF bootstrap 95% CI **0.6889–0.7843**. Holdout R² **0.7458** (patient-bootstrap 95% CI **0.6836–0.7954**), MAE **5.2012 weeks**, RMSE **6.7903 weeks**, n=242.
- Split-conformal intervals had observed 2025 holdout coverage **95.98%** for recovery percentage (half-width **4.5081 points**, calibration n=80) and **96.28%** for weeks (half-width **13.9085 weeks**, calibration n=83). Coverage is an observed sample result; exchangeability may not hold for future or shifted populations.

### 1:45–2:35 — Week-by-Week Twin and Timeline

Open **Week-by-Week Twin**, then **Weekly Recovery Timeline**. Show the same twin state progressing across weeks. Say: “The pages use the shared state-update engine. Weekly inputs are observations; the predicted line is a model trajectory. Missing values stay missing instead of being presented as measurements.”

### 2:35–3:25 — Simulation & Recalibration

Open **Simulation & Recalibration**. Demonstrate entering an observed value and leaving another metric as not observed. Say: “Recalibration updates the estimate using available observations. The evaluation is a forecast-error replay; it does not establish that adaptive changes improve health outcomes.”

If asked about the prior replay result, report it plainly: on the generated data, recalibration's patient-level forecast MAE was higher than static by **0.2072 points** on-trigger (95% cluster-bootstrap CI **0.1187–0.3057**) and **0.4206 points** when recalibrated every week (**0.2685–0.5681**), n=224. These older replay measurements have not been rerun against the latest shared trajectory update.

### 3:25–4:15 — Recommendations and Pathway Ranking

Open **Recommendations**, then **Pathway Ranking**. Say: “Pathway summaries use training-year patients only, and small cohorts are marked insufficient evidence. Similarity ranks historical matches; it is not a treatment-effect estimate.”

### 4:15–5:00 — Close with limitations

Say: “The system is a prototype. Its inputs are synthetic, the primary percentage can be a proxy, and no prospective or governed clinical validation was supplied. The metrics do not show treatment benefit or support autonomous clinical decisions. A next validation step needs an approved, de-identified clinical dataset and a prespecified external evaluation.”

## If asked about before/after performance

An earlier full-training grouped benchmark measured recovery percentage R² **0.8964** (95% CI **0.8757–0.9150**) and weeks R² **0.7577** (**0.7175–0.7923**). The current benchmark reserves 20% of training patients for independent interval calibration, so the development sample is smaller and the scores are not a controlled estimate of the effect of removing a feature. The clean baseline feature list excludes `Clinical_Est_Weeks`; its earlier controlled ablation showed small changes, not a clinically meaningful model improvement.

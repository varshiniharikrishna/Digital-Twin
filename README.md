# Dynamic Digital Twin - Orthopaedic Rehabilitation

Flask web application for data preparation, model training and evaluation,
patient-specific twin views, recovery timelines, recalibration, and pathway
mining. `app.py` is a separate Streamlit entry point for the research dashboard.

## Files

| File | Purpose |
|---|---|
| `config.py` | Data source switch (MySQL/CSV), feature lists, thresholds |
| `db.py` | Loads the application tables from MySQL or CSV |
| `data_prep.py` | Merges tables, builds features + targets, chronological train/test split |
| `modeling.py` | Trains recovery and duration ensembles; uses XGBoost when installed |
| `similarity.py` | Finds top-K similar historical patients, restricted to a given pool |
| `simulation.py` | Builds predicted weekly trajectories, Phase 6 recalibration logic |
| `evaluation.py` | Plotly figures for Phase 3 |
| `explainability.py` | Optional local SHAP explanation for the recovery prediction |
| `research_evaluation.py` | Chronological holdout, training-only CV, recalibration prediction-error comparison, and threshold-trigger sensitivity summary |
| `server.py` | Flask web application entry point |
| `app.py` | Streamlit research dashboard entry point |

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 - get your data into MySQL

For a new database only, run `import_to_mysql.py` to load the generated CSVs
into `DigitalTwinDB`. Set `DT_MYSQL_PASSWORD` and `DT_INITIAL_ADMIN_PASSWORD`
first. This importer drops and recreates the project tables, so do not rerun
it against an existing database that contains data you need.

## Step 2 - point the app at MySQL

The app defaults to MySQL (`DATA_SOURCE = "mysql"` in `config.py`).
Set the database credentials in the environment (do not commit them):

```bash
export DT_MYSQL_USER=root
export DT_MYSQL_PASSWORD='your database password'
export DT_MYSQL_HOST=localhost
export DT_MYSQL_PORT=3306
export DT_MYSQL_DB=DigitalTwinDB
```

To use the generated CSVs without MySQL, set `DT_DATA_SOURCE=csv`.
To use a different directory of CSV tables, also set `DT_CSV_DIR=/absolute/path/to/csvs`.

## Step 3 - run it

```bash
python server.py
```

Run `streamlit run app.py` only when you want the separate research dashboard.

## Authentication and access scope

Every `/api/` endpoint now requires a bearer session, except login. Sessions expire after eight hours; login attempts are rate-limited per client address. Patient reads are limited to the patient, their assigned doctor, or an administrator. Pathway mining is unavailable to patients, and doctor pathway summaries are calculated from that doctor's assigned patients only (minimum five observed outcomes). Tokens are accepted only in the Authorization header, never in URLs.

Set `DT_MYSQL_PASSWORD` before connecting. When importing accounts, set `DT_INITIAL_ADMIN_PASSWORD` to a unique value of at least 14 characters. Doctor and patient accounts receive unique random passwords; an administrator issues a one-time password through the Admin Portal. Newly registered accounts also receive a random one-time password displayed to the authorized registrar. Use **Change password** after signing in. Run the built-in server on localhost; production deployment still needs HTTPS and a production WSGI server. These controls are implementation safeguards, not a claim of GDPR/HIPAA compliance or regulatory clearance.

## Twin and pathway method

The modeled twin state is the patient's feature vector `x_t` containing the listed baseline and latest clinical measurements. A weekly observation replaces the corresponding state values, `x_(t+1)[j] = observed[j]`, while unobserved components retain their previous values. Model outputs are predicted recovery percentage and duration; the plotted weekly curve is a simulation derived from those endpoints.

Similar-patient retrieval uses a Gower-style distance over six numeric features (Age, BMI, Pain, ROM, quadriceps strength, balance) and five categorical features (gender, injury, severity, diabetes, smoking). Numeric differences are divided by each feature's range in the training pool; when a numeric range is zero, equality gives distance 0 and inequality distance 1. Categorical mismatch is 0/1. The final distance is 0.60 times the mean available numeric distance plus 0.40 times the mean available categorical mismatch; if one component has no available comparisons, the remaining component's weight is renormalized. Similarity is `1 - distance`; missing pairs are omitted.

For predicted value `p` and observation `o`, signed deviation is `d=100(o-p)/max(p, 1e-5)`. Recalibration flags a pain increase when `d > τ`, and ROM or strength decrease when `d < -τ`; `τ=15%` by default. The evaluation script reports trigger counts for 5%, 10%, 15%, 20%, and 25%. Clinical sensitivity and specificity are not calculated because the dataset has no adjudicated deterioration labels.

Pathway ranking is an exploratory retrieval score, not a treatment-benefit estimate. The code filters candidates by injury type and, when available, body region, then calculates `0.70 × mean top-10 Gower similarity + 0.30 × cohort compatibility`. Cohort compatibility is the mean of exact severity and age-band matches (each 0 or 1). The score does not use pathway success or duration: the schema does not link a patient's observed treatment assignment to a `Recovery_Pathway` row. Cohort summaries use at least five outcome records and are explicitly descriptive; they do not establish pathway efficacy or causation.

The state is `x_t=(Pain, ROM, quadriceps strength, hamstring strength, balance, walking speed, age, BMI, comorbidities, injury type, severity, and other configured baseline covariates)`. A weekly observation `o_(t+1)` replaces the corresponding available state component; unobserved components persist: `x_(t+1,j)=o_(t+1,j)` when observed and `x_(t+1,j)=x_(t,j)` otherwise. Deterministic derived features `φ(x_(t+1))` are recomputed before the fitted model produces `ŷ_recovery=f_pct(φ(x_(t+1)))` and `ŷ_weeks=f_weeks(φ(x_(t+1)))`. The plotted weekly trajectory is a generated curve from endpoint predictions, not a validated physiological simulator.

## Validation and evidence boundaries

Run `DT_DATA_SOURCE=csv python3 research_evaluation.py` from the project root. It writes `output/research_evaluation.json` without altering the input CSV files. The report includes an enrollment-date holdout, outcome-maturity-aware expanding-window temporal CV, baseline-mean comparators, patient-level bootstrap intervals, subgroup errors, landmark comparisons, and threshold flag counts. On the Desktop CSV snapshot evaluated on 2026-10-25, recovery-proxy holdout R² was 0.848 (n=242; 95% bootstrap interval 0.807–0.876; MAE 1.99 points); pooled temporal CV R² was 0.824 across three usable folds (n=196 validation predictions; two earlier folds were skipped for insufficient matured training labels). Duration holdout R² was 0.752 (MAE 5.10 weeks). Provenance of the configured CSV snapshot is unverified. These metrics are not clinical evidence.

The recovery target is either a supplied `Final_Recovery_Pct` or, when absent, a 50:50 score from final strength and final ROM using a fixed 140° ROM scale. This is a software proxy, not a validated outcome scale; a single 140° denominator is not appropriate for every joint or injury type. Reported multi-injury R² is descriptive of this proxy only. Use a validated region-specific outcome before making clinical performance claims.

Subgroup R² also needs context: on this snapshot it is low for Mild and Severe groups (0.013 and 0.095) while their target standard deviations are only about 2.55 and 2.99 points and MAE is about 1.94 and 2.20 points. R² is sensitive to within-group target range; use MAE/RMSE, uncertainty, sample size, calibration, and subgroup validation together rather than treating subgroup R² alone as a fairness verdict.

The metric audit fixed several sources of optimism/error: (1) baseline training had been using week-1 strength values collected after initial assessment; those are now excluded; (2) missing duration labels had been replaced by `Clinical_Est_Weeks`, a predictor, and missing recovery labels could fall back to baseline measurements; missing outcomes now remain missing; (3) shuffled K-fold CV mixed earlier/later dates; it is now expanding-window and requires training outcomes to have matured before each validation date; (4) enrollment date overrides inconsistent reported year values. The schema has no recorded outcome-availability timestamp, so training eligibility is conservatively estimated from enrollment date plus observed recovery duration; this estimate is not a true recorded timestamp. In the evaluated CSV snapshot, 146 of 558 train-year patients had estimated outcome completion after the 2025 holdout began and were excluded from training, and three records reported 2025 but had 2026 enrollment dates and were also excluded.

The landmark comparison reports both current app logic (the same baseline-trained model given week-4/week-8 observations) and a separate model retrained using training patients' landmark observations. On this snapshot, same-model week 4 had R² 0.847 (MAE 1.92), and week 8 had R² 0.834 (MAE 1.96; RMSE worsened), versus static baseline R² 0.848 (MAE 1.99). These are retrospective prediction errors, not evidence recalibration improves health outcomes. The retrained landmark alternative had higher proxy R², but is not the current app implementation. The three claimed pillars are not formally defined and treatment exposure is not linked to outcomes, so no valid pillar ablation or causal study can be reported.

The Evaluation tab includes a local SHAP explanation. Install the optional dependency using `python3 -m pip install shap`. SHAP describes this model's behavior relative to a training background; it is not a causal explanation or evidence of clinical validity. Global ExtraTrees feature importance is a separate model-specific summary.

## Finding realistic rehabilitation data

The included `output/*.csv` files were generated synthetically. Their metrics describe a software demonstration only and must not be presented as clinical validation. For this project, the strongest fit is a retrospective, de-identified longitudinal dataset from a physiotherapy or orthopaedic clinic, with baseline injury/severity, treatment exposure, and repeated pain, ROM, strength, mobility, and patient-reported outcomes. Ask a university hospital's orthopaedics/physiotherapy department or an investigator running an ACL/knee-rehabilitation registry; access will require institutional approval, a data-use agreement, and an appropriate ethics/IRB determination. Do not put identifiable patient data in this repository.

Public sources can help with narrower parts of the system:

* [Osteoarthritis Initiative (OAI)](https://nda.nih.gov/oai) contains longitudinal knee OA clinical measures and patient-reported outcomes, including KOOS/WOMAC, across thousands of participants. Access requires an NDA account and agreement to its data terms. It is useful for longitudinal knee symptom/function modeling, but it is not a post-operative rehabilitation pathway dataset and does not directly validate ACL or multi-injury recovery recommendations.
* [KneE-PAD on Zenodo](https://zenodo.org/records/12112951) is an open dataset of exercise sessions from 31 participants with knee pathologies, including wearable sensor signals. It can inform a future wearable exercise-monitoring layer, but it does not provide the longitudinal clinical outcomes needed to validate the recovery-percent model or pathway ranking.

Start with OAI only if you narrow the prototype to knee osteoarthritis and map available validated outcome measures. For claims about rehabilitation pathways or ACL recovery, prioritize a clinical partner dataset; no one public source located here matches the app's full patient-level weekly schema.

## Important documented assumptions

1. **`Final_Recovery_Pct` target.** The generator's exported tables
   never write out a single "final recovery %" number — `Final_Outcome`
   only stores `Actual_ROM_deg`, `Actual_Strength_pct`,
   `Actual_Pain_Score`, and a categorical `Outcome_Label`. Since Phase 2
   asks for a regression target, `data_prep.py` builds it as:
   `0.5 * Actual_Strength_pct + 0.5 * (Actual_ROM_deg / 140 * 100)`,
   clipped to [0, 100]. This is flagged in the code and is worth
   describing explicitly as a derived/proxy label in your paper's
   methodology section.

2. **Baseline muscle strength features.** `Initial_Assessment` stores a
   single combined `Muscle_Strength_pct`; it does not provide an anatomical
   quad/hamstring split. Baseline modeling now uses that assessment value
   and a documented 0.85 secondary-strength proxy. It deliberately does
   not use `Weekly_Progress` week-1 strength because that is post-baseline
   information. The split remains a modeling approximation, not a measured
   quad/hamstring assessment.

3. **Predicted weekly trajectory (Phase 5).** The trained digital twin
   predicts a single end-of-care recovery % and total weeks — it is
   not a trained week-by-week temporal model. `simulation.py` forward-
   simulates a predicted curve from the patient's initial values to the
   model's predicted endpoint using the same asymptotic
   (diminishing-returns) shape used by the current synthetic generator.
   The curve labeled "actual" comes from the configured `Weekly_Progress`
   table; in the checked-in CSVs those records are synthetic too.

4. **Recalibration (Phase 6)** is intentionally rule-based rather than
   a second trained model: entered values become the patient's new
   "current state," similarity search re-runs against the training
   pool only, and the explanation text is generated from which metrics
   deviated and by how much. This keeps the "why" fully traceable for
   a demo/paper, rather than hidden inside a black-box re-predictor.

## Notes on model choice

Recovery percentage prediction now uses a weighted ensemble of XGBoost,
ExtraTrees, and Random Forest when XGBoost is installed. Without XGBoost it
falls back to HistGradientBoosting, ExtraTrees, and Random Forest. The
reported metrics must still be re-evaluated on suitable real data; the code
does not guarantee a target R² on synthetic records or clinical cohorts.

## Ethics, privacy, regulatory status, related work, and deployment

This repository is a local research prototype. It does not establish GDPR/HIPAA compliance, informed consent, clinical safety, or regulatory authorization. Before processing identifiable health data, the responsible institution must establish the appropriate lawful basis/authorization and consent or other permitted basis; obtain ethics/IRB and data-use decisions; minimize data; define retention/deletion; implement access controls, encryption, audit and incident response; and execute any required processor/business-associate agreements. Assess missingness, calibration, and error across relevant demographic and clinical subgroups. This project has no validated bias assessment and is not production-hardened.

Regulatory classification depends on intended use, user, claims, and jurisdiction. Because this app has patient-facing and patient-specific predictive/recommendation features, do not assume it is exempt or non-device software. Obtain specialist review before any deployment. Primary references: [EU GDPR Article 9](https://eur-lex.europa.eu/legal-content/EN/TXT/?qid=1755832889478&uri=CELEX%3A32016R0679), [HHS HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html), [FDA Clinical Decision Support Software guidance (January 2026)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software), and [CDSCO Medical Device Software guidance](https://www.cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Guidance-document-on-Medical-Device-Software-under-MDR-2017.pdf). These are reference materials, not a compliance or classification conclusion.

Related methods include SHAP feature attribution ([Lundberg and Lee, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)) and privacy-preserving multi-site model training such as federated learning ([Rieke et al., npj Digital Medicine 2020](https://www.nature.com/articles/s41746-020-00323-1); [Dayan et al., Nature Medicine 2021](https://www.nature.com/articles/s41591-021-01506-3)). Neither federated learning nor wearable ingestion is implemented here. A future wearable layer could evaluate signal-quality and exercise-form summaries from datasets such as [KneE-PAD](https://zenodo.org/records/12112951), but it cannot validate longitudinal recovery outcomes by itself.

**Provisional TRL self-assessment: 3 (analytical proof of concept).** This is a tentative label for a working software prototype using synthetic demonstration data, not an external assessment or evidence of clinical validation. Reassess against the applicable TRL framework before making a formal claim.

**Commercialization/deployment gates:** define narrow intended use and population; secure a clinical data partner, ethics determination, and data permissions; validate/calibrate on a locked real-world cohort with subgroup and external-site evaluation; complete clinician/patient workflow and usability assessment; obtain jurisdiction-specific regulatory and privacy advice; then deploy with production security, monitoring, versioning, incident response, and post-deployment performance review. No clinical deployment or commercial readiness is claimed.

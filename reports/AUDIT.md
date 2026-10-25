# Technical audit — Phase 0 (read-only audit)

Audit scope: the repository sources, tests, schemas, documentation, and current data-loading/modeling paths were inspected before fixes. The only generated measurement is the model run recorded below; no project code was changed to produce this audit. `output/*.csv` contain the current CSV input snapshot. The four locally modified data files are user data and are intentionally not included in source changes.

**Audit stamp**

- Code commit inspected: `67bd4ce8c37f7bdf7c12b4bf1c5a09854b1ee5a0`.
- Data origin reported by the loader: `synthetic bundled CSV`.
- Combined SHA-256 of the current `Patient.csv`, `Injury.csv`, `Initial_Assessment.csv`, `Final_Outcome.csv`, and `Weekly_Progress.csv` files (hash each file, then hash its filename and digest in that order): `6ff765e35139a29b633be445d3f4b2a23822abe6525f8f83778d5e7f1e3d9f83`.
- Python package versions: Python 3.12; pandas 3.0.0; NumPy 2.4.2; scikit-learn 1.8.0; XGBoost 1.7.6; Streamlit 1.57.0. LightGBM, CatBoost, Optuna, SHAP and system pytest were not installed at audit time. The repository lists several of these but does not pin their versions.
- Feature list used by the production fit: `config.ALL_FEATURES` (baseline demographic, injury, assessment and engineered features; includes `Clinical_Est_Weeks`). See `config.py:49-64`.

## Current measured model numbers

Produced by code run during this audit, with `DT_DATA_SOURCE=csv`, seed `config.RANDOM_STATE`, the existing chronological split and current production `modeling.train_models`. Bootstrap intervals below are paired patient resamples (2,000 replicates). CV is the existing expanding-window, outcome-maturity-aware temporal procedure; only its three usable folds contribute. The model selected by current production code in this environment is HistGradientBoostingRegressor, because the existing XGBoost compatibility check falls back under the installed sklearn/XGBoost combination. The target is the generated composite strength-ROM proxy, not a clinical PROM.

| Target / partition | n | R² | R² 95% bootstrap CI | MAE | MAE 95% CI | RMSE | RMSE 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recovery proxy / temporal CV pooled OOF | 196 | 0.8313 | 0.7833–0.8693 | 2.2154 | 1.9767–2.4720 | 2.8127 | 2.4880–3.1448 |
| Recovery proxy / chronological holdout | 242 | 0.8509 | 0.8110–0.8807 | 2.0029 | 1.8171–2.2073 | 2.5717 | 2.2953–2.8603 |
| Recovery weeks / temporal CV pooled OOF | 196 | 0.5325 | 0.4524–0.6089 | 5.5316 | 4.7092–6.3937 | 8.2589 | 7.0070–9.4568 |
| Recovery weeks / chronological holdout | 242 | 0.7555 | 0.6912–0.8048 | 5.0662 | 4.5510–5.6038 | 6.6605 | 5.9471–7.3831 |

Two early temporal windows were skipped because outcome-matured training counts were too small. These scores are an audit of the current synthetic snapshot and current implementation, not a claim that any later benchmark candidate won.

## Findings by checklist item

| Item | Status | Evidence and finding |
|---|---|---|
| T1 Security | PARTIAL | `config.py:29-35` defaults the database user to `root`; password is empty unless supplied. SQL values are parameterized in `db.py:116-126`, but table/column identifiers are interpolated in generated SQL (`db.py:159-163`, `import_to_mysql.py`). `app.py:436-441` interpolates database-derived values into unsafe HTML without escaping. Broad exception handling/printing remains in `db.py:49-63,124-141`, `server.py`, and `app.py`; no dedicated `docs/SECURITY.md` history-purge guidance exists. |
| T2 Leakage and preparation | PARTIAL | `data_prep.py:132-142` correctly excludes weekly rows from baseline features; `data_prep.py:170-179` does not fill missing recovery weeks from `Clinical_Est_Weeks`. However the feature remains in `config.py:49-54`; `data_prep.py:138` uses suffix merging rather than an explicit source mapping/assertion; proxy uses a 140-degree denominator (`data_prep.py:154`) while the trajectory uses 145 (`simulation.py:158`). No measured with/without-`Clinical_Est_Weeks` ablation exists. Split errors raise rather than copy train (`data_prep.py:252-255`). |
| T3 Model and benchmark | PARTIAL | Benchmark candidates include XGBoost (`model_benchmark.py:29-48`), but the live model remains a hard-coded voting ensemble and feature importance assumes ExtraTrees (`modeling.py:20-54,79-95,130-143`); `app.py:128-145` retrains at startup. Current production metrics are above. Benchmark CV is repeated stratification rather than grouped-by-patient plus time-aware folds (`model_benchmark.py:71-93`), benchmarks recovery percentage only, default tuning is two trials, and no model artifact is persisted. Prediction intervals are empirical residual bands, not split conformal (`model_benchmark.py:103-109`). |
| T4 Similarity | PARTIAL | Training-fitted ranges, weights and threshold, pool-independent scoring and low-cohort metadata exist (`similarity.py:27-87,100-119,134-152`). The app fits in memory and does not load/persist the model (`app.py:145`). Existing comparison code is present, but its committed evaluation artifacts are stale relative to the current CSV snapshot; sensitivity and current results need a fresh stamped run. |
| T5 Pathways | PARTIAL | `pathway_mining.py:131-172` ranks templates using current-neighbor similarity and compatibility, and correctly disclaims treatment effects. The visible Recommendations page still sorts the global pathway table by `Success_Rate` (`app.py:830-834`); Pathway Ranking calls `mine_pathways_from_tables(tables)` on all rows (`app.py:893-899`). Mining uses a fixed 80 success cut (`pathway_mining.py:75-77`). Treatment response constants remain in `simulation.py:85-103,345-...`; UI warns they are assumptions but they still change displayed trajectories. |
| T6 Twin | PARTIAL | Fixed after initial audit: `simulation.build_predicted_trajectory` now advances `DigitalTwin` from `twin_state.py`; dashboard, treatment simulation, and server PDF call sites use that wrapper. Region-aware UI thresholds, missing-observation selectors, recorded-week bounded sliders, predicted balance, and a validated in-session baseline intake were added. Forecast calibration and some simulated protocol assumptions remain unvalidated. |
| T7 Deviation | PARTIAL | Fixed after initial audit: `deviation_analysis.py:14-53` fits GroupKFold patient-held-out outcome models, then computes metric-specific residual scales by week bucket. Threshold sweep now has patient-cluster bootstrap CIs (`deviation_analysis.py:104-132`). Detection delay and clinical threshold selection remain unestimable without adjudicated deterioration labels. |
| T8 Adaptive evaluation | PARTIAL | Replay iterates one recorded week at a time and does not use future check-in values to construct that week's updated state (`eval_adaptive_loop.py:18-34`). It does not test this invariant, predict discharge-week errors, or evaluate trajectory RMSE. Paired inference pools patient-week rows (`eval_adaptive_loop.py:35-45`) rather than patient-cluster resampling/paired patient-level aggregation. |
| T9 XAI | PARTIAL | Explanation and additivity helper exist (`explainability.py:7-43`), but the default is a generic permutation SHAP explainer, only the recovery target is explained, no original-feature aggregation or bootstrap stability is implemented, and the dashboard has a blanket `except Exception` (`app.py:908-914`). SHAP is unavailable in the audit environment. |
| T10 Engineering | PARTIAL | Tests and `Makefile`/`run_all.py` exist; the app contains pages for several analyses. `app.py` remains a single large script; report pages do not validate data/model stamps; no GitHub Actions + Ruff workflow exists. Dependencies are unpinned (`requirements.txt`), and the full runner has not completed successfully from scratch on the current checkout. |

## Phase 0 disposition

The items marked PARTIAL are the fixes to prioritize next. Items already implemented (training-only similarity scaling, no Week-1 baseline leakage, missing-target handling, chronological split, API authorization already present) should receive regression tests rather than duplicate implementations. No project source or user-edited CSV was altered to produce this Phase 0 audit.

## Post-fix verification snapshot

This run was made after the first technical fixes. It uses the current CSV snapshot, the same chronological split, the production XGBoost estimator, and 1,000 paired patient bootstrap resamples. Compared with Phase 0, the model changed from HistGradientBoosting to XGBoost, `Clinical_Est_Weeks` was removed from the feature list, and the derived proxy now uses configured region-specific ROM maxima. Therefore the score differences below are **not an isolated causal estimate of removing that feature**. The data remain synthetic.

Stamp for the measured run: source commit `67bd4ce8c37f7bdf7c12b4bf1c5a09854b1ee5a`; data-frame SHA-256 `1e740301c94022d807be52f62de7ba50a11673ee65ed8dca5fd14bd438f092d1`; Python 3.12.6, NumPy 2.4.2, pandas 3.0.0, scikit-learn 1.8.0, XGBoost 1.7.6; 22 production features, excluding `Clinical_Est_Weeks`.

| Target / partition | n | R² (95% CI) | MAE (95% CI) | RMSE (95% CI) | WAPE |
|---|---:|---:|---:|---:|---:|
| Recovery proxy / temporal CV pooled OOF | 190 | 0.7900 (0.7371–0.8319) | 2.9108 (2.5729–3.2691) | 3.8678 (3.4341–4.3226) | 3.2431% |
| Recovery proxy / chronological holdout | 224 | 0.9228 (0.8974–0.9429) | 1.6657 (1.4739–1.8768) | 2.2659 (1.9477–2.5730) | 1.8700% |
| Recovery weeks / temporal CV pooled OOF | 196 | 0.4630 (0.3662–0.5481) | 6.0107 (5.1361–6.9115) | 8.8518 (7.5164–10.0425) | 26.5698% |
| Recovery weeks / chronological holdout | 242 | 0.7546 (0.6846–0.8053) | 5.0480 (4.5248–5.6032) | 6.6723 (5.9545–7.4287) | 17.8392% |

This was the first XGBoost-only post-fix run. It is superseded by the grouped benchmark below. The recovery-weeks temporal CV R² was below 0.65. No clinical conclusion follows from these synthetic results.

Controlled `Clinical_Est_Weeks` feature ablation (same XGBoost model, same training window, same temporal folds and same holdout): adding the feature changed recovery-proxy temporal CV R² from **0.79003** to **0.80224** (Δ +0.01221) and holdout R² from **0.92283** to **0.92570** (Δ +0.00287). Bootstrap intervals overlap. The measured gain is small on this synthetic dataset; the feature remains excluded from production so the estimated-duration proxy cannot act as a shortcut. This ablation does not prove external generalization.

## Final benchmark-selected verification

The final `make all` run completed after the grouped benchmark. The current model artifact and app both select `VotingRegressor` using repeated patient-grouped training CV; the ensemble includes XGBoost 1.7.6, RandomForest, ExtraTrees, and HistGradientBoosting. The separately reported grouped benchmark selected it with mean R² **0.8964 ± 0.0174** across 15 grouped folds; grouped OOF bootstrap 95% CI **0.8757–0.9150**. Its single chronological holdout score was R² **0.9302**, MAE **1.5369**, RMSE **2.1545**, WAPE **1.7255%**, n=224; empirical 95% repeated-CV residual interval half-width 5.1427 points had holdout coverage 96.43% and no formal coverage guarantee.

The temporal procedure, which withholds later enrollment dates and also respects outcome availability, is a different evaluation and is the primary before/after row below. The before/after data sample sizes differ because region-specific ROM normalization now leaves unresolvable regions unlabeled. Both snapshots are synthetic and model/target preparation changed, so score differences are descriptive rather than a causal estimate of a code fix.

| Target / partition | Before: R² (95% CI), MAE, RMSE | Final: R² (95% CI), MAE, RMSE, WAPE | n final |
|---|---|---|---:|
| Recovery proxy / temporal CV | 0.8313 (0.7833–0.8693), 2.2154, 2.8127 | 0.8247 (0.7667–0.8642), 2.5407, 3.5346, 2.8307% | 190 |
| Recovery proxy / chronological holdout | 0.8509 (0.8110–0.8807), 2.0029, 2.5717 | 0.9303 (0.9071–0.9487), 1.5351, 2.1528, 1.7234% | 224 |
| Recovery weeks / temporal CV | 0.5325 (0.4524–0.6089), 5.5316, 8.2589 | 0.5111 (0.4262–0.5894), 5.8061, 8.4459, 25.6650% | 196 |
| Recovery weeks / chronological holdout | 0.7555 (0.6912–0.8048), 5.0662, 6.6605 | 0.7476 (0.6796–0.7978), 5.2246, 6.7673, 18.4630% | 242 |

The recovery-duration temporal CV score does **not** reach the requested R² ≥ 0.65. The controlled `Clinical_Est_Weeks` comparison for the final ensemble changed recovery-proxy R² by +0.00504 temporal CV and −0.00240 holdout when included; it stays excluded from production. Adaptive replay on 224 patients found worse patient-level mean absolute forecast error for recalibration than static: +0.2072 points (cluster bootstrap 95% CI 0.1187–0.3057) on-trigger, and +0.4206 (0.2685–0.5681) every-week. This is forecast-error evidence on synthetic data, not clinical outcome benefit.

The final experiment summary has no `not completed` statuses. Pathway effect evaluation remains not evaluable because the schema has no patient-level pathway assignment linked to outcomes; XAI remains unavailable because SHAP is not installed in this runtime; real-data evaluation is not run because no governed clinical dataset was supplied. The report stamp is commit `67bd4ce8c37f7bdf7c12b4bf1c5a09854b1ee5a`, current data-frame SHA-256 `1e740301c94022d807be52f62de7ba50a11673ee65ed8dca5fd14bd438f092d1`, Python 3.12.6, NumPy 2.4.2, pandas 3.0.0, scikit-learn 1.8.0, XGBoost 1.7.6, and 22 production features excluding `Clinical_Est_Weeks`.

Verified changes include XGBoost-compatible pipelines, benchmark-hash-checked persisted selected models, patient-grouped benchmark selection, patient-level bootstrap summaries, WAPE, empirical CV intervals in the UI, baseline schema checks, region-based ROM scale, training-pool-only pathway mining and patient-specific ranking, low-sample pathway labels, HTML/PDF escaping, blank DB credential defaults, SQL identifier validation and safe parameter binding, security and environment examples, pinned requested ML dependencies, region-aware alerts, a no-write intake prototype, patient-held-out per-week-bucket deviation sigma, threshold cluster-bootstrap intervals, and a shared twin trajectory wrapper. Regression suite: **31 passed**; Streamlit AppTest has no exceptions; full `run_all.py` completed before the latest trajectory change. Still partial: comparative time-aware CV is only for the selected model rather than every candidate; the UI artifact uses one recovery-percentage CV-selected model for both outcomes; no feature-group ablation for the three invention pillars; broad exception replacement remains incomplete in parts of server.py and app.py; held-out trajectory forecasts need re-evaluation against the shared engine; full clinical classification sensitivity remains unavailable; SHAP explanation/stability is unavailable; application page modularization and report stale-stamp warnings are not complete. Clinical validation and clinical sensitivity/specificity need governed patient records and adjudicated labels.

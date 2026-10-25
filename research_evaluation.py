"""Leakage-aware evaluation for the Digital Twin demo.

Run from the project root: DT_DATA_SOURCE=csv python3 research_evaluation.py
Optionally set DT_CSV_DIR to a read-only input folder. Only the JSON report is
written; source CSVs are never modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import config
import db
import modeling
import simulation
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame


def _metrics(y, pred):
    result = modeling.evaluate(y, pred)
    return {k: (round(float(v), 5) if v is not None and np.isfinite(v) else None)
            for k, v in result.items()}


def _bootstrap_intervals(y, pred, seed=42, samples=1000):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if len(y) < 2:
        return {"MAE_95pct_CI": None, "R2_95pct_CI": None}
    rng = np.random.default_rng(seed)
    maes, r2s = [], []
    for _ in range(samples):
        take = rng.integers(0, len(y), size=len(y))
        by, bp = y[take], pred[take]
        maes.append(mean_absolute_error(by, bp))
        if np.ptp(by) > 0:
            r2s.append(r2_score(by, bp))
    return {
        "resamples": samples,
        "MAE_95pct_CI": [round(float(x), 5) for x in np.quantile(maes, [0.025, 0.975])],
        "R2_95pct_CI": [round(float(x), 5) for x in np.quantile(r2s, [0.025, 0.975])] if r2s else None,
        "method": "paired patient-level percentile bootstrap; descriptive only",
    }


def _temporal_cv(X, y, dates, outcome_available, target_type, max_splits=5):
    """Expanding-window CV; train labels must be available before each fold."""
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna() & np.isfinite(y)
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    dates = pd.to_datetime(dates.loc[valid], errors="coerce").reset_index(drop=True)
    outcome_available = pd.to_datetime(outcome_available.loc[valid], errors="coerce").reset_index(drop=True)
    if dates.isna().any():
        raise ValueError("Temporal cross-validation requires valid enrollment dates for labelled rows.")
    unique_dates = np.sort(dates.unique())
    split_count = min(max_splits, len(unique_dates) - 1)
    if split_count < 2:
        return {"status": "Insufficient distinct dates for expanding-window CV.", "n": int(len(y))}

    # First date block is initial training history; each later block is a
    # future validation window. Patients enrolled on one date stay together.
    blocks = [block for block in np.array_split(unique_dates, split_count + 1) if len(block)]
    y_all, pred_all, baseline_all, fold_results, skipped_folds = [], [], [], [], []
    for fold_num in range(1, len(blocks)):
        valid_dates = blocks[fold_num]
        cutoff = pd.Timestamp(valid_dates[0])
        train_mask = (dates < cutoff) & outcome_available.notna() & (outcome_available < cutoff)
        valid_mask = dates.isin(valid_dates)
        if train_mask.sum() < 30 or valid_mask.sum() == 0:
            skipped_folds.append({"fold": fold_num, "training_n": int(train_mask.sum()),
                                 "validation_n": int(valid_mask.sum()),
                                 "reason": "Insufficient outcome-matured training observations."})
            continue
        estimator = modeling.build_pipeline(target_type)
        estimator.fit(X.loc[train_mask], y.loc[train_mask])
        pred = estimator.predict(X.loc[valid_mask])
        baseline_pred = np.full(int(valid_mask.sum()), float(y.loc[train_mask].mean()))
        truth = y.loc[valid_mask].to_numpy()
        y_all.extend(truth)
        pred_all.extend(pred)
        baseline_all.extend(baseline_pred)
        fold_results.append({
            "fold": fold_num,
            "train_n": int(train_mask.sum()),
            "validation_n": int(valid_mask.sum()),
            "train_end": str(dates.loc[train_mask].max().date()),
            "latest_training_outcome_available": str(outcome_available.loc[train_mask].max().date()),
            "validation_start": str(dates.loc[valid_mask].min().date()),
            "validation_end": str(dates.loc[valid_mask].max().date()),
            "metrics": _metrics(truth, pred),
        })
    return {
        "method": "expanding-window temporal CV, whole enrollment dates held together",
        "folds": len(fold_results),
        "skipped_folds": skipped_folds,
        "n_validation_total": len(y_all),
        "metrics_pooled": _metrics(y_all, pred_all) if y_all else None,
        "training_mean_baseline_pooled": _metrics(y_all, baseline_all) if y_all else None,
        "fold_metrics": fold_results,
    }


def _landmark_rows(data, weekly, cutoff):
    """State as known at a week cutoff; never read later check-ins."""
    rows = data.copy()
    week = pd.to_numeric(weekly.get("Week", pd.Series(dtype=float)), errors="coerce")
    observations = weekly.loc[(week <= cutoff) & week.notna()].copy()
    observations["Week"] = pd.to_numeric(observations["Week"], errors="coerce")
    observations["_pid"] = observations["Patient_ID"].astype(str)
    observations = observations.sort_values(["_pid", "Week"]).drop_duplicates("_pid", keep="last")
    observation_map = observations.set_index("_pid")
    fields = {
        "Pain": "Pain_Score",
        "ROM_deg": "Range_of_Motion_deg",
        "Quadriceps_Strength_pct": "Quadriceps_Strength_pct",
    }
    state_week = {}
    for idx, row in rows.iterrows():
        key = str(row["Patient_ID"])
        if key not in observation_map.index:
            continue
        obs = observation_map.loc[key]
        state_week[key] = int(obs["Week"])
        for source, feature in fields.items():
            if source in obs.index and pd.notna(obs[source]):
                rows.at[idx, feature] = obs[source]
    rows["_Landmark_Week"] = rows["Patient_ID"].astype(str).map(state_week)
    return rows


def _subgroup_metrics(test, y, pred, column, minimum_n=10):
    if column not in test:
        return {}
    results = {}
    labels = test[column].fillna("Unknown").astype(str)
    for group in sorted(labels.unique()):
        mask = labels.eq(group).to_numpy()
        n = int(mask.sum())
        if n < minimum_n:
            continue
        group_y, group_pred = np.asarray(y)[mask], np.asarray(pred)[mask]
        results[group] = {
            "n": n,
            "target_mean": round(float(np.mean(group_y)), 5),
            "target_sd": round(float(np.std(group_y, ddof=1)), 5) if n > 1 else None,
            "mean_prediction_error": round(float(np.mean(group_pred - group_y)), 5),
            **_metrics(group_y, group_pred),
        }
    return results


def run(tables=None, output_path="output/research_evaluation.json", emit=True):
    tables = db.load_all_tables() if tables is None else tables
    data = build_modelling_dataset(tables)
    duplicate_patient_rows = int(data["Patient_ID"].duplicated().sum())
    if duplicate_patient_rows:
        raise ValueError(
            f"Evaluation stopped: {duplicate_patient_rows} repeated patient rows were found. "
            "Select one prespecified index encounter per patient or use grouped episode-level validation before reporting metrics."
        )
    train, test = chronological_split(data)
    if train.empty or test.empty:
        raise RuntimeError("Temporal split is empty; add enrollment dates/years and set valid train/test years.")

    x_train = get_feature_frame(train)
    x_test = get_feature_frame(test)
    pct_model, weeks_model = modeling.train_models(
        x_train, train[config.TARGET_RECOVERY_PCT], train[config.TARGET_RECOVERY_WEEKS]
    )
    pct_pred, weeks_pred = modeling.predict(pct_model, weeks_model, x_test)

    pct_mask = pd.to_numeric(test[config.TARGET_RECOVERY_PCT], errors="coerce").notna().to_numpy()
    weeks_mask = pd.to_numeric(test[config.TARGET_RECOVERY_WEEKS], errors="coerce").notna().to_numpy()
    y_pct = pd.to_numeric(test.loc[pct_mask, config.TARGET_RECOVERY_PCT], errors="coerce").to_numpy()
    y_weeks = pd.to_numeric(test.loc[weeks_mask, config.TARGET_RECOVERY_WEEKS], errors="coerce").to_numpy()
    pred_pct_labeled, pred_weeks_labeled = pct_pred[pct_mask], weeks_pred[weeks_mask]

    pct_train_mask = pd.to_numeric(train[config.TARGET_RECOVERY_PCT], errors="coerce").notna()
    weeks_train_mask = pd.to_numeric(train[config.TARGET_RECOVERY_WEEKS], errors="coerce").notna()
    pct_mean_baseline = np.full(len(y_pct), float(train.loc[pct_train_mask, config.TARGET_RECOVERY_PCT].mean()))
    weeks_mean_baseline = np.full(len(y_weeks), float(train.loc[weeks_train_mask, config.TARGET_RECOVERY_WEEKS].mean()))
    booster_name = modeling.get_recovery_boost_model_name()
    booster_params = (
        {"max_iter": 400, "learning_rate": 0.03, "max_depth": 5, "l2_regularization": 2.0}
        if booster_name == "HistGradientBoostingRegressor"
        else {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
              "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0}
    )

    date_series = pd.to_datetime(train["Enrollment_Date"], errors="coerce")
    availability = pd.to_datetime(train["Estimated_Outcome_Availability_Date"], errors="coerce")
    pct_cv = _temporal_cv(x_train, train[config.TARGET_RECOVERY_PCT], date_series, availability, "pct")
    weeks_cv = _temporal_cv(x_train, train[config.TARGET_RECOVERY_WEEKS], date_series, availability, "weeks")

    enrollment_date = pd.to_datetime(data.get("Enrollment_Date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    mismatch = data.get("Enrollment_Year_Mismatch", pd.Series(False, index=data.index)).fillna(False)
    used_ids = set(train["Patient_ID"].astype(str)) | set(test["Patient_ID"].astype(str))
    excluded = data[~data["Patient_ID"].astype(str).isin(used_ids)]
    candidate_train = data[pd.to_numeric(data["Enrollment_Year"], errors="coerce").isin(config.TRAIN_YEARS)]
    cutoff = pd.to_datetime(test["Enrollment_Date"], errors="coerce").min()
    pre_test_train_candidates = candidate_train[pd.to_datetime(candidate_train["Enrollment_Date"], errors="coerce") < cutoff]
    pct_source_counts = data["Recovery_Pct_Target_Source"].value_counts(dropna=False).to_dict()
    weeks_source_counts = data["Recovery_Weeks_Target_Source"].value_counts(dropna=False).to_dict()

    report = {
        "data_source": config.DATA_SOURCE,
        "provenance": db.LAST_DATA_ORIGIN,
        "selected_target": {"key": config.TARGET_SELECTION, "column": config.TARGET_OUTCOME_COLUMN, "display_label": config.TARGET_DISPLAY_LABEL},
        "target_definition": {
            "recovery_pct": "Recorded Final_Recovery_Pct when present; otherwise a 50:50 proxy from both observed final strength and ROM, using a fixed 140-degree scale. This proxy is not region-normalized or validated; cross-region results are descriptive only. No baseline or imputed outcome fallback.",
            "recovery_weeks": "Observed Actual_Recovery_Time_weeks only. Missing durations remain missing and are excluded from fitting/scoring.",
        },
        "outcome_availability_note": "The source schema has no actual outcome-recorded timestamp. Training eligibility is conservatively approximated as Enrollment_Date + Actual_Recovery_Time_weeks; this estimated date is used only for backtest eligibility and never as a feature.",
        "data_quality": {
            "n_all_patients": int(len(data)),
            "duplicate_patient_rows": duplicate_patient_rows,
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_rows_excluded_outside_split": int(len(excluded)),
            "training_candidates_before_outcome_maturity_filter": int(len(pre_test_train_candidates)),
            "training_rows_excluded_because_outcome_not_yet_available": int(len(pre_test_train_candidates) - len(train)),
            "training_outcome_availability_estimated_from_duration": True,
            "estimated_training_outcome_available_through": str(pd.to_datetime(train["Estimated_Outcome_Availability_Date"]).max().date()),
            "reported_year_vs_enrollment_date_mismatch_n": int(mismatch.sum()),
            "enrollment_date_min": str(enrollment_date.min().date()) if enrollment_date.notna().any() else None,
            "enrollment_date_max": str(enrollment_date.max().date()) if enrollment_date.notna().any() else None,
            "recovery_pct_target_source_counts": {str(k): int(v) for k, v in pct_source_counts.items()},
            "duration_target_source_counts": {str(k): int(v) for k, v in weeks_source_counts.items()},
            "train_recovery_pct_labeled_n": int(pct_train_mask.sum()),
            "train_duration_labeled_n": int(weeks_train_mask.sum()),
            "test_recovery_pct_labeled_n": int(pct_mask.sum()),
            "test_duration_labeled_n": int(weeks_mask.sum()),
        },
        "split": {
            "configured_train_years": config.TRAIN_YEARS,
            "configured_test_years": config.TEST_YEARS,
            "date_is_authoritative_for_year": True,
            "training_labels_matured_before_test_start": True,
            "train_date_min": str(pd.to_datetime(train["Enrollment_Date"]).min().date()),
            "train_date_max": str(pd.to_datetime(train["Enrollment_Date"]).max().date()),
            "test_date_min": str(pd.to_datetime(test["Enrollment_Date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["Enrollment_Date"]).max().date()),
        },
        "model": {
            "recovery_pct": f"weighted VotingRegressor: {booster_name} weight 2, ExtraTrees weight 1, RandomForest weight 1",
            "recovery_weeks": "VotingRegressor: HistGradientBoosting, ExtraTrees, RandomForest; equal weights",
            "configured_hyperparameters": {
                "recovery_booster": {"estimator": booster_name, "parameters": booster_params},
                "extra_trees_pct": {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 1},
                "random_forest_pct": {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 1},
                "recovery_ensemble_weights": {booster_name: 2, "ExtraTreesRegressor": 1, "RandomForestRegressor": 1},
            },
        },
        "chronological_holdout": {
            "recovery_pct": {"model": _metrics(y_pct, pred_pct_labeled),
                             "training_mean_baseline": _metrics(y_pct, pct_mean_baseline),
                             "bootstrap_intervals": _bootstrap_intervals(y_pct, pred_pct_labeled)},
            "recovery_weeks": {"model": _metrics(y_weeks, pred_weeks_labeled),
                               "training_mean_baseline": _metrics(y_weeks, weeks_mean_baseline),
                               "bootstrap_intervals": _bootstrap_intervals(y_weeks, pred_weeks_labeled)},
        },
        "expanding_window_cross_validation": {
            "recovery_pct": pct_cv,
            "recovery_weeks": weeks_cv,
        },
        "adaptive_landmark_forecast_comparison": {},
        "deviation_threshold_sensitivity": {},
        "ablations": {
            "status": "Not estimable from current specification/data.",
            "reason": "The three claimed pillars are not operationally defined, and the data lack patient-level treatment-pathway assignment and counterfactual outcomes.",
        },
    }

    # Fair static-vs-updated forecast comparison: fit separate models at each
    # landmark using only training patients with check-ins by that week. Test
    # patients must have the same type of pre-landmark observations.
    weekly = tables.get("Weekly_Progress", pd.DataFrame())
    if not weekly.empty:
        for cutoff in (4, 8):
            landmark = _landmark_rows(data, weekly, cutoff)
            train_ids = set(train["Patient_ID"].astype(str))
            test_ids = set(test.loc[pct_mask, "Patient_ID"].astype(str))
            landmark_train = landmark[
                landmark["Patient_ID"].astype(str).isin(train_ids)
                & landmark["_Landmark_Week"].notna()
                & landmark[config.TARGET_RECOVERY_PCT].notna()
            ]
            landmark_test = landmark[
                landmark["Patient_ID"].astype(str).isin(test_ids)
                & landmark["_Landmark_Week"].notna()
                & landmark[config.TARGET_RECOVERY_PCT].notna()
            ]
            if len(landmark_train) < 2 or landmark_test.empty:
                report["adaptive_landmark_forecast_comparison"][f"week_{cutoff}"] = {
                    "status": "Insufficient observed landmark data.",
                    "n_train": int(len(landmark_train)), "n_test": int(len(landmark_test)),
                }
                continue
            dynamic_model = modeling.build_pipeline("pct")
            dynamic_model.fit(get_feature_frame(landmark_train), landmark_train[config.TARGET_RECOVERY_PCT])
            dynamic_pred = dynamic_model.predict(get_feature_frame(landmark_test))
            static_map = test.assign(_pid=test["Patient_ID"].astype(str)).set_index("_pid", drop=False)
            static_rows = static_map.loc[landmark_test["Patient_ID"].astype(str).tolist()].reset_index(drop=True).drop(columns="_pid")
            static_pred = pct_model.predict(get_feature_frame(static_rows))
            same_model_updated_pred = pct_model.predict(get_feature_frame(landmark_test))
            truth = landmark_test[config.TARGET_RECOVERY_PCT].to_numpy()
            dynamic_metrics = _metrics(truth, dynamic_pred)
            static_metrics = _metrics(truth, static_pred)
            same_model_updated_metrics = _metrics(truth, same_model_updated_pred)
            report["adaptive_landmark_forecast_comparison"][f"week_{cutoff}"] = {
                "n_train": int(len(landmark_train)),
                "n_test": int(len(landmark_test)),
                "median_latest_observation_week": float(landmark_test["_Landmark_Week"].median()),
                "static_baseline_model": static_metrics,
                "current_app_logic_same_baseline_model_with_updated_observations": same_model_updated_metrics,
                "week_specific_model_retrained_on_training_landmark_data": dynamic_metrics,
                "interpretation": "Same-model result mirrors the current recalibration method. The retrained landmark model is a separate alternative requiring deployment changes. Prediction error is not patient benefit or a causal treatment effect.",
            }

    threshold_counts = {str(float(t)): {"flagged": 0, "eligible_metric_observations": 0}
                        for t in config.DEVIATION_SENSITIVITY_THRESHOLDS_PCT}
    if not weekly.empty:
        test_ids = set(test["Patient_ID"].astype(str))
        weekly_test = weekly[weekly["Patient_ID"].astype(str).isin(test_ids)]
        for i, (_, patient) in enumerate(test.iterrows()):
            patient_weekly = weekly_test[weekly_test["Patient_ID"].astype(str) == str(patient["Patient_ID"])]
            if patient_weekly.empty:
                continue
            curve = simulation.build_predicted_trajectory(patient, pct_pred[i], weeks_pred[i])
            for _, observation in patient_weekly.iterrows():
                week_num = int(observation["Week"])
                predicted = curve.iloc[min(max(week_num, 1), len(curve)) - 1]
                actual = {"Pain": observation.get("Pain"), "ROM_deg": observation.get("ROM_deg"),
                          "Strength_pct": observation.get("Quadriceps_Strength_pct")}
                for threshold in config.DEVIATION_SENSITIVITY_THRESHOLDS_PCT:
                    flags = simulation.check_deviation(predicted, actual, threshold_pct=threshold, method="percentage")
                    key = str(float(threshold))
                    for detail in flags.values():
                        if pd.notna(detail.get("actual")) and pd.notna(detail.get("predicted")):
                            threshold_counts[key]["eligible_metric_observations"] += 1
                            threshold_counts[key]["flagged"] += int(detail["flagged"])
        for result in threshold_counts.values():
            n = result["eligible_metric_observations"]
            result["flag_rate_percent"] = round(100 * result["flagged"] / n, 3) if n else None
    report["deviation_threshold_sensitivity"] = {
        "threshold_results": threshold_counts,
        "interpretation": "Flag counts/rates only on held-out observations; no adjudicated deterioration labels exist, so this is not clinical sensitivity/specificity.",
    }

    if pct_mask.sum() >= 2:
        report["recovery_pct_subgroups"] = {
            field: _subgroup_metrics(test.loc[pct_mask].reset_index(drop=True), y_pct, pred_pct_labeled, field)
            for field in ("Body_Part", "Gender", "Injury_Type", "Severity")
        }
        age_test = test.loc[pct_mask].reset_index(drop=True).copy()
        age_test["Age_Band_Eval"] = pd.cut(
            pd.to_numeric(age_test["Age"], errors="coerce"),
            bins=[-np.inf, 25, 40, 55, np.inf],
            labels=["<25", "25-40", "40-55", "55+"],
            right=False,
        )
        report["recovery_pct_subgroups"]["Age_Band"] = _subgroup_metrics(
            age_test, y_pct, pred_pct_labeled, "Age_Band_Eval"
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if emit:
        print(json.dumps(report, indent=2, allow_nan=False))
        print(f"Saved {out}")


if __name__ == "__main__":
    run()

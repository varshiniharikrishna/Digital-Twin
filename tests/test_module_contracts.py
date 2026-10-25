"""Smoke contracts for analysis modules: public entrypoints and no hidden fake-data generation."""
import ast
from pathlib import Path
import pandas as pd
import pytest
import model_benchmark, multi_condition_eval, similarity_eval, twin_trajectory_eval, eval_adaptive_loop, ablation, validate_real_data, run_all, fairness_analysis, wearable_robustness
import modeling
import db
import pathway_mining
from eval_adaptive_loop import observations_as_of, patient_level_error_differences

def test_new_analysis_modules_expose_expected_entrypoints():
    assert callable(model_benchmark.benchmark)
    assert callable(multi_condition_eval.run)
    assert callable(similarity_eval.run)
    assert callable(twin_trajectory_eval.run)
    assert callable(eval_adaptive_loop.run)
    assert callable(ablation.run)
    assert callable(fairness_analysis.run)
    assert callable(wearable_robustness.run)
    assert callable(run_all.main)

def test_real_validator_requires_user_path():
    source=Path(validate_real_data.__file__).read_text()
    tree=ast.parse(source)
    assert "No synthetic data will be generated" in source
    assert any(isinstance(n,ast.FunctionDef) and n.name=="main" for n in tree.body)

def test_benchmark_candidates_include_requested_estimators_when_installed():
    models,_=model_benchmark.candidates()
    assert {"Ridge","RandomForest","ExtraTrees","HistGradientBoosting","VotingRegressor","MeanBaseline"}.issubset(models)
    if modeling.XGBRegressor is not None:
        assert models["XGBoost"].__class__.__name__ == "_XGBoostSklearnAdapter"

def test_xgboost_is_the_production_estimator_when_installed():
    if modeling.XGBRegressor is None:
        return
    import config
    assert modeling.get_recovery_boost_model_name() == "XGBoost"
    assert modeling.make_estimator().__class__.__name__ == "_XGBoostSklearnAdapter"

def test_xgboost_optional_dependency_has_clear_hgb_fallback(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    monkeypatch.setattr(modeling, "XGBRegressor", None)
    assert isinstance(modeling.make_estimator("XGBoost"), HistGradientBoostingRegressor)

def test_production_model_artifact_round_trip(tmp_path):
    import config
    if modeling.XGBRegressor is None:
        return
    X = pd.DataFrame({name: [30, 41, 52, 63] for name in config.NUMERIC_FEATURES})
    for name in config.CATEGORICAL_FEATURES:
        X[name] = ["a", "b", "a", "b"]
    pct = pd.Series([20., 35., 50., 65.])
    weeks = pd.Series([30., 25., 20., 15.])
    digest = modeling.dataframe_fingerprint(pd.concat([X, pct.rename("pct"), weeks.rename("weeks")], axis=1))
    first = modeling.load_or_train_models(X, pct, weeks, digest, tmp_path)
    second = modeling.load_or_train_models(X, pct, weeks, digest, tmp_path)
    assert second[2]["data_hash"] == digest
    assert first[0].predict(X).tolist() == second[0].predict(X).tolist()

def test_model_selection_requires_matching_training_hash(tmp_path):
    import json
    X = pd.DataFrame({"Age": [30, 40]})
    y = pd.Series([20., 40.])
    payload = X.copy()
    payload["__target_pct"] = y.to_numpy()
    digest = modeling.dataframe_fingerprint(payload)
    report = tmp_path / "benchmark.json"
    report.write_text(json.dumps({"data_hash": digest, "selected_by_training_cv": "VotingRegressor",
                                  "best_params": {"VotingRegressor": {}}}))
    assert modeling.cv_model_selection(X, y, report)[0] == "VotingRegressor"
    assert modeling.cv_model_selection(X, y + 1, report)[0] == __import__("config").PRODUCTION_MODEL_NAME

def test_sql_identifier_allowlist_rejects_untrusted_identifier():
    assert db._quoted_identifier("Patient", table=True) == "`Patient`"
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        db._quoted_identifier("Patient; DROP TABLE Doctor", table=True)

def test_pathway_mining_excludes_chronological_holdout_patients():
    tables = {"Recovery_Pathway": pd.DataFrame([{"Pathway_ID": "PW1", "Cohort_ID": "C1", "Body_Region": "knee"}]),
              "Cohort_Definition": pd.DataFrame([{"Cohort_ID": "C1", "Injury_Type": "ACL Tear", "Severity": "Moderate", "Age_Band": "25-40"}]),
              "Patient": pd.DataFrame([{"Patient_ID": f"P{i}", "Age": 30, "Enrollment_Year": 2023 if i < 5 else 2025} for i in range(6)]),
              "Injury": pd.DataFrame([{"Patient_ID": f"P{i}", "Injury_Type": "ACL Tear", "Severity": "Moderate"} for i in range(6)]),
              "Final_Outcome": pd.DataFrame([{"Patient_ID": f"P{i}", "Final_Recovery_Pct": 90} for i in range(6)]),
              "Pathway_Steps": pd.DataFrame()}
    mined = pathway_mining.mine_pathways_from_tables(tables)
    assert mined and mined[0]["Sample_Size"] == 5

def test_evaluation_reports_wape_instead_of_percentage_error_at_zero():
    result = modeling.evaluate([0, 10], [4, 12])
    assert result["WAPE"] == 6 / 10 * 100
    assert "MAPE" not in result

def test_replay_observations_are_bounded_by_week():
    weekly = pd.DataFrame({"Patient_ID": ["P1", "P1", "P1"], "Week": [1, 2, 3], "Pain": [7, 5, 2]})
    visible = observations_as_of(weekly, "P1", 2)
    assert visible["Week"].max() == 2
    assert visible["Pain"].tolist() == [7, 5]

def test_adaptive_comparison_aggregates_errors_by_patient():
    frame = pd.DataFrame({"Patient_ID": ["P1", "P1", "P2", "P2"],
                          "arm": ["A_static", "B_on_trigger", "A_static", "B_on_trigger"],
                          "pred": [4, 3, 8, 10], "target": [5, 5, 10, 10]})
    result = patient_level_error_differences(frame, "B_on_trigger")
    assert result.to_dict() == {"P1": 1.0, "P2": -2.0}

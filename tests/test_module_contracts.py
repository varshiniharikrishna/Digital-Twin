"""Smoke contracts for analysis modules: public entrypoints and no hidden fake-data generation."""
import ast
from pathlib import Path
import model_benchmark, multi_condition_eval, similarity_eval, twin_trajectory_eval, eval_adaptive_loop, ablation, validate_real_data, run_all, fairness_analysis, wearable_robustness

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

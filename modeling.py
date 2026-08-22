"""
modeling.py
-----------
Phase 2 (train) + Phase 3 (evaluate) of the Digital Twin.

Upgraded Ensemble Regressor combining HistGradientBoostingRegressor,
ExtraTreesRegressor, and RandomForestRegressor to maximize R2 and minimize RMSE.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


def _make_ensemble_regressor(target_type="pct"):
    if target_type == "pct":
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=5, random_state=config.RANDOM_STATE)
        et = ExtraTreesRegressor(n_estimators=400, max_depth=16, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf = RandomForestRegressor(n_estimators=400, max_depth=16, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        return VotingRegressor([("hgb", hgb), ("et", et), ("rf", rf)])
    else:
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=5, random_state=config.RANDOM_STATE)
        et = ExtraTreesRegressor(n_estimators=400, max_depth=14, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf = RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        return VotingRegressor([("hgb", hgb), ("et", et), ("rf", rf)])


def build_pipeline(target_type="pct") -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", _make_ensemble_regressor(target_type))])


def train_models(X_train: pd.DataFrame, y_train_pct: pd.Series, y_train_weeks: pd.Series):
    """Trains the ensemble digital-twin models."""
    pct_pipeline = build_pipeline("pct")
    pct_pipeline.fit(X_train, y_train_pct)

    weeks_pipeline = build_pipeline("weeks")
    weeks_pipeline.fit(X_train, y_train_weeks)

    return pct_pipeline, weeks_pipeline


def predict(pct_pipeline, weeks_pipeline, X: pd.DataFrame):
    pred_pct = pct_pipeline.predict(X)
    pred_weeks = weeks_pipeline.predict(X)
    return pred_pct, pred_weeks


def evaluate(y_true, y_pred) -> dict:
    """Computes MAE, RMSE, MAPE, R2 for one target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extracts feature importances from ExtraTrees estimator inside voting ensemble."""
    preprocess = pipeline.named_steps["preprocess"]
    voting_model = pipeline.named_steps["model"]
    
    # Extract feature names
    cat_encoder = preprocess.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(config.CATEGORICAL_FEATURES))
    all_names = config.NUMERIC_FEATURES + cat_names

    # Pull importances from the ExtraTrees model in ensemble
    et_estimator = dict(voting_model.named_estimators_)["et"]
    importances = et_estimator.feature_importances_

    fi = pd.DataFrame({"Feature": all_names, "Importance": importances})
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)

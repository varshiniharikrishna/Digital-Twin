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
from sklearn.base import is_regressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def _make_ensemble_regressor(target_type="pct"):
    if target_type == "pct":
        boosted = None
        if XGBRegressor is not None:
            candidate = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                                     subsample=0.85, colsample_bytree=0.85,
                                     reg_lambda=2.0, objective="reg:squarederror",
                                     random_state=config.RANDOM_STATE, n_jobs=-1)
            try:
                if is_regressor(candidate):
                    boosted = candidate
            except Exception:
                # Some XGBoost releases do not expose sklearn estimator tags
                # understood by newer scikit-learn VotingRegressor versions.
                pass
        if boosted is None:
            boosted = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03,
                                                    max_depth=5, l2_regularization=2.0,
                                                    random_state=config.RANDOM_STATE)
        et = ExtraTreesRegressor(n_estimators=400, max_depth=16, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf = RandomForestRegressor(n_estimators=400, max_depth=16, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        return VotingRegressor([("boosted", boosted), ("et", et), ("rf", rf)], weights=[2, 1, 1])
    else:
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=5, random_state=config.RANDOM_STATE)
        et = ExtraTreesRegressor(n_estimators=400, max_depth=14, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf = RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=1, random_state=config.RANDOM_STATE, n_jobs=-1)
        return VotingRegressor([("hgb", hgb), ("et", et), ("rf", rf)])


def get_recovery_boost_model_name():
    """Return the estimator that this environment will actually use."""
    if XGBRegressor is not None:
        try:
            if is_regressor(XGBRegressor()):
                return "XGBRegressor"
        except Exception:
            pass
    return "HistGradientBoostingRegressor"


def build_pipeline(target_type="pct") -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), config.NUMERIC_FEATURES),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                               ("encode", OneHotEncoder(handle_unknown="ignore"))]), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", _make_ensemble_regressor(target_type))])


def train_models(X_train: pd.DataFrame, y_train_pct: pd.Series, y_train_weeks: pd.Series):
    """Fit each target model only on rows with an observed target."""
    y_train_pct = pd.to_numeric(y_train_pct, errors="coerce")
    pct_mask = y_train_pct.notna() & np.isfinite(y_train_pct)
    if int(pct_mask.sum()) < 2:
        raise ValueError("At least two observed recovery-percentage targets are required for training.")
    pct_pipeline = build_pipeline("pct")
    pct_pipeline.fit(X_train.loc[pct_mask], y_train_pct.loc[pct_mask])

    y_train_weeks = pd.to_numeric(y_train_weeks, errors="coerce")
    weeks_mask = y_train_weeks.notna() & np.isfinite(y_train_weeks)
    if int(weeks_mask.sum()) < 2:
        raise ValueError("At least two observed recovery-duration targets are required for training.")
    weeks_pipeline = build_pipeline("weeks")
    weeks_pipeline.fit(X_train.loc[weeks_mask], y_train_weeks.loc[weeks_mask])

    return pct_pipeline, weeks_pipeline


def train_recovery_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Train the selected outcome model when discharge duration is unavailable."""
    y=pd.to_numeric(y_train,errors="coerce"); mask=y.notna() & np.isfinite(y)
    if int(mask.sum())<2: raise ValueError("At least two observed target values are required.")
    model=build_pipeline("pct");model.fit(X_train.loc[mask],y.loc[mask]);return model


def predict(pct_pipeline, weeks_pipeline, X: pd.DataFrame):
    pred_pct = pct_pipeline.predict(X)
    pred_weeks = weeks_pipeline.predict(X)
    return pred_pct, pred_weeks


def evaluate(y_true, y_pred) -> dict:
    """Computes MAE, RMSE, MAPE, R2 for one target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {"MAE": None, "RMSE": None, "MAPE": None, "R2": None, "n": 0}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = (np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            if nonzero.any() else None)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else None

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "n": int(len(y_true))}


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extracts feature importances from ExtraTrees estimator inside voting ensemble."""
    preprocess = pipeline.named_steps["preprocess"]
    voting_model = pipeline.named_steps["model"]
    
    # Extract feature names
    all_names = list(preprocess.get_feature_names_out())

    # ExtraTrees provides a stable global importance summary for this ensemble.
    et_estimator = dict(voting_model.named_estimators_)["et"]
    importances = et_estimator.feature_importances_

    fi = pd.DataFrame({"Feature": all_names, "Importance": importances})
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)

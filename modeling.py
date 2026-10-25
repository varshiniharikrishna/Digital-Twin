"""
modeling.py
-----------
Phase 2 (train) + Phase 3 (evaluate) of the Digital Twin.

Upgraded Ensemble Regressor combining HistGradientBoostingRegressor,
ExtraTreesRegressor, and RandomForestRegressor to maximize R2 and minimize RMSE.
"""

import numpy as np
import pandas as pd
import hashlib
import json
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


class _XGBoostSklearnAdapter(RegressorMixin, BaseEstimator):
    """Expose XGBoost as a sklearn regressor across older tag APIs."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        if XGBRegressor is None:
            raise ImportError("XGBoost is not installed; install the pinned xgboost dependency.")
        self.estimator_ = XGBRegressor(**self.params)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


def make_estimator(model_name=None, target_type="pct", params=None):
    """Build the configured estimator; unavailable optional models fail clearly."""
    name = model_name or config.PRODUCTION_MODEL_NAME
    params = dict(params or {})
    defaults = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
                "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0,
                "objective": "reg:squarederror", "random_state": config.RANDOM_STATE,
                "n_jobs": 1, "verbosity": 0}
    if name in {"XGBoost", "XGBRegressor"}:
        if XGBRegressor is None:
            return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03,
                max_depth=5, random_state=config.RANDOM_STATE)
        defaults.update(params)
        return _XGBoostSklearnAdapter(**defaults)
    if name in {"HistGradientBoosting", "HistGradientBoostingRegressor"}:
        configured = {"max_iter": 300, "learning_rate": 0.03, "max_depth": 5,
                     "random_state": config.RANDOM_STATE}
        configured.update(params)
        return HistGradientBoostingRegressor(**configured)
    if name == "ExtraTrees":
        configured = {"n_estimators": 400, "random_state": config.RANDOM_STATE, "n_jobs": 1}
        configured.update(params)
        return ExtraTreesRegressor(**configured)
    if name == "RandomForest":
        configured = {"n_estimators": 400, "random_state": config.RANDOM_STATE, "n_jobs": 1}
        configured.update(params)
        return RandomForestRegressor(**configured)
    if name == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    if name == "VotingRegressor":
        xgb = make_estimator("XGBoost", params={"n_estimators": 500, "random_state": config.RANDOM_STATE, "n_jobs": 1, "objective": "reg:squarederror", "verbosity": 0})
        return VotingRegressor([("xgb", xgb),
                                ("random_forest", RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=1)),
                                ("extra_trees", ExtraTreesRegressor(random_state=config.RANDOM_STATE, n_jobs=1)),
                                ("hist_gradient", HistGradientBoostingRegressor(random_state=config.RANDOM_STATE))])
    if name == "Stacking_XGB_LGBM_HGB_Ridge":
        estimators = [("xgb", make_estimator("XGBoost", params={"n_estimators": 500, "random_state": config.RANDOM_STATE, "n_jobs": 1, "objective": "reg:squarederror", "verbosity": 0})),
                      ("hist_gradient", HistGradientBoostingRegressor(random_state=config.RANDOM_STATE))]
        try:
            from lightgbm import LGBMRegressor
            estimators.append(("lightgbm", LGBMRegressor(random_state=config.RANDOM_STATE, verbosity=-1, n_jobs=1)))
        except ImportError:
            pass
        return StackingRegressor(estimators, final_estimator=__import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(), cv=3, n_jobs=1)
    raise ValueError(f"Unsupported production model: {name}")


def get_recovery_boost_model_name():
    """Return the estimator that this environment will actually use."""
    return "XGBoost" if XGBRegressor is not None else "HistGradientBoosting"


def build_pipeline(target_type="pct", model_name=None, params=None,
                   extra_numeric_features=None) -> Pipeline:
    numeric_features = list(config.NUMERIC_FEATURES) + list(extra_numeric_features or [])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric_features),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                               ("encode", OneHotEncoder(handle_unknown="ignore"))]), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", make_estimator(model_name, target_type, params))])


def train_models(X_train: pd.DataFrame, y_train_pct: pd.Series, y_train_weeks: pd.Series,
                 model_name=None, params=None):
    """Fit each target model only on rows with an observed target."""
    if model_name is None:
        model_name, selected_params, _ = cv_model_selection(X_train, y_train_pct)
        params = params if params is not None else selected_params
    y_train_pct = pd.to_numeric(y_train_pct, errors="coerce")
    pct_mask = y_train_pct.notna() & np.isfinite(y_train_pct)
    if int(pct_mask.sum()) < 2:
        raise ValueError("At least two observed recovery-percentage targets are required for training.")
    pct_pipeline = build_pipeline("pct", model_name, params)
    pct_pipeline.fit(X_train.loc[pct_mask], y_train_pct.loc[pct_mask])

    y_train_weeks = pd.to_numeric(y_train_weeks, errors="coerce")
    weeks_mask = y_train_weeks.notna() & np.isfinite(y_train_weeks)
    if int(weeks_mask.sum()) < 2:
        raise ValueError("At least two observed recovery-duration targets are required for training.")
    weeks_pipeline = build_pipeline("weeks", model_name, params)
    weeks_pipeline.fit(X_train.loc[weeks_mask], y_train_weeks.loc[weeks_mask])

    return pct_pipeline, weeks_pipeline


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Stable content digest used to invalidate persisted fitted models."""
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    schema = json.dumps([(str(c), str(frame[c].dtype)) for c in frame.columns], separators=(",", ":")).encode()
    return hashlib.sha256(schema + values).hexdigest()


def cv_model_selection(X_train, y_train, benchmark_path=None):
    """Use the benchmark-selected estimator only if its training hash matches."""
    y = pd.to_numeric(pd.Series(y_train, index=X_train.index), errors="coerce")
    mask = y.notna() & np.isfinite(y)
    payload = X_train.loc[mask].reset_index(drop=True).copy()
    payload["__target_pct"] = y.loc[mask].to_numpy()
    digest = dataframe_fingerprint(payload)
    path = Path(benchmark_path or "reports/model_benchmark/benchmark.json")
    if path.is_file():
        report = json.loads(path.read_text())
        if report.get("data_hash") == digest:
            name = report.get("selected_by_training_cv")
            params = {key.replace("model__", "", 1): value
                      for key, value in report.get("best_params", {}).get(name, {}).items()}
            return name, params, digest
    return config.PRODUCTION_MODEL_NAME, {}, digest


def load_or_train_models(X_train, y_train_pct, y_train_weeks, data_hash,
                         artifact_dir=None, model_name=None, params=None):
    """Load matching model artifacts, or train and persist both target models."""
    directory = Path(artifact_dir or config.MODEL_ARTIFACT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "production_metadata.json"
    model_path = directory / "production_models.joblib"
    selected_name, selected_params, pct_data_hash = cv_model_selection(
        X_train, y_train_pct, benchmark_path=directory.parent / "benchmark.json")
    if model_name:
        selected_name, selected_params = model_name, dict(params or {})
    if selected_name in {"XGBoost", "XGBRegressor"} and XGBRegressor is None:
        selected_name = "HistGradientBoosting"
    xgb_version = getattr(__import__("xgboost"), "__version__", None) if XGBRegressor else None
    benchmark_path = directory.parent / "benchmark.json"
    interval_half_width = None
    if not model_name and benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text())
        if benchmark.get("data_hash") == pct_data_hash and benchmark.get("selected_by_training_cv") == selected_name:
            interval_half_width = benchmark.get("prediction_interval", {}).get("half_width")
    if metadata_path.exists() and model_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("data_hash") == data_hash
                and metadata.get("model_name") == selected_name
                and metadata.get("params", {}) == selected_params
                and metadata.get("scikit_learn") == __import__("sklearn").__version__
                and metadata.get("xgboost") == xgb_version):
            saved = joblib.load(model_path)
            if interval_half_width is not None:
                metadata["prediction_interval_half_width"] = float(interval_half_width)
                metadata["prediction_interval_method"] = "empirical repeated grouped-CV residual; descriptive, no formal coverage guarantee"
                metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
            return saved["recovery_pct"], saved["recovery_weeks"], metadata
    pct, weeks = train_models(X_train, y_train_pct, y_train_weeks, selected_name, selected_params)
    joblib.dump({"recovery_pct": pct, "recovery_weeks": weeks}, model_path)
    metadata = {"model_name": selected_name, "params": selected_params, "data_hash": data_hash,
                "features": list(X_train.columns), "python": __import__("platform").python_version(),
                "scikit_learn": __import__("sklearn").__version__,
                "xgboost": xgb_version}
    if interval_half_width is not None:
        metadata["prediction_interval_half_width"] = float(interval_half_width)
        metadata["prediction_interval_method"] = "empirical repeated grouped-CV residual; descriptive, no formal coverage guarantee"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    return pct, weeks, metadata


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
    """Compute MAE, RMSE, WAPE and R² for one target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {"MAE": None, "RMSE": None, "WAPE": None, "R2": None, "n": 0}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denominator = float(np.abs(y_true).sum())
    wape = float(np.abs(y_true - y_pred).sum() / denominator * 100) if denominator > 0 else None
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else None

    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "R2": r2, "n": int(len(y_true))}


def bootstrap_metric_intervals(y_true, y_pred, n_resamples=1000):
    """Compute deterministic percentile intervals for paired metric resamples."""
    y = np.asarray(y_true, dtype=float); pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred); y, pred = y[valid], pred[valid]
    if len(y) < 2:
        return {key: None for key in ("R2", "MAE", "RMSE", "WAPE")}
    rng = np.random.default_rng(config.RANDOM_STATE)
    values = {key: [] for key in ("R2", "MAE", "RMSE", "WAPE")}
    for _ in range(n_resamples):
        take = rng.integers(0, len(y), size=len(y)); row = evaluate(y[take], pred[take])
        for key in values:
            if row[key] is not None and np.isfinite(row[key]): values[key].append(row[key])
    return {key: [float(np.quantile(items, .025)), float(np.quantile(items, .975))] if items else None
            for key, items in values.items()}


def get_feature_importance(pipeline: Pipeline, X_validation=None, y_validation=None) -> pd.DataFrame:
    """Return permutation importances for any fitted production pipeline."""
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    if X_validation is not None and y_validation is not None:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(pipeline, X_validation, y_validation,
                                        scoring="neg_mean_absolute_error",
                                        n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1)
        return (pd.DataFrame({"Feature": list(X_validation.columns),
                             "Importance": result.importances_mean})
                .sort_values("Importance", ascending=False).reset_index(drop=True))
    names = list(preprocess.get_feature_names_out())
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None and hasattr(estimator, "estimators_"):
        components = getattr(estimator, "named_estimators_", None)
        models = list(components.values()) if components is not None else list(estimator.estimators_)
        component_importances = [np.asarray(getattr(model, "feature_importances_"), dtype=float)
                                 for model in models if getattr(model, "feature_importances_", None) is not None]
        if component_importances:
            importances = np.mean(component_importances, axis=0)
    if importances is None and hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_)).reshape(-1)
    if importances is None and hasattr(estimator, "estimators_"):
        importances = np.zeros(len(names), dtype=float)
    if importances is None:
        raise ValueError("This estimator does not expose compatible feature importance values.")
    fi = pd.DataFrame({"Feature": names, "Importance": importances})
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)

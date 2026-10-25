"""Optional SHAP explanations for the fitted recovery-percentage ensemble."""

import numpy as np
import pandas as pd


def explain_prediction(pipeline, background: pd.DataFrame, query: pd.DataFrame, max_background: int = 20):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is optional. Install it with: python3 -m pip install shap") from exc

    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    background = background.sample(min(max_background, len(background)), random_state=42)
    background_matrix = preprocess.transform(background)
    query_matrix = preprocess.transform(query)
    if hasattr(background_matrix, "toarray"):
        background_matrix = background_matrix.toarray()
    if hasattr(query_matrix, "toarray"):
        query_matrix = query_matrix.toarray()
    feature_names = list(preprocess.get_feature_names_out())
    explainer = shap.Explainer(estimator.predict, background_matrix, feature_names=feature_names)
    values = explainer(query_matrix, max_evals=2 * len(feature_names) + 1)
    attributions = np.asarray(values.values).reshape(-1, len(feature_names))[0]
    ranked = sorted(zip(feature_names, attributions), key=lambda pair: abs(float(pair[1])), reverse=True)
    return {
        "method": "SHAP permutation explainer for fitted recovery ensemble",
        "base_value": float(np.asarray(values.base_values).reshape(-1)[0]),
        "prediction": float(estimator.predict(query_matrix)[0]),
        "features": [{"feature": name, "shap_value": float(value)} for name, value in ranked[:15]],
        "caveat": "Model explanation, not causal explanation or evidence of clinical validity.",
    }

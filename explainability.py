"""Optional SHAP explanations for the fitted recovery-percentage ensemble."""

import numpy as np
import pandas as pd

def check_additivity(values, base_value, prediction, tolerance=1e-3):
    """Return whether SHAP contributions sum to prediction minus baseline."""
    err=float(np.sum(np.asarray(values,dtype=float))-(float(prediction)-float(base_value)))
    return {"passed":bool(abs(err)<=tolerance),"error":err,"tolerance":float(tolerance)}


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
    base_value=float(np.asarray(values.base_values).reshape(-1)[0])
    prediction=float(estimator.predict(query_matrix)[0])
    additivity=check_additivity(attributions,base_value,prediction)
    additivity_error=additivity["error"]
    return {
        "method": "SHAP permutation explainer for fitted recovery ensemble",
        "base_value": base_value,
        "prediction": prediction,
        "additivity_error": additivity_error,
        "additivity_within_1e-3": additivity["passed"],
        "features": [{"feature": name, "shap_value": float(value)} for name, value in ranked[:15]],
        "caveat": "Model explanation, not causal explanation or evidence of clinical validity.",
    }

def generate_explanation_artifacts(pipeline, background: pd.DataFrame, query: pd.DataFrame, out_dir, max_rows=50):
    """Write global importance, beeswarm and per-patient waterfall where SHAP permits."""
    from pathlib import Path
    import json
    import shap
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    preprocess=pipeline.named_steps["preprocess"];estimator=pipeline.named_steps["model"]
    bg=background.sample(min(max_rows,len(background)),random_state=42)
    b=preprocess.transform(bg);q=preprocess.transform(query)
    if hasattr(b,"toarray"):b=b.toarray()
    if hasattr(q,"toarray"):q=q.toarray()
    names=list(preprocess.get_feature_names_out());explainer=shap.Explainer(estimator.predict,b,feature_names=names)
    vals=explainer(b,max_evals=2*b.shape[1]+1)
    arr=np.asarray(vals.values).reshape(len(bg),-1)
    importance=pd.DataFrame({"feature":names,"mean_abs_shap":np.mean(np.abs(arr),axis=0)}).sort_values("mean_abs_shap",ascending=False)
    importance.to_csv(out/"global_importance.csv",index=False)
    qvals=explainer(q,max_evals=2*q.shape[1]+1)
    add=check_additivity(np.asarray(qvals.values).reshape(-1),np.asarray(qvals.base_values).reshape(-1)[0],float(estimator.predict(q)[0]))
    try:
        import matplotlib.pyplot as plt
        shap.plots.beeswarm(vals,show=False,max_display=20);plt.savefig(out/"beeswarm.png",bbox_inches="tight");plt.close()
        shap.plots.waterfall(qvals[0],show=False,max_display=20);plt.savefig(out/"patient_waterfall.png",bbox_inches="tight");plt.close()
    except Exception as exc:(out/"plot_status.txt").write_text(f"SHAP plot unavailable: {exc}\n")
    result={"method":"SHAP model explanation; not causal","top_global_features":importance.head(20).to_dict(orient="records"),"patient_additivity":add,"optional_lime":"not run; LIME package/configuration not included","stability":"not evaluated: bootstrap SHAP refits are computationally expensive and require prespecified bootstrap protocol"}
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=str));return result

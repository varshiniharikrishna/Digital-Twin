"""Compare learned, uniform and legacy weights; evaluate K/tau sensitivity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import config
from similarity import SimilarityModel,find_similar_patients,cohort_outcome_estimate

def run(train,test,out_dir="reports/similarity"):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);target=config.TARGET_RECOVERY_PCT
    learned=SimilarityModel.fit(train,target);uniform=SimilarityModel.fit(train,weights={c:1 for c in learned.feature_weights});legacy=SimilarityModel.fit(train,weights={c:(.6/len([x for x in learned.feature_weights if x in config.SIM_NUM_COLS]) if c in config.SIM_NUM_COLS else .4/len([x for x in learned.feature_weights if x in config.SIM_CAT_COLS])) for c in learned.feature_weights})
    methods={"learned_train_assoc":learned,"uniform_feature":uniform,"legacy_component_0.6_0.4":legacy};rows=[]
    ytrain=pd.to_numeric(train[target],errors="coerce");tolerance=float(ytrain.quantile(.75)-ytrain.quantile(.25))*.2
    for _,q in test.iterrows():
        truth=pd.to_numeric(pd.Series([q.get(target)]),errors="coerce").iloc[0]
        for name,m in methods.items():
            n=find_similar_patients(q,train,top_k=5,model=m);estimate=cohort_outcome_estimate(n,target) if not n.empty else np.nan
            agreement=abs(estimate-truth)<=tolerance if np.isfinite(estimate) and pd.notna(truth) else np.nan
            rows.append({"Patient_ID":q.get("Patient_ID"),"method":name,"K":5,"tau":m.tau,"n_neighbors":len(n),"confidence":n.attrs.get("confidence"),"knn_prediction":estimate,"truth":truth,"absolute_error":abs(estimate-truth) if np.isfinite(estimate) and pd.notna(truth) else np.nan,"outcome_class_agreement":agreement})
    df=pd.DataFrame(rows);df.to_csv(out/"neighbor_predictions.csv",index=False)
    summary=df.groupby("method").agg(n=("absolute_error","count"),MAE=("absolute_error","mean"),precision_at_k=("outcome_class_agreement","mean"),mean_neighbors=("n_neighbors","mean")).reset_index();summary.to_csv(out/"metric_comparison.csv",index=False)
    # Sensitivity grid uses score quantiles from TRAIN pairs only; held-out patients
    # are used only to report resulting behavior, never to pick τ.
    sample=train.sample(min(100,len(train)),random_state=config.RANDOM_STATE);pool=train
    scores=[]
    for _,q in sample.iterrows(): scores.extend(learned.score(q,pool).tolist())
    tau_grid=sorted(set(float(np.quantile(scores,p)) for p in (.10,.25,.40,.55))) if scores else [learned.tau]
    sensitivity=[]
    for k in config.SIM_K_CANDIDATES:
      for tau in tau_grid:
        for _,q in test.iterrows():
            truth=pd.to_numeric(pd.Series([q.get(target)]),errors="coerce").iloc[0]
            n=find_similar_patients(q,train,top_k=k,min_k=min(config.SIM_MIN_K,k),min_score=tau,model=learned)
            pred=cohort_outcome_estimate(n,target) if len(n) else np.nan
            sensitivity.append({"K":k,"tau":tau,"n_neighbors":len(n),"confidence":n.attrs.get("confidence"),"absolute_error":abs(pred-truth) if np.isfinite(pred) and pd.notna(truth) else np.nan,"outcome_agreement":abs(pred-truth)<=tolerance if np.isfinite(pred) and pd.notna(truth) else np.nan})
    sens=pd.DataFrame(sensitivity);sens.to_csv(out/"K_tau_sensitivity.csv",index=False)
    sens_summary=sens.groupby(["K","tau"]).agg(n=("absolute_error","count"),MAE=("absolute_error","mean"),precision_at_K=("outcome_agreement","mean"),mean_neighbors=("n_neighbors","mean")).reset_index();sens_summary.to_csv(out/"K_tau_sensitivity_summary.csv",index=False)
    report={"metric":"weighted normalized Gower per feature; numeric ranges and outcome-association weights trained on training split; τ selected on training outcome-agreement pairs","tau":{"learned":learned.tau,"uniform":uniform.tau,"legacy":legacy.tau},"k_tau_grid":{"K":list(config.SIM_K_CANDIDATES),"tau_train_score_quantiles":tau_grid,"summary_file":"K_tau_sensitivity_summary.csv"},"metrics":summary.to_dict(orient="records"),"caveat":"Outcome similarity is an exploratory agreement criterion and does not imply clinical equivalence."}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2,default=str));learned.save(out/"similarity_model.json");return report

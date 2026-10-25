"""Ablation of prediction, neighbor estimate, and observation update on identical holdout."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import config, modeling
from data_prep import build_modelling_dataset,chronological_split,get_feature_frame
from similarity import SimilarityModel,find_similar_patients,cohort_outcome_estimate

def run(tables,out_dir="reports/ablation",bootstrap_n=500):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);data=build_modelling_dataset(tables);train,test=chronological_split(data)
    train=train[train[config.TARGET_RECOVERY_PCT].notna()];test=test[test[config.TARGET_RECOVERY_PCT].notna()]
    model,_=modeling.train_models(get_feature_frame(train),train[config.TARGET_RECOVERY_PCT],train[config.TARGET_RECOVERY_WEEKS]);sim_model=SimilarityModel.fit(train,config.TARGET_RECOVERY_PCT)
    records=[]; rng=np.random.default_rng(config.RANDOM_STATE)
    for _,row in test.iterrows():
        pred=float(model.predict(get_feature_frame(pd.DataFrame([row])))[0]);nei=find_similar_patients(row,train,model=sim_model);knn=cohort_outcome_estimate(nei,config.TARGET_RECOVERY_PCT)
        records.append({"patient_id":row.Patient_ID,"target":float(row[config.TARGET_RECOVERY_PCT]),"model_only":pred,"similarity_only":knn,
                        "model_plus_similarity":float(np.mean([pred,knn])) if np.isfinite(knn) else pred})
    frame=pd.DataFrame(records); variants=[c for c in frame if c not in {"patient_id","target"}];rows=[]
    for v in variants:
        valid=frame[["target",v]].dropna()
        if len(valid)<2: rows.append({"variant":v,"n":len(valid),"status":"insufficient labels"});continue
        m=modeling.evaluate(valid.target,valid[v]);ci=[]
        for _ in range(bootstrap_n):
            ix=rng.integers(0,len(valid),len(valid));ci.append(modeling.evaluate(valid.target.iloc[ix],valid[v].iloc[ix])["MAE"])
        rows.append({"variant":v,"n":len(valid),**m,"MAE_95_CI":[float(np.quantile(ci,.025)),float(np.quantile(ci,.975))],"validity_note":"simulation and pathway pillars are not isolated: no linked treatment/pathway outcome or validated weekly forecast target"})
    frame.to_csv(out/"patient_predictions.csv",index=False);pd.DataFrame(rows).to_csv(out/"ablation.csv",index=False)
    import db
    report={"origin":db.LAST_DATA_ORIGIN,"identical_split":"chronological_split from data_prep.py","rows":rows,
            "status":"partial: model and neighbor estimators are compared; simulation, recalibration and pathway effect combinations are not reported because no valid outcome/exposure target supports them"}
    (out/"ablation.json").write_text(json.dumps(report,indent=2,default=str));return report

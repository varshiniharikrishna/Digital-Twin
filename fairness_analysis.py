"""Subgroup performance with explicit group counts and synthetic provenance."""
from pathlib import Path
import json
import pandas as pd
import config, modeling
from data_prep import build_modelling_dataset,chronological_split,get_feature_frame

def run(tables,out_dir="reports/fairness",min_n=10):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);data=build_modelling_dataset(tables);tr,te=chronological_split(data)
    pm=modeling.train_recovery_model(get_feature_frame(tr),tr[config.TARGET_RECOVERY_PCT]); y=pd.to_numeric(te[config.TARGET_RECOVERY_PCT],errors="coerce");valid=y.notna();pred=pm.predict(get_feature_frame(te.loc[valid]));evaldata=te.loc[valid].copy();evaldata["_pred"]=pred;evaldata["_y"]=y[valid]
    age=pd.cut(pd.to_numeric(evaldata.get("Age"),errors="coerce"),[-float("inf"),25,40,55,float("inf")],labels=["<25","25-40","40-55","55+"])
    evaldata["Age_Band"]=age;rows=[]
    for field in ["Gender","Sex","Age_Band","Injury_Type","Severity"]:
        if field not in evaldata:continue
        for group,g in evaldata.groupby(field,dropna=False):
            m=modeling.evaluate(g._y,g._pred)
            rows.append({"group_variable":field,"group":str(group),"n":len(g),"status":"reported" if len(g)>=min_n else "small group; descriptive only",**m})
    pd.DataFrame(rows).to_csv(out/"subgroup_metrics.csv",index=False);report={"origin":"synthetic bundled data" if "synthetic" in str(getattr(__import__('db'),'LAST_DATA_ORIGIN','')) else getattr(__import__('db'),'LAST_DATA_ORIGIN','unknown'),"minimum_group_n":min_n,"groups":rows,"interpretation":"Subgroup error summaries are not proof of fairness; inspect uncertainty, measurement and access differences."}
    (out/"fairness.json").write_text(json.dumps(report,indent=2,default=str));return report

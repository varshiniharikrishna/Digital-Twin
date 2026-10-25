"""Leave-one-injury-type-out and subgroup evaluation where sample sizes permit."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import config, modeling
from data_prep import get_feature_frame

def run(data,out_dir="reports/multi_condition"):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[];types=data.get("Injury_Type",pd.Series(dtype=str)).dropna().unique()
    if len(types)<2:
        result={"status":"not evaluable","reason":"Only one injury type is present; generalization cannot be assessed.","injury_types":list(map(str,types))}
    else:
        for injury in types:
            tr=data[(data.Injury_Type!=injury)&data[config.TARGET_RECOVERY_PCT].notna()];te=data[(data.Injury_Type==injury)&data[config.TARGET_RECOVERY_PCT].notna()]
            if len(tr)<10 or len(te)<2: rows.append({"injury_type":str(injury),"status":"insufficient sample","n_train":len(tr),"n_test":len(te)});continue
            pm,wm=modeling.train_models(get_feature_frame(tr),tr[config.TARGET_RECOVERY_PCT],tr[config.TARGET_RECOVERY_WEEKS]);pred=pm.predict(get_feature_frame(te)); rows.append({"injury_type":str(injury),"status":"evaluated","n_train":len(tr),"n_test":len(te),**modeling.evaluate(te[config.TARGET_RECOVERY_PCT],pred)})
        result={"status":"evaluated where sample size allowed","n_injury_types":len(types),"groups":rows,"caveat":"The estimator uses the current configured feature schema; this is not proof of transportability."}
    pd.DataFrame(rows).to_csv(out/"leave_one_injury_out.csv",index=False);(out/"evaluation.json").write_text(json.dumps(result,indent=2,default=str));return result

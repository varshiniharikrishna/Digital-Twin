"""Week-by-week prediction replay; evaluates forecast error, not clinical benefit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import config, modeling, simulation
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame

def run(tables,out_dir="reports/adaptive_loop"):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    data=build_modelling_dataset(tables);train,test=chronological_split(data);weekly=tables.get("Weekly_Progress",pd.DataFrame())
    if weekly.empty:
        report={"status":"not evaluable","reason":"Timestamped weekly observations are required; no synthetic longitudinal records generated.","pathway_change_rate":"not available without observed patient-level pathway assignments"}
        (out/"evaluation.json").write_text(json.dumps(report,indent=2));(out/"README.md").write_text("# Adaptive-loop evaluation\n\n"+json.dumps(report,indent=2)+"\n");return report
    pm,wm=modeling.train_models(get_feature_frame(train),train[config.TARGET_RECOVERY_PCT],train[config.TARGET_RECOVERY_WEEKS])
    records=[]
    for _,p in test.iterrows():
        target=pd.to_numeric(pd.Series([p.get(config.TARGET_RECOVERY_PCT)]),errors="coerce").iloc[0]
        if pd.isna(target): continue
        base=float(pm.predict(get_feature_frame(pd.DataFrame([p])))[0]); row=p.copy()
        for _,o in weekly[weekly.Patient_ID.astype(str).eq(str(p.Patient_ID))].sort_values("Week").iterrows():
            w=int(o.get("Week",0)); obs={"Pain":o.get("Pain"),"ROM_deg":o.get("ROM_deg"),"Strength_pct":o.get("Quadriceps_Strength_pct"),"Hamstring_Strength_pct":o.get("Hamstring_Strength_pct"),"Walking_Speed_mps":o.get("Walking_Speed_mps"),"Balance_Score":o.get("Balance_Score")}
            predrow=simulation.build_predicted_trajectory(p,base,float(wm.predict(get_feature_frame(pd.DataFrame([p])))[0])); pr=predrow.iloc[min(max(w,1),len(predrow))-1]
            dev=simulation.check_deviation(pr,{k:v for k,v in obs.items() if pd.notna(v)})
            changed=p.copy()
            for source,dest in [("Pain","Pain_Score"),("ROM_deg","Range_of_Motion_deg"),("Strength_pct","Quadriceps_Strength_pct"),("Balance_Score","Balance_Score"),("Walking_Speed_mps","Walking_Speed_mps")]:
                if pd.notna(obs.get(source)): changed[dest]=obs[source]
            rec=float(pm.predict(get_feature_frame(pd.DataFrame([changed])))[0])
            trigger=any(x.get("flagged") for x in dev.values())
            records.extend([{"Patient_ID":p.Patient_ID,"week":w,"arm":"A_static","pred":base,"target":target,"trigger":False},
                            {"Patient_ID":p.Patient_ID,"week":w,"arm":"B_on_trigger","pred":rec if trigger else base,"target":target,"trigger":trigger},
                            {"Patient_ID":p.Patient_ID,"week":w,"arm":"C_every_week","pred":rec,"target":target,"trigger":True}])
    frame=pd.DataFrame(records)
    summaries=[]
    for (arm,w),g in frame.groupby(["arm","week"]): summaries.append({"arm":arm,"week":int(w),"n":len(g),"MAE":float(np.mean(np.abs(g.pred-g.target))),"RMSE":float(np.sqrt(np.mean((g.pred-g.target)**2))),"recalibrations":int(g.trigger.sum())})
    metrics=pd.DataFrame(summaries);frame.to_csv(out/"replay_predictions.csv",index=False);metrics.to_csv(out/"metrics_by_week.csv",index=False)
    paired={}
    if not frame.empty:
        wide=frame.pivot_table(index=["Patient_ID","week"],columns="arm",values="pred").dropna()
        for arm in ("B_on_trigger","C_every_week"):
            diff=np.abs(wide[arm]-frame.drop_duplicates(["Patient_ID","week"]).set_index(["Patient_ID","week"]).target.reindex(wide.index))-np.abs(wide["A_static"]-frame.drop_duplicates(["Patient_ID","week"]).set_index(["Patient_ID","week"]).target.reindex(wide.index))
            try: stat=wilcoxon(diff) if len(diff)>0 and not np.allclose(diff,0) else None
            except ValueError: stat=None
            paired[arm]={"median_paired_absolute_error_difference":float(diff.median()),"wilcoxon_pvalue":float(stat.pvalue) if stat else None,"n":int(len(diff))}
    import db
    report={"origin":db.LAST_DATA_ORIGIN,"arms":["A static","B recalibrate when configured trigger fires","C recalibrate each week"],"paired_tests":paired,"pathway_change_rate":"not available: source schema contains no linked patient treatment/pathway assignment history","trajectory_rmse":"not evaluated: source schema does not include a calibrated predicted metric trajectory target; the app curve is hand-coded","interpretation":"Prediction-error replay only; no causal or patient-benefit inference."}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2,default=str));(out/"README.md").write_text("# Adaptive-loop evaluation\n\n"+json.dumps(report,indent=2,default=str)+"\n")
    return report

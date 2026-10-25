"""Week-by-week prediction replay; evaluates forecast error, not clinical benefit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import config, modeling, simulation
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame


def observations_as_of(weekly, patient_id, week):
    """Return only the selected patient's recorded observations through week t."""
    weeks = pd.to_numeric(weekly.get("Week", pd.Series(dtype=float)), errors="coerce")
    mask = (weekly["Patient_ID"].astype(str) == str(patient_id)) & weeks.notna() & (weeks <= int(week))
    return weekly.loc[mask].sort_values("Week").copy()


def patient_level_error_differences(frame, arm):
    """Return paired patient mean-absolute-error differences (arm minus static)."""
    scored = frame.assign(abs_error=(frame["pred"] - frame["target"]).abs())
    by_patient = scored.groupby(["Patient_ID", "arm"])["abs_error"].mean().unstack("arm")
    if "A_static" not in by_patient or arm not in by_patient:
        return pd.Series(dtype=float)
    return (by_patient[arm] - by_patient["A_static"]).dropna()

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
        base=float(pm.predict(get_feature_frame(pd.DataFrame([p])))[0])
        duration=float(wm.predict(get_feature_frame(pd.DataFrame([p])))[0])
        patient_weeks=weekly[weekly.Patient_ID.astype(str).eq(str(p.Patient_ID))].sort_values("Week")
        if patient_weeks.empty: continue
        changed_rows=[]; as_of_rows=[]; week_records=[]
        predrow=simulation.build_predicted_trajectory(p,base,duration)
        for _,o in patient_weeks.iterrows():
            w=int(o.get("Week",0))
            if w < 1: continue
            as_of=observations_as_of(weekly,p.Patient_ID,w)
            if as_of.empty: continue
            o=as_of.iloc[-1]
            obs={"Pain":o.get("Pain"),"ROM_deg":o.get("ROM_deg"),"Strength_pct":o.get("Quadriceps_Strength_pct"),"Hamstring_Strength_pct":o.get("Hamstring_Strength_pct"),"Walking_Speed_mps":o.get("Walking_Speed_mps"),"Balance_Score":o.get("Balance_Score")}
            pr=predrow.iloc[min(max(w,1),len(predrow))-1]
            dev=simulation.check_deviation(pr,{k:v for k,v in obs.items() if pd.notna(v)})
            changed=p.copy()
            for source,dest in [("Pain","Pain_Score"),("ROM_deg","Range_of_Motion_deg"),("Strength_pct","Quadriceps_Strength_pct"),("Balance_Score","Balance_Score"),("Walking_Speed_mps","Walking_Speed_mps")]:
                if pd.notna(obs.get(source)): changed[dest]=obs[source]
            changed_rows.append(changed);as_of_rows.append(obs);week_records.append((w,dev))
        if not changed_rows: continue
        recalibrated=pm.predict(get_feature_frame(pd.DataFrame(changed_rows).reset_index(drop=True)))
        for (w,dev),obs,rec in zip(week_records,as_of_rows,recalibrated):
            trigger=any(x.get("flagged") for x in dev.values())
            records.extend([{"Patient_ID":p.Patient_ID,"week":w,"arm":"A_static","pred":base,"target":target,"trigger":False},
                            {"Patient_ID":p.Patient_ID,"week":w,"arm":"B_on_trigger","pred":rec if trigger else base,"target":target,"trigger":trigger},
                            {"Patient_ID":p.Patient_ID,"week":w,"arm":"C_every_week","pred":rec,"target":target,"trigger":True}])
    frame=pd.DataFrame(records)
    summaries=[]
    for (arm,w),g in frame.groupby(["arm","week"]): summaries.append({"arm":arm,"week":int(w),"n":len(g),"MAE":float(np.mean(np.abs(g.pred-g.target))),"RMSE":float(np.sqrt(np.mean((g.pred-g.target)**2))),"recalibrations":int(g.trigger.sum())})
    metrics=pd.DataFrame(summaries);frame.to_csv(out/"replay_predictions.csv",index=False);metrics.to_csv(out/"metrics_by_week.csv",index=False)
    paired={};rng=np.random.default_rng(config.RANDOM_STATE)
    if not frame.empty:
        for arm in ("B_on_trigger","C_every_week"):
            diff=patient_level_error_differences(frame,arm)
            try: stat=wilcoxon(diff) if len(diff)>0 and not np.allclose(diff,0) else None
            except ValueError: stat=None
            boot=np.asarray([float(rng.choice(diff.to_numpy(),size=len(diff),replace=True).mean()) for _ in range(1000)]) if len(diff) else np.array([])
            paired[arm]={"mean_paired_patient_mae_difference":float(diff.mean()) if len(diff) else None,
                         "mean_difference_cluster_bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))] if len(boot) else None,
                         "wilcoxon_pvalue":float(stat.pvalue) if stat else None,"n_patients":int(len(diff))}
    import db
    report={"origin":db.LAST_DATA_ORIGIN,"arms":["A static","B recalibrate when configured trigger fires","C recalibrate each week"],"paired_tests":paired,"pathway_change_rate":"not available: source schema contains no linked patient treatment/pathway assignment history","trajectory_rmse":"not evaluated: source schema does not include a calibrated predicted metric trajectory target; the app curve is hand-coded","interpretation":"Prediction-error replay only; no causal or patient-benefit inference."}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2,default=str));(out/"README.md").write_text("# Adaptive-loop evaluation\n\n"+json.dumps(report,indent=2,default=str)+"\n")
    return report

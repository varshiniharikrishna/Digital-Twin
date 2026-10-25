"""Standardized adverse-deviation analysis. Thresholds are engineering candidates."""
import numpy as np
import pandas as pd
import config

def sigma_from_residuals(residuals: pd.DataFrame):
    result={}
    if residuals is None: residuals=pd.DataFrame()
    for metric,floor in config.DEVIATION_SIGMA_FLOOR.items():
        vals=pd.to_numeric(residuals.get(metric,pd.Series(dtype=float)),errors="coerce").dropna()
        result[metric]=max(float(vals.std(ddof=1)) if len(vals)>1 else 0.0,float(floor))
    return result

def estimate_training_sigmas(tables, train, pct_model, weeks_model):
    """Estimate metric residual scales from training-patient weekly records only."""
    import simulation
    from data_prep import get_feature_frame
    weekly=tables.get("Weekly_Progress",pd.DataFrame());residuals={m:[] for m in config.DEVIATION_SIGMA_FLOOR}
    if weekly.empty or "Patient_ID" not in weekly:return sigma_from_residuals(pd.DataFrame())
    ids=set(train.get("Patient_ID",pd.Series(dtype=str)).astype(str));weekly=weekly[weekly.Patient_ID.astype(str).isin(ids)]
    # Batch inference once. Calling the parallel tree ensemble for each row
    # separately makes Flask startup needlessly slow on laptop-sized projects.
    x_train=get_feature_frame(train)
    pct_predictions=np.asarray(pct_model.predict(x_train),dtype=float)
    week_predictions=np.asarray(weeks_model.predict(x_train),dtype=float)
    weekly_by_patient={str(pid):group for pid,group in weekly.groupby(weekly.Patient_ID.astype(str),sort=False)}
    for pos,(_,row) in enumerate(train.iterrows()):
        if pd.isna(row.get(config.TARGET_RECOVERY_PCT)):continue
        curve=simulation.build_predicted_trajectory(row,float(pct_predictions[pos]),float(week_predictions[pos]))
        for _,obs in weekly_by_patient.get(str(row.Patient_ID),pd.DataFrame()).iterrows():
            w=int(obs.get("Week",1));pred=curve.iloc[min(max(w,1),len(curve))-1]
            for metric,actual_col,pred_col in [("Pain","Pain","Predicted_Pain"),("ROM_deg","ROM_deg","Predicted_ROM_deg"),("Strength_pct","Quadriceps_Strength_pct","Predicted_Strength_pct")]:
                actual=pd.to_numeric(pd.Series([obs.get(actual_col)]),errors="coerce").iloc[0]
                if pd.notna(actual):residuals[metric].append(float(actual)-float(pred[pred_col]))
    return sigma_from_residuals(pd.DataFrame(residuals))

def standardized_deviation(predicted, observed, sigma, metric):
    adverse = float(observed)-float(predicted) if metric == "Pain" else float(predicted)-float(observed)
    sd=max(float(sigma.get(metric,config.DEVIATION_SIGMA_FLOOR.get(metric,1.0))),1e-8)
    z=adverse/sd
    return {"z_adverse":float(z),"sigma":sd,"flagged":bool(z>config.DEVIATION_DELTA_SD),"delta_sd":config.DEVIATION_DELTA_SD}

def threshold_sweep(predicted: pd.DataFrame, observed: pd.DataFrame, labels, sigmas):
    """Return sensitivity/specificity for available paired labels; no truth labels => explicit empty result."""
    if labels is None or len(labels)==0:
        return pd.DataFrame(columns=["delta_sd","sensitivity","specificity","ppv","false_alarm_rate"])
    truth=np.asarray(labels,dtype=bool); scores=np.asarray(predicted,dtype=float)
    rows=[]
    for delta in config.DEVIATION_SD_GRID:
        pred=scores>delta
        tp=np.sum(pred&truth); fp=np.sum(pred&~truth); fn=np.sum(~pred&truth); tn=np.sum(~pred&~truth)
        rows.append({"delta_sd":delta,"sensitivity":tp/(tp+fn) if tp+fn else None,"specificity":tn/(tn+fp) if tn+fp else None,"ppv":tp/(tp+fp) if tp+fp else None,"false_alarm_rate":fp/(fp+tn) if fp+tn else None})
    return pd.DataFrame(rows)

def run_deviation_analysis(tables, out_dir="reports/deviation"):
    from pathlib import Path
    import json
    import config, modeling, simulation
    from data_prep import build_modelling_dataset, chronological_split, get_feature_frame
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    data=build_modelling_dataset(tables); train,test=chronological_split(data)
    weekly=tables.get("Weekly_Progress",pd.DataFrame())
    labeled=test[pd.to_numeric(test[config.TARGET_RECOVERY_PCT],errors="coerce").notna()].copy()
    if weekly.empty or labeled.empty:
        report={"status":"not evaluable","reason":"Timestamped weekly observations or observed final recovery labels are missing; no metrics invented."}
        (out/"deviation.json").write_text(__import__('json').dumps(report,indent=2));return report
    pm,wm=modeling.train_models(get_feature_frame(train),train[config.TARGET_RECOVERY_PCT],train[config.TARGET_RECOVERY_WEEKS])
    events=[]; residuals={m:[] for m in config.DEVIATION_SIGMA_FLOOR}
    for source, is_train in ((train,True),(labeled,False)):
      ids=set(source.Patient_ID.astype(str)); w=weekly[weekly.Patient_ID.astype(str).isin(ids)]
      for _,row in source.iterrows():
        if pd.isna(row.get(config.TARGET_RECOVERY_PCT)):continue
        curve=simulation.build_predicted_trajectory(row,float(pm.predict(get_feature_frame(pd.DataFrame([row])))[0]),float(wm.predict(get_feature_frame(pd.DataFrame([row])))[0]))
        for _,obs in w[w.Patient_ID.astype(str).eq(str(row.Patient_ID))].iterrows():
            wk=int(obs.get("Week",1)); pr=curve.iloc[min(max(wk,1),len(curve))-1]
            for metric,act_col,pred_col in [("Pain","Pain","Predicted_Pain"),("ROM_deg","ROM_deg","Predicted_ROM_deg"),("Strength_pct","Quadriceps_Strength_pct","Predicted_Strength_pct")]:
                actual=pd.to_numeric(pd.Series([obs.get(act_col)]),errors="coerce").iloc[0]
                if pd.notna(actual):
                    record={"Patient_ID":str(row.Patient_ID),"Week":wk,"Metric":metric,"actual":float(actual),"predicted":float(pr[pred_col]),"recovery_target":float(row[config.TARGET_RECOVERY_PCT])}
                    if is_train:residuals[metric].append(float(actual)-float(pr[pred_col]))
                    else:events.append(record)
    sig=sigma_from_residuals(pd.DataFrame(residuals));ev=pd.DataFrame(events)
    results=[]
    for delta in config.DEVIATION_SD_GRID:
        triggers=[]
        for _,e in ev.iterrows():
            z=standardized_deviation(e.predicted,e.actual,sig,e.Metric)["z_adverse"]
            triggers.append(z>delta)
        ev["trig"] = triggers
        by=ev.groupby("Patient_ID")["trig"].any()
        poor=(labeled.set_index(labeled.Patient_ID.astype(str))[config.TARGET_RECOVERY_PCT]<80)
        joined=pd.concat([by.rename("trigger"),poor.rename("poor")],axis=1).dropna(); tp=((joined.trigger)&(joined.poor)).sum();fp=((joined.trigger)&(~joined.poor)).sum();fn=((~joined.trigger)&(joined.poor)).sum();tn=((~joined.trigger)&(~joined.poor)).sum()
        results.append({"delta_sd":delta,"sensitivity":float(tp/(tp+fn)) if tp+fn else None,"specificity":float(tn/(tn+fp)) if tn+fp else None,"ppv":float(tp/(tp+fp)) if tp+fp else None,"false_alarm_rate":float(fp/(fp+tn)) if fp+tn else None,"n_patients":int(len(joined))})
    old=[]
    for threshold in config.DEVIATION_OLD_PCT_GRID:
        flag=ev.apply(lambda e: simulation.check_deviation(pd.Series({"Predicted_"+{"Pain":"Pain","ROM_deg":"ROM_deg","Strength_pct":"Strength_pct"}[e.Metric]:e.predicted}),{("Pain" if e.Metric=="Pain" else "ROM_deg" if e.Metric=="ROM_deg" else "Strength_pct"):e.actual},threshold_pct=threshold,method="percentage").get(e.Metric,{}).get("flagged",False),axis=1)
        old.append({"threshold_pct":threshold,"flag_rate":float(flag.mean()) if len(flag) else None,"n_observations":int(len(flag))})
    import db
    report={"origin":db.LAST_DATA_ORIGIN,
            "poor_recovery_label":"Final_Recovery_Pct < 80; engineering label only, not a validated clinical endpoint",
            "sigma_by_metric":sig,"sigma_estimation":"SD of training-patient weekly residuals against the app's hand-coded trajectory; in-sample and exploratory, not a validated per-week residual calibration",
            "sd_threshold_sweep":results,"old_percentage_threshold_sweep":old,
            "interpretation":"These are associations/trigger operating characteristics on the supplied dataset, not clinical sensitivity/specificity or treatment benefit."}
    ev.to_csv(out/"weekly_deviation_records.csv",index=False);pd.DataFrame(results).to_csv(out/"sd_threshold_sweep.csv",index=False);pd.DataFrame(old).to_csv(out/"percentage_threshold_sweep.csv",index=False)
    (out/"deviation.json").write_text(json.dumps(report,indent=2,default=str))
    try:
        import plotly.express as px
        pd.DataFrame(results).to_csv(out/"roc_pr_inputs.csv",index=False)
    except Exception: pass
    (out/"README.md").write_text("# Deviation analysis\n\n"+json.dumps(report,indent=2,default=str)+"\n")
    return report

"""Held-out within-patient comparison of fitted exponential and hand-coded asymptote."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from twin_state import asymptotic

def run(weekly,out_dir="reports/digital_twin"):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
    fields={"Pain":"Pain","ROM_deg":"ROM_deg","Strength_pct":"Quadriceps_Strength_pct","Walking_Speed_mps":"Walking_Speed_mps"}
    for pid,g in weekly.groupby("Patient_ID"):
        g=g.sort_values("Week")
        if len(g)<5:continue
        split=max(3,int(len(g)*.7));train=g.iloc[:split];test=g.iloc[split:]
        t=train.Week.to_numpy(float);tt=test.Week.to_numpy(float)
        for metric,col in fields.items():
            if col not in g:continue
            y=pd.to_numeric(train[col],errors="coerce");ok=y.notna(); y=y[ok];tx=t[ok.to_numpy()]
            yt=pd.to_numeric(test[col],errors="coerce");valid=yt.notna()
            if len(y)<3 or not valid.any():continue
            try:
                p,_=curve_fit(lambda x,a,k:asymptotic(x,y.iloc[0],a,k),tx-tx.min(),y.to_numpy(),p0=[y.iloc[-1],.1],bounds=([-1000,.001],[1000,3]),maxfev=5000)
                pred_fit=asymptotic(tt[valid]-tx.min(),y.iloc[0],p[0],p[1])
                # Hand-coded comparator uses fixed 90% convergence horizon fitted from the observed training endpoint.
                target=y.iloc[-1];horizon=max(float(tx[-1]-tx[0]),1);rate=-np.log(.1)/horizon
                pred_hand=asymptotic(tt[valid]-tx[0],y.iloc[0],target,rate)
                for model,pred in [("fitted_exponential",pred_fit),("handcoded_training_endpoint",pred_hand)]:rows.append({"Patient_ID":pid,"metric":metric,"model":model,"n_test_weeks":int(valid.sum()),"RMSE":float(np.sqrt(mean_squared_error(yt[valid],pred)))})
            except (RuntimeError,ValueError):continue
    frame=pd.DataFrame(rows);frame.to_csv(out/"trajectory_metrics.csv",index=False)
    summary=frame.groupby(["metric","model"]).agg(n_patients=("Patient_ID","nunique"),mean_RMSE=("RMSE","mean"),sd_RMSE=("RMSE","std")).reset_index() if not frame.empty else pd.DataFrame()
    summary.to_csv(out/"trajectory_summary.csv",index=False)
    report={"status":"evaluated" if not frame.empty else "not evaluable","n_heldout_patient_metric_comparisons":len(frame),"summary":summary.to_dict(orient="records"),"caveat":"Within-patient temporal split of bundled weekly data; bundled data provenance is synthetic. Not clinical validation."}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2,default=str));return report

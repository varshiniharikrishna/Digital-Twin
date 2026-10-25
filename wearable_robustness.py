"""Demo-only stress test of wearable aggregation under noise/missingness."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sensors.simulate_wearable_stream import generate_demo_stream
from sensors.ingest_wearable import aggregate_weekly

def run(out_dir="reports/wearable_robustness"):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);base=generate_demo_stream(days=84);baseline=aggregate_weekly(base);rows=[];rng=np.random.default_rng(42)
    for noise in (0,.05,.15,.30):
      for missing in (0,.1,.3,.5):
        d=base.copy()
        for c in ["knee_flexion_angle_deg","gait_speed_mps","cadence_spm","adherence"]:
            scale=max(float(d[c].std()),1e-6);d[c]=d[c]+rng.normal(0,noise*scale,len(d))
            if missing:d.loc[rng.random(len(d))<missing,c]=np.nan
        a=aggregate_weekly(d);joined=baseline.merge(a,on=["Patient_ID","week_start"],suffixes=("_base","_pert"))
        for c in ["knee_flexion_angle_deg","gait_speed_mps","cadence_spm","adherence"]:
            rows.append({"noise_sd_fraction":noise,"missing_fraction":missing,"metric":c,"weekly_mean_abs_change":float((joined[c+"_pert"]-joined[c+"_base"]).abs().mean()),"observed_fraction":float(a[c].notna().mean())})
    pd.DataFrame(rows).to_csv(out/"robustness.csv",index=False);report={"origin":"SIMULATED DEMO ONLY","n_days":len(base),"n_rows":len(rows),"interpretation":"Perturbation/aggregation sensitivity only; no patients, clinical endpoint, or clinical robustness claim."};(out/"evaluation.json").write_text(json.dumps(report,indent=2));return report

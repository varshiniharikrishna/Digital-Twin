"""CSV/JSON wearable ingest, validation, robust filtering and weekly aggregation; MQTT is a stub."""
import json
from pathlib import Path
import pandas as pd
METRICS={"knee_flexion_angle_deg":(-20,180),"step_count":(0,100000),"gait_speed_mps":(0,5),"cadence_spm":(0,300),"adherence":(0,1)}
def load_stream(path):
    p=Path(path)
    if p.suffix.lower()==".csv":df=pd.read_csv(p)
    elif p.suffix.lower()==".json":df=pd.DataFrame(json.loads(p.read_text()))
    else:raise ValueError("Only CSV and JSON stream files are supported. MQTT adapter is a documented stub.")
    return validate_stream(df)
def validate_stream(df):
    required={"Patient_ID","timestamp"}
    if not required.issubset(df):raise ValueError("Wearable stream requires Patient_ID and timestamp.")
    d=df.copy();d["timestamp"]=pd.to_datetime(d.timestamp,errors="coerce")
    if d.timestamp.isna().any():raise ValueError("Invalid timestamp in wearable stream.")
    for col,(lo,hi) in METRICS.items():
        if col in d:
            d[col]=pd.to_numeric(d[col],errors="coerce"); d.loc[~d[col].between(lo,hi),col]=float("nan")
            d[col]=d.groupby("Patient_ID")[col].transform(lambda s:s.where((s-s.rolling(7,center=True,min_periods=1).median()).abs() <= 3*s.rolling(7,center=True,min_periods=1).std().fillna(float("inf"))))
    return d
def aggregate_weekly(df):
    d=validate_stream(df);d["week_start"]=d.timestamp.dt.to_period("W").dt.start_time
    cols=[c for c in METRICS if c in d]
    if not cols:return pd.DataFrame(columns=["Patient_ID","week_start"])
    agg={c:("sum" if c=="step_count" else "mean") for c in cols}
    return d.groupby(["Patient_ID","week_start"],as_index=False).agg(agg)
def mqtt_stub(*_args,**_kwargs):raise NotImplementedError("MQTT transport is intentionally a stub; implement authenticated TLS, device identity and consent controls before deployment.")

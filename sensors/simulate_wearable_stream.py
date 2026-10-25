"""Generate a clearly labelled simulated wearable stream for UI/demo and tests only."""
import numpy as np
import pandas as pd

def generate_demo_stream(patient_id="SIMULATED-001", days=28, seed=42, start="2026-01-01"):
    rng=np.random.default_rng(seed);ts=pd.date_range(start,periods=days,freq="D")
    n=np.arange(days)
    return pd.DataFrame({"Patient_ID":patient_id,"timestamp":ts,"knee_flexion_angle_deg":np.clip(65+1.1*n+rng.normal(0,5,days),0,145),
      "step_count":np.clip(2500+90*n+rng.normal(0,500,days),0,20000).astype(int),"gait_speed_mps":np.clip(.55+.012*n+rng.normal(0,.06,days),0,2),
      "cadence_spm":np.clip(70+.6*n+rng.normal(0,5,days),0,180),"adherence":rng.uniform(.4,1,days),"data_origin":"SIMULATED DEMO ONLY"})

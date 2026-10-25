import json
import numpy as np
import pandas as pd
import pytest
from privacy import pseudonymize,append_audit_event,k_anonymity,reject_direct_identifiers
from sensors.simulate_wearable_stream import generate_demo_stream
from sensors.ingest_wearable import aggregate_weekly,validate_stream
from federated_demo import run_federated_demo

def test_privacy_helpers(tmp_path):
    assert pseudonymize("patient-1","salt")!=pseudonymize("patient-1","other")
    with pytest.raises(ValueError):pseudonymize("x","")
    with pytest.raises(ValueError):reject_direct_identifiers(pd.DataFrame(columns=["PatientName"]))
    assert k_anonymity(pd.DataFrame({"age":[20,20,30],"sex":["F","F","M"]}),["age","sex"],2)["passed"] is False
    p=tmp_path/"audit.jsonl";append_audit_event(p,"hash-user","prediction","hash-patient",{"role":"DOCTOR","note":"PHI"})
    event=json.loads(p.read_text());assert event["metadata"]=={"role":"DOCTOR"}

def test_wearable_simulation_is_labelled_and_aggregates():
    demo=generate_demo_stream(days=14);assert set(demo.data_origin)=={"SIMULATED DEMO ONLY"}
    weekly=aggregate_weekly(demo);assert len(weekly)>=2 and "gait_speed_mps" in weekly
    with pytest.raises(ValueError):validate_stream(pd.DataFrame({"Patient_ID":["x"],"timestamp":["bad"]}))

def test_federated_demo_runs_and_declares_simulation():
    rng=np.random.default_rng(42);x=rng.normal(size=(40,3));y=2*x[:,0]-x[:,1]
    r=run_federated_demo(x,y,k=3,seed=42,rounds=3)
    assert r["origin"].startswith("SIMULATED") and set(r["mse"])=={"centralized","federated","local_only"}

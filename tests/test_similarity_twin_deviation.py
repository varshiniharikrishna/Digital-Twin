import numpy as np
import pandas as pd
import config
from similarity import SimilarityModel,find_similar_patients
from twin_state import DigitalTwin,fit_hierarchical_priors
from deviation_analysis import standardized_deviation, estimate_training_sigmas
from pathway_ranking import rank_pathways,weight_sensitivity
from explainability import check_additivity
from simulation import build_predicted_trajectory

def train_frame():
    return pd.DataFrame({"Patient_ID":[f"P{i}" for i in range(8)],"Age":range(20,60,5),"BMI":[20,22,24,26,28,30,32,34],"Pain_Score":[8,7,6,5,4,3,2,1],"Primary_ROM_deg":[40,50,60,70,80,90,100,110],"Primary_Strength_pct":[20,30,40,50,60,70,80,90],"Balance_Score":[20,30,40,50,60,70,80,90],"Gender":["F","M"]*4,"Injury_Type":["knee"]*8,"Severity":["moderate"]*8,"Diabetes":[0]*8,"Smoking_Status":["No"]*8,"Final_Recovery_Pct":[40,45,50,55,60,65,70,75]})

def test_similarity_scaler_is_pool_independent_and_serializable(tmp_path):
    f=train_frame();m=SimilarityModel.fit(f,"Final_Recovery_Pct");q=f.iloc[-1];c=f.iloc[:-1]
    a=m.score(q,c)[0];b=m.score(q,c.iloc[:3])[0]
    assert a==b
    p=tmp_path/"model.json";m.save(p);loaded=SimilarityModel.load(p)
    assert np.allclose(m.score(q,c),loaded.score(q,c))
    n=find_similar_patients(q,f,model=m);assert q.Patient_ID not in set(n.Patient_ID.astype(str))
    assert n.attrs["confidence"] in {"sufficient cohort","low-confidence / insufficient cohort"}

def test_twin_updates_selected_week_and_shrinks_observation():
    row=train_frame().iloc[0].copy();row["Range_of_Motion_deg"]=40;row["Quadriceps_Strength_pct"]=20
    twin=DigitalTwin.from_baseline(row);prior=twin.state["Pain"]
    twin.update({"Pain":prior-2,"ROM_deg":50},week=3)
    assert twin.week==3 and twin.state["Pain"]<prior and twin.state["recovery_progress"]>0
    assert twin.history[-1]["innovations"]

def test_wearable_mode_missing_pain_and_strength_is_safe():
    row=train_frame().iloc[0].copy();row["Range_of_Motion_deg"]=40;row["Quadriceps_Strength_pct"]=20
    twin=DigitalTwin.from_baseline(row)
    old_pain=twin.state["Pain"];old_strength=twin.state["Strength_pct"]
    innovations=twin.update({"Pain":None,"Strength_pct":None,"ROM_deg":55,"Walking_Speed_mps":1.0},week=2)
    assert "Pain" not in innovations and "Strength_pct" not in innovations
    assert twin.state["Pain"] <= old_pain and twin.state["Strength_pct"] >= old_strength

def test_predicted_trajectory_advances_shared_twin_and_does_not_fill_missing_state():
    row=pd.Series({"Injury_Type":"ACL Tear","Pain_Score":6.0,"Range_of_Motion_deg":70.0,
                   "Quadriceps_Strength_pct":45.0,"Balance_Score":55.0})
    result=build_predicted_trajectory(row,80.0,3.0)
    assert len(result)==3
    assert result["Predicted_Pain"].iloc[-1] < result["Predicted_Pain"].iloc[0]
    assert result["Predicted_ROM_deg"].iloc[-1] > result["Predicted_ROM_deg"].iloc[0]
    assert result["Predicted_Walking_Speed_mps"].isna().all()

def test_hierarchical_prior_uses_training_weekly_records():
    rows=[]
    for pid,start in [("A",40),("B",45)]:
        for week in range(1,6):rows.append({"Patient_ID":pid,"Week":week,"ROM_deg":start+week*3,"Pain":8-week*.6,"Quadriceps_Strength_pct":30+week*4})
    priors=fit_hierarchical_priors(pd.DataFrame(rows),["A"])
    assert priors["ROM_deg"]["n_patients"]==1 and priors["ROM_deg"]["source"].startswith("training-patient")

def test_deviation_direction_and_pathway_no_effect_invention():
    assert standardized_deviation(3,5,{"Pain":1},"Pain")["z_adverse"]==2
    assert standardized_deviation(50,45,{"Strength_pct":2},"Strength_pct")["flagged"]
    assert rank_pathways({},None,[{"Pathway_ID":"A"}])==[]
    assert weight_sensitivity({"a":["A","B"],"b":["A","B"]})[0]["kendall_tau"]==1

def test_training_sigma_estimation_batches_model_predictions():
    class ConstantModel:
        def __init__(self,value):self.value=value;self.calls=0
        def predict(self,x):self.calls+=1;return np.full(len(x),self.value)
    train=pd.DataFrame({"Patient_ID":["A"],"Pain_Score":[6],"Range_of_Motion_deg":[50],
        "Quadriceps_Strength_pct":[40],"Final_Recovery_Pct":[70]})
    weekly=pd.DataFrame({"Patient_ID":["A"],"Week":[1],"Pain":[7],"ROM_deg":[52],"Quadriceps_Strength_pct":[42]})
    pct,weeks=ConstantModel(75),ConstantModel(2)
    # A single-patient set cannot be partitioned for OOF, so fail closed to configured floors.
    sigmas=estimate_training_sigmas({"Weekly_Progress":weekly},train,pct,weeks)
    assert pct.calls==0 and weeks.calls==0
    assert sigmas["estimation_method"] == "configured floors; insufficient patient groups for OOF"
    assert sigmas["by_week_bucket"] == {}

def test_deviation_uses_week_bucket_specific_oof_sigma():
    from simulation import check_deviation
    predicted = pd.Series({"Week": 7, "Predicted_Pain": 5.0})
    sigmas = {"Pain": 4.0, "by_week_bucket": {"Pain": {"1-4": 1.0, "5-8": 2.0, "9+": 3.0}}}
    result = check_deviation(predicted, {"Pain": 8.0}, residual_sigmas=sigmas, week=7)
    assert result["Pain"]["sigma"] == 2.0

def test_sigma_estimation_fits_only_patient_heldout_oof_predictions(monkeypatch):
    import modeling
    class MeanModel:
        def __init__(self, value): self.value = value
        def predict(self, X): return np.full(len(X), self.value)
    calls=[]
    def fake_train(X, y_pct, y_weeks):
        calls.append(set(X.index))
        return MeanModel(float(y_pct.mean())), MeanModel(float(y_weeks.mean()))
    monkeypatch.setattr(modeling, "train_models", fake_train)
    rows=[];observations=[]
    for i in range(6):
        rows.append({"Patient_ID": f"P{i}", "Age": 20+i*5, "Injury_Type": "ACL Tear", "Severity": "Moderate",
                     "Pain_Score": 8-i*.5, "Range_of_Motion_deg": 40+i*5,
                     "Quadriceps_Strength_pct": 20+i*8, "Balance_Score": 30+i*8,
                     "Walking_Speed_mps": .5+i*.1, "Final_Recovery_Pct": 50+i*5,
                     "Actual_Recovery_Time_weeks": 25+i})
        observations.append({"Patient_ID": f"P{i}", "Week": 2, "Pain": 7-i*.4,
                             "ROM_deg": 45+i*5, "Quadriceps_Strength_pct": 25+i*8})
    sigma=estimate_training_sigmas({"Weekly_Progress": pd.DataFrame(observations)},pd.DataFrame(rows),None,None)
    assert len(calls) == 5
    assert sigma["estimation_method"].startswith("GroupKFold patient-held-out")
    assert sigma["by_week_bucket"]["Pain"]["1-4"] >= config.DEVIATION_SIGMA_FLOOR["Pain"]

def test_shap_attributions_sum_to_prediction_minus_base():
    check=check_additivity([1.25,-.5,.25],2.0,3.0,tolerance=1e-12)
    assert check["passed"] is True and abs(check["error"])<1e-12

"""
server.py
---------
Flask REST API Backend for the Dynamic Digital Twin Web Application.
"""

import os
import io
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template, send_file

import config
from db import load_all_tables
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame
import modeling
import similarity as sim
import simulation as simmod
import pdf_generator

app = Flask(__name__, template_folder="templates", static_folder="static")

print("Initializing Digital Twin backend and training models...")
TABLES = load_all_tables()
FULL_DATASET = build_modelling_dataset(TABLES)
TRAIN_DATA, TEST_DATA = chronological_split(FULL_DATASET)

X_TRAIN = get_feature_frame(TRAIN_DATA)
X_TEST = get_feature_frame(TEST_DATA)

PCT_MODEL, WEEKS_MODEL = modeling.train_models(
    X_TRAIN, TRAIN_DATA[config.TARGET_RECOVERY_PCT], TRAIN_DATA[config.TARGET_RECOVERY_WEEKS]
)

PRED_PCT, PRED_WEEKS = modeling.predict(PCT_MODEL, WEEKS_MODEL, X_TEST)
TEST_DATA = TEST_DATA.copy()
TEST_DATA["Predicted_Recovery_Pct"] = PRED_PCT
TEST_DATA["Predicted_Recovery_Weeks"] = PRED_WEEKS

METRICS_PCT = modeling.evaluate(TEST_DATA[config.TARGET_RECOVERY_PCT], PRED_PCT)
METRICS_WEEKS = modeling.evaluate(TEST_DATA[config.TARGET_RECOVERY_WEEKS], PRED_WEEKS)
FEATURE_IMPORTANCE = modeling.get_feature_importance(PCT_MODEL)

print("Backend initialization complete!")


def sanitize_val(v):
    if isinstance(v, (np.integer, np.int64)):
        return int(v)
    elif isinstance(v, (np.floating, np.float64)):
        return None if pd.isna(v) or np.isnan(v) or np.isinf(v) else float(v)
    elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    elif isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    elif pd.isna(v):
        return None
    return v

def sanitize_dict(d):
    if isinstance(d, dict):
        return {k: sanitize_val(v) if not isinstance(v, (dict, list)) else (sanitize_dict(v) if isinstance(v, dict) else [sanitize_val(item) if not isinstance(item, dict) else sanitize_dict(item) for item in v]) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(item) if isinstance(item, (dict, list)) else sanitize_val(item) for item in d]
    return sanitize_val(d)


def get_patient_row(patient_id):
    match = TEST_DATA[TEST_DATA["Patient_ID"] == patient_id]
    if match.empty:
        return None
    return match.iloc[0]


def predict_row(row: pd.Series):
    X = get_feature_frame(pd.DataFrame([row]))
    pct = float(PCT_MODEL.predict(X)[0])
    weeks = float(WEEKS_MODEL.predict(X)[0])
    return pct, weeks


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/patients", methods=["GET"])
def get_patients():
    patient_ids = sorted(TEST_DATA["Patient_ID"].unique().tolist())
    return jsonify({"patients": patient_ids})


@app.route("/api/patient/<patient_id>", methods=["GET"])
def get_patient_details(patient_id):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    pred_pct, pred_weeks = predict_row(row)
    
    similar_df = sim.find_similar_patients(row, TRAIN_DATA, top_k=5)
    similar_list = similar_df[[
        "Patient_ID", "Similarity_Score", "Age", "Gender", "Injury_Type",
        "Severity", config.TARGET_RECOVERY_PCT, config.TARGET_RECOVERY_WEEKS
    ]].to_dict(orient="records")

    patient_dict = row.to_dict()

    return jsonify(sanitize_dict({
        "patient": patient_dict,
        "prediction": {
            "predicted_pct": round(pred_pct, 1),
            "predicted_weeks": round(pred_weeks, 1),
            "r2_score": round(METRICS_PCT["R2"], 3),
            "rmse_score": round(METRICS_PCT["RMSE"], 2)
        },
        "similar_patients": similar_list
    }))


@app.route("/api/patient/<patient_id>/weekly", methods=["GET"])
def get_patient_weekly(patient_id):
    actual_df = simmod.get_actual_trajectory(TABLES["Weekly_Progress"], patient_id)
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    pred_pct, pred_weeks = predict_row(row)
    predicted_df = simmod.build_predicted_trajectory(row, pred_pct, pred_weeks)

    return jsonify(sanitize_dict({
        "actual": actual_df.to_dict(orient="records"),
        "predicted": predicted_df.to_dict(orient="records"),
        "max_week": int(actual_df["Week"].max()) if not actual_df.empty else 1
    }))


@app.route("/api/patient/<patient_id>/week/<int:week>", methods=["GET"])
def get_patient_week_detail(patient_id, week):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    actual_df = simmod.get_actual_trajectory(TABLES["Weekly_Progress"], patient_id)
    week_match = actual_df[actual_df["Week"] == week]
    if week_match.empty:
        return jsonify({"error": f"No data recorded for week {week}"}), 404

    week_data = week_match.iloc[0].to_dict()

    # Column aliases normalization
    rom_val = float(week_data.get("ROM_deg", week_data.get("Primary_ROM_deg", 0.0)))
    quad_val = float(week_data.get("Quadriceps_Strength_pct", week_data.get("Primary_Strength_pct", 0.0)))
    ham_val = float(week_data.get("Hamstring_Strength_pct", week_data.get("Secondary_Strength_pct", quad_val)))
    balance_val = float(week_data.get("Balance_Score", week_data.get("Function_Score", 0.0)))
    speed_val = float(week_data.get("Walking_Speed_mps", 0.0)) if week_data.get("Walking_Speed_mps") is not None and not pd.isna(week_data.get("Walking_Speed_mps")) else 0.0

    week_data["ROM_deg"] = rom_val
    week_data["Quadriceps_Strength_pct"] = quad_val
    week_data["Hamstring_Strength_pct"] = ham_val
    week_data["Balance_Score"] = balance_val
    week_data["Walking_Speed_mps"] = speed_val

    # Predicted comparison row
    pred_pct, pred_weeks = predict_row(row)
    predicted_df = simmod.build_predicted_trajectory(row, pred_pct, pred_weeks)
    pred_week_match = predicted_df[predicted_df["Week"] == week]
    
    dev_dict = {}
    if not pred_week_match.empty:
        p_row = pred_week_match.iloc[0]
        cur_st = {
            "Pain": float(week_data["Pain"]),
            "ROM_deg": rom_val,
            "Strength_pct": quad_val
        }
        dev_dict = simmod.check_deviation(p_row, cur_st)

    # Muscle compensation event
    comp_df = TABLES["Muscle_Compensation"]
    comp_match = comp_df[(comp_df["Patient_ID"] == patient_id) & (comp_df["Week"] == week)]
    comp_event = comp_match.iloc[0].to_dict() if not comp_match.empty else None

    # Flag pressure points
    pressure_points = []
    if float(week_data["Pain"]) > 5.0:
        pressure_points.append({"Category": "High Pain Alert", "Actual": f"{week_data['Pain']}/10", "Flag": "Pain > 5.0 (Moderate-Severe)"})
    if rom_val < 90.0:
        pressure_points.append({"Category": "Low ROM Restrict", "Actual": f"{rom_val}°", "Flag": "ROM < 90° (Flexion deficit)"})
    if quad_val < 60.0:
        pressure_points.append({"Category": "Strength Deficit", "Actual": f"{quad_val}%", "Flag": "Strength < 60% (Atrophy risk)"})
    if balance_val < 60.0 and balance_val > 0:
        pressure_points.append({"Category": "Impaired Function", "Actual": f"{balance_val}", "Flag": "Function/Balance < 60"})
    if float(week_data.get("Compliance_pct", 100.0)) < 70.0:
        pressure_points.append({"Category": "Low Compliance", "Actual": f"{week_data['Compliance_pct']}%", "Flag": "Compliance < 70%"})
    if week_data.get("Swelling") in ["Severe", "Moderate"]:
        pressure_points.append({"Category": "Joint Swelling", "Actual": str(week_data['Swelling']), "Flag": "Persistent inflammation"})
    if comp_event:
        pressure_points.append({"Category": "Muscle Compensation", "Actual": f"{comp_event.get('Muscle_Overloaded', 'Muscle')} overloaded", "Flag": f"{comp_event.get('Compensation_Level', 'Moderate')} level risk"})

    # Active therapy plan
    tplan = TABLES["Therapy_Plan"]
    week_plan = tplan[(tplan["Patient_ID"] == patient_id) & (tplan["Week_Number"] == week)]
    plan_list = []
    if not week_plan.empty:
        elib = TABLES["Exercise_Library"].set_index("Exercise_ID")
        plan_merged = week_plan.merge(elib, left_on="Exercise_ID", right_index=True, how="left")
        
        target_col = "Target_Muscle_or_Function" if "Target_Muscle_or_Function" in plan_merged.columns else ("Target_Muscle" if "Target_Muscle" in plan_merged.columns else "Target_Region")
        plan_merged["Target_Muscle"] = plan_merged[target_col].fillna("Full Body")
        
        for c in ["Sets", "Repetitions", "Duration_min", "Therapist_Notes"]:
            if c not in plan_merged.columns:
                plan_merged[c] = "-"
                
        plan_list = plan_merged[["Exercise_Name", "Target_Muscle", "Exercise_Category", "Sets", "Repetitions", "Duration_min", "Therapist_Notes"]].to_dict(orient="records")

    return jsonify(sanitize_dict({
        "week_data": week_data,
        "expected_vs_actual": dev_dict,
        "muscle_compensation": comp_event,
        "pressure_points": pressure_points,
        "therapy_plan": plan_list
    }))


@app.route("/api/patient/<patient_id>/clones", methods=["GET"])
def get_patient_clones(patient_id):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    pred_pct, pred_weeks = predict_row(row)
    clones = simmod.build_treatment_clones(row, pred_pct, pred_weeks)

    clones_serialized = []
    for cl in clones:
        cl_copy = dict(cl)
        cl_copy["trajectory"] = cl["trajectory"].to_dict(orient="records")
        clones_serialized.append(cl_copy)

    return jsonify(sanitize_dict({"clones": clones_serialized}))


@app.route("/api/patient/<patient_id>/recalibrate", methods=["POST"])
def recalibrate_patient(patient_id):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    data = request.json or {}
    week = int(data.get("week", 1))
    cur_st = {
        "Pain": float(data.get("pain", 6.0)),
        "ROM_deg": float(data.get("rom", 75.0)),
        "Strength_pct": float(data.get("strength", 50.0))
    }

    pred_pct, pred_weeks = predict_row(row)
    predicted_df = simmod.build_predicted_trajectory(row, pred_pct, pred_weeks)
    pred_week_match = predicted_df[predicted_df["Week"] == week]
    p_row = pred_week_match.iloc[0] if not pred_week_match.empty else predicted_df.iloc[0]

    dev_dict = simmod.check_deviation(p_row, cur_st)
    res = simmod.recalibrate(
        row, cur_st, week, TRAIN_DATA, TABLES["Recovery_Pathway"],
        PCT_MODEL, WEEKS_MODEL, get_feature_frame, dev_dict
    )

    new_neighbors = res["new_neighbors"][[
        "Patient_ID", "Similarity_Score", config.TARGET_RECOVERY_PCT, config.TARGET_RECOVERY_WEEKS
    ]].to_dict(orient="records")

    rec_pw = res["recommended_pathway"].to_dict() if res["recommended_pathway"] is not None else None

    return jsonify(sanitize_dict({
        "old_predicted_pct": round(pred_pct, 1),
        "old_predicted_weeks": round(pred_weeks, 1),
        "new_predicted_pct": round(res["new_predicted_pct"], 1),
        "new_predicted_weeks": round(res["new_predicted_weeks"], 1),
        "explanation": res["explanation"],
        "new_neighbors": new_neighbors,
        "recommended_pathway": rec_pw,
        "deviation": dev_dict
    }))


@app.route("/api/treatment/taxonomy", methods=["GET"])
def get_treatment_taxonomy():
    return jsonify(sanitize_dict(simmod.TREATMENT_TAXONOMY))


@app.route("/api/patient/<patient_id>/treatment/simulate", methods=["POST"])
def simulate_treatment_route(patient_id):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    data = request.json or {}
    category = str(data.get("category", "rom"))
    exercise_type = str(data.get("exercise_type", "active_assisted_rom"))
    stage = str(data.get("stage", "Early Stage"))
    intensity = str(data.get("intensity", "Standard"))
    start_week = int(data.get("start_week", 4))
    duration_weeks = int(data.get("duration_weeks", 4))

    res = simmod.simulate_treatment_intervention(
        patient_row=row,
        category=category,
        exercise_type=exercise_type,
        stage=stage,
        intensity=intensity,
        start_week=start_week,
        duration_weeks=duration_weeks
    )
    return jsonify(sanitize_dict(res))


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    top_fi = FEATURE_IMPORTANCE.head(12).to_dict(orient="records")
    return jsonify(sanitize_dict({
        "metrics_pct": METRICS_PCT,
        "metrics_weeks": METRICS_WEEKS,
        "feature_importance": top_fi
    }))


import db

def sanitize_val(v):
    if isinstance(v, (np.integer, np.int64)):
        return int(v)
    elif isinstance(v, (np.floating, np.float64)):
        return None if pd.isna(v) else float(v)
    elif isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    elif isinstance(v, float) and np.isnan(v):
        return None
    return v

def sanitize_dict(d):
    if isinstance(d, dict):
        return {k: sanitize_val(v) if not isinstance(v, (dict, list)) else (sanitize_dict(v) if isinstance(v, dict) else [sanitize_val(item) if not isinstance(item, dict) else sanitize_dict(item) for item in v]) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(item) if isinstance(item, (dict, list)) else sanitize_val(item) for item in d]
    return sanitize_val(d)


@app.route("/api/patient/<patient_id>/weekly/save", methods=["POST"])
def save_weekly_progress(patient_id):
    global FULL_DATASET, TRAIN_DATA, TEST_DATA, X_TEST

    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    data = request.json or {}
    week = int(data.get("week", 1))

    record = {
        "Patient_ID": patient_id,
        "Week": week,
        "Pain": float(data.get("pain", 5.0)),
        "ROM_deg": float(data.get("rom", 90.0)),
        "Quadriceps_Strength_pct": float(data.get("quad_strength", 60.0)),
        "Hamstring_Strength_pct": float(data.get("ham_strength", 60.0)),
        "Balance_Score": float(data.get("balance", 70.0)),
        "Walking_Speed_mps": float(data.get("walking_speed", 1.0)),
        "Swelling": str(data.get("swelling", "None")),
        "Compliance_pct": float(data.get("compliance", 90.0)),
        "Patient_Feedback": str(data.get("patient_feedback", "Normal recovery")),
        "Clinician_Remark": str(data.get("clinician_remark", "On track"))
    }

    # Update in-memory DataFrame
    wp = TABLES["Weekly_Progress"]
    mask = (wp["Patient_ID"] == patient_id) & (wp["Week"] == week)
    if mask.any():
        for k, v in record.items():
            if k in wp.columns:
                wp.loc[mask, k] = v
    else:
        TABLES["Weekly_Progress"] = pd.concat([wp, pd.DataFrame([record])], ignore_index=True)

    # Execute Weekly Stage Dynamic Recalibration
    stage_recal = simmod.process_weekly_stage_recalibration(
        patient_row=row,
        week=week,
        weekly_entry=record,
        train_pool=TRAIN_DATA,
        pathways_table=TABLES["Recovery_Pathway"],
        pct_model=PCT_MODEL,
        weeks_model=WEEKS_MODEL,
        feature_fn=get_feature_frame
    )

    # Persist progress & recalibration log to DB or CSV
    db.save_weekly_progress_record(record)
    db.save_patient_recalibration_record(patient_id, week, stage_recal)

    # Re-sync dataset state if week 1 is modified
    if week == 1:
        FULL_DATASET = build_modelling_dataset(TABLES)
        TRAIN_DATA, TEST_DATA = chronological_split(FULL_DATASET)
        X_TEST = get_feature_frame(TEST_DATA)
        PRED_PCT, PRED_WEEKS = modeling.predict(PCT_MODEL, WEEKS_MODEL, X_TEST)
        TEST_DATA = TEST_DATA.copy()
        TEST_DATA["Predicted_Recovery_Pct"] = PRED_PCT
        TEST_DATA["Predicted_Recovery_Weeks"] = PRED_WEEKS

    return jsonify(sanitize_dict({
        "success": True,
        "message": f"Weekly progress & recalibration saved for patient {patient_id} week {week}",
        "record": record,
        "recalibration": stage_recal
    }))


@app.route("/api/patient/<patient_id>/recalibration_history", methods=["GET"])
def get_recalibration_history_route(patient_id):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    history = db.get_recalibration_history_records(patient_id)
    return jsonify(sanitize_dict({"history": history}))


@app.route("/api/patient/<patient_id>/recalibrate/save", methods=["POST"])
def save_recalibration_route(patient_id):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    data = request.json or {}
    week = int(data.get("week", 1))
    cur_st = {
        "Pain": float(data.get("pain", 6.0)),
        "ROM_deg": float(data.get("rom", 75.0)),
        "Strength_pct": float(data.get("strength", 50.0))
    }

    pred_pct, pred_weeks = predict_row(row)
    predicted_df = simmod.build_predicted_trajectory(row, pred_pct, pred_weeks)
    pred_week_match = predicted_df[predicted_df["Week"] == week]
    p_row = pred_week_match.iloc[0] if not pred_week_match.empty else predicted_df.iloc[0]

    dev_dict = simmod.check_deviation(p_row, cur_st)
    res = simmod.recalibrate(
        row, cur_st, week, TRAIN_DATA, TABLES["Recovery_Pathway"],
        PCT_MODEL, WEEKS_MODEL, get_feature_frame, dev_dict
    )

    db.save_patient_recalibration_record(patient_id, week, res)

    return jsonify({
        "success": True,
        "message": f"Recalibration persisted for {patient_id} at week {week}",
        "new_predicted_pct": round(res["new_predicted_pct"], 1),
        "new_predicted_weeks": round(res["new_predicted_weeks"], 1),
        "explanation": res["explanation"]
    })


@app.route("/api/patient/add", methods=["POST"])
def add_new_patient():
    global TABLES, FULL_DATASET, TEST_DATA, X_TEST
    data = request.json or {}

    patient_id = data.get("Patient_ID", f"P{len(TEST_DATA)+3000}")
    age = int(data.get("Age", 45))
    gender = str(data.get("Gender", "Male"))
    injury_type = str(data.get("Injury_Type", "ACL Tear"))
    body_part = str(data.get("Body_Part", "Knee"))
    severity = str(data.get("Severity", "Moderate"))
    pain = float(data.get("Pain_Score", 6.0))
    rom = float(data.get("Range_of_Motion_deg", 80.0))
    strength = float(data.get("Quadriceps_Strength_pct", 55.0))

    patient_dict = {
        "Patient_ID": patient_id, "Age": age, "Gender": gender,
        "BMI": float(data.get("BMI", 24.5)), "Occupation": str(data.get("Occupation", "Office Worker")),
        "Activity_Level": str(data.get("Activity_Level", "Moderate")),
        "Smoking_Status": str(data.get("Smoking_Status", "Never")),
        "Diabetes": int(data.get("Diabetes", 0)), "Hypertension": int(data.get("Hypertension", 0)),
        "Comorbidity_Score": int(data.get("Comorbidity_Score", 0)), "Enrollment_Date": "2025-01-10"
    }

    injury_dict = {
        "Patient_ID": patient_id, "Injury_Type": injury_type, "Body_Part": body_part,
        "Severity": severity, "Surgery_Required": int(data.get("Surgery_Required", 1)),
        "Injury_Date": "2025-01-01", "Previous_Injuries": int(data.get("Previous_Injuries", 0))
    }

    assessment_dict = {
        "Patient_ID": patient_id, "Pain_Score": pain, "Range_of_Motion_deg": rom,
        "Muscle_Strength_pct": strength, "Quadriceps_Strength_pct": strength,
        "Hamstring_Strength_pct": strength * 0.85, "Balance_Score": 65.0,
        "Walking_Speed_mps": 0.9, "Clinical_Est_Weeks": 12
    }

    db.add_new_patient_record(patient_dict, injury_dict, assessment_dict)

    # Reload tables in memory
    TABLES = db.load_all_tables()
    FULL_DATASET = build_modelling_dataset(TABLES)
    _, TEST_DATA = chronological_split(FULL_DATASET)
    X_TEST = get_feature_frame(TEST_DATA)
    PRED_PCT, PRED_WEEKS = modeling.predict(PCT_MODEL, WEEKS_MODEL, X_TEST)
    TEST_DATA = TEST_DATA.copy()
    TEST_DATA["Predicted_Recovery_Pct"] = PRED_PCT
    TEST_DATA["Predicted_Recovery_Weeks"] = PRED_WEEKS

    return jsonify({"success": True, "patient_id": patient_id, "message": f"Patient {patient_id} created successfully"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


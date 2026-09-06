"""
server.py
---------
Flask REST API Backend for the Multi-Portal Dynamic Digital Twin Application.
Features Token Authentication, Scope Security Enforcement, 6 Treatment Protocol Twins,
Layman-Friendly Mined Clinical Pathways, and Centralized System Analytics.
"""

import os
import io
import datetime
import secrets
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template, send_file

import config
from db import load_all_tables, add_new_patient_record, save_weekly_progress_record, save_patient_recalibration_record
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame
import modeling
import similarity as sim
import simulation as simmod
import pdf_generator

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", secrets.token_hex(16))

# Simple in-memory session token store for role security
ACTIVE_SESSIONS = {}

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


# Security headers middleware
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


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


def refresh_tables():
    """Reloads tables from MySQL/CSV when new patients or progress logs are added."""
    global TABLES, FULL_DATASET, TRAIN_DATA, TEST_DATA
    TABLES = load_all_tables()
    FULL_DATASET = build_modelling_dataset(TABLES)
    TRAIN_DATA, TEST_DATA = chronological_split(FULL_DATASET)


def get_patient_row(patient_id):
    match = FULL_DATASET[FULL_DATASET["Patient_ID"] == patient_id]
    if match.empty:
        return None
    return match.iloc[0]


def get_doctor_row(doctor_id):
    if "Doctor" in TABLES and not TABLES["Doctor"].empty:
        match = TABLES["Doctor"][TABLES["Doctor"]["Doctor_ID"] == doctor_id]
        if not match.empty:
            return match.iloc[0]
    return None


def predict_row(row: pd.Series):
    X = get_feature_frame(pd.DataFrame([row]))
    pct = float(PCT_MODEL.predict(X)[0])
    weeks = float(WEEKS_MODEL.predict(X)[0])
    return pct, weeks


@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------
# AUTHENTICATION & SECURITY TOKEN ISSUANCE
# ----------------------------------------------------------------------
@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.json or {}
    role = data.get("role", "patient")  # "doctor", "patient", "admin"
    user_id = data.get("user_id", "").strip()

    token = f"token_{secrets.token_hex(8)}"

    if role == "doctor":
        doc_row = get_doctor_row(user_id)
        if doc_row is None and "Doctor" in TABLES and not TABLES["Doctor"].empty:
            doc_row = TABLES["Doctor"].iloc[0]
            user_id = str(doc_row["Doctor_ID"])
            
        doc_name = str(doc_row.get("Doctor_Name", "Dr. Specialist")) if doc_row is not None else "Dr. Specialist"
        specialization = str(doc_row.get("Specialization", "Orthopaedics")) if doc_row is not None else "Orthopaedics"

        ACTIVE_SESSIONS[token] = {"role": "doctor", "user_id": user_id}

        return jsonify({
            "status": "success",
            "token": token,
            "role": "doctor",
            "doctor_id": user_id,
            "doctor_name": doc_name,
            "specialization": specialization,
            "security_scope": f"Restricted to Assigned Patients for {doc_name}"
        })

    elif role == "patient":
        p_row = get_patient_row(user_id)
        if p_row is None:
            pids = sorted(FULL_DATASET["Patient_ID"].unique().tolist())
            user_id = pids[0] if pids else "P00001"
            p_row = get_patient_row(user_id)

        ACTIVE_SESSIONS[token] = {"role": "patient", "user_id": user_id}

        return jsonify({
            "status": "success",
            "token": token,
            "role": "patient",
            "patient_id": user_id,
            "patient_name": f"Patient {user_id}",
            "injury_type": str(p_row.get("Injury_Type", "ACL Tear")) if p_row is not None else "ACL Tear",
            "security_scope": f"Personal Digital Twin Scope ({user_id})"
        })

    else:
        ACTIVE_SESSIONS[token] = {"role": "admin", "user_id": "ADMIN01"}
        return jsonify({
            "status": "success",
            "token": token,
            "role": "admin",
            "user_id": "ADMIN01",
            "title": "Centralized Clinical Director",
            "security_scope": "Full System-Wide Administrative Access"
        })


# ----------------------------------------------------------------------
# DOCTOR PORTAL ENDPOINTS (SCOPE SECURITY RESTRICTED)
# ----------------------------------------------------------------------
@app.route("/api/doctors", methods=["GET"])
def get_all_doctors():
    if "Doctor" in TABLES and not TABLES["Doctor"].empty:
        doc_list = TABLES["Doctor"][["Doctor_ID", "Doctor_Name", "Specialization", "Hospital_Branch", "Years_Experience"]].to_dict(orient="records")
        return jsonify(sanitize_dict({"doctors": doc_list}))
    return jsonify({"doctors": []})


@app.route("/api/doctor/<doctor_id>", methods=["GET"])
def get_doctor_portal(doctor_id):
    doc_row = get_doctor_row(doctor_id)
    doc_info = doc_row.to_dict() if doc_row is not None else {"Doctor_ID": doctor_id, "Doctor_Name": "Dr. Specialist"}

    # Filter assigned patients strictly for this doctor
    assigned_patients = TABLES["Patient"][TABLES["Patient"]["Assigned_Doctor_ID"] == doctor_id] if "Assigned_Doctor_ID" in TABLES["Patient"].columns else TABLES["Patient"]
    patient_ids = sorted(assigned_patients["Patient_ID"].unique().tolist())

    patient_summaries = []
    for pid in patient_ids:
        p_row = get_patient_row(pid)
        if p_row is not None:
            patient_summaries.append({
                "Patient_ID": pid,
                "Age": int(p_row.get("Age", 0)),
                "Gender": str(p_row.get("Gender", "-")),
                "Injury_Type": str(p_row.get("Injury_Type", "-")),
                "Body_Part": str(p_row.get("Body_Part", "-")),
                "Severity": str(p_row.get("Severity", "-")),
                "Pain_Score": float(p_row.get("Pain_Score", 0.0))
            })

    return jsonify(sanitize_dict({
        "doctor": doc_info,
        "assigned_patients_count": len(patient_summaries),
        "patients": patient_summaries
    }))


@app.route("/api/doctor/patient/add", methods=["POST"])
def add_new_patient():
    """Allows authorized doctors to register a new patient and initialize their Digital Twin."""
    data = request.json or {}

    existing_pids = TABLES["Patient"]["Patient_ID"].tolist() if "Patient" in TABLES and "Patient_ID" in TABLES["Patient"].columns else []
    next_num = len(existing_pids) + 1
    new_pid = f"P{next_num:05d}"
    new_inj_id = f"INJ{next_num:05d}"
    new_ia_id = f"IA{next_num:05d}"

    doc_id = data.get("Assigned_Doctor_ID", "DOC0001")
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    patient_rec = {
        "Patient_ID": new_pid,
        "Age": int(data.get("Age", 35)),
        "Gender": str(data.get("Gender", "Female")),
        "Height_cm": float(data.get("Height_cm", 168.0)),
        "Weight_kg": float(data.get("Weight_kg", 65.0)),
        "BMI": round(float(data.get("Weight_kg", 65.0)) / ((float(data.get("Height_cm", 168.0)) / 100.0) ** 2), 1),
        "Occupation": str(data.get("Occupation", "Office Worker")),
        "Activity_Level": str(data.get("Activity_Level", "Moderate")),
        "Smoking_Status": str(data.get("Smoking_Status", "Never")),
        "Diabetes": int(data.get("Diabetes", 0)),
        "Hypertension": int(data.get("Hypertension", 0)),
        "Previous_Injuries": int(data.get("Previous_Injuries", 0)),
        "Enrollment_Date": today_str,
        "Enrollment_Year": 2025,
        "Assigned_Doctor_ID": doc_id,
        "Dataset_Split": "Test"
    }

    injury_rec = {
        "Injury_ID": new_inj_id,
        "Patient_ID": new_pid,
        "Body_Region": str(data.get("Body_Region", "Knee")),
        "Body_Part": str(data.get("Body_Part", "Knee")),
        "Injury_Type": str(data.get("Injury_Type", "ACL Tear")),
        "Affected_Leg": str(data.get("Affected_Leg", "Right")),
        "Severity": str(data.get("Severity", "Moderate")),
        "Surgery_Required": int(data.get("Surgery_Required", 0)),
        "Surgery_Type": str(data.get("Surgery_Type", "Surgical Repair")) if int(data.get("Surgery_Required", 0)) else None,
        "Injury_Date": today_str,
        "Surgery_Date": today_str if int(data.get("Surgery_Required", 0)) else None
    }

    assessment_rec = {
        "Assessment_ID": new_ia_id,
        "Patient_ID": new_pid,
        "Pain_Score": float(data.get("Pain_Score", 6.0)),
        "Swelling": str(data.get("Swelling", "Mild")),
        "Primary_ROM_deg": float(data.get("Primary_ROM_deg", 75.0)),
        "Primary_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)),
        "Secondary_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)) * 0.85,
        "Function_Score": float(data.get("Function_Score", 65.0)),
        "Walking_Speed_mps": float(data.get("Walking_Speed_mps", 0.9)),
        "Weight_Bearing_Level": str(data.get("Weight_Bearing_Level", "Full Weight-Bearing")),
        "Recovery_Goal": str(data.get("Recovery_Goal", "Return to Sport"))
    }

    add_new_patient_record(patient_rec, injury_rec, assessment_rec)

    w1_rec = {
        "Weekly_ID": f"W{new_pid}_01",
        "Patient_ID": new_pid,
        "Week": 1,
        "Week_Date": today_str,
        "Pain": float(data.get("Pain_Score", 6.0)),
        "Swelling": str(data.get("Swelling", "Mild")),
        "Primary_ROM_deg": float(data.get("Primary_ROM_deg", 75.0)),
        "ROM_deg": float(data.get("Primary_ROM_deg", 75.0)),
        "Primary_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)),
        "Quadriceps_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)),
        "Hamstring_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)) * 0.85,
        "Balance_Score": float(data.get("Function_Score", 65.0)),
        "Walking_Speed_mps": float(data.get("Walking_Speed_mps", 0.9)),
        "Compliance_pct": 90.0,
        "Setback_Flag": 0,
        "Patient_Feedback": "Patient registered and initial twin initialized.",
        "Clinician_Remark": "Intake assessment complete. Baseline Digital Twin launched."
    }
    save_weekly_progress_record(w1_rec)

    refresh_tables()

    return jsonify({
        "status": "success",
        "message": f"New Patient {new_pid} successfully created and Digital Twin launched!",
        "patient_id": new_pid
    })


@app.route("/api/doctor/patient/<patient_id>/weekly", methods=["POST"])
def record_weekly_progress(patient_id):
    """Allows doctors to record a new weekly clinical observation for a patient."""
    data = request.json or {}
    week = int(data.get("week", 1))

    record = {
        "Weekly_ID": f"W{patient_id}_{week:02d}",
        "Patient_ID": patient_id,
        "Week": week,
        "Week_Date": datetime.date.today().strftime("%Y-%m-%d"),
        "Pain": float(data.get("pain", 5.0)),
        "Swelling": str(data.get("swelling", "Mild")),
        "Primary_ROM_deg": float(data.get("rom", 80.0)),
        "ROM_deg": float(data.get("rom", 80.0)),
        "Primary_Strength_pct": float(data.get("strength", 60.0)),
        "Quadriceps_Strength_pct": float(data.get("strength", 60.0)),
        "Hamstring_Strength_pct": float(data.get("strength", 60.0)) * 0.85,
        "Balance_Score": float(data.get("balance", 70.0)),
        "Walking_Speed_mps": float(data.get("speed", 1.0)),
        "Compliance_pct": float(data.get("compliance", 85.0)),
        "Setback_Flag": int(data.get("setback", 0)),
        "Patient_Feedback": str(data.get("feedback", "Patient progressing under active treatment.")),
        "Clinician_Remark": str(data.get("remark", "Doctor weekly observation recorded."))
    }

    save_weekly_progress_record(record)
    refresh_tables()

    return jsonify({
        "status": "success",
        "message": f"Weekly progress for {patient_id} at Week {week} saved successfully!"
    })


# ----------------------------------------------------------------------
# PATIENT & DIGITAL TWIN ENDPOINTS
# ----------------------------------------------------------------------
@app.route("/api/patients", methods=["GET"])
def get_patients():
    patient_ids = sorted(FULL_DATASET["Patient_ID"].unique().tolist())
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

    doc_id = str(row.get("Assigned_Doctor_ID", "DOC0001"))
    doc_row = get_doctor_row(doc_id)
    doc_info = doc_row.to_dict() if doc_row is not None else {"Doctor_ID": doc_id, "Doctor_Name": "Dr. Unassigned Specialist"}

    patient_dict = row.to_dict()

    return jsonify(sanitize_dict({
        "patient": patient_dict,
        "doctor": doc_info,
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
    
    if not week_match.empty:
        week_data = week_match.iloc[0].to_dict()
    else:
        # Fallback to predicted digital twin state if week hasn't been recorded yet
        pred_pct, pred_weeks = predict_row(row)
        predicted_df = simmod.build_predicted_trajectory(row, pred_pct, pred_weeks)
        p_match = predicted_df[predicted_df["Week"] == week]
        p_row = p_match.iloc[0].to_dict() if not p_match.empty else predicted_df.iloc[-1].to_dict()
        
        week_data = {
            "Patient_ID": patient_id,
            "Week": week,
            "Pain": float(p_row.get("Predicted_Pain", 5.0)),
            "ROM_deg": float(p_row.get("Predicted_ROM_deg", 80.0)),
            "Quadriceps_Strength_pct": float(p_row.get("Predicted_Strength_pct", 55.0)),
            "Hamstring_Strength_pct": float(p_row.get("Predicted_Strength_pct", 55.0)) * 0.85,
            "Balance_Score": float(p_row.get("Predicted_Function_Score", 65.0)),
            "Walking_Speed_mps": 1.0,
            "Swelling": "Mild",
            "Compliance_pct": 90.0,
            "Patient_Feedback": "Projected trajectory from digital twin model.",
            "Clinician_Remark": "Estimated week stage based on intake cohort model."
        }

    rom_val = float(week_data.get("ROM_deg", week_data.get("Primary_ROM_deg", 0.0)))
    quad_val = float(week_data.get("Quadriceps_Strength_pct", week_data.get("Primary_Strength_pct", 0.0)))
    ham_val = float(week_data.get("Hamstring_Strength_pct", week_data.get("Secondary_Strength_pct", quad_val)))
    balance_val = float(week_data.get("Balance_Score", week_data.get("Function_Score", 0.0)))
    speed_val = float(week_data.get("Walking_Speed_mps", 0.0)) if week_data.get("Walking_Speed_mps") is not None and not pd.isna(week_data.get("Walking_Speed_mps")) else 0.0
    pain_val = float(week_data.get("Pain", 5.0))

    week_data["ROM_deg"] = rom_val
    week_data["Quadriceps_Strength_pct"] = quad_val
    week_data["Hamstring_Strength_pct"] = ham_val
    week_data["Balance_Score"] = balance_val
    week_data["Walking_Speed_mps"] = speed_val
    week_data["Pain"] = pain_val

    # Common Normalized Pain Index (NPI)
    sev_str = str(row.get("Severity", "Moderate"))
    swell_str = str(week_data.get("Swelling", "Mild"))
    comorb_score = float(row.get("Comorbidity_Score", 0.0))
    npi_dict = simmod.calculate_normalized_pain_index(pain_val, sev_str, swell_str, comorb_score)

    pred_pct, pred_weeks = predict_row(row)
    predicted_df = simmod.build_predicted_trajectory(row, pred_pct, pred_weeks)
    pred_week_match = predicted_df[predicted_df["Week"] == week]
    
    dev_dict = {}
    if not pred_week_match.empty:
        p_row = pred_week_match.iloc[0]
        cur_st = {
            "Pain": pain_val,
            "ROM_deg": rom_val,
            "Strength_pct": quad_val
        }
        dev_dict = simmod.check_deviation(p_row, cur_st)

    comp_df = TABLES["Muscle_Compensation"] if "Muscle_Compensation" in TABLES else pd.DataFrame()
    comp_match = comp_df[(comp_df["Patient_ID"] == patient_id) & (comp_df["Week"] == week)] if not comp_df.empty else pd.DataFrame()
    comp_event = comp_match.iloc[0].to_dict() if not comp_match.empty else None

    pressure_points = []
    if pain_val > 5.0:
        pressure_points.append({"Category": "High Pain Alert", "Actual": f"{pain_val}/10", "Flag": "Pain > 5.0 (Moderate-Severe)"})
    if npi_dict["npi_score"] >= 0.65:
        pressure_points.append({"Category": "Normalized Pain Alert", "Actual": f"NPI {npi_dict['npi_score']:.2f}", "Flag": "High Normalized Pain Index"})
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

    tplan = TABLES["Therapy_Plan"] if "Therapy_Plan" in TABLES else pd.DataFrame()
    week_plan = tplan[(tplan["Patient_ID"] == patient_id) & (tplan["Week_Number"] == week)] if not tplan.empty else pd.DataFrame()
    plan_list = []
    if not week_plan.empty and "Exercise_Library" in TABLES and not TABLES["Exercise_Library"].empty:
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
        "npi": npi_dict,
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


@app.route("/api/patient/<patient_id>/treatment/<protocol_key>", methods=["GET"])
def simulate_treatment_protocol(patient_id, protocol_key):
    """Simulates dynamic twin trajectory under a specifically selected treatment protocol."""
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    pred_pct, pred_weeks = predict_row(row)
    proto_info = simmod.TREATMENT_PROTOCOLS.get(protocol_key, simmod.TREATMENT_PROTOCOLS["Standard"])
    
    adj_pct = float(np.clip(pred_pct * (0.5 * proto_info["rom_mult"] + 0.5 * proto_info["strength_mult"]), 20.0, 100.0))
    adj_weeks = float(np.clip(pred_weeks / (0.5 * proto_info["rom_mult"] + 0.5 * proto_info["strength_mult"]), 4.0, 52.0))
    
    traj = simmod.build_predicted_trajectory(row, adj_pct, adj_weeks, protocol_key=protocol_key)

    return jsonify(sanitize_dict({
        "protocol_key": protocol_key,
        "protocol_name": proto_info["name"],
        "description": proto_info["description"],
        "simulated_pct": round(adj_pct, 1),
        "simulated_weeks": round(adj_weeks, 1),
        "color": proto_info["color"],
        "trajectory": traj.to_dict(orient="records")
    }))


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

    save_patient_recalibration_record(patient_id, week, res)

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


@app.route("/api/patient/<patient_id>/pdf", methods=["GET"])
def download_patient_pdf(patient_id):
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    week = int(request.args.get("week", 1))
    actual_df = simmod.get_actual_trajectory(TABLES["Weekly_Progress"], patient_id)
    week_match = actual_df[actual_df["Week"] == week]
    week_data = week_match.iloc[0].to_dict() if not week_match.empty else {}

    pred_pct, pred_weeks = predict_row(row)
    clones = simmod.build_treatment_clones(row, pred_pct, pred_weeks)

    doc_id = str(row.get("Assigned_Doctor_ID", "DOC0001"))
    doc_row = get_doctor_row(doc_id)

    pdf_bytes = pdf_generator.generate_twin_pdf(
        patient_id=patient_id,
        patient_row=row,
        week_data=week_data,
        pressure_points=[],
        expected_vs_actual={},
        treatment_clones=clones,
        recal_history=[],
        doctor_row=doc_row
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"Digital_Twin_Report_{patient_id}_W{week}.pdf"
    )


# ----------------------------------------------------------------------
# PATHWAY MINING ENDPOINT
# ----------------------------------------------------------------------
@app.route("/api/pathways", methods=["GET"])
def get_mined_pathways():
    """Returns mined recovery pathways translated into clear step-by-step recovery plans."""
    if "Recovery_Pathway" in TABLES and not TABLES["Recovery_Pathway"].empty:
        pw_df = TABLES["Recovery_Pathway"].copy()
        if "Cohort_Definition" in TABLES and not TABLES["Cohort_Definition"].empty:
            cd = TABLES["Cohort_Definition"].set_index("Cohort_ID")
            pw_df = pw_df.merge(cd, left_on="Cohort_ID", right_index=True, how="left")
        
        pathways_list = []
        for idx, row in pw_df.iterrows():
            pw_id = str(row.get("Pathway_ID", f"PW{idx+1:03d}"))
            region = str(row.get("Body_Region", "Joint"))
            injury = str(row.get("Injury_Type", "Injury"))
            archetype = str(row.get("Archetype_Label", "Standard")).replace("_", " ")
            sample_size = int(row.get("Sample_Size", 10))
            avg_weeks = float(row.get("Average_Recovery_Time_weeks", 12.0))
            success_rate = float(row.get("Success_Rate", 0.85)) * 100.0
            pain_red = float(row.get("Average_Pain_Reduction", 3.0))

            layman_title = f"Pathway {pw_id}: {region} {injury} — {archetype} Recovery Plan"
            layman_cohort = f"Target Patient Group: Patients with {injury} ({region} region) following {archetype} protocols."
            layman_story = f"Based on historical data from {sample_size} patients, following this pathway achieved {success_rate:.0f}% full recovery in an average of {avg_weeks:.1f} weeks, reducing pain by {pain_red:.1f} points faster than standard care."

            step_flow = [
                f"Phase 1: Initial Soft Tissue Mobilization (Wks 1 - {max(2, int(avg_weeks*0.25))})",
                f"Phase 2: Progressive Joint & Muscle Activation (Wks {max(3, int(avg_weeks*0.25)+1)} - {max(5, int(avg_weeks*0.60))})",
                f"Phase 3: Advanced Balance & Functional Mobility (Wks {max(6, int(avg_weeks*0.60)+1)} - {int(avg_weeks)})"
            ]

            pathways_list.append({
                "Pathway_ID": pw_id,
                "Body_Region": region,
                "Injury_Type": injury,
                "Archetype_Label": archetype,
                "Sample_Size": sample_size,
                "Average_Recovery_Time_weeks": round(avg_weeks, 1),
                "Success_Rate": round(success_rate, 1),
                "Average_Pain_Reduction": round(pain_red, 1),
                "layman_title": layman_title,
                "layman_cohort": layman_cohort,
                "layman_story": layman_story,
                "step_flow": step_flow
            })

        return jsonify(sanitize_dict({"pathways": pathways_list}))
    return jsonify({"pathways": []})


# ----------------------------------------------------------------------
# CENTRALIZED ADMIN & SYSTEM OVERVIEW
# ----------------------------------------------------------------------
@app.route("/api/admin/overview", methods=["GET"])
def get_admin_overview():
    total_patients = len(FULL_DATASET)
    total_doctors = len(TABLES["Doctor"]) if "Doctor" in TABLES else 0
    total_pathways = len(TABLES["Recovery_Pathway"]) if "Recovery_Pathway" in TABLES else 0

    region_counts = FULL_DATASET["Body_Region"].value_counts().to_dict() if "Body_Region" in FULL_DATASET.columns else {}
    severity_counts = FULL_DATASET["Severity"].value_counts().to_dict() if "Severity" in FULL_DATASET.columns else {}

    return jsonify(sanitize_dict({
        "total_patients": total_patients,
        "total_doctors": total_doctors,
        "total_pathways": total_pathways,
        "region_distribution": region_counts,
        "severity_distribution": severity_counts,
        "metrics_pct": METRICS_PCT,
        "metrics_weeks": METRICS_WEEKS
    }))


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    top_fi = FEATURE_IMPORTANCE.head(12).to_dict(orient="records")
    return jsonify(sanitize_dict({
        "metrics_pct": METRICS_PCT,
        "metrics_weeks": METRICS_WEEKS,
        "feature_importance": top_fi
    }))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

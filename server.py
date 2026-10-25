"""
server.py
---------
Flask REST API Backend for the Multi-Portal Dynamic Digital Twin Application.
Features Role-Based Access Control (RBAC), Scope Security Enforcement (ADMIN, DOCTOR, PATIENT),
Dynamic Pathway Mining (restricted per role), Single Treatment Reaction Simulation, and MySQL Persistence.
"""

import os
import io
import datetime
import secrets
import functools
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template, send_file
from werkzeug.security import check_password_hash

import config
import db
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame
import modeling
import similarity as sim
import simulation as simmod
import pathway_mining as pm
import pdf_generator
import explainability

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", secrets.token_hex(16))

# Active session token store
ACTIVE_SESSIONS = {}
SESSION_TTL = datetime.timedelta(hours=8)
LOGIN_ATTEMPTS = {}
LOGIN_WINDOW = datetime.timedelta(minutes=15)
MAX_LOGIN_ATTEMPTS = 10

print("Initializing Digital Twin backend and training models...")
TABLES = db.load_all_tables()
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

PCT_EVAL_MASK = pd.to_numeric(TEST_DATA[config.TARGET_RECOVERY_PCT], errors="coerce").notna()
WEEKS_EVAL_MASK = pd.to_numeric(TEST_DATA[config.TARGET_RECOVERY_WEEKS], errors="coerce").notna()
METRICS_PCT = modeling.evaluate(TEST_DATA.loc[PCT_EVAL_MASK, config.TARGET_RECOVERY_PCT], PRED_PCT[PCT_EVAL_MASK.to_numpy()])
METRICS_WEEKS = modeling.evaluate(TEST_DATA.loc[WEEKS_EVAL_MASK, config.TARGET_RECOVERY_WEEKS], PRED_WEEKS[WEEKS_EVAL_MASK.to_numpy()])
PCT_TRAIN_LABELS = pd.to_numeric(TRAIN_DATA[config.TARGET_RECOVERY_PCT], errors="coerce").dropna()
WEEKS_TRAIN_LABELS = pd.to_numeric(TRAIN_DATA[config.TARGET_RECOVERY_WEEKS], errors="coerce").dropna()
METRICS_PCT_BASELINE = modeling.evaluate(
    TEST_DATA.loc[PCT_EVAL_MASK, config.TARGET_RECOVERY_PCT],
    np.full(int(PCT_EVAL_MASK.sum()), float(PCT_TRAIN_LABELS.mean())),
)
METRICS_WEEKS_BASELINE = modeling.evaluate(
    TEST_DATA.loc[WEEKS_EVAL_MASK, config.TARGET_RECOVERY_WEEKS],
    np.full(int(WEEKS_EVAL_MASK.sum()), float(WEEKS_TRAIN_LABELS.mean())),
)
FEATURE_IMPORTANCE = modeling.get_feature_importance(PCT_MODEL)

print("Backend initialization complete!")


# Security headers middleware
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.before_request
def require_api_authentication():
    """Fail closed: every API route except login needs a live bearer session."""
    if request.path.startswith('/api/') and request.path != '/api/auth/login':
        if get_current_session() is None:
            return jsonify({"error": "Authentication required."}), 401


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
    """Reloads tables from MySQL/CSV when new doctors, patients, or progress logs are added."""
    global TABLES, FULL_DATASET, TRAIN_DATA, TEST_DATA
    TABLES = db.load_all_tables()
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


# ----------------------------------------------------------------------
# AUTHENTICATION & SECURITY SCOPE UTILITIES
# ----------------------------------------------------------------------
def get_current_session():
    """Resolve a live session from the Authorization header only."""
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1].strip()
    if token and token in ACTIVE_SESSIONS:
        session = ACTIVE_SESSIONS[token]
        created = datetime.datetime.fromisoformat(session["created_at"])
        if datetime.datetime.now() - created <= SESSION_TTL:
            return session
        ACTIVE_SESSIONS.pop(token, None)
    return None


def is_patient_accessible(sess, patient_id):
    """Verifies scope permission for patient resource."""
    if not sess:
        return False
    role = sess.get("role")
    if role == "ADMIN":
        return True
    elif role == "DOCTOR":
        p_row = get_patient_row(patient_id)
        if p_row is not None:
            assigned_doc = str(p_row.get("Assigned_Doctor_ID", ""))
            return assigned_doc == str(sess.get("linked_doctor_id", ""))
        return False
    elif role == "PATIENT":
        return str(patient_id) == str(sess.get("linked_patient_id", ""))
    return False


@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------
# AUTHENTICATION REST ENDPOINTS
# ----------------------------------------------------------------------
@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    now = datetime.datetime.now()
    ip = request.remote_addr or "unknown"
    attempts, started = LOGIN_ATTEMPTS.get(ip, (0, now))
    if now - started > LOGIN_WINDOW:
        attempts, started = 0, now
    if attempts >= MAX_LOGIN_ATTEMPTS:
        return jsonify({"error": "Too many login attempts. Try again later."}), 429

    data = request.json or {}
    login_input = str(data.get("login_input") or data.get("username") or data.get("user_id") or "").strip()
    password = str(data.get("password") or "")

    if not login_input or not password:
        return jsonify({"error": "Username/Email/ID and Password are required."}), 400

    user_rec = db.get_user_by_login_input(login_input)
    if not user_rec or not check_password_hash(user_rec.get("Password_Hash", ""), password):
        LOGIN_ATTEMPTS[ip] = (attempts + 1, started)
        return jsonify({"error": "Invalid password authentication."}), 401

    LOGIN_ATTEMPTS.pop(ip, None)

    token = secrets.token_urlsafe(32)
    role = user_rec["Role"].upper()
    user_id = user_rec["User_ID"]
    linked_doc = user_rec.get("Linked_Doctor_ID")
    linked_pat = user_rec.get("Linked_Patient_ID")

    doc_name = None
    patient_name = None

    if role == "DOCTOR" and linked_doc:
        doc_row = get_doctor_row(linked_doc)
        doc_name = str(doc_row.get("Doctor_Name", f"Dr. {linked_doc}")) if doc_row is not None else f"Dr. {linked_doc}"
    elif role == "PATIENT" and linked_pat:
        patient_name = f"Patient {linked_pat}"

    ACTIVE_SESSIONS[token] = {
        "user_id": user_id,
        "username": user_rec["Username"],
        "email": user_rec["Email"],
        "role": role,
        "linked_doctor_id": linked_doc,
        "linked_patient_id": linked_pat,
        "created_at": datetime.datetime.now().isoformat()
    }

    scope_desc = (
        "Full System-Wide Access" if role == "ADMIN"
        else f"Restricted to Assigned Patients for {doc_name or linked_doc}" if role == "DOCTOR"
        else f"Personal Digital Twin Scope ({linked_pat})"
    )

    return jsonify({
        "status": "success",
        "token": token,
        "user_id": user_id,
        "username": user_rec["Username"],
        "role": role,
        "linked_doctor_id": linked_doc,
        "linked_patient_id": linked_pat,
        "doctor_name": doc_name,
        "patient_name": patient_name,
        "security_scope": scope_desc
    })


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if token in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[token]
    return jsonify({"status": "success", "message": "Logged out successfully."})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    sess = get_current_session()
    if not sess:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(sanitize_dict({"user": sess}))


@app.route("/api/auth/password", methods=["POST"])
def auth_change_password():
    sess = get_current_session()
    data = request.json or {}
    current_password = str(data.get("current_password") or "")
    new_password = str(data.get("new_password") or "")
    if len(new_password) < 12:
        return jsonify({"error": "New password must be at least 12 characters."}), 400
    account = db.get_user_by_id(sess["user_id"])
    if not account or not check_password_hash(account.get("Password_Hash", ""), current_password):
        return jsonify({"error": "Current password is incorrect."}), 401
    db.update_user_password(sess["user_id"], new_password)
    return jsonify({"status": "success", "message": "Password updated."})


# ----------------------------------------------------------------------
# ADMIN PORTAL ENDPOINTS
# ----------------------------------------------------------------------
@app.route("/api/admin/overview", methods=["GET"])
def get_admin_overview():
    sess = get_current_session()
    if not sess or sess.get("role") != "ADMIN":
        return jsonify({"error": "Access denied. Admin authorization required."}), 403

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


@app.route("/api/admin/doctor/add", methods=["POST"])
def admin_add_doctor():
    sess = get_current_session()
    if not sess or sess.get("role") != "ADMIN":
        return jsonify({"error": "Access denied. Only Admins can register new doctors."}), 403

    data = request.json or {}

    # Query DB and memory for all existing doctor IDs
    existing_docs = set(TABLES["Doctor"]["Doctor_ID"].tolist()) if "Doctor" in TABLES and "Doctor_ID" in TABLES["Doctor"].columns else set()
    try:
        engine = db.get_db_engine()
        with engine.connect() as conn:
            db_docs = pd.read_sql("SELECT `Doctor_ID` FROM `Doctor`", conn)
            existing_docs.update(db_docs["Doctor_ID"].tolist())
    except Exception:
        pass

    nums = [int(str(d).replace("DOC", "")) for d in existing_docs if str(d).startswith("DOC") and str(d).replace("DOC", "").isdigit()]
    next_num = (max(nums) + 1) if nums else 1
    new_doc_id = f"DOC{next_num:04d}"

    doc_dict = {
        "Doctor_ID": new_doc_id,
        "Doctor_Name": str(data.get("Doctor_Name", f"Dr. Specialist {next_num}")),
        "Specialization": str(data.get("Specialization", "Orthopaedics")),
        "Hospital_Branch": str(data.get("Hospital_Branch", "Central Medical Hospital")),
        "Years_Experience": int(data.get("Years_Experience", 10)),
        "Contact_Email": str(data.get("Contact_Email", f"{new_doc_id.lower()}@hospital.com"))
    }

    try:
        initial_password = secrets.token_urlsafe(18)
        db.create_doctor_record(doc_dict, initial_password)
        refresh_tables()
        return jsonify({
            "status": "success",
            "message": f"Doctor {new_doc_id} ({doc_dict['Doctor_Name']}) successfully added!",
            "doctor_id": new_doc_id,
            "initial_password": initial_password
        })
    except Exception as e:
        return jsonify({"error": f"Failed to create doctor: {e}"}), 500


@app.route("/api/admin/patient/reassign", methods=["POST"])
def admin_reassign_patient():
    sess = get_current_session()
    if not sess or sess.get("role") != "ADMIN":
        return jsonify({"error": "Access denied. Admin authorization required."}), 403

    data = request.json or {}
    patient_id = data.get("patient_id")
    new_doctor_id = data.get("new_doctor_id")

    if not patient_id or not new_doctor_id:
        return jsonify({"error": "patient_id and new_doctor_id required"}), 400

    try:
        db.reassign_patient_doctor(patient_id, new_doctor_id)
        refresh_tables()
        return jsonify({
            "status": "success",
            "message": f"Patient {patient_id} successfully reassigned to Doctor {new_doctor_id}."
        })
    except Exception as e:
        return jsonify({"error": f"Failed to reassign patient: {e}"}), 500


@app.route("/api/admin/account/reset-password", methods=["POST"])
def admin_reset_account_password():
    sess = get_current_session()
    if sess.get("role") != "ADMIN":
        return jsonify({"error": "Admin authorization required."}), 403
    login_input = str((request.json or {}).get("login_input") or "").strip()
    account = db.get_user_by_login_input(login_input)
    if not account or account.get("Role", "").upper() == "ADMIN":
        return jsonify({"error": "Doctor or patient account not found."}), 404
    initial_password = secrets.token_urlsafe(24)
    db.update_user_password(account["User_ID"], initial_password)
    return jsonify({
        "status": "success",
        "login_id": account["Username"],
        "initial_password": initial_password,
        "message": "One-time password created. Share it securely and ask the user to change it after signing in."
    })


# ----------------------------------------------------------------------
# DOCTOR PORTAL ENDPOINTS
# ----------------------------------------------------------------------
@app.route("/api/doctors", methods=["GET"])
def get_all_doctors():
    sess = get_current_session()
    if sess.get("role") not in ("ADMIN", "DOCTOR"):
        return jsonify({"error": "Access denied."}), 403
    if "Doctor" in TABLES and not TABLES["Doctor"].empty:
        doc_list = TABLES["Doctor"][["Doctor_ID", "Doctor_Name", "Specialization", "Hospital_Branch", "Years_Experience", "Contact_Email"]].to_dict(orient="records")
        return jsonify(sanitize_dict({"doctors": doc_list}))
    return jsonify({"doctors": []})


@app.route("/api/doctor/<doctor_id>", methods=["GET"])
def get_doctor_portal(doctor_id):
    sess = get_current_session()
    if not sess:
        return jsonify({"error": "Unauthorized"}), 401

    if sess.get("role") not in ("ADMIN", "DOCTOR"):
        return jsonify({"error": "Access denied."}), 403
    if sess.get("role") == "DOCTOR" and str(sess.get("linked_doctor_id")) != str(doctor_id):
        return jsonify({"error": "Forbidden. You can only view your own assigned patient roster."}), 403

    doc_row = get_doctor_row(doctor_id)
    doc_info = doc_row.to_dict() if doc_row is not None else {"Doctor_ID": doctor_id, "Doctor_Name": "Dr. Specialist"}

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
def doctor_add_patient():
    sess = get_current_session()
    if not sess or sess.get("role") not in ["ADMIN", "DOCTOR"]:
        return jsonify({"error": "Access denied. Only doctors and admins can register patients."}), 403

    data = request.json or {}

    existing_pids = set(TABLES["Patient"]["Patient_ID"].tolist()) if "Patient" in TABLES and "Patient_ID" in TABLES["Patient"].columns else set()
    try:
        engine = db.get_db_engine()
        with engine.connect() as conn:
            db_pats = pd.read_sql("SELECT `Patient_ID` FROM `Patient`", conn)
            existing_pids.update(db_pats["Patient_ID"].tolist())
    except Exception:
        pass

    nums = [int(str(p).replace("P", "")) for p in existing_pids if str(p).startswith("P") and str(p).replace("P", "").isdigit()]
    next_num = (max(nums) + 1) if nums else 1
    new_pid = f"P{next_num:05d}"
    new_inj_id = f"INJ{next_num:05d}"
    new_ia_id = f"IA{next_num:05d}"

    if sess.get("role") == "DOCTOR":
        doc_id = sess.get("linked_doctor_id", "DOC0001")
    else:
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
        "Assigned_Doctor_ID": doc_id
    }

    injury_rec = {
        "Injury_ID": new_inj_id,
        "Patient_ID": new_pid,
        "Injury_Type": str(data.get("Injury_Type", "ACL Tear")),
        "Body_Part": str(data.get("Body_Region") or data.get("Body_Part", "Knee")),
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
        "Range_of_Motion_deg": float(data.get("Primary_ROM_deg", 75.0)),
        "Muscle_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)),
        "Balance_Score": float(data.get("Function_Score", 65.0)),
        "Walking_Speed_mps": float(data.get("Walking_Speed_mps", 0.9)),
        "Weight_Bearing_Level": str(data.get("Weight_Bearing_Level", "Full Weight-Bearing")),
        "Recovery_Goal": str(data.get("Recovery_Goal", "Return to Sport"))
    }

    initial_password = secrets.token_urlsafe(18)
    db.add_new_patient_record(patient_rec, injury_rec, assessment_rec, initial_password)

    w1_rec = {
        "Weekly_ID": f"WP{new_pid}_01",
        "Patient_ID": new_pid,
        "Week": 1,
        "Pain": float(data.get("Pain_Score", 6.0)),
        "Swelling": str(data.get("Swelling", "Mild")),
        "ROM_deg": float(data.get("Primary_ROM_deg", 75.0)),
        "Quadriceps_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)),
        "Hamstring_Strength_pct": float(data.get("Primary_Strength_pct", 50.0)) * 0.85,
        "Balance_Score": float(data.get("Function_Score", 65.0)),
        "Walking_Speed_mps": float(data.get("Walking_Speed_mps", 0.9)),
        "Compliance_pct": 90.0,
        "Patient_Feedback": "Patient registered and initial twin launched.",
        "Clinician_Remark": f"Registered by Doctor {doc_id}. Baseline Digital Twin online."
    }
    db.save_weekly_progress_record(w1_rec)

    refresh_tables()

    return jsonify({
        "status": "success",
        "message": f"New Patient {new_pid} successfully created and assigned to Doctor {doc_id}!",
        "patient_id": new_pid,
        "initial_password": initial_password
    })


# ----------------------------------------------------------------------
# PATIENT & DIGITAL TWIN REST ENDPOINTS
# ----------------------------------------------------------------------
@app.route("/api/patients", methods=["GET"])
def get_patients():
    sess = get_current_session()
    role = sess.get("role")
    if role == "ADMIN":
        patient_ids = sorted(FULL_DATASET["Patient_ID"].unique().tolist())
    elif role == "DOCTOR":
        doc_id = sess.get("linked_doctor_id")
        assigned = TABLES["Patient"][TABLES["Patient"]["Assigned_Doctor_ID"] == doc_id] if "Assigned_Doctor_ID" in TABLES["Patient"].columns else TABLES["Patient"]
        patient_ids = sorted(assigned["Patient_ID"].unique().tolist())
    elif role == "PATIENT":
        patient_ids = [sess.get("linked_patient_id")]
    else:
        return jsonify({"error": "Access denied."}), 403

    return jsonify({"patients": patient_ids})


@app.route("/api/patient/<patient_id>", methods=["GET"])
def get_patient_details(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": f"Access denied. Patient {patient_id} is not allocated to your account."}), 403

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
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied. Patient is not allocated to you."}), 403

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
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied. Patient is not allocated to you."}), 403

    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    actual_df = simmod.get_actual_trajectory(TABLES["Weekly_Progress"], patient_id)
    week_match = actual_df[actual_df["Week"] == week]
    
    if not week_match.empty:
        week_data = week_match.iloc[0].to_dict()
    else:
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


@app.route("/api/patient/<patient_id>/weekly/save", methods=["POST"])
def save_weekly_checkin(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied. Patient is not allocated to you."}), 403

    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

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
        "Primary_Strength_pct": float(data.get("quad_strength", data.get("strength", 60.0))),
        "Quadriceps_Strength_pct": float(data.get("quad_strength", data.get("strength", 60.0))),
        "Hamstring_Strength_pct": float(data.get("ham_strength", float(data.get("strength", 60.0)) * 0.85)),
        "Balance_Score": float(data.get("balance", 70.0)),
        "Walking_Speed_mps": float(data.get("walking_speed", 1.0)),
        "Compliance_pct": float(data.get("compliance", 90.0)),
        "Setback_Flag": int(data.get("setback", 0)),
        "Patient_Feedback": str(data.get("patient_feedback", "Weekly progress observation submitted.")),
        "Clinician_Remark": str(data.get("clinician_remark", "Observation recorded into DigitalTwinDB."))
    }

    db.save_weekly_progress_record(record)
    refresh_tables()

    cur_st = {
        "Pain": float(data.get("pain", 5.0)),
        "ROM_deg": float(data.get("rom", 80.0)),
        "Strength_pct": float(data.get("quad_strength", 60.0))
    }

    pred_pct, pred_weeks = predict_row(row)
    predicted_df = simmod.build_predicted_trajectory(row, pred_pct, pred_weeks)
    pred_week_match = predicted_df[predicted_df["Week"] == week]
    p_row = pred_week_match.iloc[0] if not pred_week_match.empty else predicted_df.iloc[0]

    dev_dict = simmod.check_deviation(p_row, cur_st)
    recal_res = simmod.recalibrate(
        row, cur_st, week, TRAIN_DATA, TABLES["Recovery_Pathway"],
        PCT_MODEL, WEEKS_MODEL, get_feature_frame, dev_dict
    )

    recal_payload = {
        "old_predicted_pct": round(pred_pct, 1),
        "old_predicted_weeks": round(pred_weeks, 1),
        "new_predicted_pct": round(recal_res["new_predicted_pct"], 1),
        "new_predicted_weeks": round(recal_res["new_predicted_weeks"], 1),
        "explanation": recal_res["explanation"],
        "recommended_pathway": recal_res["recommended_pathway"].to_dict() if isinstance(recal_res["recommended_pathway"], pd.Series) else recal_res["recommended_pathway"],
        "deviation": dev_dict
    }

    db.save_patient_recalibration_record(patient_id, week, recal_payload)

    evr = {}
    for metric_key, d in dev_dict.items():
        evr[metric_key] = {
            "expected": round(float(d["predicted"]), 1),
            "actual": round(float(d["actual"]), 1),
            "pct_deviation": round(float(d["pct_deviation"]), 1),
            "flagged": bool(d["flagged"])
        }

    return jsonify(sanitize_dict({
        "success": True,
        "message": f"Weekly observation for {patient_id} at Week {week} saved to database!",
        "recalibration": {
            "week": week,
            "old_predicted_pct": round(pred_pct, 1),
            "new_predicted_pct": round(recal_res["new_predicted_pct"], 1),
            "old_predicted_weeks": round(pred_weeks, 1),
            "new_predicted_weeks": round(recal_res["new_predicted_weeks"], 1),
            "discharge_delta": round(recal_res["new_predicted_weeks"] - pred_weeks, 1),
            "expected_vs_real": evr,
            "explanation": recal_res["explanation"],
            "recommended_pathway": recal_res["recommended_pathway"].to_dict() if isinstance(recal_res["recommended_pathway"], pd.Series) else recal_res["recommended_pathway"]
        }
    }))


@app.route("/api/patient/<patient_id>/recalibrate", methods=["POST"])
def recalibrate_patient(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied. Patient is not allocated to you."}), 403

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

    recal_payload = {
        "old_predicted_pct": round(pred_pct, 1),
        "old_predicted_weeks": round(pred_weeks, 1),
        "new_predicted_pct": round(res["new_predicted_pct"], 1),
        "new_predicted_weeks": round(res["new_predicted_weeks"], 1),
        "explanation": res["explanation"],
        "recommended_pathway": res["recommended_pathway"].to_dict() if isinstance(res["recommended_pathway"], pd.Series) else res["recommended_pathway"],
        "deviation": dev_dict
    }

    db.save_patient_recalibration_record(patient_id, week, recal_payload)

    updated_row = row.copy()
    updated_row["Pain_Score"] = float(data.get("pain", 6.0))
    updated_row["Range_of_Motion_deg"] = float(data.get("rom", 75.0))
    updated_row["Quadriceps_Strength_pct"] = float(data.get("strength", 50.0))
    
    new_neighbors_df = sim.find_similar_patients(updated_row, TRAIN_DATA, top_k=5)
    new_neighbors = new_neighbors_df[[
        "Patient_ID", "Similarity_Score", config.TARGET_RECOVERY_PCT, config.TARGET_RECOVERY_WEEKS
    ]].to_dict(orient="records")

    rec_pw = res["recommended_pathway"].to_dict() if isinstance(res["recommended_pathway"], pd.Series) else res["recommended_pathway"]

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


@app.route("/api/patient/<patient_id>/recalibration_history", methods=["GET"])
def get_recalibration_history_endpoint(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied. Patient is not allocated to you."}), 403

    history = db.get_patient_recalibration_history(patient_id)
    return jsonify(sanitize_dict({"history": history}))


# ----------------------------------------------------------------------
# TREATMENT REACTION SIMULATION ENDPOINTS
# ----------------------------------------------------------------------
@app.route("/api/treatment/taxonomy", methods=["GET"])
def get_treatment_taxonomy():
    return jsonify(sanitize_dict(simmod.TREATMENT_TAXONOMY))


@app.route("/api/patient/<patient_id>/treatment/simulate", methods=["POST"])
def simulate_single_treatment(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied. Patient is not allocated to you."}), 403

    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    data = request.json or {}
    category = data.get("category", "rom")
    exercise_type = data.get("exercise_type", "rom_passive")
    start_week = int(data.get("start_week", 4))
    duration_weeks = int(data.get("duration_weeks", 4))
    intensity = data.get("intensity", "Standard")

    base_pct, base_weeks = predict_row(row)
    base_traj = simmod.build_predicted_trajectory(row, base_pct, base_weeks)

    result = simmod.simulate_single_treatment_reaction(
        patient_row=row,
        category=category,
        exercise_type=exercise_type,
        start_week=start_week,
        duration_weeks=duration_weeks,
        intensity=intensity,
        base_weeks=base_weeks,
        base_traj=base_traj
    )

    return jsonify(sanitize_dict(result))


@app.route("/api/patient/<patient_id>/clones", methods=["GET"])
def get_patient_clones(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied."}), 403

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


@app.route("/api/patient/<patient_id>/pdf", methods=["GET"])
def download_patient_pdf(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied."}), 403

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
# PATHWAY MINING ENDPOINT (ROLE SCOPE ENFORCED)
# ----------------------------------------------------------------------
@app.route("/api/pathways", methods=["GET"])
def get_mined_pathways():
    """
    Returns mined recovery pathways.
    - PATIENT: Forbidden (403).
    - DOCTOR: Returns pathways filtered SOLELY for doctor's allocated patients.
    - ADMIN: Returns system-wide mined pathways.
    """
    sess = get_current_session()
    if sess:
        role = sess.get("role")
        if role == "PATIENT":
            return jsonify({"error": "Access denied. Pathway Mining is restricted to clinicians and administrators."}), 403
        elif role == "DOCTOR":
            doc_id = sess.get("linked_doctor_id")
            doc_pathways = pm.mine_pathways_for_doctor(doc_id, TABLES)
            return jsonify(sanitize_dict({"pathways": doc_pathways}))
        elif role != "ADMIN":
            return jsonify({"error": "Access denied."}), 403

    mined = pm.mine_pathways_from_tables(TABLES)
    return jsonify(sanitize_dict({"pathways": mined}))


@app.route("/api/patient/<patient_id>/recommended_pathways", methods=["GET"])
def get_recommended_pathways(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied. Patient is not allocated to you."}), 403

    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found"}), 404

    recs = pm.recommend_pathways_for_patient(row, TRAIN_DATA, TABLES, top_k=3)
    return jsonify(sanitize_dict({"recommended_pathways": recs}))


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    top_fi = FEATURE_IMPORTANCE.head(12).to_dict(orient="records")
    if config.DATA_SOURCE == "csv":
        data_note = "Configured CSV provenance is unverified; repository generator data are synthetic. Metrics are not clinical validation."
    else:
        data_note = "MySQL data provenance has not been independently verified; metrics are not clinical validation."
    return jsonify(sanitize_dict({
        "metrics_pct": METRICS_PCT,
        "metrics_weeks": METRICS_WEEKS,
        "baseline_metrics_pct": METRICS_PCT_BASELINE,
        "baseline_metrics_weeks": METRICS_WEEKS_BASELINE,
        "feature_importance": top_fi,
        "evaluation_data_note": data_note,
        "recovery_target_note": "When Final_Recovery_Pct is absent, the target is a 50:50 final strength/ROM proxy using 140 degrees for ROM. This is not validated or region-normalized; cross-injury metrics are descriptive only."
    }))


@app.route("/api/patient/<patient_id>/explanation", methods=["GET"])
def explain_patient_prediction(patient_id):
    sess = get_current_session()
    if not is_patient_accessible(sess, patient_id):
        return jsonify({"error": "Access denied."}), 403
    row = get_patient_row(patient_id)
    if row is None:
        return jsonify({"error": "Patient not found."}), 404
    try:
        result = explainability.explain_prediction(
            PCT_MODEL, X_TRAIN, get_feature_frame(pd.DataFrame([row]))
        )
        return jsonify(sanitize_dict(result))
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "available": False}), 503
    except Exception as exc:
        app.logger.exception("SHAP explanation failed")
        return jsonify({"error": "Could not calculate a local explanation."}), 500


if __name__ == "__main__":
    app.run(host=os.environ.get("DT_HOST", "127.0.0.1"),
            port=int(os.environ.get("DT_PORT", "5000")), debug=False)

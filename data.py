"""
data.py
-------
Dynamic Digital Twin for Personalized Orthopaedic Rehabilitation
Multi-region synthetic data generator (v2).

Generates 14 relational tables covering multi-region orthopaedic
rehabilitation journeys across 12 body regions and 24 injury types.
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import timedelta

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEED = 42
N_PATIENTS = 2200
OUT_DIR = "output"
rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_WEIGHTS = {2023: 0.35, 2024: 0.35, 2025: 0.30}

INJURY_CATALOG = [
    ("ACL Tear",                 "Knee",            "Knee",           0.080,  24,         0.65),
    ("Meniscus Tear",            "Knee",            "Knee",           0.050,  10,         0.30),
    ("MCL Tear",                 "Knee",            "Knee",           0.040,   8,         0.10),
    ("Patellar Fracture",        "Knee",            "Knee",           0.030,  14,         0.55),
    ("Rotator Cuff Tear",        "Shoulder",        "Shoulder",       0.060,  16,         0.45),
    ("Shoulder Dislocation",     "Shoulder",        "Shoulder",       0.030,  10,         0.15),
    ("Labral Tear",              "Shoulder",        "Shoulder",       0.030,  14,         0.35),
    ("Elbow Fracture",           "Elbow",           "Elbow",          0.030,  12,         0.50),
    ("Tennis Elbow",             "Elbow",           "Elbow",          0.030,   8,         0.05),
    ("Distal Radius Fracture",   "Wrist",           "Wrist",          0.040,  10,         0.40),
    ("Wrist Sprain",             "Wrist",           "Wrist",          0.030,   5,         0.02),
    ("Metacarpal Fracture",      "Hand",            "Hand",           0.020,   6,         0.20),
    ("Finger Tendon Injury",     "Hand",            "Hand",           0.020,   8,         0.25),
    ("Hip Fracture",             "Hip",             "Hip",            0.030,  20,         0.75),
    ("Hip Labral Tear",          "Hip",             "Hip",            0.020,  14,         0.30),
    ("Ankle Fracture",           "Ankle",           "Ankle",          0.050,  12,         0.35),
    ("Achilles Tendon Rupture",  "Ankle",           "Ankle",          0.040,  24,         0.55),
    ("Severe Ankle Sprain",      "Ankle",           "Ankle",          0.050,   6,         0.03),
    ("Foot Fracture",            "Foot",            "Foot",           0.030,  10,         0.25),
    ("Plantar Fascia Injury",    "Foot",            "Foot",           0.030,   8,         0.02),
    ("Cervical Strain",          "Cervical Spine",  "Neck",           0.040,   6,         0.02),
    ("Cervical Disc Injury",     "Cervical Spine",  "Neck",           0.020,  12,         0.15),
    ("Thoracic Spine Strain",    "Thoracic Spine",  "Upper Back",     0.020,   6,         0.01),
    ("Lumbar Disc Herniation",   "Lumbar Spine",    "Lower Back",     0.040,  14,         0.20),
    ("Lumbar Vertebral Fracture","Lumbar Spine",    "Lower Back",     0.010,  20,         0.40),
    ("Lumbar Muscle Strain",     "Lumbar Spine",    "Lower Back",     0.050,   6,         0.01),
]
_w = np.array([r[3] for r in INJURY_CATALOG], dtype=float)
INJURY_WEIGHTS = _w / _w.sum()

BODY_REGIONS = sorted(set(r[1] for r in INJURY_CATALOG))

REGION_PROFILE = {
    "Knee":            dict(rom_label="Knee ROM",       rom_healthy=145, strength_label="Quadriceps Strength",
                             secondary_strength_label="Hamstring Strength", function_label="Balance Score",
                             uses_walking=True,  uses_weight_bearing=True),
    "Shoulder":        dict(rom_label="Shoulder ROM",   rom_healthy=180, strength_label="Shoulder Strength",
                             secondary_strength_label=None, function_label="Functional Reach Score",
                             uses_walking=False, uses_weight_bearing=False),
    "Elbow":           dict(rom_label="Elbow ROM",      rom_healthy=145, strength_label="Arm Strength",
                             secondary_strength_label="Grip Strength", function_label="Functional Arm Score",
                             uses_walking=False, uses_weight_bearing=False),
    "Wrist":           dict(rom_label="Wrist ROM",      rom_healthy=80,  strength_label="Grip Strength",
                             secondary_strength_label=None, function_label="Dexterity Score",
                             uses_walking=False, uses_weight_bearing=False),
    "Hand":            dict(rom_label="Finger ROM",     rom_healthy=90,  strength_label="Grip Strength",
                             secondary_strength_label=None, function_label="Hand Function Score",
                             uses_walking=False, uses_weight_bearing=False),
    "Hip":             dict(rom_label="Hip ROM",        rom_healthy=120, strength_label="Hip Strength",
                             secondary_strength_label=None, function_label="Functional Mobility Score",
                             uses_walking=True,  uses_weight_bearing=True),
    "Ankle":           dict(rom_label="Ankle ROM",      rom_healthy=50,  strength_label="Calf Strength",
                             secondary_strength_label=None, function_label="Balance Score",
                             uses_walking=True,  uses_weight_bearing=True),
    "Foot":            dict(rom_label="Foot ROM",       rom_healthy=40,  strength_label="Foot Strength",
                             secondary_strength_label=None, function_label="Balance Score",
                             uses_walking=True,  uses_weight_bearing=True),
    "Cervical Spine":  dict(rom_label="Cervical ROM",   rom_healthy=80,  strength_label="Neck Strength",
                             secondary_strength_label=None, function_label="Posture Score",
                             uses_walking=False, uses_weight_bearing=False),
    "Thoracic Spine":  dict(rom_label="Thoracic ROM",   rom_healthy=45,  strength_label="Core Strength",
                             secondary_strength_label=None, function_label="Posture Score",
                             uses_walking=False, uses_weight_bearing=False),
    "Lumbar Spine":    dict(rom_label="Lumbar ROM",     rom_healthy=60,  strength_label="Core Strength",
                             secondary_strength_label=None, function_label="Functional Mobility Score",
                             uses_walking=True,  uses_weight_bearing=False),
}

OCCUPATIONS = ["Athlete", "Office Worker", "Manual Labor", "Student", "Retired", "Healthcare Worker"]
ACTIVITY_LEVELS = ["Sedentary", "Moderate", "Active", "Highly Active"]
RECOVERY_GOALS = ["Return to Sport", "Return to Work", "Walk/Move Normally", "Independent Daily Living"]
SEVERITIES = ["Mild", "Moderate", "Severe"]
SEVERITY_NUM = {"Mild": 0, "Moderate": 1, "Severe": 2}

ARCHETYPES = {
    "Fast_Recovery":     dict(rate_mult=1.30, setback_p=0.04, setback_mag=0.6),
    "Standard_Recovery": dict(rate_mult=1.00, setback_p=0.06, setback_mag=0.8),
    "Cautious_Recovery": dict(rate_mult=0.80, setback_p=0.07, setback_mag=0.9),
    "Slow_Recovery":     dict(rate_mult=0.60, setback_p=0.08, setback_mag=1.0),
    "Setback_Prone":     dict(rate_mult=0.85, setback_p=0.22, setback_mag=1.6),
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())

PHASE_ORDER = ["Early", "Early-Mid", "Mid", "Mid-Late", "Late"]

CLINICIAN_REMARKS = [
    "Progressing as expected", "Slight delay, monitor closely", "Excellent compliance and progress",
    "Compliance low, discussed barriers", "Swelling higher than expected", "Ready to advance phase",
    "Patient reports increased confidence", "Mild setback, reduced load this week",
    "Setback discussed, plan adjusted", "Steady, unremarkable progress",
]
PATIENT_FEEDBACK = [
    "Feeling stronger", "Still some discomfort", "Noticeable improvement in mobility",
    "Struggled with home exercises", "Confident with daily activities", "Fatigued after sessions",
    "Motivated and on track", "Anxious about re-injury", "Had a rough week, pain flared up",
    "Back on track after last week's setback",
]

EXERCISE_LIBRARY_RAW = [
    ("Quad Sets", "Knee", "Quadriceps Strength", "Easy", "Standard Physiotherapy", "Early", 10, "Severe swelling"),
    ("Straight Leg Raise", "Knee", "Quadriceps Strength", "Easy", "Standard Physiotherapy", "Early", 10, "ROM < 30 deg"),
    ("Heel Slides", "Knee", "Hamstring Strength", "Easy", "Standard Physiotherapy", "Early", 10, "Pain score > 7"),
    ("Bridge", "Knee", "Glute Strength", "Moderate", "Strength Training", "Mid", 10, "Pain score > 6"),
    ("Single Leg Balance", "Knee", "Balance Score", "Moderate", "Balance Training", "Mid", 10, "Weight bearing restricted"),
    ("Step Ups", "Knee", "Quadriceps Strength", "Hard", "Strength Training", "Late", 15, "Strength < 60%"),
    ("Single Leg Hop", "Knee", "Balance Score", "Hard", "Return to Activity Drills", "Late", 15, "Strength < 80%"),
    ("Pendulum Exercise", "Shoulder", "Shoulder Strength", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Assisted Shoulder Flexion", "Shoulder", "Shoulder Strength", "Easy", "Standard Physiotherapy", "Early", 10, "ROM < 30 deg"),
    ("External Rotation", "Shoulder", "Shoulder Strength", "Moderate", "Strength Training", "Mid", 10, "Pain score > 6"),
    ("Scapular Retraction", "Shoulder", "Functional Reach Score", "Moderate", "Strength Training", "Mid", 10, None),
    ("Shoulder Stabilization Drills", "Shoulder", "Functional Reach Score", "Hard", "Functional Training", "Late", 15, "Strength < 70%"),
    ("Elbow ROM Stretch", "Elbow", "Arm Strength", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Forearm Isometrics", "Elbow", "Grip Strength", "Easy", "Standard Physiotherapy", "Early", 8, None),
    ("Resistance Band Elbow Flexion", "Elbow", "Arm Strength", "Moderate", "Strength Training", "Mid", 10, "Pain score > 6"),
    ("Grip and Forearm Strengthening", "Elbow", "Grip Strength", "Hard", "Strength Training", "Late", 10, "Strength < 60%"),
    ("Wrist Flexion", "Wrist", "Grip Strength", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Wrist Extension", "Wrist", "Grip Strength", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Grip Training", "Wrist", "Grip Strength", "Moderate", "Strength Training", "Mid", 10, "ROM < 30 deg"),
    ("Dexterity Exercises", "Wrist", "Dexterity Score", "Hard", "Functional Training", "Late", 10, "Strength < 60%"),
    ("Tendon Gliding", "Hand", "Hand Function Score", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Finger Mobility Drills", "Hand", "Hand Function Score", "Easy", "Standard Physiotherapy", "Early", 8, None),
    ("Hand Grip Training", "Hand", "Grip Strength", "Moderate", "Strength Training", "Mid", 10, "ROM < 30 deg"),
    ("Dexterity Drills", "Hand", "Dexterity Score", "Hard", "Functional Training", "Late", 10, "Strength < 60%"),
    ("Hip Isometrics", "Hip", "Hip Strength", "Easy", "Standard Physiotherapy", "Early", 10, "Pain score > 7"),
    ("Hip Abduction", "Hip", "Hip Strength", "Moderate", "Strength Training", "Mid", 10, "Weight bearing restricted"),
    ("Single Leg Balance (Hip)", "Hip", "Functional Mobility Score", "Moderate", "Balance Training", "Mid", 10, "Weight bearing restricted"),
    ("Functional Mobility Drills", "Hip", "Functional Mobility Score", "Hard", "Functional Training", "Late", 15, "Strength < 70%"),
    ("Ankle Pumps", "Ankle", "Calf Strength", "Easy", "Standard Physiotherapy", "Early", 5, None),
    ("Ankle Alphabet", "Ankle", "Ankle ROM", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Calf Raises", "Ankle", "Calf Strength", "Moderate", "Strength Training", "Mid", 10, "Weight bearing restricted"),
    ("Balance Board Training", "Ankle", "Balance Score", "Moderate", "Balance Training", "Mid", 10, "Weight bearing restricted"),
    ("Agility Ladder Drills", "Ankle", "Balance Score", "Hard", "Return to Activity Drills", "Late", 15, "Strength < 70%"),
    ("Toe Curls", "Foot", "Foot Strength", "Easy", "Standard Physiotherapy", "Early", 6, "Pain score > 7"),
    ("Plantar Fascia Stretch", "Foot", "Foot Strength", "Easy", "Standard Physiotherapy", "Early", 8, None),
    ("Foot Strengthening", "Foot", "Foot Strength", "Moderate", "Strength Training", "Mid", 10, "Weight bearing restricted"),
    ("Balance and Gait Training", "Foot", "Balance Score", "Hard", "Functional Training", "Late", 15, "Strength < 60%"),
    ("Chin Tucks", "Cervical Spine", "Neck Strength", "Easy", "Standard Physiotherapy", "Early", 6, "Pain score > 7"),
    ("Cervical ROM Exercises", "Cervical Spine", "Cervical ROM", "Easy", "Standard Physiotherapy", "Early", 8, None),
    ("Postural Exercises (Neck)", "Cervical Spine", "Posture Score", "Moderate", "Functional Training", "Mid", 10, None),
    ("Neck Stabilization", "Cervical Spine", "Neck Strength", "Hard", "Strength Training", "Late", 10, "Pain score > 6"),
    ("Thoracic Mobility Drills", "Thoracic Spine", "Thoracic ROM", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Postural Training", "Thoracic Spine", "Posture Score", "Moderate", "Functional Training", "Mid", 10, None),
    ("Core Activation (Thoracic)", "Thoracic Spine", "Core Strength", "Hard", "Strength Training", "Late", 10, "Pain score > 6"),
    ("Pelvic Tilts", "Lumbar Spine", "Core Strength", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Core Activation (Lumbar)", "Lumbar Spine", "Core Strength", "Easy", "Standard Physiotherapy", "Early", 8, "Pain score > 7"),
    ("Controlled Mobility Drills", "Lumbar Spine", "Functional Mobility Score", "Moderate", "Functional Training", "Mid", 10, None),
    ("Spinal Stabilization", "Lumbar Spine", "Core Strength", "Hard", "Strength Training", "Late", 10, "Pain score > 6"),
]

def build_exercise_library():
    df = pd.DataFrame(EXERCISE_LIBRARY_RAW, columns=[
        "Exercise_Name", "Target_Region", "Target_Muscle_or_Function", "Difficulty",
        "Exercise_Category", "Recovery_Phase", "Typical_Duration_min", "Contraindication"
    ])
    df.insert(0, "Exercise_ID", [f"EX{i:03d}" for i in range(1, len(df) + 1)])
    return df

COMPENSATION_PATTERNS = {
    "Knee": [("Quadriceps", "Hamstring compensation"), ("Quadriceps", "Hip stabilizer compensation")],
    "Shoulder": [("Rotator Cuff", "Upper trapezius compensation"), ("Scapular Stabilizers", "Scapular stabilizer weakness")],
    "Elbow": [("Arm Flexors", "Forearm compensation"), ("Grip", "Shoulder compensation")],
    "Wrist": [("Grip", "Forearm compensation"), ("Grip", "Grip imbalance")],
    "Hand": [("Grip", "Forearm compensation"), ("Grip", "Grip imbalance")],
    "Hip": [("Hip Flexors", "Gluteal compensation"), ("Hip Abductors", "Hip flexor compensation")],
    "Ankle": [("Calf", "Hip stabilizer compensation"), ("Calf", "Gait compensation")],
    "Foot": [("Foot Intrinsics", "Gait compensation"), ("Foot Intrinsics", "Calf overload")],
    "Cervical Spine": [("Neck Stabilizers", "Upper trapezius overload"), ("Neck Stabilizers", "Cervical stabilizer weakness")],
    "Thoracic Spine": [("Core", "Paraspinal overload"), ("Core", "Postural compensation")],
    "Lumbar Spine": [("Core", "Paraspinal overload"), ("Core", "Hip compensation")],
}


def generate_patients(n):
    ages = rng.integers(15, 80, n)
    genders = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    heights = np.where(genders == "Male", rng.normal(175, 7, n), rng.normal(162, 6, n)).round(1)
    weights = np.where(genders == "Male", rng.normal(80, 13, n), rng.normal(67, 12, n)).round(1)
    bmi = (weights / ((heights / 100) ** 2)).round(1)

    occupation = rng.choice(OCCUPATIONS, n, p=[0.15, 0.30, 0.20, 0.15, 0.15, 0.05])
    activity_level = rng.choice(ACTIVITY_LEVELS, n, p=[0.20, 0.35, 0.30, 0.15])
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.65, 0.20, 0.15])
    diabetes = rng.choice([0, 1], n, p=[0.87, 0.13])
    hypertension = rng.choice([0, 1], n, p=[0.78, 0.22])
    prev_injuries = rng.poisson(0.4, n)

    years = np.array(list(YEAR_WEIGHTS.keys()))
    yweights = np.array(list(YEAR_WEIGHTS.values()))
    enroll_year = rng.choice(years, n, p=yweights)
    day_offset = rng.integers(0, 330, n)
    anchor_date = [pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=int(d)) for y, d in zip(enroll_year, day_offset)]

    df = pd.DataFrame({
        "Patient_ID": [f"P{i:05d}" for i in range(1, n + 1)],
        "Age": ages,
        "Gender": genders,
        "Height_cm": heights,
        "Weight_kg": weights,
        "BMI": bmi,
        "Occupation": occupation,
        "Activity_Level": activity_level,
        "Smoking_Status": smoking,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Previous_Injuries": prev_injuries,
        "_anchor_date": anchor_date,
    })
    return df


def generate_injuries(patients):
    n = len(patients)
    idx = rng.choice(len(INJURY_CATALOG), n, p=INJURY_WEIGHTS)
    injury_type = [INJURY_CATALOG[i][0] for i in idx]
    body_region = [INJURY_CATALOG[i][1] for i in idx]
    body_part = [INJURY_CATALOG[i][2] for i in idx]
    base_weeks = np.array([INJURY_CATALOG[i][4] for i in idx], dtype=float)
    surgery_base_rate = np.array([INJURY_CATALOG[i][5] for i in idx])

    lateral_region = np.isin(body_region, ["Knee", "Shoulder", "Elbow", "Wrist", "Hand", "Hip", "Ankle", "Foot"])
    affected_side = rng.choice(["Left", "Right"], n)
    affected_side = np.where(lateral_region, affected_side, "N/A")

    severity = rng.choice(SEVERITIES, n, p=[0.32, 0.43, 0.25])
    sev_num = np.array([SEVERITY_NUM[s] for s in severity])

    surgery_p = np.clip(surgery_base_rate + (sev_num - 1) * 0.20, 0.01, 0.95)
    surgery_required = (rng.random(n) < surgery_p).astype(int)
    surgery_type = [f"{it} Surgical Repair" if sr else None for it, sr in zip(injury_type, surgery_required)]

    injury_date = patients["_anchor_date"].values
    injury_date = pd.to_datetime(injury_date)

    surgery_lag = rng.integers(3, 21, n)
    surgery_date = injury_date + pd.to_timedelta(surgery_lag, unit="D")
    surgery_date = pd.Series(surgery_date).where(pd.Series(surgery_required == 1), pd.NaT)

    post_event_lag = rng.integers(3, 14, n)
    base_for_enrollment = np.where(surgery_required == 1, surgery_date.values, injury_date.values)
    enrollment_date = pd.to_datetime(base_for_enrollment) + pd.to_timedelta(post_event_lag, unit="D")
    enrollment_date = pd.Series(enrollment_date)
    enrollment_year = enrollment_date.dt.year

    df = pd.DataFrame({
        "Injury_ID": [f"INJ{i:05d}" for i in range(1, n + 1)],
        "Patient_ID": patients["Patient_ID"].values,
        "Body_Region": body_region,
        "Body_Part": body_part,
        "Injury_Type": injury_type,
        "Affected_Side": affected_side,
        "Severity": severity,
        "Surgery_Required": surgery_required,
        "Surgery_Type": surgery_type,
        "Injury_Date": injury_date.date,
        "Surgery_Date": surgery_date.dt.date,
        "Base_Recovery_Weeks": base_weeks,
        "_enrollment_date": enrollment_date,
        "_enrollment_year": enrollment_year,
    })
    return df


def generate_initial_assessments(patients, injuries):
    n = len(patients)
    sev_num = injuries["Severity"].map(SEVERITY_NUM).values
    surgery = injuries["Surgery_Required"].values
    region = injuries["Body_Region"].values

    pain = np.clip(rng.normal(5 + sev_num * 1.2 + surgery * 0.6, 1.3, n), 0, 10).round(0).astype(int)
    swelling = np.array(["None", "Mild", "Moderate", "Severe"])[
        np.clip(sev_num + surgery + rng.integers(0, 2, n), 0, 3)
    ]

    rom_healthy = np.array([REGION_PROFILE[r]["rom_healthy"] for r in region])
    rom_frac = np.clip(rng.normal(0.55 - sev_num * 0.13 - surgery * 0.08, 0.08, n), 0.10, 0.85)
    rom = (rom_healthy * rom_frac).round(0).astype(int)

    strength = np.clip(rng.normal(55 - sev_num * 10 - surgery * 6, 10, n), 10, 100).round(0).astype(int)
    function_score = np.clip(rng.normal(55 - sev_num * 9, 11, n), 5, 100).round(0).astype(int)

    walk_speed = np.where(
        [REGION_PROFILE[r]["uses_walking"] for r in region],
        np.clip(rng.normal(0.8 - sev_num * 0.15, 0.15, n), 0.1, 1.6).round(2),
        np.nan,
    )
    weight_bearing = np.where(
        [REGION_PROFILE[r]["uses_weight_bearing"] for r in region],
        rng.choice(["Non-Weight-Bearing", "Partial Weight-Bearing", "Full Weight-Bearing"], n, p=[0.2, 0.45, 0.35]),
        "N/A",
    )
    recovery_goal = rng.choice(RECOVERY_GOALS, n, p=[0.30, 0.25, 0.25, 0.20])

    df = pd.DataFrame({
        "Assessment_ID": [f"IA{i:05d}" for i in range(1, n + 1)],
        "Patient_ID": patients["Patient_ID"].values,
        "Pain_Score": pain,
        "Swelling": swelling,
        "Primary_ROM_deg": rom,
        "Primary_ROM_Label": [REGION_PROFILE[r]["rom_label"] for r in region],
        "Primary_Strength_pct": strength,
        "Primary_Strength_Label": [REGION_PROFILE[r]["strength_label"] for r in region],
        "Function_Score": function_score,
        "Function_Label": [REGION_PROFILE[r]["function_label"] for r in region],
        "Walking_Speed_mps": walk_speed,
        "Weight_Bearing_Level": weight_bearing,
        "Recovery_Goal": recovery_goal,
        "Assessment_Date": injuries["_enrollment_date"].dt.date.values,
    })
    return df


def assign_archetypes(patients, injuries):
    n = len(patients)
    scores = {name: np.zeros(n) for name in ARCHETYPE_NAMES}

    age = patients["Age"].values
    activity = patients["Activity_Level"].values
    diabetes = patients["Diabetes"].values
    smoking = patients["Smoking_Status"].values
    bmi = patients["BMI"].values
    prev_inj = patients["Previous_Injuries"].values
    sev_num = injuries["Severity"].map(SEVERITY_NUM).values
    surgery = injuries["Surgery_Required"].values

    scores["Fast_Recovery"] += (age < 30) * 1.4 + (activity == "Highly Active") * 1.3 + (sev_num == 0) * 1.0 - surgery * 0.5
    scores["Slow_Recovery"] += (diabetes == 1) * 1.8 + (age > 55) * 1.2 + (smoking == "Current") * 0.9 + (sev_num == 2) * 1.0 + (bmi > 30) * 0.6
    scores["Cautious_Recovery"] += (sev_num == 1) * 1.0 + ((age >= 30) & (age <= 55)) * 0.7 + surgery * 0.4
    scores["Setback_Prone"] += (prev_inj >= 1) * 1.0 + (smoking == "Current") * 0.5 + (activity == "Sedentary") * 0.5
    scores["Standard_Recovery"] += 1.0

    for name in ARCHETYPE_NAMES:
        scores[name] += rng.normal(0, 1.0, n)

    score_matrix = np.vstack([scores[name] for name in ARCHETYPE_NAMES])
    assigned_idx = np.argmax(score_matrix, axis=0)
    return np.array(ARCHETYPE_NAMES)[assigned_idx]


def _step_up(current, ceiling, k, compliance_factor, noise_scale):
    gap = max(0.0, ceiling - current)
    return current + gap * k * compliance_factor * noise_scale

def _step_down(current, floor, k, compliance_factor, noise_scale):
    gap = max(0.0, current - floor)
    return current - gap * k * compliance_factor * noise_scale


def generate_weekly_progress(patients, injuries, assessments, archetypes):
    rows = []
    traj_store = {}

    pat = patients.set_index("Patient_ID")
    inj = injuries.set_index("Patient_ID")
    ia = assessments.set_index("Patient_ID")
    swelling_levels = ["Severe", "Moderate", "Mild", "None"]

    for pid, arche in zip(patients["Patient_ID"], archetypes):
        p = pat.loc[pid]
        i = inj.loc[pid]
        a = ia.loc[pid]
        region = i["Body_Region"]
        prof = REGION_PROFILE[region]
        arch = ARCHETYPES[arche]

        sev_num = SEVERITY_NUM[i["Severity"]]
        surgery = int(i["Surgery_Required"])
        age = int(p["Age"])
        diabetes = int(p["Diabetes"])
        smoking_current = 1 if p["Smoking_Status"] == "Current" else 0
        bmi = float(p["BMI"])
        activity_highly = 1 if p["Activity_Level"] == "Highly Active" else 0

        rom_healthy = prof["rom_healthy"]
        age_pen = 0.06 if age > 60 else (0.03 if age > 45 else 0.0)

        rom_ceiling = rom_healthy * float(np.clip(rng.normal(0.97 - sev_num * 0.09 - age_pen - diabetes * 0.03 - smoking_current * 0.02, 0.05), 0.55, 0.99))
        strength_ceiling = float(np.clip(rng.normal(100 - sev_num * 9 - age_pen * 60 - diabetes * 4 - smoking_current * 3, 6), 55, 100))
        function_ceiling = float(np.clip(rng.normal(100 - sev_num * 8 - age_pen * 50 - diabetes * 3, 7), 55, 100))
        pain_floor = float(np.clip(rng.normal(sev_num * 0.7 + surgery * 0.2, 0.4), 0, 4))
        walk_ceiling = (float(np.clip(rng.normal(1.5 - sev_num * 0.08, 0.1), 0.8, 1.8)) if prof["uses_walking"] else np.nan)

        sev_dur_mult = {0: 0.75, 1: 1.0, 2: 1.4}[sev_num]
        comorbid_mult = 1.0 + 0.10 * (age > 55) + 0.10 * diabetes + 0.05 * smoking_current + 0.05 * (bmi > 30)
        target_weeks = i["Base_Recovery_Weeks"] * sev_dur_mult * comorbid_mult / arch["rate_mult"]
        n_weeks = int(np.clip(rng.normal(target_weeks, target_weeks * 0.18), 4, 52))

        compliance_base = np.clip(rng.normal(80 - smoking_current * 5 + activity_highly * 4 - (sev_num == 2) * 3, 12), 20, 100)
        noise_scale = rng.uniform(0.75, 1.25)

        pain = float(a["Pain_Score"])
        swelling_idx = swelling_levels.index(a["Swelling"]) if a["Swelling"] in swelling_levels else 1
        rom = float(a["Primary_ROM_deg"])
        strength = float(a["Primary_Strength_pct"])
        function_score = float(a["Function_Score"])
        walk = float(a["Walking_Speed_mps"]) if prof["uses_walking"] else np.nan

        weekly_traj = {}
        enrollment_date = i["_enrollment_date"]

        for week in range(1, n_weeks + 1):
            compliance = float(np.clip(rng.normal(compliance_base, 8), 0, 100))
            compliance_factor = compliance / 100
            is_setback = rng.random() < arch["setback_p"]

            if is_setback:
                mag = arch["setback_mag"]
                pain = min(10, pain + rng.uniform(0.5, 1.5) * mag)
                swelling_idx = max(0, swelling_idx - (1 if rng.random() < 0.5 else 0))
                rom = max(0, rom - rng.uniform(1, 4) * mag)
                strength = max(0, strength - rng.uniform(1, 4) * mag)
                function_score = max(0, function_score - rng.uniform(1, 4) * mag)
                if prof["uses_walking"]:
                    walk = max(0.1, walk - rng.uniform(0.02, 0.08) * mag)
                compliance = max(10, compliance - rng.uniform(5, 20))
            else:
                k_base = rng.uniform(0.11, 0.18)
                k_mult = arch["rate_mult"] * (1 - 0.05 * sev_num) * (1 - 0.04 * (age > 55) - 0.04 * diabetes - 0.02 * smoking_current - 0.02 * (bmi > 30))
                k = max(0.03, k_base * k_mult)
                pain = _step_down(pain, pain_floor, rng.uniform(0.09, 0.15) * k_mult, compliance_factor, noise_scale)
                rom = _step_up(rom, rom_ceiling, k, compliance_factor, noise_scale)
                strength = _step_up(strength, strength_ceiling, k, compliance_factor, noise_scale)
                function_score = _step_up(function_score, function_ceiling, k, compliance_factor, noise_scale)
                if prof["uses_walking"]:
                    walk = _step_up(walk, walk_ceiling, k, compliance_factor, noise_scale)
                if rng.random() < 0.20 * arch["rate_mult"] * compliance_factor:
                    swelling_idx = min(3, swelling_idx + 1)

            week_date = enrollment_date + pd.Timedelta(days=7 * week)
            weekly_traj[week] = dict(pain=pain, swelling_idx=swelling_idx, rom=rom, strength=strength,
                                      function_score=function_score, walk=walk, compliance=compliance,
                                      setback=is_setback)

            secondary_strength = None
            if prof["secondary_strength_label"] is not None:
                secondary_strength = round(max(0, strength - rng.uniform(0, 8)), 1)

            # Map backward compatibility columns for legacy single-quad/hamstring schema
            quad_strength = round(strength, 1)
            ham_strength = secondary_strength if secondary_strength is not None else round(max(0, strength - rng.uniform(0, 8)), 1)

            rows.append({
                "Weekly_ID": f"W{pid}_{week:02d}",
                "Patient_ID": pid,
                "Week": week,
                "Week_Date": week_date.date(),
                "Pain": round(pain, 1),
                "Swelling": swelling_levels[swelling_idx],
                "Primary_ROM_deg": round(rom, 1),
                "ROM_deg": round(rom, 1),
                "Primary_Strength_pct": round(strength, 1),
                "Quadriceps_Strength_pct": quad_strength,
                "Hamstring_Strength_pct": ham_strength,
                "Secondary_Strength_pct": secondary_strength,
                "Function_Score": round(function_score, 1),
                "Balance_Score": round(function_score, 1),
                "Walking_Speed_mps": (round(walk, 2) if prof["uses_walking"] else None),
                "Compliance_pct": round(compliance, 1),
                "Setback_Flag": int(is_setback),
                "Patient_Feedback": rng.choice(PATIENT_FEEDBACK) if not is_setback else
                                     rng.choice(["Had a rough week, pain flared up", "Struggled with home exercises",
                                                 "Fatigued after sessions", "Anxious about re-injury"]),
                "Clinician_Remark": rng.choice(CLINICIAN_REMARKS) if not is_setback else
                                     rng.choice(["Setback discussed, plan adjusted", "Mild setback, reduced load this week",
                                                 "Swelling higher than expected", "Compliance low, discussed barriers"]),
            })

        traj_store[pid] = dict(archetype=arche, n_weeks=n_weeks, weekly=weekly_traj, region=region,
                                rom_ceiling=rom_ceiling, strength_ceiling=strength_ceiling,
                                function_ceiling=function_ceiling, pain_floor=pain_floor,
                                walk_ceiling=walk_ceiling, compliance_base=compliance_base,
                                enrollment_date=enrollment_date)

    return pd.DataFrame(rows), traj_store


def generate_therapy_plan(patients, injuries, assessments, traj_store, exercise_lib):
    rows = []
    ia = assessments.set_index("Patient_ID")
    lib_by_region = {r: g for r, g in exercise_lib.groupby("Target_Region")}

    for pid in patients["Patient_ID"]:
        info = traj_store[pid]
        region = info["region"]
        candidates_all = lib_by_region.get(region)
        if candidates_all is None or len(candidates_all) == 0:
            continue
        weight_bearing_label = ia.loc[pid, "Weight_Bearing_Level"]

        for week in range(1, info["n_weeks"] + 1):
            wk = info["weekly"][week]
            rom_frac = wk["rom"] / info["rom_ceiling"] if info["rom_ceiling"] else 0
            strength_frac = wk["strength"] / info["strength_ceiling"] if info["strength_ceiling"] else 0
            progress_frac = np.clip((rom_frac + strength_frac) / 2, 0, 1.2)

            if progress_frac < 0.30: phase = "Early"
            elif progress_frac < 0.45: phase = "Early-Mid"
            elif progress_frac < 0.65: phase = "Mid"
            elif progress_frac < 0.85: phase = "Mid-Late"
            else: phase = "Late"
            phase_idx = PHASE_ORDER.index(phase)

            in_phase_band = candidates_all[candidates_all["Recovery_Phase"].map(lambda p: abs(PHASE_ORDER.index(p) - phase_idx) <= 1)]
            swelling_label = ["Severe", "Moderate", "Mild", "None"][wk["swelling_idx"]]
            allowed = in_phase_band
            if len(allowed) == 0: allowed = candidates_all[candidates_all["Recovery_Phase"] == "Early"]
            if len(allowed) == 0: continue

            n_ex = int(rng.integers(1, 3))
            chosen = allowed.sample(n=min(n_ex, len(allowed)), random_state=int(rng.integers(0, 1_000_000)))
            for _, ex in chosen.iterrows():
                rows.append({
                    "Plan_ID": f"PL{pid}_{week:02d}_{ex['Exercise_ID']}",
                    "Patient_ID": pid,
                    "Week_Number": week,
                    "Exercise_ID": ex["Exercise_ID"],
                    "Repetitions": int(rng.choice([8, 10, 12, 15])),
                    "Sets": int(rng.choice([2, 3, 4])),
                    "Duration_min": ex["Typical_Duration_min"],
                    "Therapist_Notes": rng.choice([
                        "Good form maintained", "Reduce load if pain increases",
                        "Progress next week if tolerated", "Monitor for compensation patterns"]),
                })
    return pd.DataFrame(rows)


def generate_muscle_compensation(weekly_progress, traj_store):
    rows = []
    for _, row in weekly_progress.iterrows():
        info = traj_store[row["Patient_ID"]]
        region = info["region"]
        patterns = COMPENSATION_PATTERNS.get(region, [])
        if not patterns: continue
        strength_ceiling = info["strength_ceiling"] or 100
        strength_deficit = max(0.0, 1 - row["Primary_Strength_pct"] / strength_ceiling)
        risk_score = strength_deficit * (row["Compliance_pct"] / 100)
        if rng.random() < risk_score * 0.30:
            muscle, comp_desc = patterns[rng.integers(0, len(patterns))]
            comp_level = rng.choice(["Mild", "Moderate", "High"], p=[0.5, 0.35, 0.15])
            risk_level = {"Mild": "Low", "Moderate": "Medium", "High": "High"}[comp_level]
            rows.append({
                "Compensation_ID": f"MC{row['Weekly_ID']}",
                "Patient_ID": row["Patient_ID"],
                "Week": row["Week"],
                "Body_Region": region,
                "Muscle_Overloaded": muscle,
                "Compensation_Pattern": comp_desc,
                "Compensation_Level": comp_level,
                "Risk_Level": risk_level,
            })
    return pd.DataFrame(rows)


def _final_metrics(info):
    last = info["weekly"][info["n_weeks"]]
    rom_frac = last["rom"] / info["rom_ceiling"] if info["rom_ceiling"] else 0
    strength_frac = last["strength"] / info["strength_ceiling"] if info["strength_ceiling"] else 0
    function_frac = last["function_score"] / info["function_ceiling"] if info["function_ceiling"] else 0
    pain_ok = last["pain"] <= info["pain_floor"] + 1.8
    recovery_pct = float(np.clip(100 * (0.35 * rom_frac + 0.35 * strength_frac + 0.20 * function_frac + 0.10 * (1 if pain_ok else 0.4)), 0, 100))
    if rom_frac >= 0.85 and strength_frac >= 0.85 and pain_ok: label = "Full Recovery"
    elif rom_frac >= 0.60 or strength_frac >= 0.60: label = "Partial Recovery"
    else: label = "Incomplete Recovery"
    return dict(last=last, rom_frac=rom_frac, strength_frac=strength_frac, function_frac=function_frac,
                pain_ok=pain_ok, recovery_pct=recovery_pct, label=label)


ARCHETYPE_SEQUENCES = {
    "Fast_Recovery":     (["Standard Physiotherapy", "Balance Training", "Strength Training", "Functional Training", "Return to Activity Drills"], [0.00, 0.15, 0.35, 0.55, 0.75]),
    "Standard_Recovery": (["Standard Physiotherapy", "Balance Training", "Strength Training", "Functional Training", "Return to Activity Drills"], [0.00, 0.20, 0.40, 0.65, 0.85]),
    "Cautious_Recovery": (["Standard Physiotherapy", "Balance Training", "Strength Training", "Functional Training"], [0.00, 0.25, 0.50, 0.75]),
    "Slow_Recovery":     (["Standard Physiotherapy", "Balance Training", "Strength Training"], [0.00, 0.30, 0.60]),
    "Setback_Prone":     (["Standard Physiotherapy", "Balance Training", "Strength Training", "Functional Training"], [0.00, 0.20, 0.45, 0.70]),
}


def build_cohorts_and_pathways(patients, injuries, traj_store):
    merged = patients.merge(injuries, on="Patient_ID")
    merged["Age_Band"] = pd.cut(merged["Age"], bins=[0, 25, 40, 55, 100], labels=["<25", "25-40", "40-55", "55+"])

    cohort_rows, pathway_rows, step_rows = [], [], []
    cohort_id_counter = pathway_id_counter = 1

    grouped = merged.groupby(["Age_Band", "Body_Region", "Injury_Type", "Severity"], observed=True)
    for (age_band, region, injury_type, severity), group in grouped:
        if len(group) < 8: continue
        cohort_id = f"C{cohort_id_counter:03d}"
        cohort_id_counter += 1
        cohort_rows.append({
            "Cohort_ID": cohort_id, "Age_Band": age_band, "Body_Region": region,
            "Injury_Type": injury_type, "Severity": severity, "Cohort_Size": len(group),
        })

        for arche, sub in group.groupby(group["Patient_ID"].map(lambda p: traj_store[p]["archetype"])):
            if len(sub) < 4: continue
            pids = sub["Patient_ID"].tolist()
            metrics = [_final_metrics(traj_store[p]) for p in pids]
            final_weeks = [traj_store[p]["n_weeks"] for p in pids]
            success_rate = np.mean([1 if m["label"] == "Full Recovery" else 0 for m in metrics])
            avg_pain_reduction = np.mean([float(traj_store[p]["weekly"][1]["pain"]) - m["last"]["pain"] for p, m in zip(pids, metrics)])

            pathway_id = f"PW{pathway_id_counter:03d}"
            pathway_id_counter += 1
            pathway_rows.append({
                "Pathway_ID": pathway_id, "Cohort_ID": cohort_id, "Body_Region": region,
                "Archetype_Label": arche, "Sample_Size": len(pids),
                "Average_Recovery_Time_weeks": round(float(np.mean(final_weeks)), 1),
                "Success_Rate": round(float(success_rate), 2),
                "Average_Pain_Reduction": round(float(avg_pain_reduction), 1),
                "Average_Recovery_Pct": round(float(np.mean([m["recovery_pct"] for m in metrics])), 1),
                "Average_ROM_deg": round(float(np.mean([m["last"]["rom"] for m in metrics])), 1),
            })

            seq, fracs = ARCHETYPE_SEQUENCES[arche]
            avg_weeks = float(np.mean(final_weeks))
            for step_order, (cat, frac) in enumerate(zip(seq, fracs), start=1):
                step_rows.append({
                    "Pathway_ID": pathway_id, "Step_Order": step_order,
                    "Exercise_Category": cat, "Week_Offset": round(frac * avg_weeks, 1),
                })

    return pd.DataFrame(cohort_rows), pd.DataFrame(pathway_rows), pd.DataFrame(step_rows)


def compute_similar_patients(patients, injuries, assessments, top_k=5):
    merged = patients.merge(injuries, on="Patient_ID").merge(assessments, on="Patient_ID")
    num_cols = ["Age", "BMI", "Pain_Score", "Primary_ROM_deg", "Primary_Strength_pct", "Function_Score"]
    rows = []
    for region, grp in merged.groupby("Body_Region"):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < 2: continue
        norm = grp[num_cols].copy()
        for c in num_cols:
            span = (norm[c].max() - norm[c].min()) or 1
            norm[c] = (norm[c] - norm[c].min()) / span
        num_matrix = norm.values
        cat_cols = ["Gender", "Injury_Type", "Severity", "Diabetes", "Smoking_Status"]
        cat_data = grp[cat_cols]
        pids = grp["Patient_ID"].values

        for i in range(n):
            num_dist = np.sqrt(((num_matrix - num_matrix[i]) ** 2).sum(axis=1))
            cat_dist = np.zeros(n)
            for c in cat_cols:
                cat_dist += (cat_data[c].values != cat_data[c].values[i]).astype(float)
            cat_dist /= len(cat_cols)
            total_dist = 0.6 * (num_dist / (num_dist.max() or 1)) + 0.4 * cat_dist
            similarity = 1 - total_dist
            order = np.argsort(-similarity)
            order = order[order != i][:top_k]
            for rank, j in enumerate(order, start=1):
                rows.append({
                    "Similarity_ID": f"SIM{pids[i]}_{rank}",
                    "Patient_ID": pids[i],
                    "Similar_Patient_ID": pids[j],
                    "Body_Region": region,
                    "Similarity_Score": round(float(similarity[j]), 3),
                    "Similarity_Basis": "Initial-Profile (same body region)",
                    "Rank": rank,
                })
    return pd.DataFrame(rows)


def generate_simulations_and_recommendations(patients, traj_store, pathways):
    sim_rows, rec_rows = [], []
    sim_counter = rec_counter = 1
    decision_weeks = [0, 4, 8, 12]
    pathways_by_region = {r: g for r, g in pathways.groupby("Body_Region")}

    for pid in patients["Patient_ID"]:
        info = traj_store[pid]
        n_weeks = info["n_weeks"]
        true_arche = info["archetype"]
        region = info["region"]
        region_pathways = pathways_by_region.get(region)
        if region_pathways is None or len(region_pathways) == 0: continue
        candidate_pathways = region_pathways.sample(n=min(3, len(region_pathways)), random_state=int(rng.integers(0, 1_000_000)))

        for dweek in decision_weeks:
            if dweek >= n_weeks: continue
            current = info["weekly"].get(dweek, info["weekly"][1])
            sims_this_round = []
            for _, prow in candidate_pathways.iterrows():
                match_bonus = 0.15 if prow["Archetype_Label"] == true_arche else 0.0
                confidence = float(np.clip(rng.normal(0.6 + match_bonus, 0.1), 0.2, 0.95))
                predicted_recovery_time = prow["Average_Recovery_Time_weeks"] * rng.uniform(0.9, 1.1)
                predicted_recovery_pct = min(100, prow["Average_Recovery_Pct"] * rng.uniform(0.95, 1.05))
                predicted_recovery_label = "Full Recovery" if prow["Success_Rate"] > 0.55 else "Partial Recovery"

                sim_id = f"SM{sim_counter:06d}"
                sim_counter += 1
                sim_rows.append({
                    "Simulation_ID": sim_id, "Patient_ID": pid, "Week_Number": dweek,
                    "Pathway_ID": prow["Pathway_ID"],
                    "Predicted_Recovery": predicted_recovery_label,
                    "Predicted_Recovery_Time_weeks": round(float(predicted_recovery_time), 1),
                    "Predicted_Recovery_Pct": round(float(predicted_recovery_pct), 1),
                    "Confidence": round(confidence, 2),
                })
                sims_this_round.append((sim_id, prow["Pathway_ID"], confidence))

            if not sims_this_round: continue
            best_sim_id, best_pathway_id, best_conf = max(sims_this_round, key=lambda x: x[2])
            decision = rng.choice(["Accepted", "Accepted", "Accepted", "Modified", "Overridden"])
            override_reason = None
            if decision != "Accepted":
                override_reason = rng.choice([
                    "Clinician judgment based on patient tolerance", "Patient preference for different pace",
                    "Scheduling constraints", "Recent flare-up in symptoms"])

            rec_rows.append({
                "Recommendation_ID": f"REC{rec_counter:06d}", "Patient_ID": pid, "Week": dweek,
                "Current_Status": f"Pain {round(current['pain'],1)}, ROM {round(current['rom'],1)}, "
                                   f"Strength {round(current['strength'],1)}%",
                "Simulation_ID": best_sim_id, "Recommended_Pathway": best_pathway_id,
                "Reason": "Best simulated outcome among candidate pathways for same-region cohort",
                "Confidence": round(best_conf, 2),
                "Expected_Next_Milestone": rng.choice([
                    "ROM improves toward target by next review", "Strength improves toward target by next review",
                    "Pain trending toward target by next review", "Function score improves by next review"]),
                "Clinician_Decision": decision, "Override_Reason": override_reason,
            })
            rec_counter += 1

    return pd.DataFrame(sim_rows), pd.DataFrame(rec_rows)


def generate_final_outcomes(patients, injuries, traj_store):
    rows = []
    for pid in patients["Patient_ID"]:
        info = traj_store[pid]
        m = _final_metrics(info)
        last = m["last"]
        enrollment_date = info["enrollment_date"]
        discharge_date = enrollment_date + pd.Timedelta(days=7 * info["n_weeks"])
        return_to_activity = 1 if (m["label"] == "Full Recovery" and m["function_frac"] >= 0.80) else 0

        rows.append({
            "Outcome_ID": f"FO{pid}", "Patient_ID": pid,
            "Discharge_Week": info["n_weeks"], "Discharge_Date": discharge_date.date(),
            "Actual_Recovery_Time_weeks": info["n_weeks"],
            "Actual_Primary_ROM_deg": round(last["rom"], 1),
            "Actual_ROM_deg": round(last["rom"], 1),
            "Actual_Primary_Strength_pct": round(last["strength"], 1),
            "Actual_Strength_pct": round(last["strength"], 1),
            "Actual_Function_Score": round(last["function_score"], 1),
            "Actual_Pain_Score": round(last["pain"], 1),
            "Final_Recovery_Pct": round(m["recovery_pct"], 1),
            "Return_To_Activity_Flag": return_to_activity,
            "Outcome_Label": m["label"],
        })
    return pd.DataFrame(rows)


def main():
    print("Generating patients...")
    patients = generate_patients(N_PATIENTS)

    print("Generating injuries (multi-region across 12 body regions)...")
    injuries = generate_injuries(patients)

    print("Generating region-aware initial assessments...")
    assessments = generate_initial_assessments(patients, injuries)

    print("Building region-aware exercise library...")
    exercise_lib = build_exercise_library()

    print("Assigning recovery archetypes...")
    archetypes = assign_archetypes(patients, injuries)

    print("Generating weekly progress trajectories...")
    weekly_progress, traj_store = generate_weekly_progress(patients, injuries, assessments, archetypes)

    print("Generating therapy plans...")
    therapy_plan = generate_therapy_plan(patients, injuries, assessments, traj_store, exercise_lib)

    print("Generating muscle compensation events...")
    muscle_comp = generate_muscle_compensation(weekly_progress, traj_store)

    print("Building cohorts and pathways...")
    cohorts, pathways, pathway_steps = build_cohorts_and_pathways(patients, injuries, traj_store)

    print("Computing similar patients...")
    similar_patients = compute_similar_patients(patients, injuries, assessments)

    print("Generating simulations and recommendations...")
    simulations, recommendations = generate_simulations_and_recommendations(patients, traj_store, pathways)

    print("Generating final outcomes...")
    final_outcomes = generate_final_outcomes(patients, injuries, traj_store)

    patients_out = patients.drop(columns=["_anchor_date"]).copy()
    patients_out["Enrollment_Date"] = injuries["_enrollment_date"].dt.date.values
    patients_out["Enrollment_Year"] = injuries["_enrollment_year"].values
    patients_out["Dataset_Split"] = np.where(patients_out["Enrollment_Year"] == 2025, "Test", "Train")

    injuries_out = injuries.drop(columns=["Base_Recovery_Weeks", "_enrollment_date", "_enrollment_year"]).copy()

    tables = {
        "Patient": patients_out,
        "Injury": injuries_out,
        "Initial_Assessment": assessments,
        "Exercise_Library": exercise_lib,
        "Therapy_Plan": therapy_plan,
        "Weekly_Progress": weekly_progress,
        "Muscle_Compensation": muscle_comp,
        "Cohort_Definition": cohorts,
        "Recovery_Pathway": pathways,
        "Pathway_Steps": pathway_steps,
        "Similar_Patients": similar_patients,
        "Simulation": simulations,
        "Weekly_Recommendation": recommendations,
        "Final_Outcome": final_outcomes,
    }

    for name, df in tables.items():
        path = os.path.join(OUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(df)} rows)")

    print("\nGeneration complete. All 14 tables written to ./output/")
    return tables

if __name__ == "__main__":
    main()
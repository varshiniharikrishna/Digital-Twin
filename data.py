import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEED = 42
N_PATIENTS = 800
OUT_DIR = "output"

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# REFERENCE DATA
# ----------------------------------------------------------------------

# Clinical baseline recovery windows (weeks), for a MODERATE-severity,
# non-comorbid patient on a Standard_Track pace. These are broad
# real-world ballpark figures used only to shape synthetic data:
#   ACL reconstruction: ~9-12 months to full clearance
#   Meniscus tear: ~6-16 weeks depending on repair vs. trim
#   Ankle fracture: ~10-16 weeks
#   Rotator cuff repair: ~16-24 weeks (strength lags longer)
#   Achilles rupture: ~24-32 weeks
#   Tibial fracture: ~14-22 weeks
INJURY_TYPES = [
    # name, body_part, prevalence, base_weeks, min_weeks, max_weeks
    ("ACL Tear",               "Knee",      0.55, 32, 20, 52),
    ("Meniscus Tear",          "Knee",      0.15, 10,  6, 18),
    ("Ankle Fracture",         "Ankle",     0.10, 14,  8, 24),
    ("Rotator Cuff Tear",      "Shoulder",  0.10, 20, 12, 30),
    ("Achilles Tendon Rupture","Ankle",     0.05, 28, 20, 40),
    ("Tibial Fracture",        "Lower Leg", 0.05, 18, 10, 28),
]
INJURY_LOOKUP = {r[0]: {"body_part": r[1], "base_weeks": r[3], "min_weeks": r[4], "max_weeks": r[5]}
                  for r in INJURY_TYPES}

OCCUPATIONS = ["Athlete", "Office Worker", "Manual Labor", "Student", "Retired", "Healthcare Worker"]
ACTIVITY_LEVELS = ["Sedentary", "Moderate", "Active", "Highly Active"]
RECOVERY_GOALS = ["Return to Sport", "Walk Normally", "Climb Stairs", "Return to Work"]
SEVERITIES = ["Mild", "Moderate", "Severe"]
SEVERITY_DURATION_MULT = {"Mild": 0.75, "Moderate": 1.0, "Severe": 1.35}
SEVERITY_TARGET_PENALTY = {"Mild": 0, "Moderate": 5, "Severe": 14}  # deg/pct shaved off ceiling

# Recovery archetypes: named therapy sequence templates. rate_mult now
# speeds/slows convergence AND compresses/stretches total duration.
# week_offset_frac are FRACTIONS of the patient's own recovery length,
# not fixed absolute weeks, so a 10-week and a 40-week patient both get
# sensible phase transitions.
ARCHETYPES = {
    "Fast_Track": {
        "rate_mult": 1.25,
        "sequence": ["Standard Physiotherapy", "Hydrotherapy", "Strength Training", "Return to Sport Drills"],
        "week_offset_frac": [0.0, 0.20, 0.45, 0.70],
    },
    "Standard_Track": {
        "rate_mult": 1.0,
        "sequence": ["Standard Physiotherapy", "Strength Training", "Functional Training"],
        "week_offset_frac": [0.0, 0.35, 0.65],
    },
    "Cautious_Track": {
        "rate_mult": 0.85,
        "sequence": ["Standard Physiotherapy", "Hydrotherapy", "Balance Training", "Strength Training"],
        "week_offset_frac": [0.0, 0.25, 0.55, 0.80],
    },
    "Slow_Comorbid_Track": {
        "rate_mult": 0.7,
        "sequence": ["Standard Physiotherapy", "Balance Training", "Low-Impact Strength Training"],
        "week_offset_frac": [0.0, 0.40, 0.70],
    },
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())

EXERCISE_LIBRARY = [
    ("Quad Sets", "Quadriceps", "Easy", "Standard Physiotherapy", "Early Recovery", 10, "Severe swelling"),
    ("Straight Leg Raise", "Quadriceps", "Easy", "Standard Physiotherapy", "Early Recovery", 10, "ROM < 30 deg"),
    ("Ankle Pumps", "Calf", "Easy", "Standard Physiotherapy", "Early Recovery", 5, None),
    ("Heel Slides", "Hamstring", "Easy", "Standard Physiotherapy", "Early Recovery", 10, "Pain score > 7"),
    ("Pool Walking", "Full Body", "Easy", "Hydrotherapy", "Early-Mid Recovery", 20, "Open wound"),
    ("Aqua Jogging", "Full Body", "Moderate", "Hydrotherapy", "Mid Recovery", 20, "Open wound"),
    ("Water Resistance Kicks", "Quadriceps", "Moderate", "Hydrotherapy", "Mid Recovery", 15, "Open wound"),
    ("Single Leg Balance", "Stabilizers", "Moderate", "Balance Training", "Mid Recovery", 10, "Weight bearing restricted"),
    ("Bosu Ball Squats", "Stabilizers", "Moderate", "Balance Training", "Mid Recovery", 15, "ROM < 90 deg"),
    ("Tandem Stance", "Stabilizers", "Easy", "Balance Training", "Early-Mid Recovery", 10, None),
    ("Bridge", "Glutes", "Moderate", "Strength Training", "Mid Recovery", 10, "Pain score > 6"),
    ("Leg Press (Light)", "Quadriceps", "Moderate", "Low-Impact Strength Training", "Mid Recovery", 15, "ROM < 90 deg"),
    ("Wall Sits", "Quadriceps", "Moderate", "Strength Training", "Mid Recovery", 10, "Pain score > 6"),
    ("Resistance Band Squats", "Quadriceps", "Moderate", "Strength Training", "Mid-Late Recovery", 15, "ROM < 100 deg"),
    ("Step Ups", "Quadriceps", "Hard", "Strength Training", "Late Recovery", 15, "Strength < 60%"),
    ("Lateral Band Walks", "Hip Abductors", "Moderate", "Functional Training", "Mid-Late Recovery", 10, None),
    ("Agility Ladder Drills", "Full Body", "Hard", "Functional Training", "Late Recovery", 20, "Strength < 70%"),
    ("Single Leg Hop", "Full Body", "Hard", "Return to Sport Drills", "Late Recovery", 15, "Strength < 80%"),
    ("Cutting Drills", "Full Body", "Hard", "Return to Sport Drills", "Late Recovery", 20, "Strength < 85%"),
    ("Sport-Specific Drills", "Full Body", "Hard", "Return to Sport Drills", "Late Recovery", 25, "Strength < 85%"),
]

CLINICIAN_REMARKS = [
    "Progressing as expected", "Slight delay, monitor closely", "Excellent compliance and progress",
    "Compliance low, discussed barriers", "Swelling higher than expected", "Ready to advance phase",
    "Patient reports increased confidence", "Mild setback, reduced load this week",
]
PATIENT_FEEDBACK = [
    "Feeling stronger", "Still some discomfort", "Noticeable improvement in mobility",
    "Struggled with home exercises", "Confident with daily activities", "Fatigued after sessions",
    "Motivated and on track", "Anxious about re-injury",
]

SWELLING_LEVELS = ["Severe", "Moderate", "Mild", "None"]  # index 0 -> 3 = improving

# ----------------------------------------------------------------------
# DOCTOR REFERENCE DATA
# ----------------------------------------------------------------------
# Five realistic orthopaedic-rehab subspecialties. Every INJURY_TYPES entry
# above maps to exactly one "home" specialty; mild, non-surgical cases are
# sometimes co-managed by the general rehab specialty instead (see
# assign_doctors_to_patients), which is why that specialty exists even
# though no injury type maps to it directly.
DOCTOR_SPECIALTIES = [
    # name, department, num_doctors
    ("Sports Medicine & Knee Ligament Reconstruction", "Orthopaedics - Sports Medicine", 5),
    ("Foot & Ankle Orthopaedics", "Orthopaedics - Foot & Ankle", 4),
    ("Shoulder & Upper Limb Orthopaedics", "Orthopaedics - Upper Limb", 3),
    ("Trauma & Fracture Surgery", "Orthopaedics - Trauma", 4),
    ("General Musculoskeletal Rehabilitation & Sports Injuries", "Physical Medicine & Rehabilitation", 4),
]

# Which specialty is the "home" specialty for each injury type.
INJURY_TO_SPECIALTY = {
    "ACL Tear": "Sports Medicine & Knee Ligament Reconstruction",
    "Meniscus Tear": "Sports Medicine & Knee Ligament Reconstruction",
    "Ankle Fracture": "Foot & Ankle Orthopaedics",
    "Achilles Tendon Rupture": "Foot & Ankle Orthopaedics",
    "Rotator Cuff Tear": "Shoulder & Upper Limb Orthopaedics",
    "Tibial Fracture": "Trauma & Fracture Surgery",
}
GENERAL_REHAB_SPECIALTY = "General Musculoskeletal Rehabilitation & Sports Injuries"

HOSPITAL_BRANCHES = [
    "Riverside Orthopaedic & Sports Medicine Center",
    "Lakeside Rehabilitation Hospital - Main Campus",
    "Northgate Orthopaedic Trauma Center",
    "Fieldstone Sports Medicine Clinic",
]

DOCTOR_FIRST_NAMES = [
    "Sarah", "Michael", "Priya", "James", "Aisha", "Daniel", "Emily", "Carlos",
    "Grace", "Kevin", "Natasha", "Robert", "Mei", "Andrew", "Fatima", "Brian",
    "Laura", "Samuel", "Chloe", "Thomas", "Nadia", "Peter", "Hannah", "Marcus",
]
DOCTOR_LAST_NAMES = [
    "Coleman", "Tran", "Nair", "Whitfield", "Okonkwo", "Reyes", "Bennett", "Alvarez",
    "Sinclair", "Park", "Dubois", "Kowalski", "Chen", "Morrison", "Haddad", "Foster",
    "Mitchell", "Delgado", "Ivanova", "Blackwood", "Osei", "Baptiste", "Larsen", "Vance",
]


def generate_doctors():
    """
    Builds a small, realistic Doctor roster (~20 doctors across 5 orthopaedic
    rehab subspecialties). Names are synthetically combined from first/last
    name pools -- not real people. Years_Experience and Hospital_Branch are
    used later, in assign_doctors_to_patients(), to route severe cases toward
    more experienced doctors, which is why they're generated here rather
    than left purely cosmetic.
    """
    rows = []
    doctor_num = 1
    used_names = set()

    for specialty, department, count in DOCTOR_SPECIALTIES:
        for _ in range(count):
            # avoid duplicate full names within the roster
            while True:
                first = rng.choice(DOCTOR_FIRST_NAMES)
                last = rng.choice(DOCTOR_LAST_NAMES)
                full_name = f"Dr. {first} {last}"
                if full_name not in used_names:
                    used_names.add(full_name)
                    break

            years_experience = int(rng.integers(3, 31))
            branch = rng.choice(HOSPITAL_BRANCHES)
            license_number = f"ORTHO-{rng.integers(10000, 99999)}"
            email = f"{first.lower()}.{last.lower()}@{branch.split()[0].lower()}health.org"
            joined_year = min(2025, 2025 - years_experience + int(rng.integers(0, 4)))
            joined_date = date(max(joined_year, 1995), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
            max_caseload = int(rng.integers(25, 70))

            rows.append({
                "Doctor_ID": f"DOC{doctor_num:04d}",
                "Doctor_Name": full_name,
                "Specialization": specialty,
                "Department": department,
                "Hospital_Branch": branch,
                "Years_Experience": years_experience,
                "License_Number": license_number,
                "Contact_Email": email,
                "Joined_Date": joined_date,
                "Max_Patient_Caseload": max_caseload,
            })
            doctor_num += 1

    return pd.DataFrame(rows)


def assign_doctors_to_patients(patients: pd.DataFrame, injuries: pd.DataFrame, doctors: pd.DataFrame) -> pd.Series:
    """
    Assigns each patient a single treating doctor (Assigned_Doctor_ID),
    matched by injury type to that doctor's specialty -- the same clinical
    routing logic a real rehab hospital would use:
      - Mild, non-surgical cases have a 40% chance of being routed to a
        General Musculoskeletal Rehabilitation doctor instead of the
        injury-specific specialist (realistic co-management pattern).
      - Severe cases are weighted toward the more experienced doctors within
        the matched specialty, rather than picked uniformly at random.
    Returns a Series aligned to `patients.index`, ready to assign as a new
    Patient column.
    """
    inj_by_patient = injuries.set_index("Patient_ID")
    assigned = []

    for pid in patients["Patient_ID"]:
        inj = inj_by_patient.loc[pid]
        base_specialty = INJURY_TO_SPECIALTY[inj["Injury_Type"]]

        if inj["Severity"] == "Mild" and inj["Surgery_Required"] == 0 and rng.random() < 0.4:
            specialty = GENERAL_REHAB_SPECIALTY
        else:
            specialty = base_specialty

        pool = doctors[doctors["Specialization"] == specialty]

        if inj["Severity"] == "Severe":
            # weight toward more experienced doctors for severe cases
            weights = pool["Years_Experience"].to_numpy(dtype=float)
            weights = weights / weights.sum()
            choice = rng.choice(pool["Doctor_ID"].to_numpy(), p=weights)
        else:
            choice = rng.choice(pool["Doctor_ID"].to_numpy())

        assigned.append(choice)

    return pd.Series(assigned, index=patients.index, name="Assigned_Doctor_ID")


# ----------------------------------------------------------------------
# 1. PATIENT
# ----------------------------------------------------------------------
def generate_patients(n):
    ages = rng.integers(15, 70, n)
    genders = rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    heights = np.where(genders == "Male",
                        rng.normal(175, 7, n), rng.normal(162, 6, n)).round(1)
    weights = np.where(genders == "Male",
                        rng.normal(78, 12, n), rng.normal(65, 11, n)).round(1)
    bmi = (weights / ((heights / 100) ** 2)).round(1)

    occupation = rng.choice(OCCUPATIONS, n, p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05])
    activity_level = rng.choice(ACTIVITY_LEVELS, n, p=[0.2, 0.35, 0.3, 0.15])
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.65, 0.2, 0.15])
    diabetes = rng.choice([0, 1], n, p=[0.85, 0.15])
    hypertension = rng.choice([0, 1], n, p=[0.8, 0.2])
    prev_injuries = rng.poisson(0.4, n)

    enroll_year = rng.choice([2023, 2024, 2025], n, p=[0.35, 0.35, 0.3])
    enroll_date = [date(int(y), int(rng.integers(1, 13)), int(rng.integers(1, 28))) for y in enroll_year]

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
        "Enrollment_Date": enroll_date,
        "Enrollment_Year": enroll_year,
    })
    return df

# ----------------------------------------------------------------------
# 2. INJURY
# ----------------------------------------------------------------------
def generate_injuries(patients):
    n = len(patients)
    types_idx = rng.choice(len(INJURY_TYPES), n, p=[t[2] for t in INJURY_TYPES])
    injury_type = [INJURY_TYPES[i][0] for i in types_idx]
    body_part = [INJURY_TYPES[i][1] for i in types_idx]
    affected_leg = rng.choice(["Left", "Right"], n)
    affected_leg = [al if bp in ("Knee", "Ankle", "Lower Leg") else "N/A"
                    for al, bp in zip(affected_leg, body_part)]

    severity = rng.choice(SEVERITIES, n, p=[0.3, 0.45, 0.25])
    surgery_required = np.array([1 if (s == "Severe" or (s == "Moderate" and rng.random() < 0.4)) else 0
                                  for s in severity])
    surgery_type = [
        (f"{it} Reconstruction" if sr else None)
        for it, sr in zip(injury_type, surgery_required)
    ]

    injury_date = list(patients["Enrollment_Date"] - pd.to_timedelta(rng.integers(0, 14, n), unit="D"))
    surgery_date = [
        (idt + timedelta(days=int(rng.integers(3, 21)))) if sr else None
        for idt, sr in zip(injury_date, surgery_required)
    ]

    df = pd.DataFrame({
        "Injury_ID": [f"INJ{i:05d}" for i in range(1, n + 1)],
        "Patient_ID": patients["Patient_ID"],
        "Injury_Type": injury_type,
        "Body_Part": body_part,
        "Affected_Leg": affected_leg,
        "Severity": severity,
        "Surgery_Required": surgery_required,
        "Surgery_Type": surgery_type,
        "Injury_Date": injury_date,
        "Surgery_Date": surgery_date,
    })
    return df

# ----------------------------------------------------------------------
# 3. INITIAL ASSESSMENT
# ----------------------------------------------------------------------
def generate_initial_assessments(patients, injuries):
    n = len(patients)
    sev_penalty = injuries["Severity"].map({"Mild": 0, "Moderate": 1, "Severe": 2}).values

    pain = np.clip(rng.normal(5 + sev_penalty * 1.2, 1.3, n), 0, 10).round(0).astype(int)
    swelling = np.array(["None", "Mild", "Moderate", "Severe"])[
        np.clip((sev_penalty + rng.integers(0, 2, n)), 0, 3)
    ]
    rom = np.clip(rng.normal(70 - sev_penalty * 12, 10, n), 10, 140).round(0).astype(int)
    strength = np.clip(rng.normal(55 - sev_penalty * 10, 10, n), 10, 100).round(0).astype(int)
    balance = np.clip(rng.normal(60 - sev_penalty * 8, 12, n), 5, 100).round(0).astype(int)
    walk_speed = np.clip(rng.normal(0.8 - sev_penalty * 0.15, 0.15, n), 0.1, 1.6).round(2)
    weight_bearing = rng.choice(["Non-Weight-Bearing", "Partial Weight-Bearing", "Full Weight-Bearing"], n,
                                 p=[0.2, 0.45, 0.35])
    recovery_goal = rng.choice(RECOVERY_GOALS, n, p=[0.35, 0.25, 0.2, 0.2])

    df = pd.DataFrame({
        "Assessment_ID": [f"IA{i:05d}" for i in range(1, n + 1)],
        "Patient_ID": patients["Patient_ID"],
        "Pain_Score": pain,
        "Swelling": swelling,
        "Range_of_Motion_deg": rom,
        "Muscle_Strength_pct": strength,
        "Balance_Score": balance,
        "Walking_Speed_mps": walk_speed,
        "Weight_Bearing_Level": weight_bearing,
        "Recovery_Goal": recovery_goal,
    })
    return df

# ----------------------------------------------------------------------
# 4. EXERCISE LIBRARY
# ----------------------------------------------------------------------
def build_exercise_library():
    df = pd.DataFrame(EXERCISE_LIBRARY, columns=[
        "Exercise_Name", "Target_Muscle", "Difficulty", "Exercise_Category",
        "Suitable_Recovery_State", "Typical_Duration_min", "Contraindication"
    ])
    df.insert(0, "Exercise_ID", [f"EX{i:03d}" for i in range(1, len(df) + 1)])
    return df

# ----------------------------------------------------------------------
# ARCHETYPE ASSIGNMENT (hidden generative logic, not a schema table)
# ----------------------------------------------------------------------
def assign_archetypes(patients, injuries):
    n = len(patients)
    scores = {name: np.zeros(n) for name in ARCHETYPE_NAMES}

    age = patients["Age"].values
    activity = patients["Activity_Level"].values
    diabetes = patients["Diabetes"].values
    smoking = patients["Smoking_Status"].values
    severity = injuries["Severity"].map({"Mild": 0, "Moderate": 1, "Severe": 2}).values

    scores["Fast_Track"] += (age < 30) * 1.5 + (activity == "Highly Active") * 1.5 + (severity == 0) * 1.0
    scores["Slow_Comorbid_Track"] += (diabetes == 1) * 2.0 + (age > 50) * 1.2 + (smoking == "Current") * 1.0 + (severity == 2) * 1.0
    scores["Cautious_Track"] += (severity == 1) * 1.0 + ((age >= 30) & (age <= 50)) * 0.8
    scores["Standard_Track"] += 1.0

    for name in ARCHETYPE_NAMES:
        scores[name] += rng.normal(0, 1.0, n)

    score_matrix = np.vstack([scores[name] for name in ARCHETYPE_NAMES])
    assigned_idx = np.argmax(score_matrix, axis=0)
    assigned = np.array(ARCHETYPE_NAMES)[assigned_idx]
    return assigned

# ----------------------------------------------------------------------
# 6. WEEKLY PROGRESS (+ per-patient trajectory used elsewhere)
#
# Clinical model: every metric moves toward an injury/severity-specific
# TARGET using an asymptotic (diminishing-returns) update:
#     value += (target - value) * step_fraction
# step_fraction is derived so that, at *reference* pace (full compliance,
# no comorbidities), the patient closes ~90% of the gap to target by
# their own expected n_weeks. Compliance and comorbidities scale that
# pace up/down, which is what naturally produces Full / Partial /
# Incomplete recovery outcomes -- not an arbitrary global week cap.
# ----------------------------------------------------------------------
def generate_weekly_progress(patients, injuries, assessments, archetypes):
    rows = []
    traj_store = {}

    ia = assessments.set_index("Patient_ID")
    inj = injuries.set_index("Patient_ID")
    pat = patients.set_index("Patient_ID")

    for pid, arche in zip(patients["Patient_ID"], archetypes):
        base = ia.loc[pid]
        injury_row = inj.loc[pid]
        patient_row = pat.loc[pid]
        rate = ARCHETYPES[arche]["rate_mult"]

        itype = injury_row["Injury_Type"]
        severity = injury_row["Severity"]
        surgery = bool(injury_row["Surgery_Required"])
        clinical = INJURY_LOOKUP[itype]

        # ---- recovery duration: injury-type baseline, scaled by severity,
        # comorbidities, and archetype pace; clipped to the injury's own
        # clinically plausible window (NOT a global 16-week cap) ----
        comorbidity_duration_mult = (
            1
            + 0.15 * patient_row["Diabetes"]
            + 0.10 * (patient_row["Smoking_Status"] == "Current")
            + 0.10 * (patient_row["Age"] > 60)
        )
        mean_weeks = (clinical["base_weeks"] * SEVERITY_DURATION_MULT[severity]
                      * comorbidity_duration_mult / rate)
        n_weeks = int(np.clip(
            rng.normal(mean_weeks, mean_weeks * 0.15),
            clinical["min_weeks"], clinical["max_weeks"]
        ))

        # ---- per-patient targets (clinical ceiling each metric converges to) ----
        target_penalty = SEVERITY_TARGET_PENALTY[severity] + (5 if surgery else 0)
        target_rom = max(95, 145 - target_penalty)
        target_strength = max(55, 100 - target_penalty)
        target_balance = max(55, 98 - target_penalty * 0.8)
        target_walk = max(0.9, 1.65 - target_penalty * 0.012)
        target_pain = {"Mild": 0.5, "Moderate": 1.0, "Severe": 1.8}[severity]

        # ---- per-patient random effects ----
        compliance_base = np.clip(rng.normal(80, 12), 20, 100)
        noise_scale = rng.uniform(0.85, 1.15)

        comorbidity_rate_mult = (
            1
            - 0.10 * patient_row["Diabetes"]
            - 0.08 * (patient_row["Smoking_Status"] == "Current")
            - 0.05 * (patient_row["Age"] > 60)
        )

        # reference weekly fraction: closes ~90% of gap by n_weeks at full compliance
        F = 1 - 0.1 ** (1 / n_weeks)

        pain = float(base["Pain_Score"])
        swelling_idx = SWELLING_LEVELS.index(base["Swelling"]) if base["Swelling"] in SWELLING_LEVELS else 1
        rom = float(base["Range_of_Motion_deg"])
        strength = float(base["Muscle_Strength_pct"])
        balance = float(base["Balance_Score"])
        walk = float(base["Walking_Speed_mps"])

        weekly_traj = {}
        for week in range(1, n_weeks + 1):
            compliance = np.clip(rng.normal(compliance_base, 8), 0, 100)
            comp_factor = np.clip(compliance / 85, 0.5, 1.3)
            week_noise = rng.uniform(0.85, 1.15)
            step = F * comp_factor * comorbidity_rate_mult * noise_scale * week_noise
            step = float(np.clip(step, 0.02, 0.6))

            pain = max(target_pain * 0.6, pain - (pain - target_pain) * step)
            rom = min(145, rom + (target_rom - rom) * step)
            strength = min(100, strength + (target_strength - strength) * step)
            balance = min(100, balance + (target_balance - balance) * step)
            walk = min(1.8, walk + (target_walk - walk) * step)
            if rng.random() < step:
                swelling_idx = min(3, swelling_idx + 1)

            weekly_traj[week] = dict(pain=pain, swelling_idx=swelling_idx, rom=rom,
                                      strength=strength, balance=balance, walk=walk,
                                      compliance=compliance)

            rows.append({
                "Weekly_ID": f"W{pid}_{week:02d}",
                "Patient_ID": pid,
                "Week": week,
                "Pain": round(pain, 1),
                "Swelling": SWELLING_LEVELS[swelling_idx],
                "ROM_deg": round(rom, 1),
                "Quadriceps_Strength_pct": round(strength, 1),
                "Hamstring_Strength_pct": round(max(0, strength - rng.uniform(0, 8)), 1),
                "Balance_Score": round(balance, 1),
                "Walking_Speed_mps": round(walk, 2),
                "Compliance_pct": round(compliance, 1),
                "Patient_Feedback": rng.choice(PATIENT_FEEDBACK),
                "Clinician_Remark": rng.choice(CLINICIAN_REMARKS),
            })

        traj_store[pid] = {"archetype": arche, "n_weeks": n_weeks, "weekly": weekly_traj,
                            "compliance_base": compliance_base,
                            "target_rom": target_rom, "target_strength": target_strength,
                            "target_pain": target_pain}

    return pd.DataFrame(rows), traj_store

# ----------------------------------------------------------------------
# 5. THERAPY PLAN (phase transitions scale with each patient's n_weeks)
# ----------------------------------------------------------------------
def generate_therapy_plan(patients, traj_store, exercise_lib):
    rows = []
    lib_by_cat = exercise_lib.groupby("Exercise_Category")

    for pid in patients["Patient_ID"]:
        info = traj_store[pid]
        arche = info["archetype"]
        seq = ARCHETYPES[arche]["sequence"]
        fracs = ARCHETYPES[arche]["week_offset_frac"]
        n_weeks = info["n_weeks"]
        offsets = [max(1, round(f * n_weeks)) for f in fracs]

        for week in range(1, n_weeks + 1):
            phase_idx = 0
            for i, off in enumerate(offsets):
                if week >= off:
                    phase_idx = i
            category = seq[min(phase_idx, len(seq) - 1)]
            if category not in lib_by_cat.groups:
                continue
            candidates = lib_by_cat.get_group(category)
            n_ex = rng.integers(1, 3)
            chosen = candidates.sample(n=min(n_ex, len(candidates)), random_state=int(rng.integers(0, 1_000_000)))
            for _, ex in chosen.iterrows():
                rows.append({
                    "Plan_ID": f"PL{pid}_{week:02d}_{ex['Exercise_ID']}",
                    "Patient_ID": pid,
                    "Week_Number": week,
                    "Exercise_ID": ex["Exercise_ID"],
                    "Repetitions": int(rng.choice([8, 10, 12, 15])),
                    "Sets": int(rng.choice([2, 3, 4])),
                    "Duration_min": ex["Typical_Duration_min"],
                    "Therapist_Notes": rng.choice(
                        ["Good form maintained", "Reduce load if pain increases",
                         "Progress next week if tolerated", "Monitor for compensation patterns"]),
                })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 7. MUSCLE COMPENSATION
# ----------------------------------------------------------------------
def generate_muscle_compensation(weekly_progress, exercise_lib):
    rows = []
    corrective_by_muscle = {
        "Quadriceps": exercise_lib[exercise_lib["Target_Muscle"] == "Glutes"],
        "Hamstring": exercise_lib[exercise_lib["Target_Muscle"] == "Quadriceps"],
        "Glutes": exercise_lib[exercise_lib["Target_Muscle"] == "Hip Abductors"],
    }
    muscles = list(corrective_by_muscle.keys())

    for _, row in weekly_progress.iterrows():
        risk_score = (100 - row["Quadriceps_Strength_pct"]) / 100 * (row["Compliance_pct"] / 100)
        if rng.random() < risk_score * 0.25:
            muscle = rng.choice(muscles)
            comp_level = rng.choice(["Mild", "Moderate", "High"], p=[0.5, 0.35, 0.15])
            risk_level = {"Mild": "Low", "Moderate": "Medium", "High": "High"}[comp_level]
            options = corrective_by_muscle[muscle]
            correction_ex = options.sample(1).iloc[0]["Exercise_ID"] if len(options) else None
            rows.append({
                "Compensation_ID": f"MC{row['Weekly_ID']}",
                "Patient_ID": row["Patient_ID"],
                "Week": row["Week"],
                "Muscle_Overloaded": muscle,
                "Compensation_Level": comp_level,
                "Risk_Level": risk_level,
                "Suggested_Correction_Exercise_ID": correction_ex,
            })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# COHORT DEFINITION + RECOVERY PATHWAY + PATHWAY STEPS
# ----------------------------------------------------------------------
def build_cohorts_and_pathways(patients, injuries, traj_store):
    merged = patients.merge(injuries, on="Patient_ID")
    merged["Age_Band"] = pd.cut(merged["Age"], bins=[0, 25, 40, 55, 100],
                                 labels=["<25", "25-40", "40-55", "55+"])

    cohort_rows = []
    pathway_rows = []
    step_rows = []
    cohort_id_counter = 1
    pathway_id_counter = 1

    grouped = merged.groupby(["Age_Band", "Injury_Type", "Severity"], observed=True)
    for (age_band, injury_type, severity), group in grouped:
        if len(group) < 5:
            continue
        cohort_id = f"C{cohort_id_counter:03d}"
        cohort_id_counter += 1
        cohort_rows.append({
            "Cohort_ID": cohort_id,
            "Age_Band": age_band,
            "Injury_Type": injury_type,
            "Severity": severity,
            "Cohort_Size": len(group),
        })

        for arche, sub in group.groupby(
            group["Patient_ID"].map(lambda p: traj_store[p]["archetype"])
        ):
            if len(sub) < 3:
                continue
            pids = sub["Patient_ID"].tolist()
            final_weeks = [traj_store[p]["n_weeks"] for p in pids]
            final_strengths = [traj_store[p]["weekly"][traj_store[p]["n_weeks"]]["strength"] for p in pids]
            final_pain = [traj_store[p]["weekly"][traj_store[p]["n_weeks"]]["pain"] for p in pids]
            final_rom = [traj_store[p]["weekly"][traj_store[p]["n_weeks"]]["rom"] for p in pids]
            success_rate = np.mean([1 if fs > 0.85 * traj_store[p]["target_strength"] and fp < traj_store[p]["target_pain"] + 1.0 else 0
                                     for fs, fp, p in zip(final_strengths, final_pain, pids)])

            pathway_id = f"PW{pathway_id_counter:03d}"
            pathway_id_counter += 1
            pathway_rows.append({
                "Pathway_ID": pathway_id,
                "Cohort_ID": cohort_id,
                "Archetype_Label": arche,
                "Sample_Size": len(pids),
                "Average_Recovery_Time_weeks": round(float(np.mean(final_weeks)), 1),
                "Success_Rate": round(float(success_rate), 2),
                "Average_Pain_Reduction": round(float(10 - np.mean(final_pain)), 1),
                "Average_ROM_deg": round(float(np.mean(final_rom)), 1),
            })

            for step_order, (cat, frac) in enumerate(
                zip(ARCHETYPES[arche]["sequence"], ARCHETYPES[arche]["week_offset_frac"]), start=1
            ):
                avg_weeks = float(np.mean(final_weeks))
                step_rows.append({
                    "Pathway_ID": pathway_id,
                    "Step_Order": step_order,
                    "Exercise_Category": cat,
                    "Week_Offset": max(1, round(frac * avg_weeks)),
                })

    return (pd.DataFrame(cohort_rows), pd.DataFrame(pathway_rows), pd.DataFrame(step_rows))

# ----------------------------------------------------------------------
# SIMILAR PATIENTS (simple Gower-like distance on initial profile)
# ----------------------------------------------------------------------
def compute_similar_patients(patients, injuries, assessments, top_k=5):
    merged = patients.merge(injuries, on="Patient_ID").merge(assessments, on="Patient_ID")
    num_cols = ["Age", "BMI", "Pain_Score", "Range_of_Motion_deg", "Muscle_Strength_pct", "Balance_Score"]
    norm = merged[num_cols].copy()
    for c in num_cols:
        rng_ = (norm[c].max() - norm[c].min()) or 1
        norm[c] = (norm[c] - norm[c].min()) / rng_

    cat_cols = ["Gender", "Injury_Type", "Severity", "Diabetes", "Smoking_Status"]
    cat_data = merged[cat_cols].reset_index(drop=True)
    num_data = norm.reset_index(drop=True)
    pids = merged["Patient_ID"].values

    n = len(merged)
    rows = []
    num_matrix = num_data.values
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
                "Similarity_Score": round(float(similarity[j]), 3),
                "Similarity_Basis": "Initial-Profile",
                "Rank": rank,
            })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# SIMULATION + WEEKLY RECOMMENDATION
# ----------------------------------------------------------------------
def generate_simulations_and_recommendations(patients, traj_store, pathways):
    sim_rows = []
    rec_rows = []
    sim_counter = 1
    rec_counter = 1

    for pid in patients["Patient_ID"]:
        info = traj_store[pid]
        n_weeks = info["n_weeks"]
        true_arche = info["archetype"]
        decision_weeks = sorted(set([0] + [w for w in (4, 8, int(n_weeks * 0.5)) if w < n_weeks]))
        candidate_pathways = pathways.sample(n=min(3, len(pathways)),
                                              random_state=int(rng.integers(0, 1_000_000)))

        for dweek in decision_weeks:
            current = info["weekly"].get(dweek, info["weekly"][1])
            sims_this_round = []
            for _, prow in candidate_pathways.iterrows():
                match_bonus = 0.15 if prow["Archetype_Label"] == true_arche else 0.0
                confidence = float(np.clip(rng.normal(0.6 + match_bonus, 0.1), 0.2, 0.95))
                predicted_recovery_time = prow["Average_Recovery_Time_weeks"] * rng.uniform(0.9, 1.1)
                predicted_rom = min(145, prow["Average_ROM_deg"] * rng.uniform(0.95, 1.05))
                predicted_strength = min(100, current["strength"] + rng.uniform(10, 30))
                predicted_recovery_label = "Full Recovery" if prow["Success_Rate"] > 0.6 else "Partial Recovery"

                sim_id = f"SM{sim_counter:06d}"
                sim_counter += 1
                sim_rows.append({
                    "Simulation_ID": sim_id,
                    "Patient_ID": pid,
                    "Week_Number": dweek,
                    "Pathway_ID": prow["Pathway_ID"],
                    "Predicted_Recovery": predicted_recovery_label,
                    "Predicted_Recovery_Time_weeks": round(float(predicted_recovery_time), 1),
                    "Predicted_ROM_deg": round(float(predicted_rom), 1),
                    "Predicted_Strength_pct": round(float(predicted_strength), 1),
                    "Confidence": round(confidence, 2),
                })
                sims_this_round.append((sim_id, prow["Pathway_ID"], confidence))

            best_sim_id, best_pathway_id, best_conf = max(sims_this_round, key=lambda x: x[2])
            decision = rng.choice(["Accepted", "Accepted", "Accepted", "Modified", "Overridden"])
            override_reason = None
            if decision != "Accepted":
                override_reason = rng.choice([
                    "Clinician judgment based on patient tolerance",
                    "Patient preference for different pace",
                    "Scheduling constraints",
                    "Recent flare-up in symptoms",
                ])

            rec_rows.append({
                "Recommendation_ID": f"REC{rec_counter:06d}",
                "Patient_ID": pid,
                "Week": dweek,
                "Current_Status": f"Pain {round(current['pain'],1)}, ROM {round(current['rom'],1)}, "
                                   f"Strength {round(current['strength'],1)}%",
                "Simulation_ID": best_sim_id,
                "Recommended_Pathway": best_pathway_id,
                "Reason": "Best simulated outcome among candidate pathways for similar cohort",
                "Confidence": round(best_conf, 2),
                "Expected_Next_Milestone": rng.choice([
                    "ROM > 100 deg by next review", "Strength > 70% by next review",
                    "Pain < 3 by next review", "Full weight-bearing by next review"]),
                "Clinician_Decision": decision,
                "Override_Reason": override_reason,
            })
            rec_counter += 1

    return pd.DataFrame(sim_rows), pd.DataFrame(rec_rows)

# ----------------------------------------------------------------------
# FINAL OUTCOME
#
# Thresholds are now RELATIVE to each patient's own clinical target
# (which already reflects severity/surgery), not one-size-fits-all
# absolute numbers.
# ----------------------------------------------------------------------
def generate_final_outcomes(patients, injuries, traj_store):
    rows = []
    for pid in patients["Patient_ID"]:
        info = traj_store[pid]
        last_week = info["n_weeks"]
        final = info["weekly"][last_week]
        target_strength = info["target_strength"]
        target_rom = info["target_rom"]
        target_pain = info["target_pain"]

        strength_ratio = final["strength"] / target_strength
        rom_ratio = final["rom"] / target_rom
        pain_ok = final["pain"] <= target_pain + 1.0

        return_to_activity = 1 if (strength_ratio >= 0.85 and rom_ratio >= 0.85 and pain_ok) else 0

        if strength_ratio >= 0.9 and rom_ratio >= 0.9 and final["pain"] <= target_pain + 0.5:
            label = "Full Recovery"
        elif strength_ratio >= 0.7 and rom_ratio >= 0.75:
            label = "Partial Recovery"
        else:
            label = "Incomplete Recovery"

        rows.append({
            "Outcome_ID": f"FO{pid}",
            "Patient_ID": pid,
            "Discharge_Week": last_week,
            "Actual_Recovery_Time_weeks": last_week,
            "Actual_ROM_deg": round(final["rom"], 1),
            "Actual_Strength_pct": round(final["strength"], 1),
            "Actual_Pain_Score": round(final["pain"], 1),
            "Return_To_Activity_Flag": return_to_activity,
            "Outcome_Label": label,
        })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("Generating patients...")
    patients = generate_patients(N_PATIENTS)

    print("Generating injuries...")
    injuries = generate_injuries(patients)

    print("Generating doctor roster...")
    doctors = generate_doctors()

    print("Assigning treating doctors to patients (by injury specialty)...")
    patients["Assigned_Doctor_ID"] = assign_doctors_to_patients(patients, injuries, doctors)

    print("Generating initial assessments...")
    assessments = generate_initial_assessments(patients, injuries)

    print("Building exercise library...")
    exercise_lib = build_exercise_library()

    print("Assigning hidden recovery archetypes...")
    archetypes = assign_archetypes(patients, injuries)

    print("Generating weekly progress trajectories (clinically calibrated)...")
    weekly_progress, traj_store = generate_weekly_progress(patients, injuries, assessments, archetypes)

    print("Generating therapy plans...")
    therapy_plan = generate_therapy_plan(patients, traj_store, exercise_lib)

    print("Generating muscle compensation events...")
    muscle_comp = generate_muscle_compensation(weekly_progress, exercise_lib)

    print("Building cohorts and mined recovery pathways...")
    cohorts, pathways, pathway_steps = build_cohorts_and_pathways(patients, injuries, traj_store)

    print("Computing similar patients...")
    similar_patients = compute_similar_patients(patients, injuries, assessments)

    print("Generating simulations and weekly recommendations...")
    simulations, recommendations = generate_simulations_and_recommendations(patients, traj_store, pathways)

    print("Generating final outcomes...")
    final_outcomes = generate_final_outcomes(patients, injuries, traj_store)

    tables = {
        "Doctor": doctors,
        "Patient": patients,
        "Injury": injuries,
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

    print("\nDone. All tables written to ./output/")

if __name__ == "__main__":
    main()
"""
config.py
---------
Central configuration for the Digital Twin application.

CHANGES vs. previous version
  * MySQL password is no longer hard-coded (set DT_MYSQL_PASSWORD in your environment).
  * Added formal, tunable parameters for: patient similarity, deviation rule,
    digital-twin update gains, and pathway-ranking weights (patent review items).
"""

import os

# ---------------------------------------------------------------- DATA SOURCE
DATA_SOURCE = os.environ.get("DT_DATA_SOURCE", "mysql")  # "mysql" or "csv"

MYSQL_CONFIG = {
    "user": os.environ.get("DT_MYSQL_USER", "root"),
    "password": os.environ.get("DT_MYSQL_PASSWORD", ""),
    "host": os.environ.get("DT_MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("DT_MYSQL_PORT", "3306")),
    "database": os.environ.get("DT_MYSQL_DB", "DigitalTwinDB"),
}

CSV_DIR = os.environ.get("DT_CSV_DIR", "data")

TABLE_NAMES = [
    "Doctor", "Patient", "Injury", "Initial_Assessment", "Exercise_Library",
    "Therapy_Plan", "Weekly_Progress", "Muscle_Compensation", "Cohort_Definition",
    "Recovery_Pathway", "Pathway_Steps", "Similar_Patients", "Simulation",
    "Weekly_Recommendation", "Final_Outcome",
]

TRAIN_YEARS = [2023, 2024]
TEST_YEARS = [2025]

NUMERIC_FEATURES = [
    "Age", "BMI", "Previous_Injuries", "Pain_Score", "Range_of_Motion_deg",
    "Quadriceps_Strength_pct", "Hamstring_Strength_pct", "Balance_Score",
    "Walking_Speed_mps", "Quad_Ham_Ratio", "Comorbidity_Score", "Pain_ROM_Ratio",
    "Mobility_Index", "Clinical_Est_Weeks",
]

CATEGORICAL_FEATURES = [
    "Gender", "Activity_Level", "Smoking_Status", "Diabetes", "Hypertension",
    "Injury_Type", "Severity", "Surgery_Required",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_RECOVERY_PCT = "Final_Recovery_Pct"
TARGET_RECOVERY_WEEKS = "Actual_Recovery_Time_weeks"

DEVIATION_THRESHOLD_PCT = 15.0
# Recalibration trigger sensitivity analysis values; these are engineering
# settings to evaluate, not clinically validated decision thresholds.
DEVIATION_SENSITIVITY_THRESHOLDS_PCT = (5.0, 10.0, 15.0, 20.0, 25.0)
# Similarity dimensions and fixed component weights are shared by retrieval
# and pathway ranking. Distances are computed only from the training pool.
SIMILARITY_NUMERIC_WEIGHT = 0.60
SIMILARITY_CATEGORICAL_WEIGHT = 0.40
PATHWAY_MIN_SAMPLE_SIZE = 5
RANDOM_STATE = 42

# ------------------------------------------------- PATIENT SIMILARITY (claim)
# sim(q, c) = 1 - ( W_NUM * d_num(q, c) + W_CAT * d_cat(q, c) )
#   d_num : Euclidean distance on min-max scaled numeric features / sqrt(#features)  in [0, 1]
#   d_cat : fraction of categorical features that differ                               in [0, 1]
# Retrieval rule: keep neighbours with sim >= SIM_MIN_SCORE, at least SIM_MIN_K,
# at most SIM_TOP_K.
SIM_NUM_COLS = ["Age", "BMI", "Pain_Score", "Range_of_Motion_deg",
                "Quadriceps_Strength_pct", "Balance_Score"]
SIM_CAT_COLS = ["Gender", "Injury_Type", "Severity", "Diabetes", "Smoking_Status"]
SIM_W_NUM = 0.6
SIM_W_CAT = 0.4
SIM_MIN_SCORE = 0.70
SIM_MIN_K = 3
SIM_TOP_K = 5

# ------------------------------------------------------ DEVIATION RULE (claim)
# Week t is "deviant" if, for >= DEVIATION_MIN_METRICS of {pain, ROM, strength},
#   harmful-direction relative deviation  >  DEVIATION_THRESHOLD_PCT  (percent).
# Recalibration is triggered when the week is deviant for
#   DEVIATION_PERSISTENCE_WEEKS consecutive weeks.
DEVIATION_THRESHOLD_PCT = 15.0
DEVIATION_PERSISTENCE_WEEKS = 2
DEVIATION_MIN_METRICS = 1
DEVIATION_PAIN_DENOM_FLOOR = 1.0   # avoids huge % when predicted pain is ~0.5

# ------------------------------------------- DIGITAL-TWIN UPDATE (claim)
# x_hat_t = x_pred_t + K * (z_t - x_pred_t),  K in [0, 1] per state variable
TWIN_GAIN = {"pain": 0.7, "rom": 0.7, "strength": 0.7, "speed": 0.5}

# ------------------------------------------- PATHWAY RANKING (claim)
# J(p) = w_rec * Rec_p/100 + w_dur * (1 - Weeks_p/W_MAX) - w_risk * Risk_p + w_evid * Evid_p
RANK_WEIGHTS = {"recovery": 0.45, "duration": 0.25, "risk": 0.15, "evidence": 0.15}
RANK_WEEKS_MAX = 52.0

# ---------------------------------------------------------------- ABLATION
ABLATION_EVAL_WEEK = 4
ABLATION_BLEND_RECAL = 0.7        # weight on recalibrated model in "recal + cohort"
UNDERPERFORMER_DELTA_PCT = 10.0   # actual final % below predicted by this many points

THEME_COLORS = {
    "bg_light": "#FFFFFF",        # Pure White Background
    "surface_bg": "#F8FAFC",     # Light Slate Surface
    "card_bg": "#FFFFFF",        # Card Background
    "card_border": "#E2E8F0",    # Crisp Border
    "text_main": "#0F172A",      # Dark Slate Text
    "text_muted": "#64748B",     # Muted Text
    "primary": "#10B981",        # Emerald Green Accent
    "primary_dark": "#059669",
    "amber": "#F59E0B",          # Warm Amber
    "crimson": "#E11D48",        # Crimson Pain Spot
    "purple": "#8B5CF6",         # Secondary Graph Accent
}

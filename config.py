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
DATA_SOURCE = os.environ.get("DT_DATA_SOURCE", "mysql").strip().lower()  # mysql, csv, real
REAL_DATA_PATH = os.environ.get("DT_REAL_DATA_PATH", "")
TARGET_SELECTION = os.environ.get("DT_TARGET", "composite_proxy").strip().lower()
TARGET_CANDIDATES = {
    "composite_proxy": "Final_Recovery_Pct",
    "fim": "FIM_Final",
    "vas_pain": "VAS_Pain_Final",
    "dash": "DASH_Final",
    "koos": "KOOS_Final",
    "lefs": "LEFS_Final",
}
TARGET_OUTCOME_COLUMN = TARGET_CANDIDATES.get(TARGET_SELECTION, TARGET_CANDIDATES["composite_proxy"])
TARGET_DISPLAY_LABEL = {"composite_proxy": "Composite recovery proxy (%)", "fim": "FIM final score", "vas_pain": "Final VAS pain", "dash": "DASH final score", "koos": "KOOS final score", "lefs": "LEFS final score"}.get(TARGET_SELECTION, "Composite recovery proxy (%)")

MYSQL_CONFIG = {
    # No usable database credentials are assumed. MySQL mode requires explicit
    # environment configuration in the process that starts the application.
    "user": os.environ.get("DT_MYSQL_USER", ""),
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
    "Age", "BMI", "Previous_Injuries", "Pain_Score", "Primary_ROM_deg",
    "Primary_Strength_pct", "Secondary_Strength_pct", "Balance_Score",
    "Walking_Speed_mps", "Quad_Ham_Ratio", "Comorbidity_Score", "Pain_ROM_Ratio",
    "Mobility_Index",
]

CATEGORICAL_FEATURES = [
    "Gender", "Sex", "Activity_Level", "Smoking_Status", "Diabetes", "Hypertension",
    "Injury_Type", "Severity", "Surgery_Required",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PRODUCTION_MODEL_NAME = os.environ.get("DT_MODEL", "XGBoost")
MODEL_ARTIFACT_DIR = os.environ.get("DT_MODEL_ARTIFACT_DIR", "reports/model_benchmark/artifacts")
ROM_MAX_BY_REGION = {"knee": 140.0, "shoulder": 180.0, "hip": 130.0, "ankle": 50.0, "spine": 120.0, "wrist": 80.0}
REGION_ALERT_THRESHOLDS = {
    "knee": {"pain": 5.0, "rom_min": 90.0, "strength_min": 60.0, "balance_min": 60.0, "compliance_min": 70.0},
    "shoulder": {"pain": 5.0, "rom_min": 90.0, "strength_min": 60.0, "balance_min": 60.0, "compliance_min": 70.0},
    "hip": {"pain": 5.0, "rom_min": 90.0, "strength_min": 60.0, "balance_min": 60.0, "compliance_min": 70.0},
    "ankle": {"pain": 5.0, "rom_min": 20.0, "strength_min": 60.0, "balance_min": 60.0, "compliance_min": 70.0},
    "spine": {"pain": 5.0, "rom_min": 30.0, "strength_min": 60.0, "balance_min": 60.0, "compliance_min": 70.0},
    "wrist": {"pain": 5.0, "rom_min": 30.0, "strength_min": 60.0, "balance_min": 60.0, "compliance_min": 70.0},
}

TARGET_RECOVERY_PCT = "Final_Recovery_Pct"
TARGET_RECOVERY_WEEKS = "Actual_Recovery_Time_weeks"

# Recalibration trigger sensitivity analysis values; these are engineering
# settings to evaluate, not clinically validated decision thresholds.
DEVIATION_SENSITIVITY_THRESHOLDS_PCT = (5.0, 10.0, 15.0, 20.0, 25.0)
# Similarity dimensions and fixed component weights are shared by retrieval
# and pathway ranking. Distances are computed only from the training pool.
SIMILARITY_NUMERIC_WEIGHT = 0.60
SIMILARITY_CATEGORICAL_WEIGHT = 0.40
PATHWAY_MIN_SAMPLE_SIZE = 5
PATHWAY_SUCCESS_THRESHOLD_PCT = 80.0  # descriptive proxy threshold only; not an efficacy claim
RANDOM_STATE = 42

# ------------------------------------------------- PATIENT SIMILARITY (claim)
# sim(q, c) = 1 - ( W_NUM * d_num(q, c) + W_CAT * d_cat(q, c) )
#   d_num : Euclidean distance on min-max scaled numeric features / sqrt(#features)  in [0, 1]
#   d_cat : fraction of categorical features that differ                               in [0, 1]
# Retrieval rule: keep neighbours with sim >= SIM_MIN_SCORE, at least SIM_MIN_K,
# at most SIM_TOP_K.
SIM_NUM_COLS = ["Age", "BMI", "Pain_Score", "Primary_ROM_deg",
                "Primary_Strength_pct", "Balance_Score"]
SIM_CAT_COLS = ["Gender", "Injury_Type", "Severity", "Diabetes", "Smoking_Status"]
SIM_W_NUM = 0.6
SIM_W_CAT = 0.4
SIM_MIN_SCORE = 0.70
SIM_MIN_K = 3
SIM_TOP_K = 5
SIMILARITY_MODEL_PATH = os.environ.get("DT_SIMILARITY_MODEL_PATH", "reports/similarity/model.json")
SIM_K_CANDIDATES = (3, 5, 10, 20)
SIM_TAU_PERCENTILE = 0.25
DEVIATION_DELTA_SD = 1.5
DEVIATION_PERSISTENCE_WEEKS = 2
DEVIATION_SIGMA_FLOOR = {"Pain": 0.5, "ROM_deg": 2.0, "Strength_pct": 2.0}
DEVIATION_SD_GRID = tuple(x / 2 for x in range(1, 7))
DEVIATION_OLD_PCT_GRID = (5, 10, 15, 20, 25, 30)
MCID_PLACEHOLDERS = {"Pain": None, "ROM_deg": None, "Strength_pct": None}  # TODO: cite validated sources
REGION_MEASURES = {
    "knee": {"primary_strength": "Quadriceps_Strength_pct", "secondary_strength": "Hamstring_Strength_pct", "rom": "Range_of_Motion_deg", "prom": "KOOS_Final"},
    "shoulder": {"primary_strength": "Shoulder_Strength_pct", "secondary_strength": None, "rom": "Shoulder_ROM_deg", "prom": "DASH_Final"},
    "hip": {"primary_strength": "Hip_Strength_pct", "secondary_strength": None, "rom": "Hip_ROM_deg", "prom": "LEFS_Final"},
    "ankle": {"primary_strength": "Ankle_Strength_pct", "secondary_strength": None, "rom": "Ankle_ROM_deg", "prom": "LEFS_Final"},
    "spine": {"primary_strength": "Trunk_Strength_pct", "secondary_strength": None, "rom": "Spine_ROM_deg", "prom": "FIM_Final"},
}

# ------------------------------------------------------ DEVIATION RULE (claim)
# Legacy percentage rule remains available for analysis. The live trigger uses
# direction-aware adverse residual z-scores, with delta from DEVIATION_DELTA_SD.
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
RANK_WEIGHTS = {"recovery": 0.40, "weeks": 0.20, "risk": 0.20, "evidence": 0.20}
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

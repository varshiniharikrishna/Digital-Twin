"""
config.py
---------
Central configuration for the Digital Twin application.
"""

import os

# DATA SOURCE
DATA_SOURCE = os.environ.get("DT_DATA_SOURCE", "mysql")  # "mysql" or "csv"

MYSQL_CONFIG = {
    "user": os.environ.get("DT_MYSQL_USER", "root"),
    "password": os.environ.get("DT_MYSQL_PASSWORD", "varshu2508"),
    "host": os.environ.get("DT_MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("DT_MYSQL_PORT", "3306")),
    "database": os.environ.get("DT_MYSQL_DB", "DigitalTwinDB"),
}

CSV_DIR = os.environ.get("DT_CSV_DIR", "data")

TABLE_NAMES = [
    "Doctor",
    "Patient",
    "Injury",
    "Initial_Assessment",
    "Exercise_Library",
    "Therapy_Plan",
    "Weekly_Progress",
    "Muscle_Compensation",
    "Cohort_Definition",
    "Recovery_Pathway",
    "Pathway_Steps",
    "Similar_Patients",
    "Simulation",
    "Weekly_Recommendation",
    "Final_Outcome",
]

TRAIN_YEARS = [2023, 2024]
TEST_YEARS = [2025]

NUMERIC_FEATURES = [
    "Age",
    "BMI",
    "Previous_Injuries",
    "Pain_Score",
    "Range_of_Motion_deg",
    "Quadriceps_Strength_pct",
    "Hamstring_Strength_pct",
    "Balance_Score",
    "Walking_Speed_mps",
    "Quad_Ham_Ratio",
    "Comorbidity_Score",
    "Pain_ROM_Ratio",
    "Mobility_Index",
    "Clinical_Est_Weeks",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Activity_Level",
    "Smoking_Status",
    "Diabetes",
    "Hypertension",
    "Injury_Type",
    "Severity",
    "Surgery_Required",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_RECOVERY_PCT = "Final_Recovery_Pct"
TARGET_RECOVERY_WEEKS = "Actual_Recovery_Time_weeks"

DEVIATION_THRESHOLD_PCT = 15.0
RANDOM_STATE = 42

# Light Clinical Professional Theme Colors (White Background, Zero Blue)
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
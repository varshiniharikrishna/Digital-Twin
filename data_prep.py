"""
data_prep.py
------------
Builds the single modelling dataset from raw tables for multi-region injuries.
"""

import numpy as np
import pandas as pd
import config

INJURY_BASE_WEEKS = {
    "ACL Tear": 32,
    "Meniscus Tear": 10,
    "Ankle Fracture": 14,
    "Rotator Cuff Tear": 20,
    "Achilles Tendon Rupture": 28,
    "Tibial Fracture": 18,
    "Frozen Shoulder": 18,
    "Lumbar Disc Herniation": 20,
    "Hip Fracture": 20,
    "Distal Radius Fracture": 10,
}
SEVERITY_MULT = {"Mild": 0.75, "Moderate": 1.0, "Severe": 1.35}


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds clinical feature engineering for multi-region datasets."""
    df = df.copy()

    # Resolve Quadriceps / Primary Strength
    quad_col = None
    for col in ["Quadriceps_Strength_pct", "Primary_Strength_pct_x", "Primary_Strength_pct", "Primary_Strength_pct_y", "Muscle_Strength_pct", "Primary_Strength_pct_w1"]:
        if col in df.columns and df[col].notna().any():
            quad_col = col
            break
    if quad_col is not None:
        df["Quadriceps_Strength_pct"] = df[quad_col].astype(float)
    else:
        df["Quadriceps_Strength_pct"] = 50.0

    # Resolve Hamstring / Secondary Strength
    ham_col = None
    for col in ["Hamstring_Strength_pct", "Secondary_Strength_pct", "Secondary_Strength_pct_y", "Secondary_Strength_pct_w1"]:
        if col in df.columns and df[col].notna().any():
            ham_col = col
            break
    if ham_col is not None:
        df["Hamstring_Strength_pct"] = df[ham_col].fillna(df["Quadriceps_Strength_pct"] * 0.85).astype(float)
    else:
        df["Hamstring_Strength_pct"] = (df["Quadriceps_Strength_pct"] * 0.85).astype(float)

    # Resolve Range of Motion
    rom_col = None
    for col in ["Range_of_Motion_deg", "Primary_ROM_deg_x", "Primary_ROM_deg", "Primary_ROM_deg_y", "Primary_ROM_deg_w1", "ROM_deg"]:
        if col in df.columns and df[col].notna().any():
            rom_col = col
            break
    if rom_col is not None:
        df["Range_of_Motion_deg"] = df[rom_col].astype(float)
    else:
        df["Range_of_Motion_deg"] = 100.0

    # Resolve Balance / Function Score
    bal_col = None
    for col in ["Balance_Score", "Function_Score", "Function_Score_x", "Balance_Score_x"]:
        if col in df.columns and df[col].notna().any():
            bal_col = col
            break
    if bal_col is not None:
        df["Balance_Score"] = df[bal_col].astype(float)
    else:
        df["Balance_Score"] = 70.0

    # Resolve Walking Speed
    speed_col = None
    for col in ["Walking_Speed_mps", "Walking_Speed_mps_x"]:
        if col in df.columns and df[col].notna().any():
            speed_col = col
            break
    if speed_col is not None:
        df["Walking_Speed_mps"] = df[speed_col].fillna(1.0).astype(float)
    else:
        df["Walking_Speed_mps"] = 1.0

    # Resolve Pain Score
    pain_col = None
    for col in ["Pain_Score", "Pain_Score_x", "Pain_Score_ia", "Actual_Pain_Score", "Pain"]:
        if col in df.columns and df[col].notna().any():
            pain_col = col
            break
    if pain_col is not None:
        df["Pain_Score"] = df[pain_col].astype(float)
    else:
        df["Pain_Score"] = 5.0

    quad = df["Quadriceps_Strength_pct"].astype(float)
    ham = df["Hamstring_Strength_pct"].astype(float)
    df["Quad_Ham_Ratio"] = (quad / (ham + 1e-5)).round(3)
    
    diabetes = df["Diabetes"].astype(float) if "Diabetes" in df.columns else 0.0
    hypertension = df["Hypertension"].astype(float) if "Hypertension" in df.columns else 0.0
    smoking = (df["Smoking_Status"] == "Current").astype(float) if "Smoking_Status" in df.columns else 0.0
    age = df["Age"].astype(float) if "Age" in df.columns else 40.0
    age_flag = (age > 60).astype(float)
    
    df["Comorbidity_Score"] = (diabetes * 1.5 + hypertension * 1.0 + smoking * 1.2 + age_flag * 1.0).round(2)
    
    pain = df["Pain_Score"].astype(float)
    rom = df["Range_of_Motion_deg"].astype(float)
    df["Pain_ROM_Ratio"] = (pain / (rom + 1e-5)).round(3)
    
    speed = df["Walking_Speed_mps"].astype(float)
    balance = df["Balance_Score"].astype(float)
    df["Mobility_Index"] = (speed * balance / (pain + 1.0)).round(2)
    
    base_wks = df["Injury_Type"].map(INJURY_BASE_WEEKS).fillna(14.0) if "Injury_Type" in df.columns else 14.0
    sev_m = df["Severity"].map(SEVERITY_MULT).fillna(1.0) if "Severity" in df.columns else 1.0
    comorb_mult = 1.0 + 0.15 * diabetes + 0.10 * smoking + 0.10 * age_flag
    df["Clinical_Est_Weeks"] = (base_wks * sev_m * comorb_mult).round(1)
    
    return df


def build_modelling_dataset(tables: dict) -> pd.DataFrame:
    patient = tables["Patient"].copy()
    injury = tables["Injury"].copy()
    assessment = tables["Initial_Assessment"].copy()
    outcome = tables["Final_Outcome"].copy()
    weekly = tables["Weekly_Progress"].copy()

    week1 = (
        weekly[weekly["Week"] == 1]
        [["Patient_ID", "Primary_Strength_pct", "Secondary_Strength_pct"]]
        .drop_duplicates("Patient_ID")
        if "Primary_Strength_pct" in weekly.columns
        else weekly[weekly["Week"] == 1][["Patient_ID", "Quadriceps_Strength_pct", "Hamstring_Strength_pct"]].drop_duplicates("Patient_ID")
    )

    df = (
        patient
        .merge(injury, on="Patient_ID", how="inner", suffixes=("", "_inj"))
        .merge(assessment, on="Patient_ID", how="inner", suffixes=("", "_ia"))
        .merge(week1, on="Patient_ID", how="left")
        .merge(outcome, on="Patient_ID", how="inner", suffixes=("", "_out"))
    )

    df = compute_derived_features(df)

    if "Final_Recovery_Pct" not in df.columns:
        if "Actual_Primary_Strength_pct" in df.columns:
            str_col = df["Actual_Primary_Strength_pct"]
            rom_col = df["Actual_Primary_ROM_deg"]
        else:
            str_col = df["Actual_Strength_pct"]
            rom_col = df["Actual_ROM_deg"]

        df[config.TARGET_RECOVERY_PCT] = np.clip(
            0.5 * str_col + 0.5 * (rom_col / 140 * 100),
            0, 100,
        )

    df["Enrollment_Year"] = df["Enrollment_Year"].astype(int)
    return df


def chronological_split(df: pd.DataFrame):
    train = df[df["Enrollment_Year"].isin(config.TRAIN_YEARS)].reset_index(drop=True)
    test = df[df["Enrollment_Year"].isin(config.TEST_YEARS)].reset_index(drop=True)
    return train, test


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = compute_derived_features(df) if "Quad_Ham_Ratio" not in df.columns else df
    X = df_clean[config.ALL_FEATURES].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return X

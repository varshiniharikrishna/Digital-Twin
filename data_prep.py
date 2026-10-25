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
    outcome = tables["Final_Outcome"].copy() if "Final_Outcome" in tables else pd.DataFrame()
    # Baseline predictors must not contain measurements collected after
    # enrollment. Weekly_Progress starts after Initial_Assessment, so do not
    # join week-1 values into the baseline prediction dataset.

    df = (
        patient
        .merge(injury, on="Patient_ID", how="left", suffixes=("", "_inj"))
        .merge(assessment, on="Patient_ID", how="left", suffixes=("", "_ia"))
        .merge(outcome, on="Patient_ID", how="left", suffixes=("", "_out"))
    )

    df = compute_derived_features(df)

    actual_strength = pd.to_numeric(
        df.get("Actual_Strength_pct", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    actual_rom = pd.to_numeric(
        df.get("Actual_ROM_deg", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    proxy_valid = actual_strength.notna() & actual_rom.notna()
    proxy_pct = pd.Series(np.nan, index=df.index, dtype=float)
    proxy_pct.loc[proxy_valid] = np.clip(
        0.5 * actual_strength.loc[proxy_valid]
        + 0.5 * (actual_rom.loc[proxy_valid] / 140.0 * 100.0), 0, 100
    )
    recorded_pct = pd.to_numeric(
        df.get(config.TARGET_RECOVERY_PCT, pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    df["Recovery_Pct_Target_Source"] = np.select(
        [recorded_pct.notna(), proxy_pct.notna()],
        ["recorded percentage", "derived strength-ROM proxy"],
        default="missing",
    )
    df[config.TARGET_RECOVERY_PCT] = recorded_pct.combine_first(proxy_pct)

    # Never substitute Clinical_Est_Weeks (a predictor) for a missing observed
    # outcome; doing so would make the duration target leak directly from X.
    df[config.TARGET_RECOVERY_WEEKS] = pd.to_numeric(
        df.get(config.TARGET_RECOVERY_WEEKS, pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    df["Recovery_Weeks_Target_Source"] = np.where(
        df[config.TARGET_RECOVERY_WEEKS].notna(), "observed duration", "missing"
    )

    if "Enrollment_Date" in df:
        enrollment_date = pd.to_datetime(df["Enrollment_Date"], errors="coerce")
        reported_year = pd.to_numeric(
            df.get("Enrollment_Year", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        date_year = enrollment_date.dt.year
        df["Enrollment_Year_Mismatch"] = (
            reported_year.notna() & date_year.notna() & reported_year.ne(date_year)
        )
        df["Enrollment_Year"] = date_year.combine_first(reported_year)
    else:
        df["Enrollment_Year"] = pd.to_numeric(
            df.get("Enrollment_Year", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
    if "Enrollment_Date" in df:
        # The schema has a discharge week but no completion timestamp. Keep an
        # explicitly estimated availability date for leakage-aware backtests;
        # it is not a recorded clinical date and is never a model feature.
        enrollment_date = pd.to_datetime(df["Enrollment_Date"], errors="coerce")
        duration_weeks = pd.to_numeric(df[config.TARGET_RECOVERY_WEEKS], errors="coerce")
        available_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        valid_availability = enrollment_date.notna() & duration_weeks.notna()
        available_date.loc[valid_availability] = enrollment_date.loc[valid_availability] + pd.to_timedelta(
            duration_weeks.loc[valid_availability] * 7, unit="D"
        )
        df["Estimated_Outcome_Availability_Date"] = available_date
    return df


def chronological_split(df: pd.DataFrame):
    """Make a genuinely disjoint temporal split; never reuse train as test."""
    year = pd.to_numeric(df.get("Enrollment_Year", pd.Series(np.nan, index=df.index)), errors="coerce")
    train = df[year.isin(config.TRAIN_YEARS)]
    test = df[year.isin(config.TEST_YEARS)]
    if not train.empty and not test.empty:
        if "Enrollment_Date" not in df:
            return train.reset_index(drop=True), test.reset_index(drop=True)
        dates = pd.to_datetime(df["Enrollment_Date"], errors="coerce")
        train_dates, test_dates = dates.loc[train.index].dropna(), dates.loc[test.index].dropna()
        if train_dates.empty or test_dates.empty or train_dates.max() < test_dates.min():
            cutoff = test_dates.min() if not test_dates.empty else None
            if cutoff is not None and "Estimated_Outcome_Availability_Date" in df:
                available = pd.to_datetime(df["Estimated_Outcome_Availability_Date"], errors="coerce")
                train = train.loc[available.loc[train.index].notna() & (available.loc[train.index] < cutoff)]
            return train.reset_index(drop=True), test.reset_index(drop=True)

    # Configured years may not exist in a new database. Fall back to the most
    # recent fifth of unique enrollment dates, preserving calendar-date groups.
    if "Enrollment_Date" in df:
        dates = pd.to_datetime(df["Enrollment_Date"], errors="coerce")
        unique_dates = np.sort(dates.dropna().unique())
        if len(unique_dates) >= 2:
            n_test_dates = max(1, int(np.ceil(len(unique_dates) * 0.20)))
            cutoff = pd.Timestamp(unique_dates[-n_test_dates])
            train_mask = dates < cutoff
            test_mask = dates >= cutoff
            if train_mask.any() and test_mask.any():
                train = df.loc[train_mask]
                test = df.loc[test_mask]
                if "Estimated_Outcome_Availability_Date" in df:
                    available = pd.to_datetime(df["Estimated_Outcome_Availability_Date"], errors="coerce")
                    train = train.loc[available.loc[train.index].notna() & (available.loc[train.index] < cutoff)]
                return train.reset_index(drop=True), test.reset_index(drop=True)

    known_years = np.sort(year.dropna().unique())
    if len(known_years) >= 2:
        cutoff_year = known_years[max(1, int(np.floor(len(known_years) * 0.8)))] if len(known_years) > 2 else known_years[-1]
        train = df[year < cutoff_year]
        test = df[year >= cutoff_year]
        if not train.empty and not test.empty:
            return train.reset_index(drop=True), test.reset_index(drop=True)
    raise ValueError("A disjoint temporal split cannot be created: provide at least two enrollment dates or years.")


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Always recompute engineered features. In particular, weekly observations
    # can change pain, ROM, strength, balance, and walking speed on a row that
    # already contains the derived columns from the baseline state.
    df_clean = compute_derived_features(df)
    X = df_clean[config.ALL_FEATURES].copy()
    for col in config.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return X

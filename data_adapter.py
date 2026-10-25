"""Strict adapter for de-identified, flat rehabilitation CSV/Parquet exports.

The adapter deliberately refuses identifiers and direct identifiers rather than
trying to clean them silently. It does not fabricate missing outcomes.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import yaml
import config

SCHEMA_PATH = Path(__file__).parent / "schema" / "patient_schema.yaml"
DIRECT_ID_RE = re.compile(r"name|email|phone|mobile|mrn|medical.?record|ssn|date.?of.?birth|(^|_)dob($|_)|address", re.I)

class DataValidationError(ValueError):
    pass

def load_real_records(path: str | Path) -> pd.DataFrame:
    if str(path).strip().lower() in {"mysql", "mysql://"}:
        import os
        import re
        from sqlalchemy import create_engine, text
        url=os.environ.get("DT_REAL_MYSQL_URL","")
        table=os.environ.get("DT_REAL_MYSQL_TABLE","")
        if not url or not re.fullmatch(r"[A-Za-z0-9_]+",table):
            raise DataValidationError("MySQL mode needs DT_REAL_MYSQL_URL and a simple DT_REAL_MYSQL_TABLE identifier; credentials remain in environment variables.")
        try:
            engine=create_engine(url)
            with engine.connect() as conn: frame=pd.read_sql(text(f"SELECT * FROM `{table}`"),conn)
        except Exception as exc:
            raise DataValidationError(f"Could not read configured real-data MySQL export ({type(exc).__name__}); connection details were omitted.") from None
        return validate_real_records(frame)
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Real-data file not found: {p}. No synthetic data was generated.")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(p)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(p)
    else:
        raise DataValidationError("Supported real-data export formats are CSV and Parquet.")
    return validate_real_records(frame)

def validate_real_records(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError("Real-data export is empty.")
    sensitive = [str(c) for c in frame.columns if DIRECT_ID_RE.search(str(c).strip().lower())]
    if sensitive:
        raise DataValidationError("Potential direct identifiers detected; remove columns before use: " + ", ".join(sensitive))
    required = {"Patient_ID", "Age", "Injury_Type", "Severity", "Enrollment_Date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))
    out = frame.copy()
    if out["Patient_ID"].isna().any() or out["Patient_ID"].astype(str).str.strip().eq("").any():
        raise DataValidationError("Patient_ID must be a non-empty pseudonymous identifier.")
    if out["Patient_ID"].duplicated().any():
        raise DataValidationError("Expected one row per pseudonymous patient; duplicate Patient_ID values found.")
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    invalid_age = out["Age"].notna() & ~out["Age"].between(0, 120)
    if invalid_age.any():
        raise DataValidationError("Age values must be between 0 and 120 years.")
    if out["Age"].isna().mean() > .20:
        raise DataValidationError("More than 20% of Age values are missing or non-numeric.")
    out["Enrollment_Date"] = pd.to_datetime(out["Enrollment_Date"], errors="coerce")
    if out["Enrollment_Date"].isna().any():
        raise DataValidationError("Enrollment_Date contains missing or invalid dates.")
    for c, low, high in [("Pain_Score", 0, 10), ("VAS_Pain_Final", 0, 10),
                         ("Range_of_Motion_deg", 0, 360), ("Quadriceps_Strength_pct", 0, 200),
                         ("Hamstring_Strength_pct", 0, 200), ("BMI", 8, 100),
                         ("FIM_Final", 18, 126), ("DASH_Final", 0, 100),
                         ("KOOS_Final", 0, 100), ("LEFS_Final", 0, 80)]:
        if c in out:
            vals = pd.to_numeric(out[c], errors="coerce")
            bad = vals.notna() & ~vals.between(low, high)
            if bad.any():
                raise DataValidationError(f"{c} must be within {low}..{high} (documented unit/range).")
            out[c] = vals
    if out["Injury_Type"].isna().any() or out["Severity"].isna().any():
        raise DataValidationError("Injury_Type and Severity may not be missing.")
    return out

def to_tables(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Map canonical one-row-per-patient data into build_modelling_dataset tables."""
    records = validate_real_records(records)
    patient_cols = [c for c in ["Patient_ID", "Age", "Sex", "Gender", "BMI", "Enrollment_Date", "Enrollment_Year", "Activity_Level", "Smoking_Status", "Diabetes", "Hypertension", "Previous_Injuries"] if c in records]
    injury_cols = [c for c in ["Patient_ID", "Injury_Type", "Severity", "Body_Part", "Body_Region", "Surgery_Required", "Injury_Date"] if c in records]
    assessment_cols = [c for c in ["Patient_ID", "Pain_Score", "Range_of_Motion_deg", "Quadriceps_Strength_pct", "Hamstring_Strength_pct", "Balance_Score", "Walking_Speed_mps"] if c in records]
    outcome_cols = [c for c in ["Patient_ID", "Outcome_Recorded_Date", "Final_Recovery_Pct", "Actual_Recovery_Time_weeks", "Actual_Strength_pct", "Actual_ROM_deg", "FIM_Final", "VAS_Pain_Final", "DASH_Final", "KOOS_Final", "LEFS_Final"] if c in records]
    empty = pd.DataFrame(columns=["Patient_ID", "Week"])
    tables={name:pd.DataFrame() for name in config.TABLE_NAMES}
    tables.update({"Patient": records[patient_cols].copy(), "Injury": records[injury_cols].copy(),
                   "Initial_Assessment": records[assessment_cols].copy(), "Final_Outcome": records[outcome_cols].copy(),
                   "Weekly_Progress": empty})
    for name in ("Injury","Initial_Assessment","Final_Outcome"):
        if tables[name].empty and "Patient_ID" not in tables[name]:tables[name]=pd.DataFrame(columns=["Patient_ID"])
    return tables

def describe_dataset(records: pd.DataFrame) -> dict:
    r = validate_real_records(records)
    miss = (r.isna().mean() * 100).round(2).to_dict()
    sex_col = "Sex" if "Sex" in r else "Gender" if "Gender" in r else None
    return {"data_origin": "real, user-supplied; de-identification asserted by provider and checked for direct identifiers",
            "n": int(len(r)), "injury_type_counts": r["Injury_Type"].value_counts(dropna=False).to_dict(),
            "age": {"n": int(r["Age"].notna().sum()), "mean": float(r["Age"].mean()), "sd": float(r["Age"].std()) if r["Age"].notna().sum()>1 else None,
                    "median": float(r["Age"].median()), "min": float(r["Age"].min()), "max": float(r["Age"].max())},
            "sex_distribution": r[sex_col].value_counts(dropna=False).to_dict() if sex_col else {"unavailable": int(len(r))},
            "missingness_percent": miss,
            "outcome_distribution": {c: {"n": int(pd.to_numeric(r[c], errors="coerce").notna().sum()), "mean": float(pd.to_numeric(r[c], errors="coerce").mean()) if pd.to_numeric(r[c], errors="coerce").notna().any() else None} for c in ["Final_Recovery_Pct", "Actual_Recovery_Time_weeks", "FIM_Final", "VAS_Pain_Final", "DASH_Final", "KOOS_Final", "LEFS_Final"] if c in r}}

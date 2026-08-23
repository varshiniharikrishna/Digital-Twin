"""
db.py
-----
Loads all 14 tables into a dict of pandas DataFrames, either from MySQL
or from CSV files, depending on config.DATA_SOURCE.

Usage:
    from db import load_all_tables
    tables = load_all_tables()
    patients = tables["Patient"]
"""

import os
import pandas as pd
import config


def _load_from_csv():
    tables = {}
    for name in config.TABLE_NAMES:
        path = os.path.join(config.CSV_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected CSV not found: {path}. "
                f"Make sure your generator's ./output CSVs are copied into "
                f"'{config.CSV_DIR}/' or switch DT_DATA_SOURCE to 'mysql'."
            )
        tables[name] = pd.read_csv(path)
    return tables


def _load_from_mysql():
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError, ProgrammingError

    cfg = config.MYSQL_CONFIG
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    engine = create_engine(url)
    tables = {}
    try:
        with engine.connect() as conn:
            for name in config.TABLE_NAMES:
                try:
                    tables[name] = pd.read_sql(f"SELECT * FROM `{name}`", conn)
                except (OperationalError, ProgrammingError) as e:
                    raise RuntimeError(
                        f"Could not read table `{name}` from MySQL database "
                        f"'{cfg['database']}'. Make sure you've imported it "
                        f"(see import_to_mysql.py) and that the table name "
                        f"matches exactly. Original error: {e}"
                    ) from e
    except OperationalError as e:
        raise RuntimeError(
            f"Could not connect to MySQL at {cfg['host']}:{cfg['port']} as "
            f"user '{cfg['user']}'. Check MYSQL_CONFIG in config.py (or the "
            f"DT_MYSQL_* environment variables) - host, port, user, password, "
            f"and that the MySQL server is actually running. "
            f"Original error: {e}"
        ) from e
    return tables


def load_all_tables():
    """Returns dict {table_name: DataFrame} for all 14 tables."""
    if config.DATA_SOURCE == "mysql":
        tables = _load_from_mysql()
    elif config.DATA_SOURCE == "csv":
        tables = _load_from_csv()
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {config.DATA_SOURCE!r}")

    # normalize date columns so downstream code can rely on real dtypes
    if "Enrollment_Date" in tables["Patient"].columns:
        tables["Patient"]["Enrollment_Date"] = pd.to_datetime(
            tables["Patient"]["Enrollment_Date"], errors="coerce"
        )
    if "Injury_Date" in tables["Injury"].columns:
        tables["Injury"]["Injury_Date"] = pd.to_datetime(
            tables["Injury"]["Injury_Date"], errors="coerce"
        )

    return tables


def _get_engine():
    from sqlalchemy import create_engine
    cfg = config.MYSQL_CONFIG
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url)


def save_weekly_progress_record(record_dict):
    """
    Saves or updates a weekly progress observation record in MySQL or CSV.
    """
    patient_id = record_dict.get("Patient_ID")
    week = int(record_dict.get("Week", 1))

    if config.DATA_SOURCE == "mysql":
        engine = _get_engine()
        from sqlalchemy import text
        cols = list(record_dict.keys())
        cols_str = ", ".join([f"`{c}`" for c in cols])
        vals_str = ", ".join([f":{c}" for c in cols])
        update_str = ", ".join([f"`{c}` = :{c}" for c in cols if c not in ["Patient_ID", "Week"]])
        
        sql = text(f"""
            INSERT INTO `Weekly_Progress` ({cols_str})
            VALUES ({vals_str})
            ON DUPLICATE KEY UPDATE {update_str}
        """)
        with engine.begin() as conn:
            conn.execute(sql, record_dict)
    else:
        csv_path = os.path.join(config.CSV_DIR, "Weekly_Progress.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            mask = (df["Patient_ID"] == patient_id) & (df["Week"] == week)
            if mask.any():
                for k, v in record_dict.items():
                    if k in df.columns:
                        df.loc[mask, k] = v
            else:
                new_row = pd.DataFrame([record_dict])
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)


def save_patient_recalibration_record(patient_id, week, recal_dict):
    """
    Persists a recalibration outcome record in MySQL or CSV.
    Includes per-metric expected vs actual values and discharge timeline delta.
    """
    evr = recal_dict.get("expected_vs_real", {})

    def _ev(metric, key):
        return float(evr[metric][key]) if metric in evr and key in evr[metric] else None

    rec_data = {
        "Patient_ID": patient_id,
        "Week": int(week),
        "Old_Predicted_Weeks": float(recal_dict.get("old_predicted_weeks", 0.0)),
        "Recalibrated_Recovery_Pct": float(recal_dict.get("new_predicted_pct", 0.0)),
        "Recalibrated_Recovery_Weeks": float(recal_dict.get("new_predicted_weeks", 0.0)),
        "Discharge_Delta": float(recal_dict.get("discharge_delta", 0.0)),
        "Pain_Expected": _ev("Pain", "expected"),
        "Pain_Actual": _ev("Pain", "actual"),
        "Pain_Dev_Pct": _ev("Pain", "pct_deviation"),
        "ROM_Expected": _ev("ROM_deg", "expected"),
        "ROM_Actual": _ev("ROM_deg", "actual"),
        "ROM_Dev_Pct": _ev("ROM_deg", "pct_deviation"),
        "Strength_Expected": _ev("Strength_pct", "expected"),
        "Strength_Actual": _ev("Strength_pct", "actual"),
        "Strength_Dev_Pct": _ev("Strength_pct", "pct_deviation"),
        "Explanation": str(recal_dict.get("explanation", "")),
        "Updated_Pathway": str(
            recal_dict.get("recommended_pathway", {}).get("Archetype_Label", "Standard")
            if isinstance(recal_dict.get("recommended_pathway"), dict) else "Standard"
        )
    }

    if config.DATA_SOURCE == "mysql":
        engine = _get_engine()
        from sqlalchemy import text
        sql = text("""
            CREATE TABLE IF NOT EXISTS `Recalibration_Log` (
                `Patient_ID` VARCHAR(50),
                `Week` INT,
                `Recalibrated_Recovery_Pct` FLOAT,
                `Recalibrated_Recovery_Weeks` FLOAT,
                `Explanation` TEXT,
                `Updated_Pathway` VARCHAR(255),
                PRIMARY KEY (`Patient_ID`, `Week`)
            );
        """)
        with engine.begin() as conn:
            conn.execute(sql)
            cols = list(rec_data.keys())
            cols_str = ", ".join([f"`{c}`" for c in cols])
            vals_str = ", ".join([f":{c}" for c in cols])
            update_str = ", ".join([f"`{c}` = :{c}" for c in cols if c not in ["Patient_ID", "Week"]])
            upsert_sql = text(f"""
                INSERT INTO `Recalibration_Log` ({cols_str})
                VALUES ({vals_str})
                ON DUPLICATE KEY UPDATE {update_str}
            """)
            conn.execute(upsert_sql, rec_data)
    else:
        csv_path = os.path.join(config.CSV_DIR, "Recalibration_Log.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            mask = (df["Patient_ID"] == patient_id) & (df["Week"] == week)
            if mask.any():
                for k, v in rec_data.items():
                    df.loc[mask, k] = v
            else:
                new_row = pd.DataFrame([rec_data])
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)
        else:
            pd.DataFrame([rec_data]).to_csv(csv_path, index=False)


def add_new_patient_record(patient_dict, injury_dict, assessment_dict):
    """
    Creates a new patient profile across Patient, Injury, and Initial_Assessment.
    """
    if config.DATA_SOURCE == "mysql":
        engine = _get_engine()
        from sqlalchemy import text
        with engine.begin() as conn:
            for tbl, data in [("Patient", patient_dict), ("Injury", injury_dict), ("Initial_Assessment", assessment_dict)]:
                cols = list(data.keys())
                cols_str = ", ".join([f"`{c}`" for c in cols])
                vals_str = ", ".join([f":{c}" for c in cols])
                sql = text(f"INSERT INTO `{tbl}` ({cols_str}) VALUES ({vals_str})")
                conn.execute(sql, data)
    else:
        for tbl, data in [("Patient", patient_dict), ("Injury", injury_dict), ("Initial_Assessment", assessment_dict)]:
            csv_path = os.path.join(config.CSV_DIR, f"{tbl}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                df.to_csv(csv_path, index=False)


def get_recalibration_history_records(patient_id):
    """
    Retrieves the full chronological list of recalibration logs for a patient.
    """
    if config.DATA_SOURCE == "mysql":
        engine = _get_engine()
        from sqlalchemy import text
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text("SELECT * FROM `Recalibration_Log` WHERE `Patient_ID` = :pid ORDER BY `Week` ASC"), conn, params={"pid": patient_id})
                return df.to_dict(orient="records")
        except Exception:
            return []
    else:
        csv_path = os.path.join(config.CSV_DIR, "Recalibration_Log.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            match = df[df["Patient_ID"] == patient_id].sort_values("Week")
            return match.to_dict(orient="records")
        return []


if __name__ == "__main__":
    t = load_all_tables()
    for name, df in t.items():
        print(f"{name:<22} {df.shape}")


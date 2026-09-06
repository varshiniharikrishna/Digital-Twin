"""
db.py
-----
Loads all 15 tables into a dict of pandas DataFrames, either from MySQL
or from CSV files, depending on config.DATA_SOURCE with automatic CSV fallback.
"""

import os
import pandas as pd
import config


def _load_from_csv():
    tables = {}
    for name in config.TABLE_NAMES:
        path = os.path.join("output", f"{name}.csv")
        if not os.path.exists(path):
            path = os.path.join(config.CSV_DIR, f"{name}.csv")
        if os.path.exists(path):
            tables[name] = pd.read_csv(path)
        else:
            tables[name] = pd.DataFrame()
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
    with engine.connect() as conn:
        for name in config.TABLE_NAMES:
            try:
                tables[name] = pd.read_sql(f"SELECT * FROM `{name}`", conn)
            except (OperationalError, ProgrammingError) as e:
                # If a specific table doesn't exist in MySQL, try loading from CSV
                csv_path = os.path.join("output", f"{name}.csv")
                if not os.path.exists(csv_path):
                    csv_path = os.path.join(config.CSV_DIR, f"{name}.csv")
                if os.path.exists(csv_path):
                    tables[name] = pd.read_csv(csv_path)
                else:
                    tables[name] = pd.DataFrame()
    return tables


def load_all_tables():
    """Returns dict {table_name: DataFrame} for all tables."""
    tables = None
    if config.DATA_SOURCE == "mysql":
        try:
            tables = _load_from_mysql()
        except Exception as e:
            print(f"MySQL connection failed ({e}). Falling back to CSV data files...")
            tables = _load_from_csv()
    else:
        tables = _load_from_csv()

    if tables is None or not tables:
        tables = _load_from_csv()

    # Normalize date columns so downstream code can rely on real dtypes
    if "Patient" in tables and not tables["Patient"].empty and "Enrollment_Date" in tables["Patient"].columns:
        tables["Patient"]["Enrollment_Date"] = pd.to_datetime(
            tables["Patient"]["Enrollment_Date"], errors="coerce"
        )
    if "Injury" in tables and not tables["Injury"].empty and "Injury_Date" in tables["Injury"].columns:
        tables["Injury"]["Injury_Date"] = pd.to_datetime(
            tables["Injury"]["Injury_Date"], errors="coerce"
        )

    return tables


def add_new_patient_record(patient_dict, injury_dict, assessment_dict):
    """Inserts a newly registered patient into Patient, Injury, and Initial_Assessment tables (MySQL or CSV)."""
    if config.DATA_SOURCE == "mysql":
        try:
            from sqlalchemy import create_engine, text
            cfg = config.MYSQL_CONFIG
            url = f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
            engine = create_engine(url)
            with engine.begin() as conn:
                for tbl, data in [("Patient", patient_dict), ("Injury", injury_dict), ("Initial_Assessment", assessment_dict)]:
                    cols = list(data.keys())
                    cols_str = ", ".join([f"`{c}`" for c in cols])
                    vals_str = ", ".join([f":{c}" for c in cols])
                    sql = text(f"INSERT INTO `{tbl}` ({cols_str}) VALUES ({vals_str})")
                    conn.execute(sql, data)
        except Exception as e:
            print(f"MySQL insert failed ({e}). Falling back to CSV...")

    # Always persist to CSV as well (or as fallback)
    for tbl, data in [("Patient", patient_dict), ("Injury", injury_dict), ("Initial_Assessment", assessment_dict)]:
        csv_path = os.path.join("output", f"{tbl}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(config.CSV_DIR, f"{tbl}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(csv_path, index=False)
        else:
            pd.DataFrame([data]).to_csv(csv_path, index=False)


def save_weekly_progress_record(record_dict):
    """Inserts or updates a weekly progress observation in Weekly_Progress table."""
    patient_id = record_dict.get("Patient_ID")
    week = int(record_dict.get("Week", 1))

    if config.DATA_SOURCE == "mysql":
        try:
            from sqlalchemy import create_engine, text
            cfg = config.MYSQL_CONFIG
            url = f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
            engine = create_engine(url)
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
        except Exception as e:
            print(f"MySQL weekly record insert failed ({e}). Falling back to CSV...")

    # Always persist to CSV
    csv_path = os.path.join("output", "Weekly_Progress.csv")
    if not os.path.exists(csv_path):
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
    else:
        pd.DataFrame([record_dict]).to_csv(csv_path, index=False)


def save_patient_recalibration_record(patient_id, week, recal_dict):
    """Persists a recalibration outcome in MySQL or CSV."""
    rec_data = {
        "Patient_ID": patient_id,
        "Week": int(week),
        "Recalibrated_Recovery_Pct": float(recal_dict.get("new_predicted_pct", 0.0)),
        "Recalibrated_Recovery_Weeks": float(recal_dict.get("new_predicted_weeks", 0.0)),
        "Explanation": str(recal_dict.get("explanation", "")),
        "Updated_Pathway": str(recal_dict.get("recommended_pathway", {}).get("Archetype_Label", "Standard") if isinstance(recal_dict.get("recommended_pathway"), dict) else "Standard")
    }

    if config.DATA_SOURCE == "mysql":
        try:
            from sqlalchemy import create_engine, text
            cfg = config.MYSQL_CONFIG
            url = f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
            engine = create_engine(url)
            sql = text("""
                CREATE TABLE IF NOT EXISTS `Recalibration_Log` (
                    `Patient_ID` VARCHAR(64),
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
        except Exception as e:
            print(f"MySQL recalibration record insert failed ({e}). Falling back to CSV...")

    csv_path = os.path.join("output", "Recalibration_Log.csv")
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

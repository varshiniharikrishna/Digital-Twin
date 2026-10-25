"""
db.py
-----
Loads all 15 tables + User_Account into pandas DataFrames or queries MySQL directly.
Provides CRUD and authentication operations for the Digital Twin application.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash
import config

LAST_DATA_ORIGIN = "unknown"


def get_db_engine():
    """Returns a SQLAlchemy engine connecting to MySQL DigitalTwinDB."""
    cfg = config.MYSQL_CONFIG
    if not cfg["password"]:
        raise RuntimeError("Set DT_MYSQL_PASSWORD in the environment before connecting to MySQL.")
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url)


def _load_from_csv():
    tables = {}
    for name in config.TABLE_NAMES:
        # Honor an explicitly configured CSV directory first; otherwise use
        # the project's bundled output/ files as the normal CSV mode source.
        path = os.path.join(config.CSV_DIR, f"{name}.csv")
        if not os.path.exists(path):
            path = os.path.join("output", f"{name}.csv")
        if os.path.exists(path):
            tables[name] = pd.read_csv(path)
        else:
            tables[name] = pd.DataFrame()
    return tables


def _load_from_mysql():
    engine = get_db_engine()
    tables = {}
    with engine.connect() as conn:
        for name in config.TABLE_NAMES:
            try:
                tables[name] = pd.read_sql(f"SELECT * FROM `{name}`", conn)
            except Exception:
                csv_path = os.path.join("output", f"{name}.csv")
                if not os.path.exists(csv_path):
                    csv_path = os.path.join(config.CSV_DIR, f"{name}.csv")
                if os.path.exists(csv_path):
                    tables[name] = pd.read_csv(csv_path)
                else:
                    tables[name] = pd.DataFrame()
        
        try:
            tables["User_Account"] = pd.read_sql("SELECT * FROM `User_Account`", conn)
        except Exception:
            tables["User_Account"] = pd.DataFrame()

    return tables


def load_all_tables():
    """Returns dict {table_name: DataFrame} for all tables."""
    global LAST_DATA_ORIGIN
    tables = None
    if config.DATA_SOURCE == "real":
        if not config.REAL_DATA_PATH:
            raise RuntimeError("DT_DATA_SOURCE=real requires DT_REAL_DATA_PATH; no synthetic fallback is used.")
        from data_adapter import load_real_records, to_tables
        tables = to_tables(load_real_records(config.REAL_DATA_PATH))
        LAST_DATA_ORIGIN = "real user-supplied"
    elif config.DATA_SOURCE == "mysql":
        try:
            tables = _load_from_mysql()
            LAST_DATA_ORIGIN = os.environ.get("DT_MYSQL_PROVENANCE", "unknown MySQL provenance")
        except Exception as e:
            print(f"MySQL connection error ({e}). Using CSV fallback.")
            tables = _load_from_csv()
            external_csv=os.path.exists(os.path.join(config.CSV_DIR,"Patient.csv"))
            LAST_DATA_ORIGIN = os.environ.get("DT_CSV_PROVENANCE") or ("CSV provenance unverified" if external_csv else "synthetic bundled CSV fallback")
    else:
        tables = _load_from_csv()
        external_csv=os.path.exists(os.path.join(config.CSV_DIR,"Patient.csv"))
        LAST_DATA_ORIGIN = os.environ.get("DT_CSV_PROVENANCE") or ("CSV provenance unverified" if external_csv else "synthetic bundled CSV")

    if tables is None or not tables:
        tables = _load_from_csv()

    if "Patient" in tables and not tables["Patient"].empty and "Enrollment_Date" in tables["Patient"].columns:
        tables["Patient"]["Enrollment_Date"] = pd.to_datetime(
            tables["Patient"]["Enrollment_Date"], errors="coerce"
        )
    if "Injury" in tables and not tables["Injury"].empty and "Injury_Date" in tables["Injury"].columns:
        tables["Injury"]["Injury_Date"] = pd.to_datetime(
            tables["Injury"]["Injury_Date"], errors="coerce"
        )

    return tables


def get_user_by_login_input(login_input: str):
    """
    Looks up user in User_Account by Username, Email, Linked_Doctor_ID, or Linked_Patient_ID.
    Case-insensitive matching.
    """
    if not login_input:
        return None

    engine = get_db_engine()
    query = text("""
        SELECT * FROM `User_Account`
        WHERE LOWER(`Username`) = LOWER(:inp)
           OR LOWER(`Email`) = LOWER(:inp)
           OR LOWER(`Linked_Doctor_ID`) = LOWER(:inp)
           OR LOWER(`Linked_Patient_ID`) = LOWER(:inp)
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            res = conn.execute(query, {"inp": login_input.strip()}).mappings().first()
            if res:
                return dict(res)
    except Exception as e:
        print(f"Error querying User_Account: {e}")
    return None


def get_user_by_id(user_id: str):
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM `User_Account` WHERE `User_ID` = :id"), {"id": user_id}).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        print(f"Error querying account: {e}")
        return None


def update_user_password(user_id: str, new_password: str):
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE `User_Account` SET `Password_Hash` = :password WHERE `User_ID` = :id"),
            {"password": generate_password_hash(new_password), "id": user_id},
        )


def create_doctor_record(doctor_dict: dict, initial_password: str):
    """Inserts a new doctor into Doctor table and creates/updates a User_Account."""
    doc_id = doctor_dict["Doctor_ID"]
    engine = get_db_engine()
    with engine.begin() as conn:
        cols = list(doctor_dict.keys())
        cols_str = ", ".join([f"`{c}`" for c in cols])
        vals_str = ", ".join([f":{c}" for c in cols])
        update_doc = ", ".join([f"`{c}` = :{c}" for c in cols if c != "Doctor_ID"])
        conn.execute(text(f"INSERT INTO `Doctor` ({cols_str}) VALUES ({vals_str}) ON DUPLICATE KEY UPDATE {update_doc}"), doctor_dict)

        user_rec = {
            "User_ID": f"USR_{doc_id}",
            "Username": doc_id.lower(),
            "Email": doctor_dict.get("Contact_Email", f"{doc_id.lower()}@hospital.com"),
            "Password_Hash": generate_password_hash(initial_password),
            "Role": "DOCTOR",
            "Linked_Doctor_ID": doc_id,
            "Linked_Patient_ID": None
        }
        u_cols = list(user_rec.keys())
        u_cols_str = ", ".join([f"`{c}`" for c in u_cols])
        u_vals_str = ", ".join([f":{c}" for c in u_cols])
        u_update = ", ".join([f"`{c}` = :{c}" for c in u_cols if c != "User_ID"])
        conn.execute(text(f"INSERT INTO `User_Account` ({u_cols_str}) VALUES ({u_vals_str}) ON DUPLICATE KEY UPDATE {u_update}"), user_rec)

    csv_path = os.path.join("output", "Doctor.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([doctor_dict])], ignore_index=True)
        df.to_csv(csv_path, index=False)


def reassign_patient_doctor(patient_id: str, new_doctor_id: str):
    """Updates Assigned_Doctor_ID for a patient in MySQL and CSV."""
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE `Patient` SET `Assigned_Doctor_ID` = :doc_id WHERE `Patient_ID` = :p_id"),
            {"doc_id": new_doctor_id, "p_id": patient_id}
        )

    csv_path = os.path.join("output", "Patient.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.loc[df["Patient_ID"] == patient_id, "Assigned_Doctor_ID"] = new_doctor_id
        df.to_csv(csv_path, index=False)


def add_new_patient_record(patient_dict, injury_dict, assessment_dict, initial_password: str):
    """Inserts a newly registered patient into Patient, Injury, Initial_Assessment, and User_Account."""
    p_id = patient_dict["Patient_ID"]
    engine = get_db_engine()
    with engine.begin() as conn:
        for tbl, data in [("Patient", patient_dict), ("Injury", injury_dict), ("Initial_Assessment", assessment_dict)]:
            cols = list(data.keys())
            cols_str = ", ".join([f"`{c}`" for c in cols])
            vals_str = ", ".join([f":{c}" for c in cols])
            update_str = ", ".join([f"`{c}` = :{c}" for c in cols if c != cols[0]])
            sql = text(f"INSERT INTO `{tbl}` ({cols_str}) VALUES ({vals_str}) ON DUPLICATE KEY UPDATE {update_str}")
            conn.execute(sql, data)

        user_rec = {
            "User_ID": f"USR_{p_id}",
            "Username": p_id,
            "Email": f"{p_id.lower()}@digitaltwin.com",
            "Password_Hash": generate_password_hash(initial_password),
            "Role": "PATIENT",
            "Linked_Doctor_ID": None,
            "Linked_Patient_ID": p_id
        }
        u_cols = list(user_rec.keys())
        u_cols_str = ", ".join([f"`{c}`" for c in u_cols])
        u_vals_str = ", ".join([f":{c}" for c in u_cols])
        u_update = ", ".join([f"`{c}` = :{c}" for c in u_cols if c != "User_ID"])
        conn.execute(text(f"INSERT INTO `User_Account` ({u_cols_str}) VALUES ({u_vals_str}) ON DUPLICATE KEY UPDATE {u_update}"), user_rec)

    for tbl, data in [("Patient", patient_dict), ("Injury", injury_dict), ("Initial_Assessment", assessment_dict)]:
        csv_path = os.path.join("output", f"{tbl}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(csv_path, index=False)


def save_weekly_progress_record(record_dict):
    """Inserts or updates a weekly progress observation in Weekly_Progress table."""
    patient_id = record_dict.get("Patient_ID")
    week = int(record_dict.get("Week", 1))

    engine = get_db_engine()
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

    csv_path = os.path.join("output", "Weekly_Progress.csv")
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
    """Persists a recalibration outcome in Recalibration_Log in MySQL and CSV."""
    dev = recal_dict.get("deviation") or {}
    
    pain_dev = dev.get("Pain") if isinstance(dev.get("Pain"), dict) else {}
    rom_dev = dev.get("ROM_deg") if isinstance(dev.get("ROM_deg"), dict) else {}
    str_dev = dev.get("Strength_pct") if isinstance(dev.get("Strength_pct"), dict) else {}

    updated_pw = "Standard"
    rec_pw = recal_dict.get("recommended_pathway")
    if isinstance(rec_pw, dict):
        updated_pw = str(rec_pw.get("Archetype_Label") or rec_pw.get("Pathway_Name") or "Standard")
    elif isinstance(rec_pw, pd.Series):
        updated_pw = str(rec_pw.get("Archetype_Label", "Standard"))

    rec_data = {
        "Patient_ID": str(patient_id),
        "Week": int(week),
        "Old_Predicted_Pct": float(recal_dict.get("old_predicted_pct") or 0.0),
        "Old_Predicted_Weeks": float(recal_dict.get("old_predicted_weeks") or 0.0),
        "Recalibrated_Recovery_Pct": float(recal_dict.get("new_predicted_pct") or 0.0),
        "Recalibrated_Recovery_Weeks": float(recal_dict.get("new_predicted_weeks") or 0.0),
        "Pain_Expected": float(pain_dev.get("predicted") or 0.0),
        "Pain_Actual": float(pain_dev.get("actual") or 0.0),
        "ROM_Expected": float(rom_dev.get("predicted") or 0.0),
        "ROM_Actual": float(rom_dev.get("actual") or 0.0),
        "Strength_Expected": float(str_dev.get("predicted") or 0.0),
        "Strength_Actual": float(str_dev.get("actual") or 0.0),
        "Explanation": str(recal_dict.get("explanation") or ""),
        "Updated_Pathway": updated_pw
    }

    engine = get_db_engine()
    create_tbl_sql = text("""
        CREATE TABLE IF NOT EXISTS `Recalibration_Log` (
            `Patient_ID` VARCHAR(64),
            `Week` INT,
            `Old_Predicted_Pct` FLOAT,
            `Old_Predicted_Weeks` FLOAT,
            `Recalibrated_Recovery_Pct` FLOAT,
            `Recalibrated_Recovery_Weeks` FLOAT,
            `Pain_Expected` FLOAT,
            `Pain_Actual` FLOAT,
            `ROM_Expected` FLOAT,
            `ROM_Actual` FLOAT,
            `Strength_Expected` FLOAT,
            `Strength_Actual` FLOAT,
            `Explanation` TEXT,
            `Updated_Pathway` VARCHAR(255),
            PRIMARY KEY (`Patient_ID`, `Week`)
        );
    """)

    with engine.begin() as conn:
        conn.execute(create_tbl_sql)

        col_types = {
            "Old_Predicted_Pct": "FLOAT",
            "Old_Predicted_Weeks": "FLOAT",
            "Pain_Expected": "FLOAT",
            "Pain_Actual": "FLOAT",
            "ROM_Expected": "FLOAT",
            "ROM_Actual": "FLOAT",
            "Strength_Expected": "FLOAT",
            "Strength_Actual": "FLOAT"
        }
        for col_n, col_t in col_types.items():
            try:
                conn.execute(text(f"ALTER TABLE `Recalibration_Log` ADD COLUMN `{col_n}` {col_t}"))
            except Exception:
                pass

        cols = list(rec_data.keys())
        cols_str = ", ".join([f"`{c}`" for c in cols])
        vals_str = ", ".join([f":{c}`" if False else f":{c}" for c in cols])
        update_str = ", ".join([f"`{c}` = :{c}" for c in cols if c not in ["Patient_ID", "Week"]])
        upsert_sql = text(f"""
            INSERT INTO `Recalibration_Log` ({cols_str})
            VALUES ({vals_str})
            ON DUPLICATE KEY UPDATE {update_str}
        """)
        conn.execute(upsert_sql, rec_data)

    csv_path = os.path.join("output", "Recalibration_Log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        mask = (df["Patient_ID"] == patient_id) & (df["Week"] == week)
        if mask.any():
            for k, v in rec_data.items():
                if k in df.columns:
                    df.loc[mask, k] = v
        else:
            new_row = pd.DataFrame([rec_data])
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
    else:
        pd.DataFrame([rec_data]).to_csv(csv_path, index=False)


def get_patient_recalibration_history(patient_id: str) -> list:
    """Retrieves full recalibration log for a patient from MySQL or CSV."""
    engine = get_db_engine()
    query = text("SELECT * FROM `Recalibration_Log` WHERE `Patient_ID` = :p_id ORDER BY `Week` ASC")
    try:
        with engine.connect() as conn:
            rows = conn.execute(query, {"p_id": patient_id}).mappings().all()
            if rows:
                return [dict(r) for r in rows]
    except Exception as e:
        print(f"Recalibration_Log DB query notice: {e}")

    csv_path = os.path.join("output", "Recalibration_Log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df_p = df[df["Patient_ID"] == patient_id].sort_values("Week")
        return df_p.to_dict(orient="records")

    return []

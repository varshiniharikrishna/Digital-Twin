"""
import_to_mysql.py
-------------------
Loads all generated CSVs from ./output/ into MySQL database (DigitalTwinDB),
creates VARCHAR primary keys, foreign keys for relational integrity,
and initializes the User_Account authentication table with hashed passwords.
"""

import argparse
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text, VARCHAR
from werkzeug.security import generate_password_hash

import config

OUT_DIR = "output"

TABLE_ORDER = [
    "Doctor",
    "Patient",
    "Injury",
    "Initial_Assessment",
    "Exercise_Library",
    "Weekly_Progress",
    "Therapy_Plan",
    "Muscle_Compensation",
    "Cohort_Definition",
    "Recovery_Pathway",
    "Pathway_Steps",
    "Similar_Patients",
    "Simulation",
    "Weekly_Recommendation",
    "Final_Outcome",
]

PRIMARY_KEYS = {
    "Doctor": "Doctor_ID",
    "Patient": "Patient_ID",
    "Injury": "Injury_ID",
    "Initial_Assessment": "Assessment_ID",
    "Exercise_Library": "Exercise_ID",
    "Weekly_Progress": "Weekly_ID",
    "Therapy_Plan": "Plan_ID",
    "Muscle_Compensation": "Compensation_ID",
    "Cohort_Definition": "Cohort_ID",
    "Recovery_Pathway": "Pathway_ID",
    "Similar_Patients": "Similarity_ID",
    "Simulation": "Simulation_ID",
    "Weekly_Recommendation": "Recommendation_ID",
    "Final_Outcome": "Outcome_ID",
}

FOREIGN_KEYS = [
    ("Patient", "Assigned_Doctor_ID", "Doctor", "Doctor_ID"),
    ("Injury", "Patient_ID", "Patient", "Patient_ID"),
    ("Initial_Assessment", "Patient_ID", "Patient", "Patient_ID"),
    ("Weekly_Progress", "Patient_ID", "Patient", "Patient_ID"),
    ("Therapy_Plan", "Patient_ID", "Patient", "Patient_ID"),
    ("Therapy_Plan", "Exercise_ID", "Exercise_Library", "Exercise_ID"),
    ("Muscle_Compensation", "Patient_ID", "Patient", "Patient_ID"),
    ("Recovery_Pathway", "Cohort_ID", "Cohort_Definition", "Cohort_ID"),
    ("Pathway_Steps", "Pathway_ID", "Recovery_Pathway", "Pathway_ID"),
    ("Similar_Patients", "Patient_ID", "Patient", "Patient_ID"),
    ("Similar_Patients", "Similar_Patient_ID", "Patient", "Patient_ID"),
    ("Simulation", "Patient_ID", "Patient", "Patient_ID"),
    ("Simulation", "Pathway_ID", "Recovery_Pathway", "Pathway_ID"),
    ("Weekly_Recommendation", "Patient_ID", "Patient", "Patient_ID"),
    ("Weekly_Recommendation", "Recommended_Pathway", "Recovery_Pathway", "Pathway_ID"),
    ("Final_Outcome", "Patient_ID", "Patient", "Patient_ID"),
]


def get_args():
    cfg = config.MYSQL_CONFIG
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("DB_HOST", cfg["host"]))
    p.add_argument("--port", default=os.environ.get("DB_PORT", str(cfg["port"])))
    p.add_argument("--user", default=os.environ.get("DB_USER", cfg["user"]))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD", cfg["password"]))
    p.add_argument("--database", default=os.environ.get("DB_NAME", cfg["database"]))
    return p.parse_args()


def unique_email(email, fallback, used_emails):
    candidate = str(email).strip() if pd.notna(email) else ""
    if not candidate or candidate.casefold() in used_emails:
        candidate = fallback

    if candidate.casefold() in used_emails:
        local_part, separator, domain = fallback.rpartition("@")
        suffix = 2
        while candidate.casefold() in used_emails:
            candidate = (
                f"{local_part}_{suffix}@{domain}"
                if separator
                else f"{fallback}_{suffix}"
            )
            suffix += 1

    used_emails.add(candidate.casefold())
    return candidate


def main():
    args = get_args()
    url = f"mysql+pymysql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}"
    
    try:
        engine = create_engine(url, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))

            for name in reversed(TABLE_ORDER):
                conn.execute(text(f"DROP TABLE IF EXISTS `{name}`"))
            conn.execute(text("DROP TABLE IF EXISTS `User_Account`"))
            print("Dropped any existing tables.")

            for name in TABLE_ORDER:
                path = os.path.join(OUT_DIR, f"{name}.csv")
                if not os.path.exists(path):
                    print(f"  [skip] {path} not found")
                    continue
                df = pd.read_csv(path)

                dtype_map = {}
                for col in df.columns:
                    if col.endswith("_ID") or col in (
                        "Patient_ID", "Similar_Patient_ID", "Recommended_Pathway",
                        "Exercise_ID", "Pathway_ID", "Cohort_ID"
                    ):
                        dtype_map[col] = VARCHAR(64)

                df.to_sql(name, con=conn, if_exists="append", index=False,
                          chunksize=2000, dtype=dtype_map)
                print(f"  loaded {name} ({len(df)} rows)")

            for table, col in PRIMARY_KEYS.items():
                try:
                    conn.execute(text(f"ALTER TABLE `{table}` ADD PRIMARY KEY (`{col}`)"))
                except Exception as e:
                    print(f"  [warn] could not add PK on {table}.{col}: {e}")

            for table, col, ref_table, ref_col in FOREIGN_KEYS:
                fk_name = f"fk_{table}_{col}"
                try:
                    conn.execute(text(
                        f"ALTER TABLE `{table}` ADD CONSTRAINT `{fk_name}` "
                        f"FOREIGN KEY (`{col}`) REFERENCES `{ref_table}`(`{ref_col}`)"
                    ))
                except Exception as e:
                    print(f"  [warn] could not add FK {table}.{col} -> {ref_table}.{ref_col}: {e}")

            # ------------------------------------------------------------------
            # CREATE AND SEED USER_ACCOUNT AUTHENTICATION TABLE
            # ------------------------------------------------------------------
            print("Creating User_Account table...")
            create_user_tbl_sql = text("""
                CREATE TABLE `User_Account` (
                    `User_ID` VARCHAR(64) PRIMARY KEY,
                    `Username` VARCHAR(128) UNIQUE NOT NULL,
                    `Email` VARCHAR(128) UNIQUE NOT NULL,
                    `Password_Hash` VARCHAR(255) NOT NULL,
                    `Role` VARCHAR(32) NOT NULL,
                    `Linked_Doctor_ID` VARCHAR(64) NULL,
                    `Linked_Patient_ID` VARCHAR(64) NULL,
                    `Created_At` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT `fk_user_doctor` FOREIGN KEY (`Linked_Doctor_ID`) REFERENCES `Doctor`(`Doctor_ID`) ON DELETE SET NULL,
                    CONSTRAINT `fk_user_patient` FOREIGN KEY (`Linked_Patient_ID`) REFERENCES `Patient`(`Patient_ID`) ON DELETE SET NULL
                );
            """)
            conn.execute(create_user_tbl_sql)

            user_records = []
            
            # 1. Admin / Central Account
            user_records.append({
                "User_ID": "USR_ADMIN_01",
                "Username": "admin",
                "Email": "admin@digitaltwin.com",
                "Password_Hash": generate_password_hash("admin123"),
                "Role": "ADMIN",
                "Linked_Doctor_ID": None,
                "Linked_Patient_ID": None
            })
            used_emails = {"admin@digitaltwin.com"}

            # 2. Doctors Accounts
            doc_df = pd.read_csv(os.path.join(OUT_DIR, "Doctor.csv"))
            for _, doc in doc_df.iterrows():
                doc_id = str(doc["Doctor_ID"])
                email = unique_email(
                    doc.get("Contact_Email", ""),
                    f"{doc_id.lower()}@hospital.com",
                    used_emails,
                )
                user_records.append({
                    "User_ID": f"USR_{doc_id}",
                    "Username": doc_id.lower(), # e.g. doc0001
                    "Email": email,
                    "Password_Hash": generate_password_hash("doc123"),
                    "Role": "DOCTOR",
                    "Linked_Doctor_ID": doc_id,
                    "Linked_Patient_ID": None
                })

            # 3. Patients Accounts
            patient_df = pd.read_csv(os.path.join(OUT_DIR, "Patient.csv"))
            for _, pat in patient_df.iterrows():
                p_id = str(pat["Patient_ID"])
                email = unique_email(
                    f"{p_id.lower()}@digitaltwin.com",
                    f"{p_id.lower()}@digitaltwin.com",
                    used_emails,
                )
                user_records.append({
                    "User_ID": f"USR_{p_id}",
                    "Username": p_id, # e.g. P00001
                    "Email": email,
                    "Password_Hash": generate_password_hash("patient123"),
                    "Role": "PATIENT",
                    "Linked_Doctor_ID": None,
                    "Linked_Patient_ID": p_id
                })

            users_df = pd.DataFrame(user_records)
            users_df.to_sql("User_Account", con=conn, if_exists="append", index=False, chunksize=1000)
            print(f"Seeded User_Account table with {len(users_df)} authenticated users.")

            conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

        print("\nMySQL Database setup and import complete. All 16 tables loaded successfully into MySQL DigitalTwinDB.")
    except Exception as e:
        print(f"\n[Error] Could not connect or setup MySQL server at {args.host}:{args.port} as user '{args.user}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
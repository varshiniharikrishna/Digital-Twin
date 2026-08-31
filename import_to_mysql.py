"""
import_to_mysql.py
-------------------
Loads the 15 generated CSVs (14 original + Doctor) from ./output/ into a
MySQL database (DigitalTwinDB), creating VARCHAR primary keys and foreign
keys for relational integrity, including Patient -> Doctor.

Rerun-safe: drops all 15 tables (in reverse dependency order, with FK checks
disabled for the whole session) before recreating anything, so you can run
this script as many times as you want without leftover-constraint errors.
"""

import argparse
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text, VARCHAR

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
    # Every patient is now linked to the doctor treating them.
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


def main():
    args = get_args()
    url = f"mysql+pymysql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}"
    # autocommit avoids implicit-transaction issues with DDL + SET session vars
    engine = create_engine(url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        # Disable FK checks for this whole session BEFORE touching any table.
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))

        # Drop every table first, in reverse dependency order, so no
        # leftover constraint from a previous run can block a CREATE TABLE.
        for name in reversed(TABLE_ORDER):
            conn.execute(text(f"DROP TABLE IF EXISTS `{name}`"))
        print("Dropped any existing tables.")

        # Now load each CSV fresh.
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

        # Primary keys
        for table, col in PRIMARY_KEYS.items():
            try:
                conn.execute(text(f"ALTER TABLE `{table}` ADD PRIMARY KEY (`{col}`)"))
            except Exception as e:
                print(f"  [warn] could not add PK on {table}.{col}: {e}")

        # Foreign keys
        for table, col, ref_table, ref_col in FOREIGN_KEYS:
            fk_name = f"fk_{table}_{col}"
            try:
                conn.execute(text(
                    f"ALTER TABLE `{table}` ADD CONSTRAINT `{fk_name}` "
                    f"FOREIGN KEY (`{col}`) REFERENCES `{ref_table}`(`{ref_col}`)"
                ))
            except Exception as e:
                print(f"  [warn] could not add FK {table}.{col} -> {ref_table}.{ref_col}: {e}")

        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

    print("\nMySQL Import complete. All 15 tables loaded successfully.")


if __name__ == "__main__":
    main()
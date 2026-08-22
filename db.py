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


if __name__ == "__main__":
    t = load_all_tables()
    for name, df in t.items():
        print(f"{name:<22} {df.shape}")

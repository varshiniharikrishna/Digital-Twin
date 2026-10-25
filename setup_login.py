"""Create or reset one MySQL login without re-importing or dropping app tables."""
from __future__ import annotations

import argparse
import getpass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import generate_password_hash

import db


def configure_login(role: str, login: str, password: str) -> str:
    role = role.upper().strip()
    login = login.strip()
    if role not in {"ADMIN", "DOCTOR", "PATIENT"}:
        raise ValueError("Role must be ADMIN, DOCTOR, or PATIENT.")
    if not login:
        raise ValueError("Login ID cannot be empty.")
    if len(password) < 12:
        raise ValueError("Choose a password with at least 12 characters.")

    engine = db.get_db_engine()
    with engine.begin() as conn:
        # Safe on existing databases: creates only the authentication table if
        # it is absent; it never drops or rewrites clinical tables.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS `User_Account` (
                `User_ID` VARCHAR(64) PRIMARY KEY,
                `Username` VARCHAR(128) UNIQUE NOT NULL,
                `Email` VARCHAR(128) UNIQUE NOT NULL,
                `Password_Hash` VARCHAR(255) NOT NULL,
                `Role` VARCHAR(32) NOT NULL,
                `Linked_Doctor_ID` VARCHAR(64) NULL,
                `Linked_Patient_ID` VARCHAR(64) NULL,
                `Created_At` DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))

        doctor_id = patient_id = None
        source_email = None
        if role == "DOCTOR":
            record = conn.execute(text(
                "SELECT * FROM `Doctor` WHERE LOWER(`Doctor_ID`) = LOWER(:login) LIMIT 1"
            ), {"login": login}).mappings().first()
            if not record:
                raise ValueError(f"Doctor ID {login!r} was not found in the Doctor table.")
            doctor_id = str(record["Doctor_ID"])
            account_id = f"USR_{doctor_id}"
            username = doctor_id.lower()
            source_email = record.get("Contact_Email")
            existing = conn.execute(text(
                "SELECT `User_ID`, `Email` FROM `User_Account` WHERE `Linked_Doctor_ID` = :id LIMIT 1"
            ), {"id": doctor_id}).mappings().first()
        elif role == "PATIENT":
            record = conn.execute(text(
                "SELECT * FROM `Patient` WHERE LOWER(`Patient_ID`) = LOWER(:login) LIMIT 1"
            ), {"login": login}).mappings().first()
            if not record:
                raise ValueError(f"Patient ID {login!r} was not found in the Patient table.")
            patient_id = str(record["Patient_ID"])
            account_id = f"USR_{patient_id}"
            username = patient_id
            source_email = record.get("Email")
            existing = conn.execute(text(
                "SELECT `User_ID`, `Email` FROM `User_Account` WHERE `Linked_Patient_ID` = :id LIMIT 1"
            ), {"id": patient_id}).mappings().first()
        else:
            account_id = "USR_ADMIN_01"
            username = login
            existing = conn.execute(text(
                "SELECT `User_ID`, `Email` FROM `User_Account` WHERE UPPER(`Role`) = 'ADMIN' LIMIT 1"
            )).mappings().first()

        if existing:
            account_id = existing["User_ID"]
            email = existing["Email"]
        else:
            email = str(source_email or f"{username.casefold()}@digitaltwin.local").strip()
            collision = conn.execute(text(
                "SELECT `User_ID` FROM `User_Account` WHERE LOWER(`Email`) = LOWER(:email) LIMIT 1"
            ), {"email": email}).first()
            if collision:
                email = f"{username.casefold()}.{role.casefold()}@digitaltwin.local"

        row = {"User_ID": account_id, "Username": username, "Email": email,
               "Password_Hash": generate_password_hash(password), "Role": role,
               "Linked_Doctor_ID": doctor_id, "Linked_Patient_ID": patient_id}
        conn.execute(text("""
            INSERT INTO `User_Account`
                (`User_ID`, `Username`, `Email`, `Password_Hash`, `Role`, `Linked_Doctor_ID`, `Linked_Patient_ID`)
            VALUES
                (:User_ID, :Username, :Email, :Password_Hash, :Role, :Linked_Doctor_ID, :Linked_Patient_ID)
            ON DUPLICATE KEY UPDATE
                `Username` = VALUES(`Username`), `Email` = VALUES(`Email`),
                `Password_Hash` = VALUES(`Password_Hash`), `Role` = VALUES(`Role`),
                `Linked_Doctor_ID` = VALUES(`Linked_Doctor_ID`),
                `Linked_Patient_ID` = VALUES(`Linked_Patient_ID`)
        """), row)
    return username


def main() -> int:
    parser = argparse.ArgumentParser(description="Create/reset one app login in MySQL without re-importing project data.")
    parser.add_argument("--role", required=True, choices=["ADMIN", "DOCTOR", "PATIENT"])
    parser.add_argument("--login", required=True, help="Use admin, a Doctor_ID such as DOC0001, or a Patient_ID such as P00001.")
    args = parser.parse_args()
    password = getpass.getpass("Set a password (12+ characters): ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        parser.error("Passwords do not match.")
    try:
        username = configure_login(args.role, args.login, password)
    except (RuntimeError, ValueError, SQLAlchemyError) as exc:
        parser.error(str(exc))
    print(f"{args.role} login {username!r} is ready. The password was stored as a hash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

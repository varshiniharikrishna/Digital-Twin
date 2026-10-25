# Five-minute HTML app demo

This is the walkthrough for the main Flask-served HTML frontend, not the optional Streamlit research dashboard.

## Start the app

Open Terminal and run these commands from the project directory:

```bash
cd "/Users/varshini/Desktop/Digital twin"
python3 -m pip install -r requirements-flask.txt
export DT_DATA_SOURCE=mysql
export DT_MYSQL_USER="your_mysql_username"
read -s 'DT_MYSQL_PASSWORD?MySQL password: '; export DT_MYSQL_PASSWORD
export DT_MYSQL_HOST=localhost
export DT_MYSQL_PORT=3306
export DT_MYSQL_DB=DigitalTwinDB
```

Set passwords for the accounts you will use. Each command prompts for a hidden password twice. Repeat with any other doctor or patient IDs you want to demo:

```bash
python3 setup_login.py --role ADMIN --login admin
python3 setup_login.py --role DOCTOR --login DOC0001
python3 setup_login.py --role PATIENT --login P00001
```

Start Flask and open the local address:

```bash
python3 server.py
```

Open **http://127.0.0.1:5000** in Chrome. Leave this Terminal window running while you use the app. Stop the app with `Control+C`.

`setup_login.py` only creates or resets the selected MySQL login and stores a password hash. It does not re-import/drop project tables. Do not run `import_to_mysql.py` on an existing database to fix credentials; it drops and recreates tables. You need database permissions for account writes and existing `Doctor`/`Patient` rows for those roles.

## Walkthrough

### 0:00–0:45 — Show the three sign-in modes

Point to **Patient Login**, **Doctor Portal**, and **Central Admin**. Say: “This is the actual HTML frontend. The same Flask server serves this page and the API behind it. The page sends credentials to `/api/auth/login`; the backend checks the MySQL `User_Account` password hash and role.”

### 0:45–1:45 — Patient access

Sign in with a Patient ID and the password you set for it. Show the patient’s own recovery record, digital twin, weekly timeline, and recommendations. Say: “A patient account can access only the linked patient record.”

### 1:45–2:45 — Doctor access

Log out, then sign in through **Doctor Portal** with a Doctor ID and its configured password. Show the assigned patient roster and weekly observation/recalibration workflow. Say: “The server checks patient assignment on the API routes; the frontend visibility controls are not the only access check.”

### 2:45–3:30 — Admin access

Log out and sign in through **Central Admin**. Show the admin overview, doctor roster, patient reassignment, and account reset tools. Say: “Admin actions are separately restricted by backend role checks.”

### 3:30–4:15 — Pathway Mining

Open **Pathway Mining** from the Doctor view if that tab is available. Say: “Pathway data are limited to the training-year pool and small samples are marked insufficient evidence. These are historical similarity summaries, not proven treatment effects.”

### 4:15–5:00 — Model numbers and limits

Open the metrics area. On the bundled snapshot, the research benchmark selected VotingRegressor for recovery percentage and recovery weeks. Its 2025 holdout R² values were **0.9227** and **0.7458**, respectively. The backend reports that the bundled CSV data are synthetic and not clinical validation. Say: “These scores describe generated demo data only. The percentage can be a strength/ROM proxy, and this prototype does not establish treatment benefit or clinical safety.”

For exact grouped-CV results, confidence intervals, and calibration coverage, see the generated `reports/model_benchmark/benchmark.json` and `reports/model_benchmark/weeks/benchmark.json`. Regenerate reports before presenting if source data change; the app itself is started with `python3 server.py`.

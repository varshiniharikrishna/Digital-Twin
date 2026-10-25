# Digital Twin — orthopaedic rehabilitation research prototype

The app to run is the browser-based HTML frontend in `templates/index.html`, served by the Flask backend in `server.py`. It includes Patient, Doctor, and Admin access modes. **Streamlit is not part of the run steps below.** The checked-in `output/*.csv` files are synthetic demo records, not patient data or clinical validation.

## Run the HTML app with MySQL

### 1. Install the app dependencies once

In Terminal:

```zsh
cd "/Users/varshini/Desktop/Digital twin"
python3 -m pip install -r requirements-flask.txt
```

This installs the Flask app's Python dependencies into the Python installation used by `python3`; no virtual environment or Streamlit setup is needed.

### 2. Set your MySQL connection details

Create the local settings file and open it:

```zsh
cp .env.example .env
open -e .env
```

Fill in the values for the MySQL server and database you already use, then save the file:

```env
DT_DATA_SOURCE=mysql
DT_MYSQL_USER=your_mysql_username
DT_MYSQL_PASSWORD=your_mysql_password
DT_MYSQL_HOST=localhost
DT_MYSQL_PORT=3306
DT_MYSQL_DB=DigitalTwinDB
```

Replace the example username, password, and database name with your actual MySQL details. `.env` is ignored by Git, so these credentials are kept out of commits. Keep your MySQL server running.

### 3. Start the app

```zsh
./run_app.command
```

The launcher reads `.env`, asks in Terminal for the MySQL username or password if either is blank, and starts Flask. Leave that Terminal window open and visit **http://127.0.0.1:5000** in your browser. Press **Control+C** in Terminal to stop it. You can also double-click `run_app.command` in Finder.

### 4. Log in

The login page checks accounts in the MySQL `User_Account` table. Use an existing account's ID/username and password for its matching Patient, Doctor, or Admin card. If you need to create or reset a demo account, run the relevant command from the project folder:

```zsh
python3 setup_login.py --role ADMIN --login admin
python3 setup_login.py --role DOCTOR --login DOC0001
python3 setup_login.py --role PATIENT --login P00001
```

Run only the command for the role/account you need. It prompts you to enter a password twice. A Doctor_ID or Patient_ID must already exist in its respective MySQL table. These commands create or reset the selected login; they do not import or drop your application tables.

If the page reports a MySQL connection error, check that MySQL is running and the `.env` username, password, host, port, and database name match your setup. **Do not run `import_to_mysql.py` against your existing database to fix a login:** that importer drops and recreates project tables.

## Real-data workflow

See [`schema/patient_schema.yaml`](schema/patient_schema.yaml) and [`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md). The adapter accepts CSV/Parquet, or `mysql` with `DT_REAL_MYSQL_URL` and `DT_REAL_MYSQL_TABLE` set in the environment. It rejects obvious direct-identifier columns; this is not a de-identification certification. The data provider/institution must authorize access and confirm appropriate governance.

```bash
python3 validate_real_data.py /path/to/approved_deidentified_export.csv
```

Or set `DT_DATA_SOURCE=real` and `DT_REAL_DATA_PATH=/path/to/file`. If the requested file is absent or invalid, validation stops with an error and does not create mock “real” records. `DT_TARGET` may select `composite_proxy`, `fim`, `vas_pain`, `dash`, `koos`, or `lefs`. The composite proxy is a generated strength/ROM formula and must not be described as a PROM or clinical recovery measure. PROM results retain their instrument-specific units.

## Reproduce reports and tests

```bash
make all
make test
```

`make all` runs the fixed-seed training/holdout, nested repeated-CV benchmark, similarity comparisons, deviation and adaptive-loop replay, available ablation, multi-condition, fairness, wearable robustness, federated-learning demo and twin-trajectory comparison. It writes experiment artifacts under `reports/<experiment>/` and an index at [`reports/SUMMARY.md`](reports/SUMMARY.md). To skip the compute-intensive benchmark explicitly, run `python3 run_all.py --skip-benchmark`.

All synthetic results are labelled synthetic/demo. If there is no patient-level pathway exposure/outcome linkage, no pathway treatment effect is estimated. If there are no real clinical records, the real-data result is “not run.”

## Method and limitations

- Baseline features exclude weekly observations. Missing measurements remain missing and are imputed inside training-fitted pipelines; no baseline values are filled with invented clinical constants.
- Similarity uses training-fitted numeric ranges, per-feature Gower distances, training-only feature/outcome association weights, a persisted JSON model and a training-only selected threshold. The learned metric is exploratory and requires external validation.
- Deviation uses direction-aware standardized residuals with scales estimated from training-patient weekly residuals (and metric-specific floors). This exploratory fit is not a validated per-week calibration; without adjudicated deterioration labels, sensitivity results are engineering analyses only.
- `twin_state.py` provides the shared exponential state transition and shrinkage update used by the app trajectory wrapper. It remains a prototype, not a physiologically validated simulator.
- Treatment response multipliers are labelled assumed demo inputs because the bundled schema has no linked treatment exposure/outcome data; discharge duration is not shortened by the demo simulator.
- See [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), [`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md) and the DRAFT specification documents for what remains unvalidated.

## Main files

| Module | Purpose |
|---|---|
| `templates/index.html`, `static/app.js`, `server.py` | Main HTML frontend, role-specific access UI, and Flask API/backend |
| `app.py` | Optional Streamlit research dashboard |
| `setup_login.py` | Create or reset a selected MySQL login without re-importing project tables |
| `data_adapter.py`, `validate_real_data.py` | Real-data schema checking and validation CLI |
| `data_prep.py`, `leakage_audit.py`, `modeling.py` | Baseline-only preparation, audit and models |
| `model_benchmark.py`, `research_evaluation.py` | Training-fold benchmark and chronological evaluation |
| `similarity.py`, `pathway_mining.py`, `pathway_ranking.py` | Retrieval, descriptive cohort templates and linked-evidence ranking |
| `simulation.py`, `twin_state.py`, `deviation_analysis.py` | Observation update, state representation and trigger analysis |
| `sensors/`, `federated_demo.py`, `privacy.py` | Wearable interface, simulated FedAvg and privacy utilities |
| `run_all.py`, `Makefile`, `tests/` | Reproducibility and test entry points |

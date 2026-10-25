# Digital Twin — orthopaedic rehabilitation research prototype

The main application is the browser-based HTML frontend in `templates/index.html` served by the Flask API/backend in `server.py`. It has Patient, Doctor, and Admin access modes. `app.py` is a separate optional Streamlit research dashboard; it is not required to run the browser app. The checked-in `output/*.csv` files are **synthetic demo records**, not patient data or clinical validation.

## Run the app

Install only the Flask app dependencies (Streamlit is not needed):

```bash
python3 -m pip install -r requirements-flask.txt
```

First-time setup: copy the sample config. The launcher asks for your MySQL username and password at startup; it does not save the password. Edit `DT_MYSQL_DB` in `.env` only if your database has a different name. The real `.env` file is ignored by Git.

```bash
cp .env.example .env
```

Start the HTML app with one command:

```bash
./run_app.command
```

You can also double-click `run_app.command` in Finder. It starts the same Flask server that serves the HTML frontend and `/api` backend. Install dependencies once with `python3 -m pip install -r requirements-flask.txt` if needed.

Before the first login, make sure `User_Account` exists and configure a password for each account you plan to demo. These commands create or reset only the specified login; they do not import or drop your application tables:

```bash
python3 setup_login.py --role ADMIN --login admin
python3 setup_login.py --role DOCTOR --login DOC0001
python3 setup_login.py --role PATIENT --login P00001
```

Each command prompts twice for a password (12+ characters). Use the matching ID and password in that role's login card. Repeat the Doctor/Patient command with other IDs when you need more demo accounts. `setup_login.py` requires that the selected Doctor_ID or Patient_ID already exists in the corresponding MySQL table.

Start the HTML app:

```bash
python3 server.py
```

Open **http://127.0.0.1:5000**. The Patient, Doctor, and Admin cards call the Flask `/api` routes in this same app. Patient accounts are limited to their own record, Doctors to assigned patients, and Admins to system-wide management. The HTML app requires MySQL for login and account lookup; CSV mode can load synthetic data for inspection but cannot authenticate users.

If you want to inspect the bundled data without logging in, you can start the server in CSV mode:

```bash
DT_DATA_SOURCE=csv python3 server.py
```

The research dashboard is optional and separate: `DT_DATA_SOURCE=csv python3 -m streamlit run app.py`. The Flask launcher loads `.env`; when starting `server.py` directly, the app reads settings from that Terminal's environment.

**Do not run `import_to_mysql.py` against an existing database to fix a login.** That importer drops and recreates project tables. Use `setup_login.py` to configure one account without replacing patient or injury data.

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

# Digital Twin — orthopaedic rehabilitation research prototype

This repository contains a Flask multi-portal app (`server.py`) and a separate Streamlit research dashboard (`app.py`), baseline outcome models, exploratory cohort similarity, an observation-updated state prototype, and report-generation scripts. The checked-in `output/*.csv` files are **synthetic demo records**, not patient data or clinical validation.

## Run the app

Install the project requirements once:

```bash
python3 -m pip install -r requirements.txt
```

For the browser-based Flask app, set your MySQL details in the Terminal session (password is not stored in source files), then start it:

```bash
export DT_DATA_SOURCE=mysql
export DT_MYSQL_USER='your_mysql_user'
read -s 'DT_MYSQL_PASSWORD?MySQL password: '; export DT_MYSQL_PASSWORD
export DT_MYSQL_HOST=localhost
export DT_MYSQL_PORT=3306
export DT_MYSQL_DB=DigitalTwinDB
python3 server.py
```

To use the bundled synthetic CSVs without MySQL:

```bash
DT_DATA_SOURCE=csv python3 server.py
```

The HTML frontend is served at `http://127.0.0.1:5000`. The separate Streamlit dashboard is started with `DT_DATA_SOURCE=csv streamlit run app.py` (or set MySQL environment variables first). `.env.example` lists supported variables; copy it to `.env` only if your launcher loads dotenv. The application reads environment variables directly and does not automatically parse `.env`.

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
- `twin_state.py` provides an exponential state transition and shrinkage update. The app’s hand-coded display curve remains a prototype, not a physiological simulator.
- Treatment response multipliers are labelled assumed demo inputs because the bundled schema has no linked treatment exposure/outcome data; discharge duration is not shortened by the demo simulator.
- See [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), [`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md) and the DRAFT specification documents for what remains unvalidated.

## Main files

| Module | Purpose |
|---|---|
| `app.py`, `server.py` | Streamlit research dashboard and Flask browser app |
| `data_adapter.py`, `validate_real_data.py` | Real-data schema checking and validation CLI |
| `data_prep.py`, `leakage_audit.py`, `modeling.py` | Baseline-only preparation, audit and models |
| `model_benchmark.py`, `research_evaluation.py` | Training-fold benchmark and chronological evaluation |
| `similarity.py`, `pathway_mining.py`, `pathway_ranking.py` | Retrieval, descriptive cohort templates and linked-evidence ranking |
| `simulation.py`, `twin_state.py`, `deviation_analysis.py` | Observation update, state representation and trigger analysis |
| `sensors/`, `federated_demo.py`, `privacy.py` | Wearable interface, simulated FedAvg and privacy utilities |
| `run_all.py`, `Makefile`, `tests/` | Reproducibility and test entry points |

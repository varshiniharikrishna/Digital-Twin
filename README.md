# Dynamic Digital Twin - Orthopaedic Rehabilitation

A single Streamlit application implementing all 7 phases: data prep,
model training, evaluation, per-patient digital twin, weekly recovery
timelines, live recalibration, and a final multi-page dashboard.

## Files

| File | Purpose |
|---|---|
| `config.py` | Data source switch (MySQL/CSV), feature lists, thresholds |
| `db.py` | Loads all 14 tables from MySQL or CSV |
| `data_prep.py` | Merges tables, builds features + targets, chronological train/test split |
| `modeling.py` | Trains the two RandomForest (or XGBoost) regressors, evaluation metrics |
| `similarity.py` | Finds top-K similar historical patients, restricted to a given pool |
| `simulation.py` | Builds predicted weekly trajectories, Phase 6 recalibration logic |
| `evaluation.py` | Plotly figures for Phase 3 |
| `app.py` | Streamlit dashboard - the entry point |

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 - get your data into MySQL

If you haven't already, run `import_to_mysql.py` (also included here)
to load your 14 generator CSVs into the `DigitalTwinDB` database.
See the comments at the top of that file — you just edit
`MYSQL_PASSWORD` and run it.

## Step 2 - point the app at MySQL

The app defaults to MySQL (`DATA_SOURCE = "mysql"` in `config.py`).
Open `config.py` and edit the `MYSQL_CONFIG` dict with your real
credentials:

```python
MYSQL_CONFIG = {
    "user": "root",
    "password": "your_password_here",
    "host": "localhost",
    "port": 3306,
    "database": "DigitalTwinDB",
}
```

(Or, if you prefer, set environment variables `DT_MYSQL_USER`,
`DT_MYSQL_PASSWORD`, `DT_MYSQL_HOST`, `DT_MYSQL_PORT`, `DT_MYSQL_DB`
instead of editing the file.)

A `data/` folder of CSVs is also included as a fallback/testing path —
set `DATA_SOURCE = "csv"` in `config.py` (or `export DT_DATA_SOURCE=csv`)
if you ever want to run the app without MySQL running at all.

## Step 3 - run it

```bash
streamlit run app.py
```

## Important documented assumptions

1. **`Final_Recovery_Pct` target.** The generator's exported tables
   never write out a single "final recovery %" number — `Final_Outcome`
   only stores `Actual_ROM_deg`, `Actual_Strength_pct`,
   `Actual_Pain_Score`, and a categorical `Outcome_Label`. Since Phase 2
   asks for a regression target, `data_prep.py` builds it as:
   `0.5 * Actual_Strength_pct + 0.5 * (Actual_ROM_deg / 145 * 100)`,
   clipped to [0, 100]. This is flagged in the code and is worth
   describing explicitly as a derived/proxy label in your paper's
   methodology section.

2. **Quadriceps/Hamstring strength as initial features.**
   `Initial_Assessment` only stores one combined `Muscle_Strength_pct`
   at baseline — the quad/hamstring split only exists week-by-week in
   `Weekly_Progress`. `data_prep.py` pulls each patient's **Week 1**
   quad/hamstring split as a stand-in for "initial" strength, falling
   back to the combined score if Week 1 data is missing for any
   patient.

3. **Predicted weekly trajectory (Phase 5).** The trained digital twin
   predicts a single end-of-care recovery % and total weeks — it is
   not a trained week-by-week temporal model. `simulation.py` forward-
   simulates a predicted curve from the patient's initial values to the
   model's predicted endpoint using the same asymptotic
   (diminishing-returns) shape the synthetic generator itself uses.
   The "actual" curve plotted alongside it is 100% real
   `Weekly_Progress` data. This distinction is called out in the
   dashboard caption on the Weekly Recovery page, and should be, too,
   wherever you describe this in the paper.

4. **Recalibration (Phase 6)** is intentionally rule-based rather than
   a second trained model: entered values become the patient's new
   "current state," similarity search re-runs against the training
   pool only, and the explanation text is generated from which metrics
   deviated and by how much. This keeps the "why" fully traceable for
   a demo/paper, rather than hidden inside a black-box re-predictor.

## Notes on model choice

`modeling.py` defaults to `RandomForestRegressor` (zero extra system
dependencies). Set `USE_XGBOOST = True` at the top of `modeling.py` and
install `xgboost` if you'd rather use that — nothing else in the app
needs to change.

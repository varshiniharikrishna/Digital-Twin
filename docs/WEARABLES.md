# Wearable / IoT integration — DRAFT

`sensors/ingest_wearable.py` accepts CSV/JSON records, validates timestamped IMU-derived fields, drops out-of-range values, applies a rolling median outlier screen and aggregates to patient-week means (step count is summed). The MQTT function is deliberately a stub; production requires authenticated device identity, transport security, consent, clock synchronization, provenance, quality flags and monitoring.

`simulate_wearable_stream.py` emits explicitly labelled SIMULATED DEMO ONLY values. These are not clinical measurements and do not validate the model. Weekly simulated observations may be used by the Streamlit demo to show an ingest flow; pain and strength remain missing if not measured. See `reports/wearable_robustness/` for simulated noise/missingness stress checks.

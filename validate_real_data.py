"""Validate and evaluate a supplied, de-identified real dataset; never synthesizes inputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import config
from data_adapter import load_real_records, describe_dataset, to_tables
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame
from modeling import train_recovery_model, evaluate
from leakage_audit import audit_feature_names
from privacy import k_anonymity

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", help="CSV/Parquet conforming to schema/patient_schema.yaml")
    parser.add_argument("--output", default=str(ROOT / "reports" / "real_data"))
    args = parser.parse_args()
    source = args.data or config.REAL_DATA_PATH
    if not source:
        parser.error("Provide a real-data file path or set DT_REAL_DATA_PATH. No synthetic data will be generated.")
    records = load_real_records(source)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    stats = describe_dataset(records)
    stats["k_anonymity_diagnostic"] = k_anonymity(records,[c for c in ["Age","Sex","Gender","Injury_Type","Severity"] if c in records],k=5)
    (out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, default=str))
    tables = to_tables(records)
    full = build_modelling_dataset(tables)
    audit_feature_names()
    target = pd.to_numeric(full[config.TARGET_RECOVERY_PCT], errors="coerce")
    full = full.loc[target.notna()].copy()
    train, test = chronological_split(full)
    if len(train) < 2 or len(test) < 2:
        raise ValueError("Insufficient labeled observations for a disjoint chronological holdout; dataset statistics were saved.")
    Xtr, Xte = get_feature_frame(train), get_feature_frame(test)
    pm = train_recovery_model(Xtr, train[config.TARGET_RECOVERY_PCT])
    pred = pm.predict(Xte)
    metrics = {"data_origin": "real, user-supplied", "target_selection": config.TARGET_SELECTION,
               "holdout": evaluate(test[config.TARGET_RECOVERY_PCT], pred),
               "n_train": len(train), "n_test": len(test),
               "discharge_duration": "not evaluated: target unavailable in the selected schema" if train[config.TARGET_RECOVERY_WEEKS].notna().sum()<2 else "available in source; separate fit required",
               "similarity": "not scored: no longitudinal/outcome-linked pathway records in the declared one-row schema",
               "deviation_and_adaptive_loop": "not evaluable from this schema without timestamped longitudinal observations"}
    (out / "evaluation.json").write_text(json.dumps(metrics, indent=2, default=str))
    # Real-data validation modules each write their own explicit evaluated or
    # not-evaluable report. No synthetic longitudinal observations are created.
    import model_benchmark, similarity_eval, deviation_analysis, eval_adaptive_loop
    if len(train)>=10:
        model_benchmark.benchmark(Xtr,train[config.TARGET_RECOVERY_PCT],Xte,test[config.TARGET_RECOVERY_PCT],
                                  origin="real user-supplied",out_dir=out/"model_benchmark")
    else:
        (out/"model_benchmark").mkdir(parents=True,exist_ok=True)
        (out/"model_benchmark"/"status.json").write_text(json.dumps({"status":"not evaluable","reason":"At least 10 labeled training records required."},indent=2))
    similarity_eval.run(train,test,out/"similarity")
    deviation_analysis.run_deviation_analysis(tables,out/"deviation")
    eval_adaptive_loop.run(tables,out/"adaptive_loop")
    (out / "README.md").write_text("# Real-data validation\n\nOrigin: real user-supplied data. De-identification is the provider's responsibility; this script only rejects obvious direct-identifier columns.\n\n" + json.dumps(metrics, indent=2, default=str) + "\n")
    print(json.dumps(metrics, indent=2, default=str))

if __name__ == "__main__": main()

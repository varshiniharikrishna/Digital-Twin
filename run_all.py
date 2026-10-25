"""Reproduce the experiment/report bundle from configured data; no data are synthesized here."""
from __future__ import annotations
import argparse, json, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import config, db, modeling
from data_prep import build_modelling_dataset,chronological_split,get_feature_frame

ROOT=Path(__file__).resolve().parent; REPORTS=ROOT/"reports"
def _write_status(name,status,error=None):
    d=REPORTS/name;d.mkdir(parents=True,exist_ok=True);data={"status":status,"error":error}
    (d/"run_status.json").write_text(json.dumps(data,indent=2));return data

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--skip-benchmark",action="store_true",help="Skip the compute-intensive 5x3 nested model benchmark");args=ap.parse_args()
    REPORTS.mkdir(exist_ok=True); tables=db.load_all_tables(); origin=db.LAST_DATA_ORIGIN
    data=build_modelling_dataset(tables);train,test=chronological_split(data)
    summary={"data_origin":origin,"seed":config.RANDOM_STATE,"target":config.TARGET_SELECTION,"experiments":{},"important_boundary":"All bundled output CSVs are synthetic; real data require explicit governed input and DATA_SOURCE=real."}
    def run(name,fn):
        try: result=fn();summary["experiments"][name]={"status":"complete","result":result if isinstance(result,(str,int,float,bool,dict,list,type(None))) else str(type(result))}
        except Exception as exc:
            summary["experiments"][name]={"status":"not completed","error":f"{type(exc).__name__}: {exc}"};_write_status(name,"not completed",str(exc))
    if not args.skip_benchmark:
        def do_bench():
            import model_benchmark
            results={}
            for target_name,target_col,out_dir in (
                ("recovery_pct",config.TARGET_RECOVERY_PCT,REPORTS/"model_benchmark"),
                ("recovery_weeks",config.TARGET_RECOVERY_WEEKS,REPORTS/"model_benchmark"/"weeks"),
            ):
                report,_=model_benchmark.benchmark(
                    get_feature_frame(train),train[target_col],get_feature_frame(test),test[target_col],
                    origin,out_dir=out_dir,groups=train["Patient_ID"],target_name=target_name,
                    groups_test=test["Patient_ID"])
                results[target_name]={"selected_model":report["selected_by_training_cv"],
                    "holdout":report["chronological_holdout"],
                    "cv_mean_R2":report["models"][report["selected_by_training_cv"]]["fold_mean_sd"]["R2"]["mean"]}
            return results
        run("model_benchmark",do_bench)
    else:_write_status("model_benchmark","skipped by flag")
    try:
        import research_evaluation
        run("research_evaluation",lambda:research_evaluation.run(tables,REPORTS/"research_evaluation"/"research_evaluation.json",emit=False))
    except Exception as exc: summary["experiments"]["research_evaluation"]={"status":"not completed","error":str(exc)}
    def do_similarity():
        import similarity_eval
        return similarity_eval.run(train,test,REPORTS/"similarity")
    run("similarity",do_similarity)
    def do_deviation():
        import deviation_analysis
        return deviation_analysis.run_deviation_analysis(tables,REPORTS/"deviation")
    run("deviation",do_deviation)
    def do_adaptive():
        import eval_adaptive_loop
        return eval_adaptive_loop.run(tables,REPORTS/"adaptive_loop")
    run("adaptive_loop",do_adaptive)
    def do_ablation():
        import ablation
        return ablation.run(tables,REPORTS/"ablation")
    run("ablation",do_ablation)
    def do_twin():
        import twin_trajectory_eval
        return twin_trajectory_eval.run(tables.get("Weekly_Progress",pd.DataFrame()),REPORTS/"digital_twin")
    run("digital_twin",do_twin)
    def do_multi():
        import multi_condition_eval
        return multi_condition_eval.run(data,REPORTS/"multi_condition")
    run("multi_condition",do_multi)
    def do_fairness():
        import fairness_analysis
        return fairness_analysis.run(tables,REPORTS/"fairness")
    run("fairness",do_fairness)
    def do_wearable():
        import wearable_robustness
        return wearable_robustness.run(REPORTS/"wearable_robustness")
    run("wearable_robustness",do_wearable)
    def do_federated():
        from federated_demo import run_federated_demo
        sub=train[train[config.TARGET_RECOVERY_PCT].notna()]
        x=get_feature_frame(sub)[config.NUMERIC_FEATURES].apply(pd.to_numeric,errors="coerce")
        x=x.fillna(x.median()).fillna(0).to_numpy();y=pd.to_numeric(sub[config.TARGET_RECOVERY_PCT],errors="coerce").to_numpy()
        result=run_federated_demo(x,y,seed=config.RANDOM_STATE)
        d=REPORTS/"federated";d.mkdir(parents=True,exist_ok=True);(d/"evaluation.json").write_text(json.dumps(result,indent=2))
        return result
    run("federated",do_federated)
    # Current generated schema has templates but no linked patient-level pathway exposure/outcome table.
    pathway_dir=REPORTS/"pathway_ranking";pathway_dir.mkdir(parents=True,exist_ok=True)
    outcomes=tables.get("Final_Outcome",pd.DataFrame())
    if "Pathway_ID" not in outcomes.columns:
        pstatus={"status":"not evaluable","reason":"Final_Outcome has no patient-level Pathway_ID linkage; treatment ranking effects cannot be estimated."}
    else:pstatus={"status":"needs exposure/outcome data","reason":"Pathway IDs exist but must include linked recovery gain, duration, and adverse-event outcomes."}
    (pathway_dir/"evaluation.json").write_text(json.dumps(pstatus,indent=2));summary["experiments"]["pathway_ranking"]=pstatus
    xai_dir=REPORTS/"xai";xai_dir.mkdir(parents=True,exist_ok=True)
    try:
        import explainability
        from modeling import train_recovery_model
        labeled=train[train[config.TARGET_RECOVERY_PCT].notna()];model=train_recovery_model(get_feature_frame(labeled),labeled[config.TARGET_RECOVERY_PCT])
        background=get_feature_frame(labeled);query=get_feature_frame(test).head(1)
        result=explainability.explain_prediction(model,background.head(20),query)
        (xai_dir/"example.json").write_text(json.dumps(result,indent=2))
        global_result=explainability.generate_explanation_artifacts(model,background,query,xai_dir,max_rows=20)
        summary["experiments"]["xai"]={"status":"complete","additivity_error":result["additivity_error"],"global_feature_count":len(global_result["top_global_features"])}
    except Exception as exc:
        xai={"status":"unavailable","reason":str(exc),"note":"No fabricated SHAP explanation saved."};(xai_dir/"status.json").write_text(json.dumps(xai,indent=2));summary["experiments"]["xai"]=xai
    if config.DATA_SOURCE=="real":
        summary["experiments"]["real_data"]={"status":"validated by user-supplied source","path":config.REAL_DATA_PATH}
    else:
        summary["experiments"]["real_data"]={"status":"not run","reason":"No real dataset was supplied. Run validate_real_data.py PATH; this runner never substitutes synthetic data for real data."}
    links=[]
    for name in sorted(summary["experiments"]):
        links.append(f"- [{name}](./{name}/) — {summary['experiments'][name].get('status','status unavailable')}")
    (REPORTS/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    md="# Reproducible experiment summary\n\n"+f"Data origin: **{origin}**. Seed: `{config.RANDOM_STATE}`. Selected target: `{config.TARGET_SELECTION}`.\n\n"+"\n".join(links)+"\n\nBundled data are synthetic. Results on them are software/analysis checks, not clinical validation. See each linked report for failures, unavailable analyses, confidence intervals and assumptions. No report is made from missing data.\n"
    (REPORTS/"SUMMARY.md").write_text(md)
    print(md)
    failures=[n for n,v in summary["experiments"].items() if v.get("status") in {"not completed"}]
    return 1 if failures else 0

if __name__=="__main__":raise SystemExit(main())

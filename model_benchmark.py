"""Training-only repeated nested-CV benchmark plus untouched chronological holdout."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint, uniform
import config, modeling

OUT=Path(__file__).resolve().parent/"reports"/"model_benchmark"

def _encoder():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _prep():
    return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("scale",StandardScaler())]),config.NUMERIC_FEATURES),
                              ("cat",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="__MISSING__",keep_empty_features=True)),("ohe",_encoder())]),config.CATEGORICAL_FEATURES)],sparse_threshold=0)

def candidates():
    regs={"MeanBaseline":DummyRegressor(strategy="mean"),"Ridge":Ridge(),"RandomForest":RandomForestRegressor(random_state=config.RANDOM_STATE,n_jobs=1),
          "ExtraTrees":ExtraTreesRegressor(random_state=config.RANDOM_STATE,n_jobs=1),
          "HistGradientBoosting":HistGradientBoostingRegressor(random_state=config.RANDOM_STATE)}
    try:
        from xgboost import XGBRegressor
        regs["XGBoost"]=XGBRegressor(random_state=config.RANDOM_STATE,n_jobs=1,objective="reg:squarederror",verbosity=0)
    except ImportError: pass
    try:
        from lightgbm import LGBMRegressor
        regs["LightGBM"]=LGBMRegressor(random_state=config.RANDOM_STATE,verbosity=-1,n_jobs=1)
    except ImportError: pass
    try:
        from catboost import CatBoostRegressor
        regs["CatBoost"]=CatBoostRegressor(random_seed=config.RANDOM_STATE,verbose=False,thread_count=1)
    except ImportError: pass
    voting=VotingRegressor([(n,regs[n]) for n in ("RandomForest","ExtraTrees","HistGradientBoosting")],n_jobs=1)
    regs["VotingRegressor"] = voting
    stack=StackingRegressor([(n,regs[n]) for n in ("XGBoost","LightGBM","HistGradientBoosting") if n in regs],final_estimator=Ridge(),cv=3,n_jobs=1)
    if len(stack.estimators)>=2: regs["Stacking_XGB_LGBM_HGB_Ridge"]=stack
    spaces={
      "Ridge":{"model__alpha":uniform(0.01,20)},
      "RandomForest":{"model__n_estimators":randint(100,301),"model__max_depth":[None,8,16],"model__min_samples_leaf":randint(1,5)},
      "ExtraTrees":{"model__n_estimators":randint(100,301),"model__max_depth":[None,8,16],"model__min_samples_leaf":randint(1,5)},
      "HistGradientBoosting":{"model__max_iter":randint(100,301),"model__learning_rate":uniform(.02,.15),"model__max_leaf_nodes":randint(7,32)},
    }
    return regs,spaces

def _metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p); y,p=y[ok],p[ok]
    return {"R2":float(r2_score(y,p)) if len(y)>1 else None,"MAE":float(mean_absolute_error(y,p)) if len(y) else None,
            "RMSE":float(np.sqrt(mean_squared_error(y,p))) if len(y) else None,
            "MAPE":float(np.mean(np.abs((y[y!=0]-p[y!=0])/y[y!=0]))*100) if (y!=0).any() else None,"n":int(len(y))}

def _bootstrap(y,p,seed=config.RANDOM_STATE,n=1000):
    rng=np.random.default_rng(seed); y=np.asarray(y,float); p=np.asarray(p,float); out={k:[] for k in ("R2","MAE","RMSE","MAPE")}
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y)); m=_metrics(y[ix],p[ix])
        for k in out:
            if m[k] is not None and np.isfinite(m[k]): out[k].append(m[k])
    return {k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] if v else None for k,v in out.items()}

def benchmark(X_train,y_train,X_test=None,y_test=None,origin="unspecified",out_dir=OUT,outer_splits=5,repeats=3,n_iter=2):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    y=pd.to_numeric(pd.Series(y_train,index=X_train.index),errors="coerce"); valid=y.notna(); X_train=X_train.loc[valid].reset_index(drop=True); y=y.loc[valid].reset_index(drop=True)
    if len(y)<10: raise ValueError("Need at least 10 labeled training examples for stratified repeated CV.")
    bins=pd.qcut(y.rank(method="first"),q=min(5,max(2,len(y)//5)),labels=False,duplicates="drop").astype(int)
    splitter=RepeatedStratifiedKFold(n_splits=min(outer_splits,max(2,int(bins.value_counts().min()))),n_repeats=repeats,random_state=config.RANDOM_STATE)
    regs,spaces=candidates(); results={}; predictions={}; best_params={}; search_space={k:str(v) for k,v in spaces.items()}
    for name,est in regs.items():
        fold_rows=[]; oof_y=[]; oof_p=[]
        for fold,(tr,va) in enumerate(splitter.split(X_train,bins)):
            tr_bins=bins.iloc[tr]
            inner=StratifiedKFold(n_splits=min(3,int(tr_bins.value_counts().min())),shuffle=True,random_state=config.RANDOM_STATE+fold)
            pipe=Pipeline([("prep",_prep()),("model",est)])
            dist=spaces.get(name,{})
            if dist:
                try:
                    search=RandomizedSearchCV(pipe,dist,n_iter=n_iter,scoring="r2",cv=inner,random_state=config.RANDOM_STATE+fold,n_jobs=1,error_score="raise")
                    search.fit(X_train.iloc[tr],y.iloc[tr]); fit=search.best_estimator_
                except Exception as exc:
                    warnings.warn(f"Tuning {name} failed in fold {fold}: {exc}; using fixed estimator")
                    fit=pipe.fit(X_train.iloc[tr],y.iloc[tr])
            else: fit=pipe.fit(X_train.iloc[tr],y.iloc[tr])
            pred=fit.predict(X_train.iloc[va]); actual=y.iloc[va].to_numpy(); fold_rows.append(_metrics(actual,pred)); oof_y.extend(actual);oof_p.extend(pred)
        results[name]={"fold_mean_sd":{metric:{"mean":float(np.nanmean([r[metric] for r in fold_rows if r[metric] is not None])),"sd":float(np.nanstd([r[metric] for r in fold_rows if r[metric] is not None],ddof=1)) if len(fold_rows)>1 else None} for metric in ("R2","MAE","RMSE","MAPE")},
                       "folds":fold_rows,"oof_metrics":_metrics(oof_y,oof_p),"oof_bootstrap_95_ci":_bootstrap(oof_y,oof_p)}
        predictions[name]={"y":oof_y,"pred":oof_p}
    winner=max(results,key=lambda n:results[n]["fold_mean_sd"]["R2"]["mean"])
    final_pipe=Pipeline([("prep",_prep()),("model",regs[winner])]); final_pipe.fit(X_train,y)
    if spaces.get(winner):
        inner=StratifiedKFold(n_splits=min(5,int(bins.value_counts().min())),shuffle=True,random_state=config.RANDOM_STATE)
        search=RandomizedSearchCV(Pipeline([("prep",_prep()),("model",regs[winner])]),spaces[winner],n_iter=max(5,n_iter),scoring="r2",cv=inner,random_state=config.RANDOM_STATE,n_jobs=1)
        search.fit(X_train,y); final_pipe=search.best_estimator_;best_params[winner]=search.best_params_
    cv_resid=np.abs(np.asarray(predictions[winner]["y"])-np.asarray(predictions[winner]["pred"]))
    q=float(np.quantile(cv_resid,.95)) if len(cv_resid) else None
    holdout=None
    if X_test is not None and y_test is not None:
        yt=pd.to_numeric(pd.Series(y_test),errors="coerce"); mask=yt.notna()
        pred=final_pipe.predict(X_test.loc[mask])
        holdout={"metrics":_metrics(yt[mask],pred),"prediction_interval":{"method":"empirical 95th percentile of repeated-CV absolute residuals","half_width":q,"observed_coverage":float(np.mean(np.abs(yt[mask].to_numpy()-pred)<=q)) if q is not None else None,"formal_guarantee":False}}
        pd.DataFrame({"y_true":yt[mask].to_numpy(),"y_pred":pred,"residual":yt[mask].to_numpy()-pred,
                      "lower":pred-q if q is not None else np.nan,"upper":pred+q if q is not None else np.nan}).to_csv(out_dir/"holdout_predictions.csv",index=False)
        try:
            import plotly.express as px
            fig=px.scatter(x=yt[mask],y=pred,labels={"x":"Observed","y":"Predicted"},title=f"{winner} calibration / observed vs predicted")
            fig.write_image(str(out_dir/"calibration.png"))
            fig2=px.histogram(x=yt[mask].to_numpy()-pred,title="Holdout residual distribution");fig2.write_image(str(out_dir/"residuals.png"))
        except Exception as exc:
            (out_dir/"plots_status.txt").write_text(f"Plot export unavailable: {exc}\n")
    report={"data_origin":origin,"target":"Final_Recovery_Pct alias; configured target="+config.TARGET_SELECTION,
            "seed":config.RANDOM_STATE,"cv":"nested training-fold RandomizedSearchCV inside repeated stratified 5-fold x 3 repeats (where feasible)",
            "outer_cv_folds":len(next(iter(results.values()))["folds"]),"selected_by_training_cv":winner,
            "models":results,"chronological_holdout":holdout,"search_space":search_space,"limitations":["Optional estimators absent from installed dependencies are omitted.","Bootstrap intervals are descriptive patient resamples of out-of-fold predictions.","Prediction interval coverage is empirical, not a guaranteed conformal coverage claim.","Synthetic-source results are not clinical validation."]}
    (out_dir/"benchmark.json").write_text(json.dumps(report,indent=2,default=str))
    (out_dir/"best_params.json").write_text(json.dumps(best_params,indent=2,default=str))
    (out_dir/"search_space.json").write_text(json.dumps(search_space,indent=2,default=str))
    holdout_r2=(holdout or {}).get("metrics",{}).get("R2")
    is_synthetic="synthetic" in origin.casefold();is_real="real" in origin.casefold()
    cv_r2=results[winner]["fold_mean_sd"]["R2"]["mean"]
    synthetic_text=f"R²={cv_r2:.5f} (CV), holdout R²={holdout_r2}" if is_synthetic else "not evaluated"
    real_text=f"R²={cv_r2:.5f} (CV), holdout R²={holdout_r2}" if is_real else "not evaluated (no explicitly identified real dataset supplied)"
    (out_dir/"README.md").write_text("# Model benchmark\n\nOrigin: **"+origin+"**.\n\n- Synthetic data: "+synthetic_text+". Reached R² > 0.65: "+str(bool(is_synthetic and cv_r2>.65))+" (CV criterion).\n- Real data: "+real_text+". Reached R² > 0.65: "+str(bool(is_real and cv_r2>.65))+" (CV criterion).\n\nSelected by training-period CV: `"+winner+"`. CV mean R²="+f"{cv_r2:.5f}"+"; chronological holdout R²="+str(holdout_r2)+". A high synthetic score may reflect generator structure and is not evidence of clinical validity.\n")
    return report,final_pipe

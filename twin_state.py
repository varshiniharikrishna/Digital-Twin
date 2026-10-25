"""Interpretable parametric digital-twin state and Bayesian-style observation update.

The transition is a proof-of-concept model. Parameters must be fitted and
externally validated before clinical use.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

METRICS = ("Pain", "ROM_deg", "Strength_pct", "Hamstring_Strength_pct", "Walking_Speed_mps", "Balance_Score")

def asymptotic(t, start, asymptote, rate):
    return asymptote + (start - asymptote) * np.exp(-max(float(rate), 1e-8) * np.asarray(t, dtype=float))

@dataclass
class DigitalTwin:
    state: dict[str, float]
    static_covariates: dict = field(default_factory=dict)
    theta: dict[str, dict[str, float]] = field(default_factory=dict)
    covariance: dict[str, float] = field(default_factory=dict)
    week: int = 0
    history: list = field(default_factory=list)

    @classmethod
    def from_baseline(cls, row, patient_id=None, priors=None):
        mapping = {"Pain": "Pain_Score", "ROM_deg": "Range_of_Motion_deg", "Strength_pct": "Quadriceps_Strength_pct",
                   "Hamstring_Strength_pct": "Hamstring_Strength_pct", "Walking_Speed_mps": "Walking_Speed_mps", "Balance_Score": "Balance_Score"}
        state = {m: float(row[k]) for m,k in mapping.items() if k in row and np.isfinite(float(row[k]))}
        covars = {k: row.get(k) for k in ("Age", "Gender", "Injury_Type", "Severity", "BMI", "Surgery_Required") if k in row}
        if patient_id: covars["Patient_ID"] = patient_id
        # Latent progress is initialized to the enrollment origin, not measured data.
        state["recovery_progress"] = 0.0
        return cls(state=state, static_covariates=covars, theta=dict(priors or {}), covariance={m: 1.0 for m in state})

    def predict(self, weeks=1, treatment=None):
        nxt = {}
        for metric, x in self.state.items():
            if metric=="recovery_progress":
                rate=float(self.theta.get(metric,{}).get("rate",0.08))
                nxt[metric]=float(1-(1-x)*np.exp(-rate*weeks));continue
            p = self.theta.get(metric, {})
            target = float(p.get("asymptote", x))
            rate = float(p.get("rate", 0.08))
            if metric == "Pain": target = min(target, x)
            else: target = max(target, x)
            nxt[metric] = float(asymptotic(weeks, x, target, rate))
        return nxt

    def update(self, observations: dict, week: int, observation_variance: dict | None = None):
        predicted = self.predict(max(1, week - self.week))
        obs_var = observation_variance or {}
        innovations = {}
        for metric, value in predicted.items():
            self.state[metric]=value
            self.covariance[metric]=float(self.covariance.get(metric,1.0))+0.05
        for metric, observed in observations.items():
            if metric not in predicted or observed is None or not np.isfinite(float(observed)): continue
            prior_var = max(float(self.covariance.get(metric, 1.0)), 1e-8)
            r = max(float(obs_var.get(metric, prior_var)), 1e-8)
            gain = prior_var / (prior_var + r)
            innovation = float(observed) - predicted[metric]
            self.state[metric] = predicted[metric] + gain * innovation
            self.covariance[metric] = max((1-gain)*prior_var, 1e-8)
            innovations[metric] = {"predicted": predicted[metric], "observed": float(observed), "innovation": innovation, "gain": gain}
        self.week = int(week)
        self.history.append({"week": self.week, "innovations": innovations})
        self._refit_rates()
        return innovations

    def _refit_rates(self):
        for metric in self.state:
            obs = [(h["week"], h["innovations"][metric]["observed"]) for h in self.history if metric in h["innovations"]]
            if len(obs) < 3: continue
            t = np.asarray([w for w,_ in obs], dtype=float); y = np.asarray([v for _,v in obs], dtype=float)
            start = y[0]
            try:
                params, _ = curve_fit(lambda tt, a, k: asymptotic(tt, start, a, k), t-t.min(), y,
                                      p0=[y[-1], .1], bounds=([-1e4, .001],[1e4, 3]), maxfev=3000)
                self.theta[metric] = {"asymptote": float(params[0]), "rate": float(params[1])}
            except (RuntimeError, ValueError):
                pass

def fit_hierarchical_priors(weekly: pd.DataFrame, patient_ids=None):
    """Estimate empirical-Bayes starting asymptote/rate medians from training patients."""
    if weekly is None or weekly.empty:return {}
    frame=weekly.copy()
    if patient_ids is not None:frame=frame[frame["Patient_ID"].astype(str).isin(set(map(str,patient_ids)))]
    columns={"Pain":"Pain","ROM_deg":"ROM_deg","Strength_pct":"Quadriceps_Strength_pct",
             "Hamstring_Strength_pct":"Hamstring_Strength_pct","Walking_Speed_mps":"Walking_Speed_mps","Balance_Score":"Balance_Score"}
    fits={m:[] for m in columns}
    for _,group in frame.groupby("Patient_ID"):
        group=group.sort_values("Week")
        if "Week" not in group:continue
        for metric,col in columns.items():
            if col not in group:continue
            y=pd.to_numeric(group[col],errors="coerce");ok=y.notna()
            if ok.sum()<3:continue
            t=pd.to_numeric(group.loc[ok,"Week"],errors="coerce").to_numpy(float);v=y[ok].to_numpy(float);start=float(v[0])
            try:
                p,_=curve_fit(lambda x,a,k:asymptotic(x,start,a,k),t-t.min(),v,p0=[v[-1],.1],bounds=([-1e4,.001],[1e4,3]),maxfev=3000)
                fits[metric].append((float(p[0]),float(p[1])))
            except (RuntimeError,ValueError):continue
    priors={}
    for metric,vals in fits.items():
        if vals:
            priors[metric]={"asymptote":float(np.median([x[0] for x in vals])),"rate":float(np.median([x[1] for x in vals])),
                            "prior_sd_asymptote":float(np.std([x[0] for x in vals],ddof=1)) if len(vals)>1 else None,
                            "n_patients":len(vals),"source":"training-patient empirical prior"}
    return priors

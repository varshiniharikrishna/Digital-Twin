"""Transparent pathway ranking from linked observed exposure/outcome data only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
import config

def weight_sensitivity(rankings_by_weight_set):
    """Kendall rank stability across explicitly supplied objective weights."""
    from scipy.stats import kendalltau
    names=list(rankings_by_weight_set); pairs=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            ra={str(x):j for j,x in enumerate(rankings_by_weight_set[a])};rb={str(x):j for j,x in enumerate(rankings_by_weight_set[b])}
            common=sorted(set(ra)&set(rb))
            tau=float(kendalltau([ra[x] for x in common],[rb[x] for x in common]).statistic) if len(common)>1 else None
            pairs.append({"weights_a":a,"weights_b":b,"n_common":len(common),"kendall_tau":tau})
    return pairs

def wilson_lower(successes, n, z=1.96):
    if n <= 0: return 0.0
    p = successes/n; den = 1+z*z/n
    return max(0.0, (p+z*z/(2*n)-z*np.sqrt((p*(1-p)+z*z/(4*n))/n))/den)

def rank_pathways(patient, cohort, pathways, weights=None, min_n=5):
    """Rank from exposure-linked data. Unavailable effect terms are neutral and flagged assumed."""
    w = {**config.RANK_WEIGHTS, **(weights or {})}
    if not pathways or cohort is None or cohort.empty:
        return []
    rows=[]
    for path in pathways:
        pid=str(path.get("Pathway_ID", ""))
        observed=cohort[cohort.get("Pathway_ID", pd.Series(index=cohort.index,dtype=str)).astype(str).eq(pid)].copy()
        recovery=pd.to_numeric(observed.get("Recovery_Gain", pd.Series(dtype=float)),errors="coerce").dropna()
        duration=pd.to_numeric(observed.get("Actual_Recovery_Time_weeks",pd.Series(dtype=float)),errors="coerce").dropna()
        risk=pd.to_numeric(observed.get("Pain_Flare",pd.Series(dtype=float)),errors="coerce").dropna()
        n=int(max(len(recovery),len(duration),len(risk)))
        supported=n>=min_n
        rg=float(recovery.mean()) if len(recovery) else 0.0
        rg_score=float(np.clip(rg/100 if rg>1 else rg,0,1))
        dur=float(duration.mean()) if len(duration) else 0.0
        risk_rate=float(risk.mean()) if len(risk) else 0.0
        # Conservative evidence shrinkage. No causal effect is asserted.
        lcb=wilson_lower(int(risk.sum()),len(risk)) if len(risk) and risk.between(0,1).all() else 0.0
        outcome_success=int((recovery>0).sum())
        evidence_lcb=wilson_lower(outcome_success,len(recovery)) if len(recovery) else 0.0
        score=w["recovery"]*rg_score - w["weeks"]*(dur/52) - w["risk"]*max(risk_rate,lcb) + w["evidence"]*evidence_lcb
        rows.append({**path,"score":float(score),"n_linked":n,"evidence_status":"observed cohort association" if supported else "assumed (expert prior) / insufficient linked data",
                     "risk_lower_bound":lcb,"evidence_success_lcb":evidence_lcb,"causal_claim":False})
    return sorted(rows,key=lambda r:(-r["score"],str(r.get("Pathway_ID"))))

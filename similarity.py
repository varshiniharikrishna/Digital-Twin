"""Training-fitted, pool-independent Gower similarity and cohort retrieval.

Fit SimilarityModel once on the training split, save it as JSON, then pass it to
all retrieval/evaluation calls. Outcome-derived weights are learned from training
only. This is research software; learned weights need external validation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import config

NUM_COLS = list(config.SIM_NUM_COLS)
CAT_COLS = list(config.SIM_CAT_COLS)

class SimilarityModel:
    def __init__(self, numeric_ranges=None, feature_weights=None, tau=None, k_min=None, k_max=None, metadata=None):
        self.numeric_ranges = numeric_ranges or {}
        self.feature_weights = feature_weights or {}
        self.tau = float(config.SIM_MIN_SCORE if tau is None else tau)
        self.k_min = int(config.SIM_MIN_K if k_min is None else k_min)
        self.k_max = int(config.SIM_TOP_K if k_max is None else k_max)
        self.metadata = metadata or {"weight_method": "uniform fallback; no labeled training outcomes supplied"}

    @classmethod
    def fit(cls, train: pd.DataFrame, outcome: str | None = None, weights: dict | None = None):
        cols = [c for c in NUM_COLS + CAT_COLS if c in train]
        if not cols:
            raise ValueError("Training data contains no configured similarity features.")
        ranges = {}
        for c in NUM_COLS:
            if c in train:
                x = pd.to_numeric(train[c], errors="coerce")
                ranges[c] = [float(x.min()), float(x.max())] if x.notna().any() else [0.0, 1.0]
        raw = {}
        method = "uniform"
        y = pd.to_numeric(train[outcome], errors="coerce") if outcome and outcome in train else None
        if weights:
            raw = {c: max(0.0, float(weights.get(c, 0))) for c in cols}; method = "provided training-derived weights"
        elif y is not None and y.notna().sum() >= 3:
            # Feature-outcome association computed only in the training data.
            # Categorical association is the largest absolute dummy/target rank correlation.
            for c in cols:
                if c in NUM_COLS:
                    x = pd.to_numeric(train[c], errors="coerce")
                    valid = x.notna() & y.notna()
                    corr = spearmanr(x[valid], y[valid]).statistic if valid.sum() >= 3 else 0
                    raw[c] = abs(float(corr)) if np.isfinite(corr) else 0.0
                else:
                    vals = train[c].astype("string")
                    cs = []
                    for level in vals.dropna().unique():
                        valid = vals.notna() & y.notna()
                        indicator = vals[valid].eq(level).astype(float)
                        if indicator.sum() >= 3 and (1 - indicator).sum() >= 2:
                            corr = spearmanr(indicator, y[valid]).statistic
                            if np.isfinite(corr): cs.append(abs(float(corr)))
                    raw[c] = max(cs, default=0.0)
            method = "absolute Spearman feature-outcome association in training split (exploratory)"
        if not raw: raw = {c: 1.0 for c in cols}
        total = sum(raw.values())
        if total <= 0: raw = {c: 1.0 for c in cols}; total = len(cols); method = "uniform fallback: zero/undefined training association"
        w = {c: float(raw[c] / total) for c in cols}
        fitted = cls(ranges, w, metadata={"weight_method": method, "fit_rows": int(len(train)), "outcome_column": outcome if y is not None else None})
        # Train-only threshold: retain a lower quantile of similarities for pairs
        # whose outcome difference is within 20% of the training target IQR.
        # This is an engineering retrieval rule, not a clinical agreement bound.
        if y is not None and y.notna().sum() >= 5:
            valid_idx = y[y.notna()].index[:500]
            pair_scores, pair_agree = [], []
            scale = float(y.quantile(.75) - y.quantile(.25))
            tolerance = max(.20 * scale, 1e-9)
            for i in valid_idx:
                q = train.loc[i]
                candidates = train.loc[valid_idx]
                scores = fitted.score(q, candidates)
                outcomes = pd.to_numeric(candidates[outcome], errors="coerce")
                mask = outcomes.notna() & (candidates.get("Patient_ID", pd.Series(index=candidates.index, dtype=str)).astype(str) != str(q.get("Patient_ID", "__none__")))
                pair_scores.extend(scores[mask.to_numpy()].tolist())
                pair_agree.extend((np.abs(outcomes[mask].to_numpy() - float(y.loc[i])) <= tolerance).tolist())
            if pair_scores and any(pair_agree):
                agree_scores = np.asarray(pair_scores)[np.asarray(pair_agree, dtype=bool)]
                fitted.tau = float(np.quantile(agree_scores, config.SIM_TAU_PERCENTILE))
                fitted.metadata.update({"tau_selection": "training-pair outcome agreement quantile; engineering criterion only", "outcome_agreement_tolerance": tolerance, "tau": fitted.tau})
        return fitted

    def save(self, path):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"numeric_ranges": self.numeric_ranges, "feature_weights": self.feature_weights,
                                 "tau": self.tau, "k_min": self.k_min, "k_max": self.k_max,
                                 "metadata": self.metadata}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["numeric_ranges"], d["feature_weights"], d["tau"], d["k_min"], d["k_max"], d.get("metadata"))

    def score(self, query: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
        distances, weights = [], []
        for c, w in self.feature_weights.items():
            if c not in candidates or c not in query or w <= 0: continue
            vals = candidates[c]
            if c in NUM_COLS:
                v = pd.to_numeric(vals, errors="coerce"); q = pd.to_numeric(pd.Series([query[c]]), errors="coerce").iloc[0]
                lo, hi = self.numeric_ranges.get(c, [0.0, 1.0]); span = hi-lo
                d = (v.sub(q).abs().div(span).clip(0, 1) if span > 0 else v.ne(q).astype(float)).to_numpy(float)
                valid = v.notna().to_numpy() & pd.notna(q)
            else:
                v = vals.astype("string"); q = pd.NA if pd.isna(query[c]) else str(query[c])
                d = v.ne(q).fillna(False).astype(float).to_numpy(); valid = v.notna().to_numpy() & pd.notna(q)
            distances.append(np.where(valid, d, np.nan)); weights.append(w)
        if not distances: return np.zeros(len(candidates), dtype=float)
        arr = np.vstack(distances); weight = np.asarray(weights)[:, None]
        validw = np.isfinite(arr) * weight
        denom = validw.sum(axis=0)
        dist = np.divide(np.nansum(arr * weight, axis=0), denom, out=np.ones(len(candidates)), where=denom > 0)
        return np.clip(1.0 - dist, 0.0, 1.0)

    def explain(self, query: pd.Series, candidate: pd.Series) -> list[dict]:
        """Per-feature contribution to weighted Gower distance for a pair."""
        rows=[]; denominator=0.0
        for c,w in self.feature_weights.items():
            if c not in query or c not in candidate or pd.isna(query[c]) or pd.isna(candidate[c]) or w<=0: continue
            if c in NUM_COLS:
                lo,hi=self.numeric_ranges.get(c,[0.0,1.0]);span=hi-lo
                d=min(abs(float(query[c])-float(candidate[c]))/span,1.0) if span>0 else float(float(query[c])!=float(candidate[c]))
            else:d=float(str(query[c])!=str(candidate[c]))
            rows.append({"feature":c,"weight":w,"distance":d,"weighted_distance":w*d});denominator+=w
        for r in rows:r["normalized_distance_contribution"]=r["weighted_distance"]/denominator if denominator else 0.0
        return sorted(rows,key=lambda r:r["normalized_distance_contribution"],reverse=True)

def find_similar_patients(query_row: pd.Series, candidate_pool: pd.DataFrame, top_k=None,
                          min_score=None, min_k=None, model: SimilarityModel | None = None) -> pd.DataFrame:
    """Returns only neighbours meeting tau. DataFrame.attrs includes confidence status."""
    k = config.SIM_TOP_K if top_k is None else int(top_k)
    tau = config.SIM_MIN_SCORE if min_score is None else float(min_score)
    kmin = config.SIM_MIN_K if min_k is None else int(min_k)
    pool = candidate_pool.reset_index(drop=True).copy()
    if "Patient_ID" in pool and "Patient_ID" in query_row.index:
        pool = pool[pool["Patient_ID"].astype(str) != str(query_row["Patient_ID"])].reset_index(drop=True)
    if pool.empty:
        result = pool.copy(); result["Similarity_Score"] = pd.Series(dtype=float)
        result.attrs["confidence"] = "low-confidence / insufficient cohort"; return result
    fitted = model or SimilarityModel.fit(candidate_pool, config.TARGET_RECOVERY_PCT)
    pool["Similarity_Score"] = fitted.score(query_row, pool)
    pool = pool.sort_values("Similarity_Score", ascending=False)
    passing = pool[pool["Similarity_Score"] >= tau].head(k)
    passing.attrs["confidence"] = "sufficient cohort" if len(passing) >= kmin else "low-confidence / insufficient cohort"
    passing.attrs["tau"] = tau; passing.attrs["n_neighbours"] = len(passing)
    return passing.reset_index(drop=True)

def cohort_outcome_estimate(neighbours: pd.DataFrame, target_col: str) -> float:
    valid = neighbours.dropna(subset=[target_col, "Similarity_Score"])
    if valid.empty: return float("nan")
    w = valid["Similarity_Score"].astype(float).clip(lower=1e-8).to_numpy()
    return float(np.average(pd.to_numeric(valid[target_col], errors="coerce"), weights=w))

def best_pathway_for_cohort(similar_patients: pd.DataFrame, pathways: pd.DataFrame,
                            archetypes_by_patient=None, target="Final_Recovery_Pct"):
    """Return pathway ranked from actual neighbour assignments and outcomes only.

    Input mapping must be patient_id -> {pathway_id, outcome}; success-rate
    template fields are not treated as causal or patient-linked evidence.
    """
    if pathways is None or pathways.empty or not archetypes_by_patient:
        return {"pathway": None, "n": 0, "confidence": "insufficient linked pathway/outcome evidence"}
    ids = set(similar_patients.get("Patient_ID", pd.Series(dtype=str)).astype(str))
    sim_lookup = similar_patients.drop_duplicates("Patient_ID").set_index("Patient_ID")["Similarity_Score"].to_dict() if {"Patient_ID","Similarity_Score"}.issubset(similar_patients.columns) else {}
    rows = []
    for pid in ids:
        rec = archetypes_by_patient.get(pid)
        if isinstance(rec, dict) and rec.get("pathway_id") is not None and pd.notna(rec.get("outcome")):
            rows.append({"Patient_ID": pid, "Pathway_ID": str(rec["pathway_id"]), "Outcome": float(rec["outcome"]),"Weight":float(sim_lookup.get(pid,1.0))})
    if not rows:
        return {"pathway": None, "n": 0, "confidence": "insufficient linked pathway/outcome evidence"}
    observed = pd.DataFrame(rows)
    observed["Weighted_Outcome"]=observed["Outcome"]*observed["Weight"]
    score = observed.groupby("Pathway_ID").agg(weighted_outcome=("Weighted_Outcome", "sum"), weight=("Weight","sum"), n=("Outcome", "size"))
    score["outcome_mean"]=score["weighted_outcome"]/score["weight"].replace(0,np.nan)
    score = score.sort_values(["outcome_mean", "n"], ascending=[False, False])
    best_id = score.index[0]
    match = pathways[pathways["Pathway_ID"].astype(str).eq(best_id)]
    return {"pathway": match.iloc[0].to_dict() if not match.empty else {"Pathway_ID": best_id},
            "n": int(score.iloc[0]["n"]), "confidence": "descriptive observed cohort; not causal",
            "outcome_mean": float(score.iloc[0]["outcome_mean"])}

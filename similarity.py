"""
similarity.py
-------------
Formal patient-similarity metric + retrieval rule (patent claim support).

Definition
    sim(q, c) = 1 - ( W_NUM * d_num(q, c) + W_CAT * d_cat(q, c) )

    d_num(q, c) = || (q_num - c_num) / range_num ||_2 / sqrt(n_num)      in [0, 1]
                  (range_num = min-max range of each feature in the candidate pool;
                   scaled differences are clipped to [0, 1], so the score does NOT
                   depend on the farthest patient in the pool, unlike the old version)
    d_cat(q, c) = (# categorical features that differ) / n_cat           in [0, 1]

Retrieval rule
    neighbours = patients with sim >= SIM_MIN_SCORE, at least SIM_MIN_K,
                 at most SIM_TOP_K (sorted by sim, descending).

Fixes vs. previous version
    * best_pathway_for_cohort() now actually uses the cohort (similarity-weighted
      success) when similar patients carry a Pathway_ID; it previously ignored them.
    * The query patient is excluded from its own neighbours (no self-match).
"""

import numpy as np
import pandas as pd

import config

NUM_COLS = ["Age", "BMI", "Pain_Score", "Range_of_Motion_deg",
            "Quadriceps_Strength_pct", "Balance_Score"]
CAT_COLS = ["Gender", "Injury_Type", "Severity", "Diabetes", "Smoking_Status"]

def find_similar_patients(query_row: pd.Series, candidate_pool: pd.DataFrame,
                          top_k: int = None, min_score: float = None,
                          min_k: int = None) -> pd.DataFrame:
    """
    Retrieve neighbours of `query_row` from `candidate_pool` (use TRAIN data only).
    Returns pool rows + 'Similarity_Score', sorted descending.
    """
    top_k = config.SIM_TOP_K if top_k is None else top_k
    min_score = config.SIM_MIN_SCORE if min_score is None else min_score
    min_k = config.SIM_MIN_K if min_k is None else min_k

    pool = candidate_pool.reset_index(drop=True).copy()
    if "Patient_ID" in pool.columns and "Patient_ID" in query_row.index:
        pool = pool[pool["Patient_ID"] != query_row["Patient_ID"]].reset_index(drop=True)

    # Standard Gower-style distance: numeric absolute differences are scaled
    # by each feature's training-pool range; categoricals use 0/1 mismatch.
    # Missing comparisons are omitted from the relevant component average.
    num_dists = []
    for col in NUM_COLS:
        values = pd.to_numeric(pool[col], errors="coerce")
        query_value = pd.to_numeric(pd.Series([query_row.get(col)]), errors="coerce").iloc[0]
        valid = values.notna() & pd.notna(query_value)
        span = values[valid].max() - values[valid].min() if valid.any() else 0
        if span > 0:
            num_dists.append((values - query_value).abs().div(span).clip(0, 1).where(valid))
        elif valid.any():
            num_dists.append((values != query_value).astype(float).where(valid))
    numeric_distance = pd.concat(num_dists, axis=1).mean(axis=1) if num_dists else pd.Series(np.nan, index=pool.index)

    cat_dists = []
    for col in CAT_COLS:
        values = pool[col].astype("string")
        query_value = pd.NA if pd.isna(query_row.get(col)) else str(query_row.get(col))
        valid = values.notna() & pd.notna(query_value)
        cat_dists.append((values != query_value).astype(float).where(valid))
    categorical_distance = pd.concat(cat_dists, axis=1).mean(axis=1) if cat_dists else pd.Series(np.nan, index=pool.index)

    numeric_weight = config.SIMILARITY_NUMERIC_WEIGHT * numeric_distance.notna()
    categorical_weight = config.SIMILARITY_CATEGORICAL_WEIGHT * categorical_distance.notna()
    weight_total = numeric_weight + categorical_weight
    total_distance = (numeric_weight * numeric_distance.fillna(0)
                      + categorical_weight * categorical_distance.fillna(0)).div(weight_total.replace(0, np.nan))
    similarity = 1 - total_distance.fillna(1.0)
    pool["Similarity_Score"] = np.round(similarity, 3)
    pool = pool.sort_values("Similarity_Score", ascending=False)
    passing = pool[pool["Similarity_Score"] >= min_score]
    chosen = passing if len(passing) >= min_k else pool.head(min_k)
    return chosen.head(max(top_k, min_k)).reset_index(drop=True)


def cohort_outcome_estimate(neighbours: pd.DataFrame, target_col: str) -> float:
    """Similarity-weighted mean outcome of the retrieved cohort."""
    w = neighbours["Similarity_Score"].astype(float).clip(lower=1e-6).values
    y = neighbours[target_col].astype(float).values
    return float(np.sum(w * y) / np.sum(w))


def best_pathway_for_cohort(similar_patients: pd.DataFrame, pathways: pd.DataFrame,
                            archetypes_by_patient: dict = None):
    """
    Select a pathway template by observed prevalence among similar patients.
    Return None when patient-to-pathway assignments are not supplied; template
    success-rate fields are not evidence of pathway effect.
    """
    if pathways.empty or not archetypes_by_patient:
        return None
    ids = set(similar_patients.get("Patient_ID", pd.Series(dtype=str)).astype(str))
    assigned = [str(archetypes_by_patient[pid]) for pid in ids if pid in archetypes_by_patient]
    if not assigned:
        return None
    counts = pd.Series(assigned).value_counts()
    candidates = pathways[pathways["Pathway_ID"].astype(str).isin(counts.index)]
    if candidates.empty:
        return None
    candidates = candidates.copy()
    candidates["_assignment_count"] = candidates["Pathway_ID"].astype(str).map(counts)
    return candidates.sort_values(["_assignment_count", "Pathway_ID"], ascending=[False, True]).iloc[0].drop(labels="_assignment_count")


if __name__ == "__main__":
    from db import load_all_tables
    from data_prep import build_modelling_dataset, chronological_split

    tables = load_all_tables()
    full = build_modelling_dataset(tables)
    train, test = chronological_split(full)
    query = test.iloc[0]
    top = find_similar_patients(query, train)
    print(f"Query {query['Patient_ID']} ({query['Injury_Type']}, {query['Severity']})")
    print(top[["Patient_ID", "Similarity_Score", config.TARGET_RECOVERY_PCT,
               config.TARGET_RECOVERY_WEEKS]])

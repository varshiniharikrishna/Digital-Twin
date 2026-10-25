"""
similarity.py
-------------
Phase 4 requires: "Find and display the Top 5 most similar historical
patients from ONLY the 2023-2024 training data" for a chosen 2025 test
patient.

This module recomputes similarity at request time, restricted explicitly to
whatever DataFrame you pass in as the candidate pool (i.e. the training set only).
Distance measure: Gower-like blend of normalized numeric distance + categorical mismatch rate.
"""

import numpy as np
import pandas as pd
import config

NUM_COLS = ["Age", "BMI", "Pain_Score", "Range_of_Motion_deg", "Quadriceps_Strength_pct", "Balance_Score"]
CAT_COLS = ["Gender", "Injury_Type", "Severity", "Diabetes", "Smoking_Status"]


def find_similar_patients(query_row: pd.Series, candidate_pool: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    query_row: a single row (Series) from the merged modelling dataset.
    candidate_pool: DataFrame of candidate patients (should be TRAIN only).
    Returns candidate_pool rows for the top_k most similar patients, with
    an added Similarity_Score column, sorted descending.
    """
    pool = candidate_pool.reset_index(drop=True).copy()

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
    top = pool.sort_values("Similarity_Score", ascending=False).head(top_k)
    return top.reset_index(drop=True)


def best_pathway_for_cohort(similar_patients: pd.DataFrame, pathways: pd.DataFrame,
                             archetypes_by_patient: dict = None) -> pd.Series:
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
    top5 = find_similar_patients(query, train, top_k=5)
    print(f"Query patient: {query['Patient_ID']} ({query['Injury_Type']}, {query['Severity']})")
    print(top5[["Patient_ID", "Similarity_Score", config.TARGET_RECOVERY_PCT, config.TARGET_RECOVERY_WEEKS]])

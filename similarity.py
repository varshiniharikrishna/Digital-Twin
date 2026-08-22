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

    # ---- numeric distance (min-max normalized) ----
    num_matrix = pool[NUM_COLS].astype(float).values
    query_num = np.array([float(query_row[c]) for c in NUM_COLS])
    
    ranges = num_matrix.max(axis=0) - num_matrix.min(axis=0)
    ranges[ranges == 0] = 1.0
    
    norm_pool = (num_matrix - num_matrix.min(axis=0)) / ranges
    norm_query = (query_num - num_matrix.min(axis=0)) / ranges
    
    num_dist = np.sqrt(((norm_pool - norm_query) ** 2).sum(axis=1))
    num_dist = num_dist / (num_dist.max() or 1)

    # ---- categorical mismatch rate ----
    cat_dist = np.zeros(len(pool))
    for c in CAT_COLS:
        cat_dist += (pool[c].astype(str).values != str(query_row[c])).astype(float)
    cat_dist /= len(CAT_COLS)

    total_dist = 0.6 * num_dist + 0.4 * cat_dist
    similarity = 1 - total_dist

    pool["Similarity_Score"] = np.round(similarity, 3)
    top = pool.sort_values("Similarity_Score", ascending=False).head(top_k)
    return top.reset_index(drop=True)


def best_pathway_for_cohort(similar_patients: pd.DataFrame, pathways: pd.DataFrame,
                             archetypes_by_patient: dict = None) -> pd.Series:
    """
    Given a set of similar patients, pick the Recovery_Pathway row with the
    highest Success_Rate among pathways.
    """
    if pathways.empty:
        return None
    return pathways.sort_values("Success_Rate", ascending=False).iloc[0]


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

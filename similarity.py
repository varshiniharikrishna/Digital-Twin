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


def similarity_scores(query_row: pd.Series, pool: pd.DataFrame) -> np.ndarray:
    """Vector of sim(query, c) for every row c in `pool` (formal definition above)."""
    num_cols = [c for c in config.SIM_NUM_COLS if c in pool.columns]
    cat_cols = [c for c in config.SIM_CAT_COLS if c in pool.columns]

    num = pool[num_cols].astype(float).values
    q = np.array([float(query_row[c]) for c in num_cols])
    lo, hi = np.nanmin(num, axis=0), np.nanmax(num, axis=0)
    rng = np.where((hi - lo) == 0, 1.0, hi - lo)
    diff = np.clip(np.abs(num - q) / rng, 0.0, 1.0)
    d_num = np.sqrt((diff ** 2).sum(axis=1)) / np.sqrt(len(num_cols))

    d_cat = np.zeros(len(pool))
    for c in cat_cols:
        d_cat += (pool[c].astype(str).values != str(query_row[c])).astype(float)
    d_cat /= max(len(cat_cols), 1)

    return 1.0 - (config.SIM_W_NUM * d_num + config.SIM_W_CAT * d_cat)


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

    pool["Similarity_Score"] = np.round(similarity_scores(query_row, pool), 3)
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
    Pick the recovery pathway best supported by the *retrieved cohort*.

    If `similar_patients` has a 'Pathway_ID' column that matches `pathways`,
    pathways are scored by  sum_i sim_i * Success_Rate(path_i) / sum_i sim_i.
    Otherwise falls back to the highest global Success_Rate (old behaviour).
    """
    if pathways is None or pathways.empty:
        return None
    if ("Pathway_ID" in similar_patients.columns and "Pathway_ID" in pathways.columns
            and "Success_Rate" in pathways.columns):
        merged = similar_patients[["Pathway_ID", "Similarity_Score"]].merge(
            pathways[["Pathway_ID", "Success_Rate"]], on="Pathway_ID", how="inner")
        if not merged.empty:
            merged["w"] = merged["Similarity_Score"].astype(float).clip(lower=1e-6)
            score = (merged.assign(ws=merged["w"] * merged["Success_Rate"].astype(float))
                     .groupby("Pathway_ID").agg(ws=("ws", "sum"), w=("w", "sum")))
            best_id = (score["ws"] / score["w"]).idxmax()
            return pathways[pathways["Pathway_ID"] == best_id].iloc[0]
    return pathways.sort_values("Success_Rate", ascending=False).iloc[0]


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
"""Cohort summaries and exploratory pathway-template retrieval.

The bundled schema does not link a patient's observed treatment exposure to a
Recovery_Pathway row. Consequently, cohort outcomes below are descriptive only
and are never presented as a pathway effect or used as a success-rate feature
in personalized ranking.
"""

import numpy as np
import pandas as pd
import config
from similarity import find_similar_patients
from data_prep import _rom_max_for_region


AGE_BANDS = {"<25": (0, 25), "25-40": (25, 40), "40-55": (40, 55), "55+": (55, np.inf)}


def _age_band(age):
    if pd.isna(age):
        return None
    for label, (low, high) in AGE_BANDS.items():
        if low <= float(age) < high:
            return label
    return None


def _cohort_patients(cohort, patients, injuries):
    if patients.empty or injuries.empty or "Patient_ID" not in patients or "Patient_ID" not in injuries:
        return set()
    joined = patients.merge(injuries, on="Patient_ID", suffixes=("", "_inj"))
    injury_type = cohort.get("Injury_Type")
    severity = cohort.get("Severity")
    age_band = cohort.get("Age_Band")
    injury_col = "Injury_Type" if "Injury_Type" in joined else "Injury_Type_inj"
    severity_col = "Severity" if "Severity" in joined else "Severity_inj"
    if injury_type is not None and injury_col in joined:
        joined = joined[joined[injury_col].astype(str).str.casefold() == str(injury_type).casefold()]
    if severity is not None and severity_col in joined:
        joined = joined[joined[severity_col].astype(str).str.casefold() == str(severity).casefold()]
    if age_band in AGE_BANDS and "Age" in joined:
        joined = joined[joined["Age"].apply(_age_band) == age_band]
    return set(joined["Patient_ID"].astype(str))


def _recovery_percentage_values(outcomes, injuries):
    """Return observed target values, using the same region-aware proxy as modeling."""
    rows = outcomes.copy()
    injury_fields = [name for name in ("Patient_ID", "Body_Region", "Body_Part", "Injury_Type") if name in injuries]
    if not rows.empty and "Patient_ID" in injury_fields:
        rows = rows.merge(injuries[injury_fields].drop_duplicates("Patient_ID", keep="first"),
                          on="Patient_ID", how="left", suffixes=("", "_injury"))
    if config.TARGET_SELECTION != "composite_proxy":
        values = pd.to_numeric(rows.get(config.TARGET_OUTCOME_COLUMN, pd.Series(np.nan, index=rows.index)), errors="coerce")
        return rows, values, False, f"configured target: {config.TARGET_OUTCOME_COLUMN}"

    recorded = pd.to_numeric(rows.get("Final_Recovery_Pct", pd.Series(np.nan, index=rows.index)), errors="coerce")
    derived = pd.Series(np.nan, index=rows.index, dtype=float)
    if {"Actual_Strength_pct", "Actual_ROM_deg"}.issubset(rows.columns):
        strength = pd.to_numeric(rows["Actual_Strength_pct"], errors="coerce")
        rom = pd.to_numeric(rows["Actual_ROM_deg"], errors="coerce")
        rom_max = pd.to_numeric(rows.apply(_rom_max_for_region, axis=1), errors="coerce")
        valid = strength.notna() & rom.notna() & rom_max.notna()
        derived.loc[valid] = (0.5 * strength.loc[valid]
                              + 0.5 * rom.loc[valid] / rom_max.loc[valid] * 100.0).clip(0, 100)
    values = recorded.combine_first(derived)
    used_derived = bool((recorded.isna() & derived.notna()).any())
    has_recorded = bool(recorded.notna().any())
    metric = ("recorded recovery percentage with region-normalized proxy for missing rows" if has_recorded and used_derived else
              "region-normalized derived strength-ROM proxy" if used_derived else
              "recorded recovery percentage" if has_recorded else
              "no observed recovery metric")
    return rows, values, used_derived, metric


def mine_pathways_from_tables(tables: dict) -> list:
    """Build template-level summaries from training-year patients only."""
    templates = tables.get("Recovery_Pathway", pd.DataFrame())
    cohorts = tables.get("Cohort_Definition", pd.DataFrame())
    patients = tables.get("Patient", pd.DataFrame())
    injuries = tables.get("Injury", pd.DataFrame())
    outcomes = tables.get("Final_Outcome", pd.DataFrame()).copy()
    steps = tables.get("Pathway_Steps", pd.DataFrame())
    if templates.empty or cohorts.empty or outcomes.empty or "Patient_ID" not in outcomes:
        return []

    # Enforce the split at the mining boundary, even when a caller supplies
    # complete tables. The outcome table is then restricted to the same IDs.
    patients = patients.copy()
    if "Enrollment_Date" in patients:
        year = pd.to_datetime(patients["Enrollment_Date"], errors="coerce").dt.year
    elif "Enrollment_Year" in patients:
        year = pd.to_numeric(patients["Enrollment_Year"], errors="coerce")
    else:
        return []
    patients = patients.loc[year.isin(config.TRAIN_YEARS)].copy()
    training_ids = set(patients["Patient_ID"].astype(str)) if "Patient_ID" in patients else set()
    outcomes = outcomes[outcomes["Patient_ID"].astype(str).isin(training_ids)].copy()
    injuries = injuries[injuries["Patient_ID"].astype(str).isin(training_ids)].copy() if "Patient_ID" in injuries else pd.DataFrame()
    if not training_ids or outcomes.empty:
        return []

    cohort_map = cohorts.set_index("Cohort_ID").to_dict(orient="index") if "Cohort_ID" in cohorts else {}
    results = []
    for i, template in templates.iterrows():
        cohort_id = str(template.get("Cohort_ID", ""))
        cohort = cohort_map.get(cohort_id, {})
        pids = _cohort_patients(cohort, patients, injuries)
        observed = outcomes[outcomes["Patient_ID"].astype(str).isin(pids)] if pids else pd.DataFrame()
        outcome_rows, pct, used_derived_proxy, outcome_metric = _recovery_percentage_values(observed, injuries)
        pct = pd.Series(pct, index=outcome_rows.index, dtype=float)
        weeks = pd.to_numeric(outcome_rows.get("Actual_Recovery_Time_weeks", pd.Series(np.nan, index=outcome_rows.index)), errors="coerce")
        pct_n = int(pct.notna().sum())
        weeks_n = int(weeks.notna().sum())
        observed_n = int(outcome_rows.loc[pct.notna() | weeks.notna(), "Patient_ID"].astype(str).nunique()) if "Patient_ID" in outcome_rows else 0
        enough_evidence = observed_n >= config.PATHWAY_MIN_SAMPLE_SIZE
        avg_pct = float(pct.mean()) if pct_n else None
        success_count = int((pct >= config.PATHWAY_SUCCESS_THRESHOLD_PCT).sum()) if pct_n else None
        success_rate = float(success_count / pct_n * 100.0) if pct_n else None
        avg_weeks = float(weeks.mean()) if weeks.notna().any() else None
        body_region = template.get("Body_Region", template.get("Body_Part", ""))
        injury_type = cohort.get("Injury_Type", template.get("Injury_Type", "Unspecified"))
        severity = cohort.get("Severity", "Unspecified")
        archetype = str(template.get("Archetype_Label", "Unspecified")).replace("_", " ")
        pathway_steps = []
        if not steps.empty and {"Pathway_ID", "Step_Order", "Exercise_Category"}.issubset(steps.columns):
            path_steps = steps[steps["Pathway_ID"].astype(str) == str(template.get("Pathway_ID", ""))].sort_values("Step_Order")
            for _, step in path_steps.iterrows():
                week = step.get("Week_Offset")
                week_text = f"Week {int(week)}" if pd.notna(week) else "Week not specified"
                pathway_steps.append(f"{week_text}: {step['Exercise_Category']}")
        results.append({
            "Pathway_ID": str(template.get("Pathway_ID", f"PW{i+1:03d}")),
            "Cohort_ID": cohort_id,
            "Body_Region": str(body_region),
            "Injury_Type": str(injury_type),
            "Severity": str(severity),
            "Age_Band": str(cohort.get("Age_Band", "Unspecified")),
            "Archetype_Label": archetype,
            "Sample_Size": observed_n,
            "Cohort_Pool_Size": int(len(pids)),
            "Recovery_Pct_N": pct_n,
            "Recovery_Weeks_N": weeks_n,
            "Success_Count": success_count,
            "Success_Threshold_Pct": float(config.PATHWAY_SUCCESS_THRESHOLD_PCT),
            "Average_Recovery_Time_weeks": round(avg_weeks, 1) if avg_weeks is not None else None,
            "Average_Recovery_Pct": round(avg_pct, 1) if avg_pct is not None else None,
            "Success_Rate": round(success_rate, 1) if success_rate is not None else None,
            "evidence_status": "sufficient descriptive sample" if enough_evidence else f"insufficient evidence: n={observed_n}, requires {config.PATHWAY_MIN_SAMPLE_SIZE}; observed summaries are descriptive only",
            "Outcome_Metric": outcome_metric,
            "Outcome_Evidence_Scope": "Descriptive cohort outcomes only; pathway assignment is not linked to patient outcomes.",
            "title": f"Pathway template {template.get('Pathway_ID', '')}: {injury_type} — {archetype}",
            "cohort_desc": f"Cohort: {injury_type}, {severity}, age band {cohort.get('Age_Band', 'unspecified')}.",
            "story": (f"{observed_n} patients with observed outcomes match this cohort. "
                      + (f"This is below the configured minimum n={config.PATHWAY_MIN_SAMPLE_SIZE}; observed values are descriptive only. " if not enough_evidence else "")
                      + "These summaries do not establish that the listed pathway caused or improved these outcomes."),
            "step_flow": pathway_steps,
        })
    return results


def mine_pathways_for_doctor(doctor_id: str, tables: dict) -> list:
    """Summarize only records belonging to the doctor's assigned patients."""
    patients = tables.get("Patient", pd.DataFrame())
    if patients.empty or "Assigned_Doctor_ID" not in patients:
        return []
    assigned = patients[patients["Assigned_Doctor_ID"].astype(str) == str(doctor_id)].copy()
    if assigned.empty:
        return []
    ids = set(assigned["Patient_ID"].astype(str))
    scoped = dict(tables)
    scoped["Patient"] = assigned
    for name in ("Injury", "Final_Outcome"):
        frame = tables.get(name, pd.DataFrame())
        scoped[name] = frame[frame["Patient_ID"].astype(str).isin(ids)].copy() if not frame.empty and "Patient_ID" in frame else pd.DataFrame()
    return mine_pathways_from_tables(scoped)


def recommend_pathways_for_patient(patient_row: pd.Series, candidate_pool: pd.DataFrame, tables: dict, top_k: int = 3, similarity_model=None) -> list:
    """Rank compatible templates by similarity and cohort compatibility only.

    The score is a retrieval score, not a predicted treatment benefit. No
    pathway-specific success statistic is used because outcome linkage is absent.
    """
    mined = mine_pathways_from_tables(tables)
    if not mined:
        return []
    p_region = str(patient_row.get("Body_Region", patient_row.get("Body_Part", ""))).casefold()
    p_injury = str(patient_row.get("Injury_Type", "")).casefold()
    p_severity = str(patient_row.get("Severity", "")).casefold()
    p_age = _age_band(patient_row.get("Age"))
    similar = find_similar_patients(patient_row, candidate_pool, top_k=10, model=similarity_model)
    if similar.empty:
        return []
    patient_similarity = float(similar["Similarity_Score"].mean())

    ranked = []
    for pathway in mined:
        if pathway["Injury_Type"].casefold() != p_injury:
            continue
        region = pathway.get("Body_Region", "").casefold()
        if p_region and region and region != p_region:
            continue
        severity_match = bool(p_severity and pathway["Severity"].casefold() == p_severity)
        age_match = bool(p_age and pathway["Age_Band"] == p_age)
        compatibility = (float(severity_match) + float(age_match)) / 2.0
        similarity_score = 100.0 * patient_similarity
        cohort_score = 100.0 * compatibility
        score = 0.70 * similarity_score + 0.30 * cohort_score
        item = dict(pathway)
        item["match_score"] = round(score, 1)
        item["ranking_components"] = {
            "mean_top_10_gower_similarity": round(similarity_score, 1),
            "severity_and_age_band_compatibility": round(cohort_score, 1),
        }
        item["ranking_interpretation"] = "Exploratory cohort-template similarity; not a treatment efficacy estimate."
        item["neighbor_confidence"] = similar.attrs.get("confidence", "unknown")
        ranked.append(item)
    ranked.sort(key=lambda x: (-x["match_score"], x["Pathway_ID"]))
    return ranked[:top_k]

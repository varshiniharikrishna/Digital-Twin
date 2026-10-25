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


def mine_pathways_from_tables(tables: dict) -> list:
    """Build template-level cohort summaries, not treatment-effect claims."""
    templates = tables.get("Recovery_Pathway", pd.DataFrame())
    cohorts = tables.get("Cohort_Definition", pd.DataFrame())
    patients = tables.get("Patient", pd.DataFrame())
    injuries = tables.get("Injury", pd.DataFrame())
    outcomes = tables.get("Final_Outcome", pd.DataFrame()).copy()
    steps = tables.get("Pathway_Steps", pd.DataFrame())
    if templates.empty or cohorts.empty or outcomes.empty or "Patient_ID" not in outcomes:
        return []

    derived_proxy = False
    if "Final_Recovery_Pct" not in outcomes and {"Actual_Strength_pct", "Actual_ROM_deg"}.issubset(outcomes.columns):
        # The application target is a constructed score, not an observed clinical scale.
        outcomes["Final_Recovery_Pct"] = (
            0.5 * pd.to_numeric(outcomes["Actual_Strength_pct"], errors="coerce")
            + 0.5 * pd.to_numeric(outcomes["Actual_ROM_deg"], errors="coerce") / 140.0 * 100.0
        ).clip(0, 100)
        derived_proxy = True

    cohort_map = cohorts.set_index("Cohort_ID").to_dict(orient="index") if "Cohort_ID" in cohorts else {}
    results = []
    for i, template in templates.iterrows():
        cohort_id = str(template.get("Cohort_ID", ""))
        cohort = cohort_map.get(cohort_id, {})
        pids = _cohort_patients(cohort, patients, injuries)
        observed = outcomes[outcomes["Patient_ID"].astype(str).isin(pids)] if pids else pd.DataFrame()
        if len(observed) < config.PATHWAY_MIN_SAMPLE_SIZE:
            continue

        pct = pd.to_numeric(observed.get("Final_Recovery_Pct", pd.Series(dtype=float)), errors="coerce")
        avg_pct = float(pct.mean()) if pct.notna().any() else None
        success_rate = float((pct >= 80).mean() * 100) if pct.notna().any() else None
        weeks = pd.to_numeric(observed.get("Actual_Recovery_Time_weeks", pd.Series(dtype=float)), errors="coerce")
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
            "Sample_Size": int(len(observed)),
            "Average_Recovery_Time_weeks": round(avg_weeks, 1) if avg_weeks is not None else None,
            "Average_Recovery_Pct": round(avg_pct, 1) if avg_pct is not None else None,
            "Success_Rate": round(success_rate, 1) if success_rate is not None else None,
            "Outcome_Metric": "derived strength-ROM proxy" if derived_proxy else "configured recovery percentage",
            "Outcome_Evidence_Scope": "Descriptive cohort outcomes only; pathway assignment is not linked to patient outcomes.",
            "title": f"Pathway template {template.get('Pathway_ID', '')}: {injury_type} — {archetype}",
            "cohort_desc": f"Cohort: {injury_type}, {severity}, age band {cohort.get('Age_Band', 'unspecified')}.",
            "story": (f"{len(observed)} outcome records match this cohort. These are cohort-level summaries; "
                      "they do not establish that the listed pathway caused or improved these outcomes."),
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


def recommend_pathways_for_patient(patient_row: pd.Series, candidate_pool: pd.DataFrame, tables: dict, top_k: int = 3) -> list:
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
    similar = find_similar_patients(patient_row, candidate_pool, top_k=10)
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
        ranked.append(item)
    ranked.sort(key=lambda x: (-x["match_score"], x["Pathway_ID"]))
    return ranked[:top_k]

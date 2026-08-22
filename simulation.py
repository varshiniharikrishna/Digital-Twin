"""
simulation.py
-------------
Phase 5 (weekly recovery simulation) + Phase 6 (recalibration) + Treatment Clones.

This module provides:
  - forward-simulated predicted trajectory curves (Pain, ROM, Strength, Walking Speed)
  - actual trajectory retrieval from Weekly_Progress
  - 2-3 Treatment Clones (Standard Physio vs Hydrotherapy/Aggressive vs Cautious Track)
    to show how different treatment pathways cause different patient reactions
  - deviation checking & live recalibration
"""

import numpy as np
import pandas as pd
import config


def _asymptotic_curve(start_value, target_value, n_weeks, ceiling=None, floor=None):
    """Generates a week-by-week predicted curve from start to target."""
    F = 1 - 0.1 ** (1 / max(n_weeks, 1))
    values = []
    v = start_value
    for _ in range(1, n_weeks + 1):
        v = v + (target_value - v) * F
        if ceiling is not None:
            v = min(ceiling, v)
        if floor is not None:
            v = max(floor, v)
        values.append(v)
    return values


def build_predicted_trajectory(patient_row: pd.Series, predicted_pct: float, predicted_weeks: float) -> pd.DataFrame:
    """
    Builds a week-by-week predicted trajectory for Pain, ROM, Strength,
    Walking Speed, ending at values consistent with the model's predicted
    Final_Recovery_Pct at the model's predicted recovery week.
    """
    n_weeks = max(int(round(predicted_weeks)), 1)

    target_strength = np.clip(predicted_pct, 0, 100)
    target_rom = np.clip(predicted_pct / 100 * 145, 0, 145)
    target_pain = 0.5 if predicted_pct > 90 else (1.0 if predicted_pct > 75 else 1.8)
    target_walk = 0.9 + (predicted_pct / 100) * 0.75

    pain_curve = _asymptotic_curve(float(patient_row["Pain_Score"]), target_pain, n_weeks, floor=target_pain * 0.6)
    rom_curve = _asymptotic_curve(float(patient_row["Range_of_Motion_deg"]), target_rom, n_weeks, ceiling=145)
    strength_curve = _asymptotic_curve(float(patient_row["Quadriceps_Strength_pct"]), target_strength, n_weeks, ceiling=100)
    walk_curve = _asymptotic_curve(float(patient_row["Walking_Speed_mps"]), target_walk, n_weeks, ceiling=1.8)

    return pd.DataFrame({
        "Week": range(1, n_weeks + 1),
        "Predicted_Pain": pain_curve,
        "Predicted_ROM_deg": rom_curve,
        "Predicted_Strength_pct": strength_curve,
        "Predicted_Walking_Speed_mps": walk_curve,
    })


def build_treatment_clones(patient_row: pd.Series, predicted_pct: float, predicted_weeks: float) -> list:
    """
    Generates 3 Treatment Clones to demonstrate how different rehabilitation protocols
    produce different physiological reactions in the digital twin:
      Clone A: Standard Physiotherapy (Baseline)
      Clone B: Hydrotherapy & Accelerated Track (Faster strength, lower joint strain, -20% duration)
      Clone C: Cautious Balance & Low-Impact Track (Steady progression, minimal flare-up, +15% duration)
    """
    n_base = max(int(round(predicted_weeks)), 1)

    # Clone A: Standard
    traj_a = build_predicted_trajectory(patient_row, predicted_pct, n_base)
    clone_a = {
        "id": "clone_standard",
        "name": "Clone A: Standard Physiotherapy",
        "archetype": "Standard_Track",
        "weeks": n_base,
        "recovery_pct": predicted_pct,
        "final_rom": float(traj_a["Predicted_ROM_deg"].iloc[-1]),
        "final_strength": float(traj_a["Predicted_Strength_pct"].iloc[-1]),
        "final_pain": float(traj_a["Predicted_Pain"].iloc[-1]),
        "reaction": "Steady baseline progression. Moderate joint flare-up risk in mid-phase.",
        "color": "#10B981", # Emerald
        "trajectory": traj_a
    }

    # Clone B: Hydrotherapy & Accelerated
    wks_b = max(int(round(n_base * 0.8)), 1)
    pct_b = min(100.0, predicted_pct * 1.05)
    traj_b = build_predicted_trajectory(patient_row, pct_b, wks_b)
    # Accelerated pain drop due to aquatic buoyancy
    traj_b["Predicted_Pain"] = np.clip(traj_b["Predicted_Pain"] * 0.85, 0.4, 10.0)
    clone_b = {
        "id": "clone_hydro",
        "name": "Clone B: Hydrotherapy & Accelerated Track",
        "archetype": "Fast_Track",
        "weeks": wks_b,
        "recovery_pct": pct_b,
        "final_rom": float(traj_b["Predicted_ROM_deg"].iloc[-1]),
        "final_strength": float(traj_b["Predicted_Strength_pct"].iloc[-1]),
        "final_pain": float(traj_b["Predicted_Pain"].iloc[-1]),
        "reaction": "Rapid pain relief and accelerated early ROM gain due to reduced weight-bearing stress.",
        "color": "#F59E0B", # Amber
        "trajectory": traj_b
    }

    # Clone C: Cautious Balance
    wks_c = max(int(round(n_base * 1.15)), 1)
    pct_c = max(50.0, predicted_pct * 0.95)
    traj_c = build_predicted_trajectory(patient_row, pct_c, wks_c)
    clone_c = {
        "id": "clone_cautious",
        "name": "Clone C: Cautious Balance & Low-Impact",
        "archetype": "Cautious_Track",
        "weeks": wks_c,
        "recovery_pct": pct_c,
        "final_rom": float(traj_c["Predicted_ROM_deg"].iloc[-1]),
        "final_strength": float(traj_c["Predicted_Strength_pct"].iloc[-1]),
        "final_pain": float(traj_c["Predicted_Pain"].iloc[-1]),
        "reaction": "Ultra-low strain protocol. Slower timeline but near-zero risk of swelling or re-injury.",
        "color": "#E11D48", # Crimson
        "trajectory": traj_c
    }

    return [clone_a, clone_b, clone_c]


def get_actual_trajectory(weekly_progress: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    """Pulls the real Weekly_Progress rows for one patient, sorted by week."""
    actual = weekly_progress[weekly_progress["Patient_ID"] == patient_id].sort_values("Week")
    return actual[["Week", "Pain", "Swelling", "ROM_deg", "Quadriceps_Strength_pct", "Hamstring_Strength_pct", "Balance_Score", "Walking_Speed_mps", "Compliance_pct", "Patient_Feedback", "Clinician_Remark"]].reset_index(drop=True)


def check_deviation(predicted_row: dict, actual_entry: dict) -> dict:
    """
    Compares a predicted trajectory row (dict-like Series at a given week)
    to user-entered actual values. Returns a dict per-metric of
    {actual, predicted, pct_deviation, flagged}.
    """
    metric_map = {
        "Pain": ("Predicted_Pain", actual_entry.get("Pain")),
        "ROM_deg": ("Predicted_ROM_deg", actual_entry.get("ROM_deg")),
        "Strength_pct": ("Predicted_Strength_pct", actual_entry.get("Strength_pct")),
    }
    result = {}
    for metric, (pred_key, actual_val) in metric_map.items():
        if actual_val is None:
            continue
        pred_val = float(predicted_row[pred_key])
        denom = pred_val if pred_val != 0 else 1e-6
        pct_dev = (actual_val - pred_val) / denom * 100
        result[metric] = {
            "actual": float(actual_val),
            "predicted": float(pred_val),
            "pct_deviation": float(pct_dev),
            "flagged": abs(pct_dev) > config.DEVIATION_THRESHOLD_PCT,
        }
    return result


def recalibrate(patient_row: pd.Series, current_state: dict, week: int,
                 train_pool: pd.DataFrame, pathways: pd.DataFrame,
                 pct_model, weeks_model, feature_fn, deviation: dict):
    """
    Performs Phase 6 recalibration:
      - builds an updated "as-of-now" profile using the entered actual values
      - re-predicts final recovery % / weeks off that updated profile
      - re-runs similarity search restricted to train_pool using updated state
      - recommends the best pathway among the new neighbour cohort
      - returns a plain-language explanation
    """
    import similarity as sim

    updated_row = patient_row.copy()
    if "Pain" in current_state:
        updated_row["Pain_Score"] = current_state["Pain"]
    if "ROM_deg" in current_state:
        updated_row["Range_of_Motion_deg"] = current_state["ROM_deg"]
    if "Strength_pct" in current_state:
        updated_row["Quadriceps_Strength_pct"] = current_state["Strength_pct"]
        updated_row["Hamstring_Strength_pct"] = current_state["Strength_pct"]

    X_updated = feature_fn(pd.DataFrame([updated_row]))
    new_pred_pct = float(pct_model.predict(X_updated)[0])
    new_pred_weeks = float(weeks_model.predict(X_updated)[0])

    new_neighbors = sim.find_similar_patients(updated_row, train_pool, top_k=5)
    recommended_pathway = sim.best_pathway_for_cohort(new_neighbors, pathways)

    # ---- plain-language explanation ----
    flagged_metrics = [m for m, d in deviation.items() if d["flagged"]]
    reasons = []
    for m in flagged_metrics:
        d = deviation[m]
        direction = "slower" if d["pct_deviation"] < 0 else "faster"
        if m == "Pain":
            direction = "worse than expected" if d["pct_deviation"] > 0 else "better than expected"
            reasons.append(f"Pain is {direction} (actual {d['actual']:.1f} vs predicted {d['predicted']:.1f}).")
        else:
            reasons.append(
                f"{m.replace('_', ' ')} is progressing {direction} than expected "
                f"(actual {d['actual']:.1f} vs predicted {d['predicted']:.1f})."
            )

    if reasons:
        pathway_line = ""
        if recommended_pathway is not None:
            pathway_line = (
                f" Patients with similar current trajectories showed the best outcomes on the "
                f"'{recommended_pathway['Archetype_Label']}' pathway "
                f"(historical success rate {recommended_pathway['Success_Rate']*100:.0f}%)."
            )
        explanation = " ".join(reasons) + pathway_line
    else:
        explanation = "No significant deviation detected -- current pathway remains appropriate."

    return {
        "updated_row": updated_row,
        "new_predicted_pct": new_pred_pct,
        "new_predicted_weeks": new_pred_weeks,
        "new_neighbors": new_neighbors,
        "recommended_pathway": recommended_pathway,
        "explanation": explanation,
    }

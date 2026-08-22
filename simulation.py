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

TREATMENT_TAXONOMY = {
    "rom": {
        "label": "Range-of-Motion (ROM) Exercises",
        "subtypes": {
            "passive_rom": {
                "name": "Passive ROM (Therapist / Machine Assisted)",
                "description": "Therapist or machine moves joint while patient relaxes.",
                "pain_impact": -1.8, "rom_boost": 12.0, "strength_boost": 0.0, "balance_boost": 0.0, "speed_boost": 0.0,
                "preferred_stage": "Early Stage", "max_strain": "Low"
            },
            "active_assisted_rom": {
                "name": "Active-Assisted ROM (Strap / Tool Assisted)",
                "description": "Patient moves joint with assistance from non-injured limb or strap.",
                "pain_impact": -1.2, "rom_boost": 16.0, "strength_boost": 4.0, "balance_boost": 0.0, "speed_boost": 0.05,
                "preferred_stage": "Early Stage", "max_strain": "Low-Moderate"
            },
            "active_rom": {
                "name": "Active ROM (Unassisted Movement)",
                "description": "Patient moves joint entirely unassisted through full range.",
                "pain_impact": -0.8, "rom_boost": 22.0, "strength_boost": 8.0, "balance_boost": 2.0, "speed_boost": 0.1,
                "preferred_stage": "Mid Stage", "max_strain": "Moderate"
            }
        }
    },
    "strengthening": {
        "label": "Strengthening Exercises",
        "subtypes": {
            "isometric": {
                "name": "Isometric Strengthening (Static Contraction)",
                "description": "Muscle contraction without moving joint. Builds stability with minimal pain.",
                "pain_impact": -0.5, "rom_boost": 4.0, "strength_boost": 15.0, "balance_boost": 6.0, "speed_boost": 0.1,
                "preferred_stage": "Early Stage", "max_strain": "Low"
            },
            "isotonic": {
                "name": "Isotonic Strengthening (Progressive Resistance)",
                "description": "Muscles move resistance through range of motion.",
                "pain_impact": -0.3, "rom_boost": 10.0, "strength_boost": 24.0, "balance_boost": 10.0, "speed_boost": 0.25,
                "preferred_stage": "Mid Stage", "max_strain": "Moderate"
            },
            "isokinetic": {
                "name": "Isokinetic Strengthening (Machine Controlled Speed)",
                "description": "Maximal effort against machine-controlled constant speed.",
                "pain_impact": 0.5, "rom_boost": 14.0, "strength_boost": 32.0, "balance_boost": 12.0, "speed_boost": 0.35,
                "preferred_stage": "Late Stage", "max_strain": "High"
            }
        }
    },
    "balance": {
        "label": "Balance and Proprioception Training",
        "subtypes": {
            "single_leg_stands": {
                "name": "Single-Leg Balance Stands",
                "description": "Retrains joint position awareness and static balance.",
                "pain_impact": -0.4, "rom_boost": 2.0, "strength_boost": 8.0, "balance_boost": 18.0, "speed_boost": 0.2,
                "preferred_stage": "Mid Stage", "max_strain": "Low-Moderate"
            },
            "wobble_board": {
                "name": "Wobble Board Stability Training",
                "description": "Balancing on unstable surface to challenge joint & core stability.",
                "pain_impact": -0.2, "rom_boost": 5.0, "strength_boost": 12.0, "balance_boost": 28.0, "speed_boost": 0.3,
                "preferred_stage": "Mid Stage", "max_strain": "Moderate"
            },
            "catch_and_throw": {
                "name": "Dynamic Catch & Throw Drills",
                "description": "Sudden movements on balance pads to challenge core & limb reflexes.",
                "pain_impact": 0.2, "rom_boost": 8.0, "strength_boost": 15.0, "balance_boost": 35.0, "speed_boost": 0.4,
                "preferred_stage": "Late Stage", "max_strain": "Moderate-High"
            }
        }
    },
    "functional": {
        "label": "Functional and Agility Exercises",
        "subtypes": {
            "stepups_and_squats": {
                "name": "Step-ups & Functional Squats",
                "description": "Simulating everyday stair climbing and squatting movements.",
                "pain_impact": -0.5, "rom_boost": 12.0, "strength_boost": 18.0, "balance_boost": 15.0, "speed_boost": 0.35,
                "preferred_stage": "Mid Stage", "max_strain": "Moderate"
            },
            "jogging_and_cutting": {
                "name": "Jogging & Multi-Directional Cutting Drills",
                "description": "High-speed directional changes preparing for sport return.",
                "pain_impact": 1.2, "rom_boost": 15.0, "strength_boost": 25.0, "balance_boost": 25.0, "speed_boost": 0.65,
                "preferred_stage": "Late Stage", "max_strain": "High"
            }
        }
    }
}


def simulate_treatment_intervention(patient_row: pd.Series, category: str, exercise_type: str, stage: str, intensity: str = "Standard") -> dict:
    """
    Simulates how a patient's digital twin reacts to a specific clinical treatment intervention.
    """
    cat_info = TREATMENT_TAXONOMY.get(category, TREATMENT_TAXONOMY["rom"])
    ex_info = cat_info["subtypes"].get(exercise_type, list(cat_info["subtypes"].values())[0])

    base_pct = float(patient_row.get("Predicted_Recovery_Pct", 75.0))
    base_weeks = max(int(round(float(patient_row.get("Predicted_Recovery_Weeks", 16.0)))), 1)
    base_traj = build_predicted_trajectory(patient_row, base_pct, base_weeks)

    # Intensity multipliers
    int_mult = 1.3 if intensity == "Aggressive" else (0.7 if intensity == "Gentle" else 1.0)
    
    # Overload check
    is_early = "Early" in stage
    is_high_strain = ex_info["max_strain"] in ["High", "Moderate-High"]
    is_overload = is_early and is_high_strain

    if is_overload:
        appr_status = "warning"
        appr_msg = f"HIGH STRAIN ALERT: {ex_info['name']} applied prematurely in Early Stage! High risk of joint swelling and pain flare-up (+2.5 weeks delay)."
        week_delta = +2.5
        pain_modifier = +1.8
        strength_mod = 0.5
    elif ex_info["preferred_stage"] in stage or "Mid" in stage:
        appr_status = "success"
        appr_msg = f"HIGHLY APPROPRIATE: {ex_info['name']} matches current {stage} goals for {patient_row.get('Body_Part', 'injured joint')}."
        week_delta = -1.5 * int_mult
        pain_modifier = ex_info["pain_impact"] * int_mult
        strength_mod = 1.2 * int_mult
    else:
        appr_status = "info"
        appr_msg = f"MODERATE BENEFIT: {ex_info['name']} provides steady progress during {stage}."
        week_delta = -0.5 * int_mult
        pain_modifier = ex_info["pain_impact"]
        strength_mod = 1.0

    new_weeks = max(int(round(base_weeks + week_delta)), 1)
    
    # Build stimulated curves
    rx_pain = np.clip(base_traj["Predicted_Pain"].values + pain_modifier, 0.5, 10.0)
    rx_rom = np.clip(base_traj["Predicted_ROM_deg"].values + (ex_info["rom_boost"] * int_mult * 0.4), 0, 145.0)
    rx_strength = np.clip(base_traj["Predicted_Strength_pct"].values + (ex_info["strength_boost"] * strength_mod * 0.4), 0, 100.0)
    rx_speed = np.clip(base_traj["Predicted_Walking_Speed_mps"].values + (ex_info["speed_boost"] * int_mult * 0.3), 0.2, 1.8)

    rx_traj = pd.DataFrame({
        "Week": range(1, len(rx_pain) + 1),
        "Predicted_Pain": rx_pain.round(1),
        "Predicted_ROM_deg": rx_rom.round(1),
        "Predicted_Strength_pct": rx_strength.round(1),
        "Predicted_Walking_Speed_mps": rx_speed.round(2),
    })

    reaction_summary = (
        f"For {patient_row.get('Patient_ID')} ({patient_row.get('Injury_Type')}), applying {ex_info['name']} "
        f"at {intensity} intensity in {stage} results in a {rx_strength[-1]:.0f}% strength recovery "
        f"and {rx_rom[-1]:.0f}° ROM. Discharge timeline estimated at {new_weeks} weeks ({week_delta:+.1f} wks vs standard)."
    )

    return {
        "exercise": ex_info,
        "appropriateness": {"status": appr_status, "message": appr_msg},
        "impact": {
            "pain_delta": round(float(pain_modifier), 1),
            "rom_boost": round(float(ex_info["rom_boost"] * int_mult), 1),
            "strength_boost": round(float(ex_info["strength_boost"] * strength_mod), 1),
            "discharge_weeks_orig": base_weeks,
            "discharge_weeks_new": new_weeks,
            "discharge_delta": round(float(week_delta), 1)
        },
        "baseline_trajectory": base_traj.to_dict(orient="records"),
        "treatment_trajectory": rx_traj.to_dict(orient="records"),
        "reaction_summary": reaction_summary
    }


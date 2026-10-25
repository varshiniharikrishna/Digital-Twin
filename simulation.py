"""
simulation.py
-------------
Phase 5 (weekly recovery simulation) + Phase 6 (recalibration) + Treatment Clones.

This module provides:
  - forward-simulated predicted trajectory curves (Pain, ROM, Strength, Walking Speed)
  - actual trajectory retrieval from Weekly_Progress
  - Common Normalized Pain Index (NPI) calculation
  - 6 Treatment Protocol Twins (Standard, ROM Focus, Strength Focus, Balance/Proprioception, Hydrotherapy, Functional/Agility)
  - deviation checking & live recalibration
"""

import numpy as np
import pandas as pd
import config
from similarity import find_similar_patients

SEVERITY_MULT_NPI = {"Mild": 0.0, "Moderate": 0.15, "Severe": 0.30}
SWELLING_MULT_NPI = {"None": 0.0, "Mild": 0.10, "Moderate": 0.20, "Severe": 0.35}


def calculate_normalized_pain_index(pain_score: float, severity: str = "Moderate", swelling: str = "Mild", comorbidity_score: float = 0.0) -> dict:
    """
    Calculates the Common Normalized Pain Index (NPI):
    NPI = (Pain / 10.0) * (1.0 + 0.15 * Sev + 0.10 * Swell + 0.10 * Comorbid)
    """
    raw_pain = float(pain_score)
    sev_mult = SEVERITY_MULT_NPI.get(str(severity), 0.15)
    swell_mult = SWELLING_MULT_NPI.get(str(swelling), 0.10)
    comorb_val = float(comorbidity_score) if comorbidity_score is not None else 0.0

    multiplier = 1.0 + sev_mult + swell_mult + min(0.30, 0.08 * comorb_val)
    npi_score = round(min(1.0, (raw_pain / 10.0) * multiplier), 3)
    npi_pct = round(npi_score * 100.0, 1)

    if npi_score < 0.25:
        category = "Mild / Minimal Pain"
        color = "#10B981"
    elif npi_score < 0.55:
        category = "Moderate Discomfort"
        color = "#F59E0B"
    elif npi_score < 0.75:
        category = "High Pain / Inflammation Risk"
        color = "#E11D48"
    else:
        category = "Severe / Critical Pain Exacerbation"
        color = "#991B1B"

    return {
        "raw_pain": raw_pain,
        "npi_score": npi_score,
        "npi_pct": npi_pct,
        "category": category,
        "color": color
    }


TREATMENT_PROTOCOLS = {
    "Standard": {
        "name": "Standard Physiotherapy Track (Baseline Care)",
        "description": "Standard evidence-based physical therapy protocol with gradual mobility and load progression.",
        "rom_mult": 1.0,
        "strength_mult": 1.0,
        "color": "#64748B"
    },
    "ROM_Focus": {
        "name": "Range-of-Motion & Mobility Focus (Early Phase)",
        "description": "Accelerated passive and active-assisted mobilization targeting early flex/ext deficits.",
        "rom_mult": 1.20,
        "strength_mult": 0.95,
        "color": "#06B6D4"
    },
    "Strength_Focus": {
        "name": "Strengthening & Hypertrophy Focus (Mid Phase)",
        "description": "Progressive high-load resistance training to build peak muscle force and hypertrophy.",
        "rom_mult": 1.05,
        "strength_mult": 1.25,
        "color": "#10B981"
    },
    "Balance_Proprioception": {
        "name": "Balance & Proprioception Focus (Stability Track)",
        "description": "Neuromuscular perturbation training focusing on joint balance, joint position sense, and fall risk reduction.",
        "rom_mult": 1.10,
        "strength_mult": 1.10,
        "color": "#8B5CF6"
    },
    "Hydrotherapy": {
        "name": "Hydrotherapy & Aquatic Rehab Track (Low-Impact)",
        "description": "Low-impact buoyant water walking and pool exercises designed for joint offloading.",
        "rom_mult": 1.15,
        "strength_mult": 0.90,
        "color": "#3B82F6"
    },
    "Functional_Sport_Drills": {
        "name": "Functional & Sport-Specific Agility Drills (Late Phase)",
        "description": "High-velocity cutting, plyometrics, and sport-specific drills for athletic return-to-play clearance.",
        "rom_mult": 1.10,
        "strength_mult": 1.30,
        "color": "#F59E0B"
    }
}


def build_treatment_clones(patient_row: pd.Series, base_pred_pct: float, base_pred_weeks: float) -> list:
    """
    Generates 6 distinct treatment twin clones to demonstrate how different protocols
    cause different physiological reactions in the digital twin.
    """
    clones = []
    for key, p in TREATMENT_PROTOCOLS.items():
        adj_pct = float(np.clip(base_pred_pct * (0.5 * p["rom_mult"] + 0.5 * p["strength_mult"]), 20.0, 100.0))
        adj_weeks = float(np.clip(base_pred_weeks / (0.5 * p["rom_mult"] + 0.5 * p["strength_mult"]), 4.0, 52.0))
        
        traj = build_predicted_trajectory(patient_row, adj_pct, adj_weeks)
        
        clones.append({
            "key": key,
            "name": p["name"],
            "description": p["description"],
            "color": p["color"],
            "recovery_pct": round(adj_pct, 1),
            "weeks": round(adj_weeks, 1),
            "final_rom": float(traj["Predicted_ROM_deg"].iloc[-1]) if not traj.empty else 120.0,
            "final_strength": float(traj["Predicted_Strength_pct"].iloc[-1]) if not traj.empty else 80.0,
            "trajectory": traj
        })
    return clones


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


def build_predicted_trajectory(patient_row: pd.Series, predicted_pct: float, predicted_weeks: float, protocol_key: str = "Standard") -> pd.DataFrame:
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

    start_pain = float(patient_row.get("Pain_Score", 6.0))
    start_rom = float(patient_row.get("Range_of_Motion_deg", 75.0))
    start_quad = float(patient_row.get("Quadriceps_Strength_pct", 50.0))
    start_speed = float(patient_row.get("Walking_Speed_mps", 0.9))

    pain_curve = np.array(_asymptotic_curve(start_pain, target_pain, n_weeks, floor=target_pain * 0.6))
    rom_curve = np.array(_asymptotic_curve(start_rom, target_rom, n_weeks, ceiling=145))
    strength_curve = np.array(_asymptotic_curve(start_quad, target_strength, n_weeks, ceiling=100))
    walk_curve = np.array(_asymptotic_curve(start_speed, target_walk, n_weeks, ceiling=1.8))

    df = pd.DataFrame({
        "Week": range(1, n_weeks + 1),
        "Predicted_Pain": np.round(pain_curve, 1),
        "Predicted_ROM_deg": np.round(rom_curve, 1),
        "Predicted_Strength_pct": np.round(strength_curve, 1),
        "Predicted_Walking_Speed_mps": np.round(walk_curve, 2),
        "Predicted_Recovery_Pct": np.round(strength_curve, 1),
        "Predicted_Function_Score": np.round(np.clip(strength_curve * 0.9, 0, 100), 1)
    })
    return df


def get_actual_trajectory(weekly_progress_df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    """Extracts actual recorded weekly progress rows for a patient."""
    p_df = weekly_progress_df[weekly_progress_df["Patient_ID"] == patient_id].sort_values("Week").copy()
    return p_df


def check_deviation(pred_row: pd.Series, actual_dict: dict, threshold_pct: float = None) -> dict:
    """Flag pain increases or ROM/strength decreases >15% from prediction."""
    if threshold_pct is None:
        threshold_pct = config.DEVIATION_THRESHOLD_PCT
    results = {}
    
    # Pain check
    if "Pain" in actual_dict and "Predicted_Pain" in pred_row:
        p_act = float(actual_dict["Pain"])
        p_exp = float(pred_row["Predicted_Pain"])
        dev = ((p_act - p_exp) / max(p_exp, 1e-5)) * 100.0
        results["Pain"] = {
            "predicted": p_exp,
            "actual": p_act,
            "pct_deviation": round(dev, 1),
            "flagged": dev > threshold_pct
        }

    # ROM check
    if "ROM_deg" in actual_dict and "Predicted_ROM_deg" in pred_row:
        r_act = float(actual_dict["ROM_deg"])
        r_exp = float(pred_row["Predicted_ROM_deg"])
        dev = ((r_act - r_exp) / max(r_exp, 1e-5)) * 100.0
        results["ROM_deg"] = {
            "predicted": r_exp,
            "actual": r_act,
            "pct_deviation": round(dev, 1),
            "flagged": dev < -threshold_pct
        }

    # Strength check
    if "Strength_pct" in actual_dict and "Predicted_Strength_pct" in pred_row:
        s_act = float(actual_dict["Strength_pct"])
        s_exp = float(pred_row["Predicted_Strength_pct"])
        dev = ((s_act - s_exp) / max(s_exp, 1e-5)) * 100.0
        results["Strength_pct"] = {
            "predicted": s_exp,
            "actual": s_act,
            "pct_deviation": round(dev, 1),
            "flagged": dev < -threshold_pct
        }

    return results


def recalibrate(patient_row, actual_dict, current_week, train_data, pathways_df, pct_model, weeks_model, feature_extractor, dev_dict=None):
    """Re-simulates patient recovery based on actual clinical observations."""
    updated_row = patient_row.copy()
    observation_to_state = {
        "Pain": "Pain_Score",
        "ROM_deg": "Range_of_Motion_deg",
        "Strength_pct": "Quadriceps_Strength_pct",
        "Hamstring_Strength_pct": "Hamstring_Strength_pct",
        "Balance_Score": "Balance_Score",
        "Walking_Speed_mps": "Walking_Speed_mps",
    }
    for observation, state_name in observation_to_state.items():
        if observation in actual_dict and pd.notna(actual_dict[observation]):
            updated_row[state_name] = float(actual_dict[observation])

    X = feature_extractor(pd.DataFrame([updated_row]))
    new_pct = float(pct_model.predict(X)[0])
    new_weeks = float(weeks_model.predict(X)[0])

    explanation = (f"Model prediction refreshed at Week {current_week} using the supplied observation. "
                   f"Estimated recovery is {new_pct:.1f}% over {new_weeks:.1f} weeks. "
                   "This is a prediction update, not evidence that treatment or recalibration improves outcomes.")
    
    return {
        "new_predicted_pct": new_pct,
        "new_predicted_weeks": new_weeks,
        "explanation": explanation,
        "new_neighbors": find_similar_patients(updated_row, train_data, top_k=5),
        # A pathway cannot be selected from efficacy because patient-level
        # treatment assignments are absent. Never return the first arbitrary row.
        "recommended_pathway": None
    }


TREATMENT_TAXONOMY = {
    "rom": {
        "name": "Range-of-Motion (ROM) Exercises",
        "subtypes": {
            "rom_passive": {"name": "Passive Joint Mobilization", "rom_boost": 8.0, "strength_boost": 1.0, "pain_impact": -0.4, "speed_boost": 0.05, "preferred_stage": "Early Phase (Wks 1-4)"},
            "rom_active_assisted": {"name": "Active-Assisted Flexion/Extension", "rom_boost": 12.0, "strength_boost": 3.0, "pain_impact": -0.6, "speed_boost": 0.10, "preferred_stage": "Early Phase (Wks 1-4)"},
            "rom_stretching": {"name": "Static Muscle Stretching", "rom_boost": 10.0, "strength_boost": 2.0, "pain_impact": -0.5, "speed_boost": 0.05, "preferred_stage": "Early-Mid Phase"}
        }
    },
    "strengthening": {
        "name": "Strengthening & Load Exercises",
        "subtypes": {
            "strength_isometric": {"name": "Isometric Quad/Ham Contractions", "rom_boost": 3.0, "strength_boost": 8.0, "pain_impact": -0.3, "speed_boost": 0.08, "preferred_stage": "Early Phase (Wks 1-4)"},
            "strength_isotonic": {"name": "Isotonic Leg Press & Extension", "rom_boost": 5.0, "strength_boost": 15.0, "pain_impact": -0.2, "speed_boost": 0.15, "preferred_stage": "Mid Phase (Wks 5-10)"},
            "strength_eccentric": {"name": "Eccentric Load Control", "rom_boost": 4.0, "strength_boost": 18.0, "pain_impact": +0.2, "speed_boost": 0.18, "preferred_stage": "Late-Mid Phase"}
        }
    },
    "balance": {
        "name": "Balance & Proprioception Training",
        "subtypes": {
            "balance_single_leg": {"name": "Single-Leg Stance & Perturbation", "rom_boost": 4.0, "strength_boost": 6.0, "pain_impact": -0.4, "speed_boost": 0.20, "preferred_stage": "Mid-Late Phase"},
            "balance_wobble_board": {"name": "Wobble Board Dynamic Balance", "rom_boost": 6.0, "strength_boost": 8.0, "pain_impact": -0.5, "speed_boost": 0.22, "preferred_stage": "Mid-Late Phase"}
        }
    },
    "functional": {
        "name": "Functional & Agility Drills",
        "subtypes": {
            "func_aquatic": {"name": "Hydrotherapy & Aquatic Pool Walking", "rom_boost": 10.0, "strength_boost": 8.0, "pain_impact": -0.8, "speed_boost": 0.25, "preferred_stage": "Early-Mid Phase"},
            "func_agility_cutting": {"name": "Agility Ladder & Cutting Drills", "rom_boost": 5.0, "strength_boost": 20.0, "pain_impact": +0.4, "speed_boost": 0.35, "preferred_stage": "Late Clearance Phase"}
        }
    }
}


def simulate_single_treatment_reaction(patient_row, category, exercise_type, start_week, duration_weeks, intensity, base_weeks, base_traj):
    """Simulates physiological reaction of digital twin to specific treatment intervention."""
    ex_info = None
    if category in TREATMENT_TAXONOMY and exercise_type in TREATMENT_TAXONOMY[category]["subtypes"]:
        ex_info = TREATMENT_TAXONOMY[category]["subtypes"][exercise_type]
    else:
        ex_info = {"name": "General Rehabilitation Exercise", "rom_boost": 8.0, "strength_boost": 10.0, "pain_impact": -0.4, "speed_boost": 0.15, "preferred_stage": "Mid Phase"}

    int_mult = 0.75 if intensity == "Gentle" else (1.35 if intensity == "Aggressive" else 1.0)
    pain_modifier = ex_info["pain_impact"] * int_mult
    strength_mod = 1.0 * int_mult

    stage_short = f"Week {start_week}"
    appr_status = "success"
    appr_msg = f"APPROPRIATE: {ex_info['name']} introduced at Week {start_week} fits active rehabilitation goals."
    week_delta = -1.0 * int_mult

    new_weeks = max(int(round(base_weeks + week_delta)), 1)
    
    rx_pain = base_traj["Predicted_Pain"].values.copy()
    rx_rom = base_traj["Predicted_ROM_deg"].values.copy()
    rx_strength = base_traj["Predicted_Strength_pct"].values.copy()
    rx_speed = base_traj["Predicted_Walking_Speed_mps"].values.copy()

    n_total = len(rx_pain)
    end_w = min(start_week - 1 + duration_weeks, n_total)

    for idx in range(start_week - 1, n_total):
        if idx < end_w:
            weight = (idx - (start_week - 1) + 1) / float(duration_weeks)
            rx_pain[idx] = np.clip(rx_pain[idx] + pain_modifier * weight, 0.5, 10.0)
            rx_rom[idx] = np.clip(rx_rom[idx] + (ex_info["rom_boost"] * int_mult * 0.4 * weight), 0, 145.0)
            rx_strength[idx] = np.clip(rx_strength[idx] + (ex_info["strength_boost"] * strength_mod * 0.4 * weight), 0, 100.0)
            rx_speed[idx] = np.clip(rx_speed[idx] + (ex_info["speed_boost"] * int_mult * 0.3 * weight), 0.2, 1.8)
        else:
            carry = (end_w - (start_week - 1)) / float(duration_weeks)
            rx_pain[idx] = np.clip(rx_pain[idx] + pain_modifier * 0.5 * carry, 0.5, 10.0)
            rx_rom[idx] = np.clip(rx_rom[idx] + (ex_info["rom_boost"] * int_mult * 0.4 * carry), 0, 145.0)
            rx_strength[idx] = np.clip(rx_strength[idx] + (ex_info["strength_boost"] * strength_mod * 0.4 * carry), 0, 100.0)
            rx_speed[idx] = np.clip(rx_speed[idx] + (ex_info["speed_boost"] * int_mult * 0.3 * carry), 0.2, 1.8)

    rx_traj = pd.DataFrame({
        "Week": range(1, n_total + 1),
        "Predicted_Pain": rx_pain.round(1),
        "Predicted_ROM_deg": rx_rom.round(1),
        "Predicted_Strength_pct": rx_strength.round(1),
        "Predicted_Walking_Speed_mps": rx_speed.round(2),
    })

    reaction_summary = (
        f"For {patient_row.get('Patient_ID')} ({patient_row.get('Injury_Type')}), introducing {ex_info['name']} "
        f"at Week {start_week} for {duration_weeks} weeks ({intensity} intensity) results in a {rx_strength[-1]:.0f}% strength recovery "
        f"and {rx_rom[-1]:.0f}° ROM. Discharge timeline estimated at {new_weeks} weeks ({week_delta:+.1f} wks vs standard)."
    )

    return {
        "exercise": ex_info,
        "start_week": start_week,
        "duration_weeks": duration_weeks,
        "stage_eval": stage_short,
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

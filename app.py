"""
app.py
------
Phase 7 - Streamlit Digital Twin Dashboard for Orthopaedic Rehabilitation.

Run with:
    streamlit run app.py
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import config
import db
from db import load_all_tables
from data_prep import build_modelling_dataset, chronological_split, get_feature_frame
import modeling
import deviation_analysis
import similarity as sim
from twin_state import fit_hierarchical_priors
import simulation as simmod
import evaluation as ev
import pathway_mining as pm
import pdf_generator


st.set_page_config(
    page_title="Dynamic Digital Twin - Orthopaedic Rehabilitation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected Custom CSS for ultra-professional non-blue dark slate theme
st.markdown("""
<style>
/* Main Dark Theme Container */
.stApp {
    background-color: #0F172A;
    color: #F8FAFC;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155 !important;
}

/* Metric card styling */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    border-color: #10B981;
}
.metric-title {
    color: #94A3B8;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #F8FAFC;
}
.metric-sub {
    font-size: 0.8rem;
    color: #10B981;
    margin-top: 2px;
}

/* Pressure point flag box */
.flag-box {
    background: rgba(225, 29, 72, 0.12);
    border: 1px solid #E11D48;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: #FECDD3;
}
.flag-box-title {
    font-weight: 700;
    color: #F43F5E;
    font-size: 0.92rem;
}

/* Twin State card */
.twin-card-real {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
}
.twin-card-twin {
    background: #1E293B;
    border: 1px solid #059669;
    border-radius: 12px;
    padding: 18px;
}

/* SVG Body Map Container */
.svg-container {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CACHED DATA & MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading database & training Digital Twin models...")
def load_everything():
    tables = load_all_tables()
    full = build_modelling_dataset(tables)
    train, test = chronological_split(full)

    X_train = get_feature_frame(train)
    X_test = get_feature_frame(test)

    pct_model, weeks_model = modeling.train_models(
        X_train, train[config.TARGET_RECOVERY_PCT], train[config.TARGET_RECOVERY_WEEKS]
    )
    pred_pct, pred_weeks = modeling.predict(pct_model, weeks_model, X_test)

    metrics_pct = modeling.evaluate(test[config.TARGET_RECOVERY_PCT], pred_pct)
    metrics_weeks = modeling.evaluate(test[config.TARGET_RECOVERY_WEEKS], pred_weeks)
    fi = modeling.get_feature_importance(pct_model)
    similarity_model = sim.SimilarityModel.fit(train, config.TARGET_RECOVERY_PCT)
    train_ids=set(train["Patient_ID"].astype(str))
    weekly=tables.get("Weekly_Progress",pd.DataFrame())
    weekly_training=weekly[weekly["Patient_ID"].astype(str).isin(train_ids)] if "Patient_ID" in weekly else weekly
    twin_priors=fit_hierarchical_priors(weekly_training,train_ids)
    residual_sigmas=deviation_analysis.estimate_training_sigmas(tables,train,pct_model,weeks_model)

    test = test.copy()
    test["Predicted_Recovery_Pct"] = pred_pct
    test["Predicted_Recovery_Weeks"] = pred_weeks

    return {
        "tables": tables,
        "full": full,
        "train": train,
        "test": test,
        "pct_model": pct_model,
        "weeks_model": weeks_model,
        "metrics_pct": metrics_pct,
        "metrics_weeks": metrics_weeks,
        "feature_importance": fi,
        "similarity_model": similarity_model,
        "twin_priors": twin_priors,
        "residual_sigmas": residual_sigmas,
    }


state = load_everything()
tables = state["tables"]
train = state["train"]
test = state["test"]
pct_model = state["pct_model"]
weeks_model = state["weeks_model"]
similarity_model = state["similarity_model"]
twin_priors = state["twin_priors"]
residual_sigmas = state["residual_sigmas"]


# Initialize session state for recalibration history
if "recal_result" not in st.session_state:
    st.session_state["recal_result"] = None
if "recal_week" not in st.session_state:
    st.session_state["recal_week"] = None
if "recal_history" not in st.session_state:
    st.session_state["recal_history"] = []


# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.title("Orthopaedic Digital Twin")
st.sidebar.caption("Dynamic Living Patient Simulation System")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Week-by-Week Twin",
        "Treatment Clones",
        "Digital Twin Overview",
        "Patient Information",
        "Weekly Recovery Timeline",
        "Simulation & Recalibration",
        "Recommendations",
        "Evaluation Metrics",
        "Model Benchmark", "Similarity Spec / Explain", "Deviation Analysis",
        "Pathway Ranking", "Adaptive Loop Evaluation", "Explain (XAI)", "Wearable Demo",
    ]
)

st.sidebar.markdown("---")
patient_id = st.sidebar.selectbox("Select 2025 Test Patient", sorted(test["Patient_ID"].unique()))
patient_row = test[test["Patient_ID"] == patient_id].iloc[0]


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def predict_for_row(row: pd.Series):
    X = get_feature_frame(pd.DataFrame([row]))
    pct = float(pct_model.predict(X)[0])
    weeks = float(weeks_model.predict(X)[0])
    return pct, weeks


def non_blue_line_chart(actual_traj, predicted_traj, metric_actual, metric_pred, title, y_title):
    fig = go.Figure()
    if not actual_traj.empty and metric_actual in actual_traj.columns:
        fig.add_trace(go.Scatter(
            x=actual_traj["Week"], y=actual_traj[metric_actual], mode="lines+markers",
            name="Actual Recorded", line=dict(color="#10B981", width=3),
            marker=dict(size=7, color="#059669")
        ))
    if not predicted_traj.empty and metric_pred in predicted_traj.columns:
        fig.add_trace(go.Scatter(
            x=predicted_traj["Week"], y=predicted_traj[metric_pred], mode="lines",
            name="Twin Predicted Trajectory", line=dict(color="#F59E0B", width=2, dash="dash")
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title=y_title,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        height=350,
        legend=dict(orientation="h", y=1.15)
    )
    return fig


def render_svg_pain_spot(pain_score: float, body_part: str, compensation_event: dict = None):
    """
    Renders a full-body human anatomical SVG diagram displaying dynamic pain spot intensity
    mapped to exact anatomical joint locations.
    """
    if pain_score >= 8.0:
        fill_color = "#E11D48"
        glow_color = "rgba(225, 29, 72, 0.6)"
        radius = 24
    elif pain_score >= 5.0:
        fill_color = "#F59E0B"
        glow_color = "rgba(245, 158, 11, 0.5)"
        radius = 18
    elif pain_score >= 2.0:
        fill_color = "#EAB308"
        glow_color = "rgba(234, 179, 8, 0.4)"
        radius = 13
    else:
        fill_color = "#10B981"
        glow_color = "rgba(16, 185, 129, 0.3)"
        radius = 10

    bp = (body_part or "").lower()
    spot_x, spot_y = 215, 280
    part_label = "Right Knee"

    if "shoulder" in bp or "rotator" in bp:
        spot_x, spot_y, part_label = 245, 85, "Right Shoulder"
    elif "ankle" in bp or "achilles" in bp:
        spot_x, spot_y, part_label = 215, 380, "Right Ankle"
    elif "lower leg" in bp or "tibia" in bp or "calf" in bp:
        spot_x, spot_y, part_label = 215, 330, "Tibia / Calf"
    elif "arm" in bp or "elbow" in bp:
        spot_x, spot_y, part_label = 260, 125, "Elbow / Arm"
    elif "wrist" in bp or "hand" in bp:
        spot_x, spot_y, part_label = 275, 170, "Wrist"
    elif "neck" in bp or "spine" in bp:
        spot_x, spot_y, part_label = 200, 55, "Spine / Neck"
    elif "hip" in bp or "pelvis" in bp:
        spot_x, spot_y, part_label = 215, 195, "Hip Joint"
    else:
        spot_x, spot_y, part_label = 215, 280, "Right Knee Joint"

    comp_marker = ""
    if compensation_event:
        comp_m = compensation_event.get("Muscle_Overloaded", "Muscle")
        comp_l = compensation_event.get("Compensation_Level", "Moderate")
        comp_marker = f"""
        <g transform="translate(260, {spot_y - 20})">
            <rect x="0" y="0" width="110" height="38" rx="6" fill="#7F1D1D" stroke="#EF4444" stroke-width="1.5"/>
            <text x="55" y="15" fill="#FEE2E2" font-size="9" font-weight="bold" text-anchor="middle">OVERLOAD ALERT</text>
            <text x="55" y="28" fill="#FCA5A5" font-size="8" text-anchor="middle">{comp_m} ({comp_l})</text>
        </g>
        <line x1="{spot_x}" y1="{spot_y}" x2="260" y2="{spot_y}" stroke="#EF4444" stroke-width="1.5" stroke-dasharray="3,3"/>
        """

    svg = f"""
    <div class="svg-container">
        <h4 style="color:#10B981; margin-top:0;">Anatomical Visual Twin — {part_label}</h4>
        <svg width="380" height="420" viewBox="0 0 380 420" style="max-width:100%;">
            <defs>
                <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>

            <!-- Full Body Silhouette -->
            <ellipse cx="200" cy="32" rx="16" ry="20" fill="#334155" stroke="#475569" stroke-width="1.5"/>
            <rect x="195" y="52" width="10" height="12" fill="#475569"/>
            <path d="M 165 64 L 235 64 L 225 170 L 175 170 Z" fill="#334155" stroke="#475569" stroke-width="1.5"/>
            
            <!-- Left & Right Arms -->
            <path d="M 165 64 L 140 80 L 125 130" fill="none" stroke="#334155" stroke-width="12" stroke-linecap="round"/>
            <path d="M 235 64 L 260 80 L 275 130 L 285 175" fill="none" stroke="#334155" stroke-width="12" stroke-linecap="round"/>

            <!-- Hips & Legs -->
            <path d="M 175 170 L 225 170 L 230 195 L 170 195 Z" fill="#475569"/>
            <path d="M 180 195 L 180 280 L 180 380" fill="none" stroke="#334155" stroke-width="16" stroke-linecap="round"/>
            <path d="M 220 195 L 220 280 L 220 380" fill="none" stroke="#334155" stroke-width="16" stroke-linecap="round"/>

            <!-- Dynamic Pain Spot -->
            <circle cx="{spot_x}" cy="{spot_y}" r="{radius + 8}" fill="{glow_color}" filter="url(#glow)">
                <animate attributeName="r" values="{radius+5};{radius+12};{radius+5}" dur="2s" repeatCount="indefinite" />
            </circle>
            <circle cx="{spot_x}" cy="{spot_y}" r="{radius}" fill="{fill_color}" stroke="#FFFFFF" stroke-width="2"/>
            <text x="{spot_x}" y="{spot_y + 4}" fill="#FFFFFF" font-size="10" font-weight="bold" text-anchor="middle">{pain_score}</text>

            {comp_marker}
        </svg>
        <div style="font-size:0.85rem; color:#94A3B8; margin-top:4px;">
            Targeted Joint: <b>{part_label}</b> | Recorded Pain: <b>{pain_score}/10</b>
        </div>
    </div>
    """
    return svg


# ========================================================================
# PAGE 1: WEEK-BY-WEEK TWIN (MAIN RECOVERY SHOWCASE)
# ========================================================================
if page == "Week-by-Week Twin":
    st.title(f"Living Patient Twin — Week-by-Week Timeline")
    st.caption(f"Patient ID: {patient_id} | {patient_row['Injury_Type']} ({patient_row['Severity']} Severity)")

    # Fetch actual weekly progress data
    actual_traj = simmod.get_actual_trajectory(tables["Weekly_Progress"], patient_id)
    pred_pct, pred_weeks = predict_for_row(patient_row)
    predicted_traj = simmod.build_predicted_trajectory(patient_row, pred_pct, pred_weeks)

    if actual_traj.empty:
        st.warning("No weekly progress records found for this patient.")
    else:
        max_week = int(actual_traj["Week"].max())
        
        # Week Selector Slider
        col_w1, col_w2 = st.columns([3, 1])
        with col_w1:
            selected_week = st.slider("Select Timeline Week", 1, max_week, 1, key="wbw_week_slider")
        with col_w2:
            st.metric("Timeline Stage", f"Week {selected_week} of {max_week}")

        # Current selected week data row
        week_row = actual_traj[actual_traj["Week"] == selected_week].iloc[0]

        # ----------------------------------------------------------------
        # 1. TWIN STATE (Metric Cards)
        # ----------------------------------------------------------------
        st.markdown("### 1. Selected Week Twin State")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Pain Score", f"{week_row['Pain']}/10")
        m2.metric("ROM", f"{week_row['ROM_deg']}°")
        m3.metric("Quad Strength", f"{week_row['Quadriceps_Strength_pct']}%")
        m4.metric("Hamstring Strength", f"{week_row['Hamstring_Strength_pct']}%")
        m5.metric("Balance", f"{week_row['Balance_Score']}")
        m6.metric("Walking Speed", f"{week_row['Walking_Speed_mps']} m/s")

        sub1, sub2, sub3, sub4 = st.columns(4)
        sub1.metric("Swelling", str(week_row["Swelling"]))
        sub2.metric("Compliance", f"{week_row['Compliance_pct']}%")
        sub3.metric("Patient Feedback", str(week_row["Patient_Feedback"]))
        sub4.metric("Clinician Remark", str(week_row["Clinician_Remark"]))

        st.markdown("---")

        # ----------------------------------------------------------------
        # 2. DYNAMIC VISUAL TWIN & PRESSURE POINTS (Side by Side)
        # ----------------------------------------------------------------
        col_vis, col_flags = st.columns([1, 1])

        # Check muscle compensation table for this week
        comp_df = tables["Muscle_Compensation"]
        comp_match = comp_df[(comp_df["Patient_ID"] == patient_id) & (comp_df["Week"] == selected_week)]
        comp_event = comp_match.iloc[0].to_dict() if not comp_match.empty else None

        with col_vis:
            body_part = patient_row.get("Body_Part", "Knee")
            svg_html = render_svg_pain_spot(float(week_row["Pain"]), body_part, comp_event)
            st.markdown(svg_html, unsafe_allow_html=True)

        with col_flags:
            st.markdown("### 2. Flagged Recovery Pressure Points")
            flags = []

            if float(week_row["Pain"]) > 5.0:
                flags.append({"Category": "High Pain Alert", "Actual": f"{week_row['Pain']}/10", "Flag": "Pain > 5.0 (Moderate to Severe)"})
            if float(week_row["ROM_deg"]) < 90.0:
                flags.append({"Category": "Low ROM Restrict", "Actual": f"{week_row['ROM_deg']}°", "Flag": "ROM < 90° (Flexion deficit)"})
            if float(week_row["Quadriceps_Strength_pct"]) < 60.0:
                flags.append({"Category": "Quad Weakness", "Actual": f"{week_row['Quadriceps_Strength_pct']}%", "Flag": "Strength < 60% (Atrophy risk)"})
            if float(week_row["Balance_Score"]) < 60.0:
                flags.append({"Category": "Impaired Balance", "Actual": f"{week_row['Balance_Score']}", "Flag": "Balance < 60 (Fall risk)"})
            if float(week_row["Compliance_pct"]) < 70.0:
                flags.append({"Category": "Low Compliance", "Actual": f"{week_row['Compliance_pct']}%", "Flag": "Compliance < 70%"})
            if week_row["Swelling"] in ["Severe", "Moderate"]:
                flags.append({"Category": "Joint Swelling", "Actual": str(week_row['Swelling']), "Flag": "Persistent inflammation"})
            if comp_event:
                flags.append({"Category": "Muscle Compensation", "Actual": f"{comp_event['Muscle_Overloaded']} overloaded", "Flag": f"{comp_event['Compensation_Level']} level risk"})

            if flags:
                for flg in flags:
                    st.markdown(f"""
                    <div class="flag-box">
                        <div class="flag-box-title">⚠ {flg['Category']} — {flg['Flag']}</div>
                        <div style="font-size:0.88rem;">Recorded Value: <b>{flg['Actual']}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✓ No critical pressure point flags detected for this week!")

        st.markdown("---")

        # ----------------------------------------------------------------
        # 3. EXPECTED VS ACTUAL COMPARISON FOR SELECTED WEEK
        # ----------------------------------------------------------------
        st.markdown(f"### 3. Twin Expected vs. Actual (Week {selected_week})")
        pred_week_row = predicted_traj[predicted_traj["Week"] == selected_week]
        if not pred_week_row.empty:
            p_row = pred_week_row.iloc[0]
            current_state = {
                "Pain": float(week_row["Pain"]),
                "ROM_deg": float(week_row["ROM_deg"]),
                "Strength_pct": float(week_row["Quadriceps_Strength_pct"]),
            }
            dev_dict = simmod.check_deviation(p_row, current_state, residual_sigmas=residual_sigmas)
            
            c_ev1, c_ev2, c_ev3 = st.columns(3)
            for col, (met, d) in zip([c_ev1, c_ev2, c_ev3], dev_dict.items()):
                status = "⚠ Flagged Deviation" if d["flagged"] else "✓ On Track"
                col.metric(met, f"Actual: {d['actual']:.1f}", f"{d['pct_deviation']:+.1f}% vs Pred ({d['predicted']:.1f})")
                col.caption(status)

        st.markdown("---")

        # ----------------------------------------------------------------
        # 4. RECOVERY JOURNEY & THERAPY PLAN
        # ----------------------------------------------------------------
        st.markdown("### 4. Patient Recovery Journey & Active Rehabilitation Phase")
        
        # Display Pathway Phase
        tplan = tables["Therapy_Plan"]
        week_plan = tplan[(tplan["Patient_ID"] == patient_id) & (tplan["Week_Number"] == selected_week)]
        
        st.info(f"**Current Milestone:** Initial Assessment → **Week {selected_week} (Active)** → Predicted Discharge (Week {pred_weeks:.0f})")

        if not week_plan.empty:
            st.markdown("##### Prescribed Exercises for Selected Week")
            elib = tables["Exercise_Library"].set_index("Exercise_ID")
            plan_merged = week_plan.merge(elib, left_on="Exercise_ID", right_index=True, how="left")
            target_col = "Target_Muscle_or_Function" if "Target_Muscle_or_Function" in plan_merged.columns else ("Target_Muscle" if "Target_Muscle" in plan_merged.columns else "Target_Region")
            plan_merged["Target_Muscle"] = plan_merged[target_col].fillna("Full Body")
            for c in ["Sets", "Repetitions", "Duration_min", "Therapist_Notes"]:
                if c not in plan_merged.columns:
                    plan_merged[c] = "-"

            st.dataframe(
                plan_merged[["Exercise_Name", "Target_Muscle", "Exercise_Category", "Sets", "Repetitions", "Duration_min", "Therapist_Notes"]],
                width="stretch"
            )


        st.markdown("---")

        # ----------------------------------------------------------------
        # 5. GENERATE & DOWNLOAD CLINICAL PDF REPORT
        # ----------------------------------------------------------------
        st.markdown("### 5. Export Patient Twin Clinical PDF Report")
        clones = simmod.build_treatment_clones(patient_row, pred_pct, pred_weeks)
        
        pdf_bytes = pdf_generator.generate_twin_pdf(
            patient_id=patient_id,
            patient_row=patient_row,
            week_data=week_row.to_dict(),
            pressure_points=flags if 'flags' in locals() else [],
            expected_vs_actual=dev_dict if 'dev_dict' in locals() else {},
            treatment_clones=clones,
            recal_history=st.session_state.get("recal_history", [])
        )

        st.download_button(
            label="Download Clinical PDF Report",
            data=pdf_bytes,
            file_name=f"Digital_Twin_Report_{patient_id}_Week{selected_week}.pdf",
            mime="application/pdf",
            type="primary"
        )


# ========================================================================
# ========================================================================
# PAGE 2: TREATMENT REACTION SIMULATOR
# ========================================================================
elif page == "Treatment Clones":
    st.title(f"Single Digital Twin Treatment Reaction Simulator — {patient_id}")
    st.caption("Simulate how this patient's digital twin reacts in real time to specific clinical treatment interventions, exercise sub-protocols, and recovery stages.")

    c_cat, c_sub = st.columns(2)
    with c_cat:
        cat_key = st.selectbox(
            "1. Treatment Category",
            options=list(simmod.TREATMENT_TAXONOMY.keys()),
            format_func=lambda k: simmod.TREATMENT_TAXONOMY[k].get("label", simmod.TREATMENT_TAXONOMY[k]["name"])
        )
    with c_sub:
        sub_map = simmod.TREATMENT_TAXONOMY[cat_key]["subtypes"]
        sub_key = st.selectbox(
            "2. Exercise Sub-Protocol",
            options=list(sub_map.keys()),
            format_func=lambda k: f"{sub_map[k]['name']} ({sub_map[k]['preferred_stage']})"
        )

    c_w, c_dur, c_int = st.columns(3)
    with c_w:
        start_w = st.slider("3. Intervention Start Week", 1, 52, 4)
    with c_dur:
        dur_w = st.slider("4. Duration (Weeks)", 1, 12, 4)
    with c_int:
        intensity = st.selectbox("5. Intervention Intensity", ["Gentle", "Standard", "Aggressive"], index=1)

    rx_res = simmod.simulate_treatment_intervention(
        patient_row=patient_row,
        category=cat_key,
        exercise_type=sub_key,
        intensity=intensity,
        start_week=start_w,
        duration_weeks=dur_w,
        base_pred_pct=predict_for_row(patient_row)[0],
        base_pred_weeks=predict_for_row(patient_row)[1]
    )

    # Banner
    appr = rx_res["appropriateness"]
    if appr["status"] == "warning":
        st.error(appr["message"])
    elif appr["status"] == "success":
        st.success(appr["message"])
    else:
        st.info(appr["message"])

    # Impact Cards
    imp = rx_res["impact"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROM Impact", f"+{imp['rom_boost']:.1f}°")
    m2.metric("Strength Impact", f"+{imp['strength_boost']:.1f}%")
    m3.metric("Pain Delta", f"{imp['pain_delta']:+.1f}")
    m4.metric("Discharge Timeline", f"{imp['discharge_weeks_new']} wks", f"{imp['discharge_delta']:+.1f} wks vs standard")

    st.info(f"**Physiological Reaction Rationale:** {rx_res['reaction_summary']}")
    st.caption(rx_res.get("effect_evidence", "Assumed demo effects; not estimated from patient treatment outcomes."))

    st.markdown("---")
    st.markdown("### Comparative Trajectory Overlay (Standard vs Treatment-Stimulated)")

    base_df = pd.DataFrame(rx_res["baseline_trajectory"])
    tx_df = pd.DataFrame(rx_res["treatment_trajectory"])

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=base_df["Week"], y=base_df["Predicted_Strength_pct"], mode="lines",
        name="Standard Baseline (Strength %)", line=dict(color="#64748B", width=2, dash="dash")
    ))
    fig_comp.add_trace(go.Scatter(
        x=tx_df["Week"], y=tx_df["Predicted_Strength_pct"], mode="lines+markers",
        name=f"Stimulated: {rx_res['exercise']['name']} (Strength %)", line=dict(color="#10B981", width=3)
    ))
    fig_comp.add_trace(go.Scatter(
        x=tx_df["Week"], y=tx_df["Predicted_Pain"], mode="lines",
        name="Treatment Pain Curve (0-10)", line=dict(color="#E11D48", width=2)
    ))

    fig_comp.update_layout(
        title=f"Physiological Reaction Trajectory for {patient_id} ({rx_res['exercise']['name']})",
        xaxis_title="Week", yaxis_title="Strength (%) / Pain (0-10)",
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC"), height=420
    )
    st.plotly_chart(fig_comp, width="stretch")


# ========================================================================
# PAGE 3: DIGITAL TWIN OVERVIEW
# ========================================================================
elif page == "Digital Twin Overview":
    st.title(f"Digital Twin Overview — {patient_id}")
    pred_pct, pred_weeks = predict_for_row(patient_row)
    actual_traj = simmod.get_actual_trajectory(tables["Weekly_Progress"], patient_id)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="twin-card-real">
            <h4 style="color:#94A3B8; margin-top:0;">Real Patient Profile</h4>
            <div><b>Patient ID:</b> {patient_id}</div>
            <div><b>Age / Gender:</b> {patient_row['Age']} / {patient_row['Gender']}</div>
            <div><b>Injury:</b> {patient_row['Injury_Type']} ({patient_row['Severity']})</div>
            <div><b>Initial Pain / ROM:</b> {patient_row['Pain_Score']}/10 | {patient_row['Range_of_Motion_deg']}°</div>
            <div><b>Initial Strength:</b> {patient_row['Quadriceps_Strength_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="twin-card-twin">
            <h4 style="color:#10B981; margin-top:0;">Digital Twin Prediction</h4>
            <div><b>Twin ID:</b> DT_{patient_id}</div>
            <div><b>Predicted Recovery %:</b> {pred_pct:.1f}%</div>
            <div><b>Predicted Total Weeks:</b> {pred_weeks:.1f} weeks</div>
            <div><b>Model Confidence R²:</b> {state['metrics_pct']['R2']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Closest Historical Patient Matches (Training Pool Only)")
    top_sim = sim.find_similar_patients(patient_row, train, top_k=5, model=similarity_model)
    st.dataframe(
        top_sim[["Patient_ID", "Similarity_Score", "Age", "Injury_Type", "Severity", config.TARGET_RECOVERY_PCT, config.TARGET_RECOVERY_WEEKS]],
        width="stretch"
    )


# ========================================================================
# PAGE 4: PATIENT INFORMATION
# ========================================================================
elif page == "Patient Information":
    st.title(f"Patient Information — {patient_id}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Demographics & Lifestyle")
        st.write(f"**Age:** {patient_row['Age']}")
        st.write(f"**Gender:** {patient_row['Gender']}")
        st.write(f"**BMI:** {patient_row['BMI']}")
        st.write(f"**Occupation:** {patient_row['Occupation']}")
        st.write(f"**Activity Level:** {patient_row['Activity_Level']}")
        st.write(f"**Smoking Status:** {patient_row['Smoking_Status']}")
        st.write(f"**Diabetes:** {'Yes' if patient_row['Diabetes'] else 'No'}")
        st.write(f"**Hypertension:** {'Yes' if patient_row['Hypertension'] else 'No'}")

    with c2:
        st.markdown("### Injury & Surgical Background")
        st.write(f"**Injury Type:** {patient_row['Injury_Type']}")
        st.write(f"**Body Part:** {patient_row['Body_Part']}")
        st.write(f"**Severity:** {patient_row['Severity']}")
        st.write(f"**Surgery Required:** {'Yes' if patient_row['Surgery_Required'] else 'No'}")
        st.write(f"**Injury Date:** {patient_row['Injury_Date']}")

    st.markdown("---")
    st.markdown("### Initial Assessment Baseline")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Pain Score", f"{patient_row['Pain_Score']}/10")
    b2.metric("Range of Motion", f"{patient_row['Range_of_Motion_deg']}°")
    b3.metric("Quad Strength", f"{patient_row['Quadriceps_Strength_pct']}%")
    b4.metric("Walking Speed", f"{patient_row['Walking_Speed_mps']} m/s")


# ========================================================================
# PAGE 5: WEEKLY RECOVERY TIMELINE
# ========================================================================
elif page == "Weekly Recovery Timeline":
    st.title(f"Weekly Recovery Timeline — {patient_id}")

    pred_pct, pred_weeks = predict_for_row(patient_row)
    predicted_traj = simmod.build_predicted_trajectory(patient_row, pred_pct, pred_weeks)
    actual_traj = simmod.get_actual_trajectory(tables["Weekly_Progress"], patient_id)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(non_blue_line_chart(actual_traj, predicted_traj, "Pain", "Predicted_Pain", "Pain Trajectory", "Pain (0-10)"), width="stretch")
        st.plotly_chart(non_blue_line_chart(actual_traj, predicted_traj, "Quadriceps_Strength_pct", "Predicted_Strength_pct", "Strength Trajectory", "Strength (%)"), width="stretch")
    with col2:
        st.plotly_chart(non_blue_line_chart(actual_traj, predicted_traj, "ROM_deg", "Predicted_ROM_deg", "Range of Motion", "ROM (deg)"), width="stretch")
        st.plotly_chart(non_blue_line_chart(actual_traj, predicted_traj, "Walking_Speed_mps", "Predicted_Walking_Speed_mps", "Walking Speed", "Speed (m/s)"), width="stretch")


# ========================================================================
# PAGE 6: SIMULATION & RECALIBRATION
# ========================================================================
elif page == "Simulation & Recalibration":
    st.title(f"Weekly Stage Check-In & Dynamic Recalibration — {patient_id}")
    st.caption("Perform weekly check-in stage recalibrations, note recorded changes, compare Expected vs. Real outcomes, and view treatment pathway adaptations.")

    pred_pct, pred_weeks = predict_for_row(patient_row)
    predicted_traj = simmod.build_predicted_trajectory(patient_row, pred_pct, pred_weeks)
    max_w = int(predicted_traj["Week"].max())

    rec_w = st.slider("Select Weekly Stage for Check-In Recalibration", 1, max_w, min(4, max_w))

    p_match = predicted_traj[predicted_traj["Week"] == rec_w]
    p_row = p_match.iloc[0] if not p_match.empty else predicted_traj.iloc[0]

    st.markdown(f"#### Target Twin Predictions for Week {rec_w}")
    exp_cols = st.columns(5)
    exp_cols[0].metric("Expected Pain", f"{p_row['Predicted_Pain']:.1f}/10")
    exp_cols[1].metric("Expected ROM", f"{p_row['Predicted_ROM_deg']:.1f}°")
    exp_cols[2].metric("Expected Quad Strength", f"{p_row['Predicted_Strength_pct']:.1f}%")
    exp_cols[3].metric("Expected Balance", "75.0")
    exp_cols[4].metric("Expected Speed", f"{p_row['Predicted_Walking_Speed_mps']:.2f} m/s")

    st.markdown("---")
    st.markdown("### Enter Recorded Weekly Check-In Observations")
    observation_source=st.radio("Observation source",["Manual weekly entry","Wearable (simulated)"],horizontal=True,key="recal_observation_source")
    if observation_source == "Wearable (simulated)":
        from sensors.simulate_wearable_stream import generate_demo_stream
        from sensors.ingest_wearable import aggregate_weekly
        wearable=aggregate_weekly(generate_demo_stream(patient_id="SIMULATED-DEMO",days=max(7,rec_w*7),seed=config.RANDOM_STATE))
        selected=wearable.iloc[min(rec_w-1,len(wearable)-1)]
        weekly_entry={"ROM_deg":float(selected["knee_flexion_angle_deg"]),"Walking_Speed_mps":float(selected["gait_speed_mps"]),"Pain":None,"Quadriceps_Strength_pct":None,"Balance_Score":None}
        st.warning("SIMULATED DEMO ONLY. Pain and strength are unobserved and will not be substituted with predictions.")
        st.write({k:v for k,v in weekly_entry.items() if v is not None})
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            p_in = st.number_input("Recorded Pain (0-10)", 0.0, 10.0, float(p_row['Predicted_Pain']), 0.5)
            quad_in = st.number_input("Recorded Primary Strength (%)", 0.0, 100.0, float(p_row['Predicted_Strength_pct']), 1.0)
        with c2:
            rom_in = st.number_input("Recorded Primary ROM (deg)", 0.0, 360.0, float(p_row['Predicted_ROM_deg']), 1.0)
            bal_in = st.number_input("Recorded Balance Score", 0.0, 100.0, 75.0, 1.0)
        with c3:
            speed_in = st.number_input("Recorded Speed (m/s)", 0.0, 3.0, float(p_row['Predicted_Walking_Speed_mps']), 0.1)
            swelling_in = st.selectbox("Swelling Level", ["None", "Mild", "Moderate", "Severe"], index=0)
        weekly_entry={"Pain":p_in,"ROM_deg":rom_in,"Quadriceps_Strength_pct":quad_in,"Balance_Score":bal_in,"Walking_Speed_mps":speed_in,"Swelling":swelling_in}

    if st.button("Run Weekly Stage Recalibration", type="primary"):
        res = simmod.process_weekly_stage_recalibration(
            patient_row=patient_row, week=rec_w, weekly_entry=weekly_entry,
            train_pool=train, pathways_table=tables["Recovery_Pathway"],
            pct_model=pct_model, weeks_model=weeks_model, feature_fn=get_feature_frame,
            similarity_model=similarity_model, cohort_priors=twin_priors,
            residual_sigmas=residual_sigmas
        )
        st.session_state["recal_result"] = res

        # Persist to database & session history
        try:
            db.save_patient_recalibration_record(patient_id, rec_w, res)
        except Exception as exc:
            st.warning(f"Database history was not saved ({type(exc).__name__}). The result remains available in this browser session.")
        pw_label = res["recommended_pathway"]["Archetype_Label"] if res["recommended_pathway"] is not None else "Standard"
        st.session_state["recal_history"].append({
            "Week": rec_w,
            "Old_Discharge_wks": round(pred_weeks, 1),
            "Recalibrated_Discharge_wks": res["new_predicted_weeks"],
            "Delta_wks": res["discharge_delta"],
            "Pain_Expected_Real": f"{p_row['Predicted_Pain']:.1f} / {weekly_entry.get('Pain')}",
            "ROM_Expected_Real": f"{p_row['Predicted_ROM_deg']:.1f} / {weekly_entry.get('ROM_deg')}",
            "Strength_Expected_Real": f"{p_row['Predicted_Strength_pct']:.1f} / {weekly_entry.get('Quadriceps_Strength_pct')}",
            "Updated_Pathway": pw_label,
            "Notes": res["explanation"]
        })

    if st.session_state.get("recal_result"):
        res = st.session_state["recal_result"]
        st.markdown("---")
        st.markdown(f"### Week {rec_w} Dynamic Recalibration Outcome")
        r1, r2, r3 = st.columns(3)
        r1.metric("Old Discharge Timeline", f"{res['old_predicted_weeks']:.1f} wks")
        r2.metric("Recalibrated Discharge Timeline", f"{res['new_predicted_weeks']:.1f} wks", f"{res['discharge_delta']:+.1f} wks delta")
        r3.metric("Recalibrated Recovery %", f"{res['new_predicted_pct']:.1f}%", f"{res['new_predicted_pct'] - pred_pct:+.1f}% vs baseline")

        st.markdown(f"#### Expected vs Real Outcome Comparison (Week {rec_w})")
        evr_df = pd.DataFrame([
            {
                "Metric": k.replace("_", " "),
                "Expected (Twin)": v["predicted"],
                "Actual (Recorded)": v["actual"],
                "Deviation %": f"{v['pct_deviation']:+.1f}%",
                "Adverse z-score": f"{v['z_adverse']:+.2f} SD",
                "Status": "FLAGGED DEVIATION" if v["flagged"] else "On Track"
            }
            for k, v in res["expected_vs_real"].items()
        ])
        st.table(evr_df)

        st.info(f"**Recalibration Rationale:** {res['explanation']}")

        if res.get("recommended_pathway"):
            st.success(f"**Updated Prescribed Pathway:** {res['recommended_pathway'].get('Archetype_Label', 'Standard Protocol')}")

    # Display Session History Log
    st.markdown("---")
    st.markdown("### Weekly Stage-by-Stage Recalibration History Log")
    try:
        db_history = db.get_recalibration_history_records(patient_id)
    except Exception:
        db_history = []
    if db_history:
        hist_df = pd.DataFrame(db_history)
        st.dataframe(hist_df, width="stretch")
    elif st.session_state.get("recal_history"):
        st.dataframe(pd.DataFrame(st.session_state["recal_history"]), width="stretch")
    else:
        st.caption("No weekly stage check-ins recorded yet for this patient.")


# ========================================================================
# PAGE 7: RECOMMENDATIONS
# ========================================================================
elif page == "Recommendations":
    st.title("Clinical Pathway Recommendations")
    pathways = tables["Recovery_Pathway"].sort_values("Success_Rate", ascending=False)
    st.markdown("### All Mined Recovery Pathways")
    st.dataframe(pathways, width="stretch")


# ========================================================================
# PAGE 8: EVALUATION METRICS
# ========================================================================
elif page == "Evaluation Metrics":
    st.title("Model Evaluation Metrics (2025 Test Set)")

    m_pct = state["metrics_pct"]
    m_wks = state["metrics_weeks"]

    e1, e2, e3, e4 = st.columns(4)
    e1.metric(f"{config.TARGET_DISPLAY_LABEL} R²", f"{m_pct['R2']:.3f}")
    e2.metric("Recovery % RMSE", f"{m_pct['RMSE']:.2f}")
    e3.metric("Recovery Weeks R²", f"{m_wks['R2']:.3f}")
    e4.metric("Recovery Weeks RMSE", f"{m_wks['RMSE']:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(ev.actual_vs_predicted_scatter(test[config.TARGET_RECOVERY_PCT], test["Predicted_Recovery_Pct"], "Actual vs Predicted Recovery %", "Recovery %"), width="stretch")
        st.plotly_chart(ev.residual_plot(test[config.TARGET_RECOVERY_PCT], test["Predicted_Recovery_Pct"], "Residuals — Recovery %", "Recovery %"), width="stretch")
    with col2:
        st.plotly_chart(ev.actual_vs_predicted_scatter(test[config.TARGET_RECOVERY_WEEKS], test["Predicted_Recovery_Weeks"], "Actual vs Predicted Recovery Weeks", "Weeks"), width="stretch")
        st.plotly_chart(ev.error_distribution_hist(test[config.TARGET_RECOVERY_WEEKS], test["Predicted_Recovery_Weeks"], "Error Distribution — Recovery Weeks"), width="stretch")

    st.markdown("---")
    st.markdown("### Feature Importance")
    st.plotly_chart(ev.feature_importance_bar(state["feature_importance"]), width="stretch")

elif page == "Model Benchmark":
    st.title("Training-period model benchmark")
    path=Path("reports/model_benchmark/benchmark.json")
    if path.exists():
        report=json.loads(path.read_text());st.caption(f"Data origin: {report.get('data_origin')} · selected using CV: {report.get('selected_by_training_cv')}")
        st.dataframe(pd.DataFrame([{**v.get("fold_mean_sd",{}).get("R2",{}),"model":k,"holdout_R2":(report.get("chronological_holdout") or {}).get("metrics",{}).get("R2") if k==report.get("selected_by_training_cv") else None} for k,v in report.get("models",{}).items()]),width="stretch")
        st.json(report.get("chronological_holdout"))
    else: st.info("Run `make all` to generate reports/model_benchmark/benchmark.json.")
    st.caption("Synthetic data are not clinical validation. The target display name follows DT_TARGET.")

elif page == "Similarity Spec / Explain":
    st.title("Patient similarity")
    st.markdown("`d(x,y)=Σᵢwᵢdᵢ/Σᵢwᵢ`; `s=1-d`. Numeric ranges and outcome-association weights are fitted on the training partition and persisted. Retrieval requires `s ≥ τ`; cohorts below K_min are marked insufficient.")
    st.write({"threshold":similarity_model.tau,"K_min":similarity_model.k_min,"K_max":similarity_model.k_max,"fit metadata":similarity_model.metadata})
    neighbors=sim.find_similar_patients(patient_row,train,model=similarity_model)
    st.caption(neighbors.attrs.get("confidence","unknown"))
    if not neighbors.empty:
        st.dataframe(neighbors[[c for c in ["Patient_ID","Similarity_Score","Injury_Type","Severity",config.TARGET_RECOVERY_PCT] if c in neighbors]],width="stretch")
        candidate=neighbors.iloc[0];st.dataframe(pd.DataFrame(similarity_model.explain(patient_row,candidate)),width="stretch")

elif page == "Deviation Analysis":
    st.title("Deviation threshold analysis")
    path=Path("reports/deviation/deviation.json")
    if path.exists():
        report=json.loads(path.read_text());st.caption(report.get("interpretation",""));st.json(report)
    else: st.info("Run `make all` to generate the deviation threshold sweep.")
    st.caption("MCID placeholders are intentionally unset pending instrument-specific source review. Current sigma floors are engineering values.")

elif page == "Pathway Ranking":
    st.title("Pathway ranking evidence")
    path=Path("reports/pathway_ranking/evaluation.json")
    if path.exists(): st.json(json.loads(path.read_text()))
    mined=pm.mine_pathways_from_tables(tables)
    if mined: st.dataframe(pd.DataFrame(mined),width="stretch")
    st.warning("Bundled data do not link patient-level treatment exposure to outcomes. Cohort templates are descriptive; displayed intervention effects are assumed demo values, not treatment benefits.")

elif page == "Adaptive Loop Evaluation":
    st.title("Adaptive-loop replay")
    path=Path("reports/adaptive_loop/evaluation.json")
    if path.exists(): st.json(json.loads(path.read_text()))
    else: st.info("Run `make all` to generate the week-by-week prediction replay.")
    st.caption("Forecast error comparisons do not show that recalibration improves patient outcomes.")

elif page == "Explain (XAI)":
    st.title("Explain this prediction")
    try:
        explanation=__import__("explainability").explain_prediction(pct_model,background=get_feature_frame(train),query=get_feature_frame(pd.DataFrame([patient_row])))
        st.metric("Prediction",f"{explanation['prediction']:.2f}",f"SHAP additive residual {explanation['additivity_error']:+.3g}")
        st.dataframe(pd.DataFrame(explanation["features"]),width="stretch");st.caption(explanation["caveat"])
    except Exception as exc: st.info(f"SHAP explanation unavailable: {exc}. Install the optional `shap` dependency.")

elif page == "Wearable Demo":
    st.title("Wearable data demo")
    st.radio("Observation source",["Manual weekly entry","Wearable (simulated)"],horizontal=True,key="wearable_source")
    st.warning("SIMULATED DEMO ONLY. These values are generated for interface testing, not patient records or model validation.")
    from sensors.simulate_wearable_stream import generate_demo_stream
    from sensors.ingest_wearable import aggregate_weekly
    stream=generate_demo_stream(patient_id="SIMULATED-DEMO",days=56,seed=config.RANDOM_STATE)
    st.dataframe(aggregate_weekly(stream),width="stretch")

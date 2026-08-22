"""
pdf_generator.py
----------------
Generates a PDF report for a Patient's Digital Twin using ReportLab.
Includes patient demographics, weekly twin state, flagged pressure points,
expected vs actual metrics, treatment clone options, and recalibration history.
"""

import os
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_twin_pdf(patient_id: str, patient_row: pd.Series, week_data: dict,
                      pressure_points: list, expected_vs_actual: dict,
                      treatment_clones: list = None, recal_history: list = None) -> bytes:
    """
    Generates a PDF report and returns the PDF bytes. Also saves a copy to ./output/
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom non-blue palette
    PRIMARY_COLOR = colors.HexColor("#10B981")     # Emerald Green
    SECONDARY_COLOR = colors.HexColor("#0F172A")   # Dark Slate
    ACCENT_COLOR = colors.HexColor("#D97706")      # Warm Amber
    ALERT_COLOR = colors.HexColor("#E11D48")       # Crimson Red
    BG_LIGHT = colors.HexColor("#F8FAFC")          # Light Slate BG
    TEXT_DARK = colors.HexColor("#1E293B")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=SECONDARY_COLOR,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12
    )
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PRIMARY_COLOR,
        spaceBefore=12,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK
    )
    bold_body_style = ParagraphStyle(
        'BoldBodyText',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # Title Banner
    story.append(Paragraph(f"Dynamic Digital Twin Clinical Report", title_style))
    story.append(Paragraph(f"Patient ID: <b>{patient_id}</b> | Generated for Rehabilitation Tracking", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY_COLOR, spaceAfter=12))

    # Patient Demographics & Baseline Table
    story.append(Paragraph("1. Patient Profile & Clinical Baseline", section_style))
    demographics_data = [
        [
            Paragraph("<b>Age:</b>", body_style), Paragraph(str(patient_row.get("Age", "N/A")), body_style),
            Paragraph("<b>Gender:</b>", body_style), Paragraph(str(patient_row.get("Gender", "N/A")), body_style),
        ],
        [
            Paragraph("<b>Injury Type:</b>", body_style), Paragraph(str(patient_row.get("Injury_Type", "N/A")), body_style),
            Paragraph("<b>Severity:</b>", body_style), Paragraph(str(patient_row.get("Severity", "N/A")), body_style),
        ],
        [
            Paragraph("<b>Surgery Required:</b>", body_style), Paragraph("Yes" if patient_row.get("Surgery_Required") else "No", body_style),
            Paragraph("<b>BMI:</b>", body_style), Paragraph(str(patient_row.get("BMI", "N/A")), body_style),
        ],
        [
            Paragraph("<b>Activity Level:</b>", body_style), Paragraph(str(patient_row.get("Activity_Level", "N/A")), body_style),
            Paragraph("<b>Smoking Status:</b>", body_style), Paragraph(str(patient_row.get("Smoking_Status", "N/A")), body_style),
        ],
    ]
    demo_table = Table(demographics_data, colWidths=[110, 150, 110, 150])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # Selected Week Twin State
    week_num = week_data.get("Week", "N/A")
    story.append(Paragraph(f"2. Twin Recovery State — Week {week_num}", section_style))
    state_table_data = [
        [Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Recorded Value</b>", bold_body_style), Paragraph("<b>Status / Target</b>", bold_body_style)],
        [Paragraph("Pain Score", body_style), Paragraph(f"{week_data.get('Pain', 'N/A')} / 10", body_style), Paragraph("Target <= 2.0", body_style)],
        [Paragraph("Range of Motion (ROM)", body_style), Paragraph(f"{week_data.get('ROM_deg', 'N/A')}°", body_style), Paragraph("Target >= 120°", body_style)],
        [Paragraph("Quadriceps Strength", body_style), Paragraph(f"{week_data.get('Quadriceps_Strength_pct', 'N/A')}%", body_style), Paragraph("Target >= 80%", body_style)],
        [Paragraph("Hamstring Strength", body_style), Paragraph(f"{week_data.get('Hamstring_Strength_pct', 'N/A')}%", body_style), Paragraph("Target >= 80%", body_style)],
        [Paragraph("Balance Score", body_style), Paragraph(f"{week_data.get('Balance_Score', 'N/A')}", body_style), Paragraph("Target >= 75", body_style)],
        [Paragraph("Walking Speed", body_style), Paragraph(f"{week_data.get('Walking_Speed_mps', 'N/A')} m/s", body_style), Paragraph("Target >= 1.2 m/s", body_style)],
        [Paragraph("Swelling", body_style), Paragraph(str(week_data.get("Swelling", "N/A")), body_style), Paragraph("Target: None", body_style)],
        [Paragraph("Compliance", body_style), Paragraph(f"{week_data.get('Compliance_pct', 'N/A')}%", body_style), Paragraph("Target >= 80%", body_style)],
    ]
    state_table = Table(state_table_data, colWidths=[180, 160, 180])
    state_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(state_table)
    story.append(Spacer(1, 10))

    # Pressure Points & Concerns
    story.append(Paragraph("3. Pressure Points & Flagged Recovery Concerns", section_style))
    if pressure_points:
        pp_rows = [[Paragraph("<b>Concern Category</b>", bold_body_style), Paragraph("<b>Actual Value</b>", bold_body_style), Paragraph("<b>Clinical Flag</b>", bold_body_style)]]
        for pp in pressure_points:
            pp_rows.append([
                Paragraph(pp.get("Category", "N/A"), body_style),
                Paragraph(str(pp.get("Actual", "N/A")), body_style),
                Paragraph(f"<font color='#E11D48'><b>{pp.get('Flag', 'N/A')}</b></font>", body_style)
            ])
        pp_table = Table(pp_rows, colWidths=[160, 160, 200])
        pp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FEF2F2")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FCA5A5")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(pp_table)
    else:
        story.append(Paragraph("<font color='#10B981'><b>✓ No critical recovery flags detected for Week {}</b></font>".format(week_num), body_style))
    
    story.append(Spacer(1, 10))

    # Expected vs Actual Comparison
    if expected_vs_actual:
        story.append(Paragraph(f"4. Twin Expected vs. Actual Comparison (Week {week_num})", section_style))
        eva_rows = [[Paragraph("<b>Metric</b>", bold_body_style), Paragraph("<b>Expected (Twin)</b>", bold_body_style), Paragraph("<b>Actual (Recorded)</b>", bold_body_style), Paragraph("<b>Deviation</b>", bold_body_style)]]
        for metric, d in expected_vs_actual.items():
            dev_val = d.get("pct_deviation", 0.0)
            dev_str = f"{dev_val:+.1f}%"
            dev_color = "#E11D48" if d.get("flagged") else "#10B981"
            eva_rows.append([
                Paragraph(metric, body_style),
                Paragraph(f"{d.get('predicted', 0.0):.1f}", body_style),
                Paragraph(f"{d.get('actual', 0.0):.1f}", body_style),
                Paragraph(f"<font color='{dev_color}'><b>{dev_str}</b></font>", body_style)
            ])
        eva_table = Table(eva_rows, colWidths=[140, 120, 120, 140])
        eva_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(eva_table)
        story.append(Spacer(1, 10))

    # Treatment Clones Comparison
    if treatment_clones:
        story.append(Paragraph("5. Treatment Pathway Clones & Reaction Analysis", section_style))
        tc_rows = [[Paragraph("<b>Pathway Option</b>", bold_body_style), Paragraph("<b>Est. Duration</b>", bold_body_style), Paragraph("<b>Predicted Recovery</b>", bold_body_style), Paragraph("<b>Reaction Profile</b>", bold_body_style)]]
        for tc in treatment_clones:
            tc_rows.append([
                Paragraph(f"<b>{tc.get('name', 'N/A')}</b>", body_style),
                Paragraph(f"{tc.get('weeks', 'N/A')} weeks", body_style),
                Paragraph(f"{tc.get('recovery_pct', 'N/A'):.0f}%", body_style),
                Paragraph(tc.get('reaction', 'N/A'), body_style)
            ])
        tc_table = Table(tc_rows, colWidths=[150, 90, 110, 170])
        tc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0FDF4")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#86EFAC")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(tc_table)
        story.append(Spacer(1, 10))

    # Recalibration History
    if recal_history:
        story.append(Paragraph("6. Recalibration History", section_style))
        rh_rows = [[Paragraph("<b>Week</b>", bold_body_style), Paragraph("<b>Original Pred</b>", bold_body_style), Paragraph("<b>Recalibrated Pred</b>", bold_body_style), Paragraph("<b>Updated Pathway</b>", bold_body_style)]]
        for item in recal_history:
            rh_rows.append([
                Paragraph(f"Week {item.get('week')}", body_style),
                Paragraph(f"{item.get('old_pct', 0):.0f}% ({item.get('old_wks', 0):.0f}w)", body_style),
                Paragraph(f"{item.get('new_pct', 0):.0f}% ({item.get('new_wks', 0):.0f}w)", body_style),
                Paragraph(item.get('pathway', 'N/A'), body_style)
            ])
        rh_table = Table(rh_rows, colWidths=[80, 130, 130, 180])
        rh_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FFFBEB")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FDE68A")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(rh_table)

    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    story.append(Paragraph("<i>Confidential Clinical Digital Twin Simulation Report — Orthopaedic Rehabilitation Department</i>", subtitle_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()

    # Save to disk in output directory as well
    os.makedirs("output", exist_ok=True)
    file_path = os.path.join("output", f"Digital_Twin_Report_{patient_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    return pdf_bytes

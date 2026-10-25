"""
pdf_generator.py
----------------
ReportLab PDF Generator for Clinical Digital Twin Progress Reports.
Generates printable, zero-emoji PDF reports containing patient demographics,
treating doctor details, weekly progression, Normalized Pain Index, pressure points,
and active therapy plans.
"""

import io
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import simulation as simmod


def _safe(value):
    """Escape patient-controlled text before passing it to ReportLab Paragraph."""
    return escape(str(value))


def generate_twin_pdf(patient_id: str, patient_row, week_data: dict, pressure_points: list,
                      expected_vs_actual: dict, treatment_clones: list, recal_history: list, doctor_row=None) -> bytes:
    """Generates a downloadable PDF clinical report for a patient digital twin."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "DocHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    text_style = ParagraphStyle(
        "DocText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    bold_text_style = ParagraphStyle(
        "DocTextBold",
        parent=text_style,
        fontName="Helvetica-Bold",
    )

    # Document Header
    story.append(Paragraph("Dynamic Digital Twin — Clinical Progress Report", title_style))
    story.append(Paragraph("Personalized Human Orthopaedic Rehabilitation & Predictive Pathway Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    # Patient & Doctor Info Table
    doc_name = doctor_row.get("Doctor_Name", "Dr. Unassigned") if doctor_row is not None else "Attending Specialist"
    doc_spec = doctor_row.get("Specialization", "Orthopaedic Rehabilitation") if doctor_row is not None else "Orthopaedics"

    info_data = [
        [
            Paragraph("<b>Patient ID:</b>", text_style), Paragraph(_safe(patient_id), text_style),
            Paragraph("<b>Treating Doctor:</b>", text_style), Paragraph(_safe(doc_name), text_style)
        ],
        [
            Paragraph("<b>Age / Gender:</b>", text_style), Paragraph(_safe(f"{patient_row.get('Age', '-')} yrs / {patient_row.get('Gender', '-')}") , text_style),
            Paragraph("<b>Specialization:</b>", text_style), Paragraph(_safe(doc_spec), text_style)
        ],
        [
            Paragraph("<b>Injury Type:</b>", text_style), Paragraph(_safe(patient_row.get("Injury_Type", "-")), text_style),
            Paragraph("<b>Body Region / Part:</b>", text_style), Paragraph(_safe(f"{patient_row.get('Body_Region', '-')} / {patient_row.get('Body_Part', '-')}") , text_style)
        ],
        [
            Paragraph("<b>Severity:</b>", text_style), Paragraph(_safe(patient_row.get("Severity", "-")), text_style),
            Paragraph("<b>Report Stage:</b>", text_style), Paragraph(_safe(f"Week {week_data.get('Week', 1)} Review"), bold_text_style)
        ],
    ]
    t_info = Table(info_data, colWidths=[110, 160, 110, 160])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 12))

    # Normalized Pain Index (NPI) & Weekly Observations
    raw_pain = float(week_data.get("Pain", patient_row.get("Pain_Score", 5.0)))
    sev_str = str(patient_row.get("Severity", "Moderate"))
    swell_str = str(week_data.get("Swelling", "Mild"))
    comorb_val = float(patient_row.get("Comorbidity_Score", 0.0))
    npi = simmod.calculate_normalized_pain_index(raw_pain, sev_str, swell_str, comorb_val)

    story.append(Paragraph("Weekly Physiological Metrics & Normalized Pain Index (NPI)", heading_style))

    metrics_data = [
        [
            Paragraph("<b>Metric</b>", bold_text_style),
            Paragraph("<b>Recorded Value</b>", bold_text_style),
            Paragraph("<b>Clinical Target / Threshold</b>", bold_text_style)
        ],
        [Paragraph("Pain Score (0-10)", text_style), Paragraph(f"{raw_pain:.1f} / 10", text_style), Paragraph("Target <= 2.0", text_style)],
        [Paragraph("Normalized Pain Index (NPI)", bold_text_style), Paragraph(f"<b>{npi['npi_score']:.3f}</b> ({npi['category']})", bold_text_style), Paragraph("Normalized Scale [0.0 - 1.0]", text_style)],
        [Paragraph("Primary ROM (Degrees)", text_style), Paragraph(f"{week_data.get('ROM_deg', 0):.1f}°", text_style), Paragraph("Target >= 120°", text_style)],
        [Paragraph("Primary Strength (%)", text_style), Paragraph(f"{week_data.get('Quadriceps_Strength_pct', 0):.1f}%", text_style), Paragraph("Target >= 85%", text_style)],
        [Paragraph("Function / Balance Score", text_style), Paragraph(f"{week_data.get('Balance_Score', 0):.1f}", text_style), Paragraph("Target >= 75", text_style)],
        [Paragraph("Walking Speed (m/s)", text_style), Paragraph(f"{week_data.get('Walking_Speed_mps', 0):.2f} m/s", text_style), Paragraph("Target >= 1.2 m/s", text_style)],
        [Paragraph("Swelling / Compliance", text_style), Paragraph(_safe(f"{week_data.get('Swelling', '-')} / {week_data.get('Compliance_pct', '-')}%"), text_style), Paragraph("Target <= Mild / >= 80%", text_style)],
    ]
    t_metrics = Table(metrics_data, colWidths=[180, 180, 180])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 12))

    # Pressure Points & Flags
    if pressure_points:
        story.append(Paragraph("Flagged Recovery Pressure Points", heading_style))
        pp_data = [[Paragraph("<b>Category</b>", bold_text_style), Paragraph("<b>Observation</b>", bold_text_style), Paragraph("<b>Clinical Risk Flag</b>", bold_text_style)]]
        for pp in pressure_points:
            pp_data.append([Paragraph(_safe(pp.get("Category", "-")), text_style), Paragraph(_safe(pp.get("Actual", "-")), text_style), Paragraph(_safe(pp.get("Flag", "-")), text_style)])
        t_pp = Table(pp_data, colWidths=[140, 140, 260])
        t_pp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FEF2F2')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#FCA5A5')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FCA5A5')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_pp)
        story.append(Spacer(1, 12))

    # Treatment Clones Comparison Table
    if treatment_clones:
        story.append(Paragraph("Simulated Treatment Clones & Physiological Reactions", heading_style))
        cl_data = [[
            Paragraph("<b>Treatment Protocol</b>", bold_text_style),
            Paragraph("<b>Est. Duration</b>", bold_text_style),
            Paragraph("<b>Final ROM</b>", bold_text_style),
            Paragraph("<b>Final Strength</b>", bold_text_style),
            Paragraph("<b>Recovery %</b>", bold_text_style)
        ]]
        for cl in treatment_clones:
            cl_data.append([
                Paragraph(_safe(cl.get("name", "-")), text_style),
                Paragraph(f"{cl.get('weeks', 0):.1f} wks", text_style),
                Paragraph(f"{cl.get('final_rom', 0):.1f}°", text_style),
                Paragraph(f"{cl.get('final_strength', 0):.1f}%", text_style),
                Paragraph(f"<b>{cl.get('recovery_pct', 0):.1f}%</b>", bold_text_style),
            ])
        t_cl = Table(cl_data, colWidths=[170, 90, 90, 90, 100])
        t_cl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_cl)

    doc.build(story)
    return buffer.getvalue()
"""
pdf_generator.py
----------------
ReportLab PDF Generator for Clinical Digital Twin Progress Reports.
Generates printable, zero-emoji PDF reports containing patient demographics,
treating doctor details, weekly progression, Normalized Pain Index, pressure points,
and active therapy plans.
"""

import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import simulation as simmod


def generate_twin_pdf(patient_id: str, patient_row, week_data: dict, pressure_points: list,
                      expected_vs_actual: dict, treatment_clones: list, recal_history: list, doctor_row=None) -> bytes:
    """Generates a downloadable PDF clinical report for a patient digital twin."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "DocHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    text_style = ParagraphStyle(
        "DocText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    bold_text_style = ParagraphStyle(
        "DocTextBold",
        parent=text_style,
        fontName="Helvetica-Bold",
    )

    # Document Header
    story.append(Paragraph("Dynamic Digital Twin — Clinical Progress Report", title_style))
    story.append(Paragraph("Personalized Human Orthopaedic Rehabilitation & Predictive Pathway Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    # Patient & Doctor Info Table
    doc_name = doctor_row.get("Doctor_Name", "Dr. Unassigned") if doctor_row is not None else "Attending Specialist"
    doc_spec = doctor_row.get("Specialization", "Orthopaedic Rehabilitation") if doctor_row is not None else "Orthopaedics"

    info_data = [
        [
            Paragraph("<b>Patient ID:</b>", text_style), Paragraph(_safe(patient_id), text_style),
            Paragraph("<b>Treating Doctor:</b>", text_style), Paragraph(_safe(doc_name), text_style)
        ],
        [
            Paragraph("<b>Age / Gender:</b>", text_style), Paragraph(_safe(f"{patient_row.get('Age', '-')} yrs / {patient_row.get('Gender', '-')}") , text_style),
            Paragraph("<b>Specialization:</b>", text_style), Paragraph(_safe(doc_spec), text_style)
        ],
        [
            Paragraph("<b>Injury Type:</b>", text_style), Paragraph(_safe(patient_row.get("Injury_Type", "-")), text_style),
            Paragraph("<b>Body Region / Part:</b>", text_style), Paragraph(_safe(f"{patient_row.get('Body_Region', '-')} / {patient_row.get('Body_Part', '-')}") , text_style)
        ],
        [
            Paragraph("<b>Severity:</b>", text_style), Paragraph(_safe(patient_row.get("Severity", "-")), text_style),
            Paragraph("<b>Report Stage:</b>", text_style), Paragraph(_safe(f"Week {week_data.get('Week', 1)} Review"), bold_text_style)
        ],
    ]
    t_info = Table(info_data, colWidths=[110, 160, 110, 160])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 12))

    # Normalized Pain Index (NPI) & Weekly Observations
    raw_pain = float(week_data.get("Pain", patient_row.get("Pain_Score", 5.0)))
    sev_str = str(patient_row.get("Severity", "Moderate"))
    swell_str = str(week_data.get("Swelling", "Mild"))
    comorb_val = float(patient_row.get("Comorbidity_Score", 0.0))
    npi = simmod.calculate_normalized_pain_index(raw_pain, sev_str, swell_str, comorb_val)

    story.append(Paragraph("Weekly Physiological Metrics & Normalized Pain Index (NPI)", heading_style))

    metrics_data = [
        [
            Paragraph("<b>Metric</b>", bold_text_style),
            Paragraph("<b>Recorded Value</b>", bold_text_style),
            Paragraph("<b>Clinical Target / Threshold</b>", bold_text_style)
        ],
        [Paragraph("Pain Score (0-10)", text_style), Paragraph(f"{raw_pain:.1f} / 10", text_style), Paragraph("Target <= 2.0", text_style)],
        [Paragraph("Normalized Pain Index (NPI)", bold_text_style), Paragraph(f"<b>{npi['npi_score']:.3f}</b> ({npi['category']})", bold_text_style), Paragraph("Normalized Scale [0.0 - 1.0]", text_style)],
        [Paragraph("Primary ROM (Degrees)", text_style), Paragraph(f"{week_data.get('ROM_deg', 0):.1f}°", text_style), Paragraph("Target >= 120°", text_style)],
        [Paragraph("Primary Strength (%)", text_style), Paragraph(f"{week_data.get('Quadriceps_Strength_pct', 0):.1f}%", text_style), Paragraph("Target >= 85%", text_style)],
        [Paragraph("Function / Balance Score", text_style), Paragraph(f"{week_data.get('Balance_Score', 0):.1f}", text_style), Paragraph("Target >= 75", text_style)],
        [Paragraph("Walking Speed (m/s)", text_style), Paragraph(f"{week_data.get('Walking_Speed_mps', 0):.2f} m/s", text_style), Paragraph("Target >= 1.2 m/s", text_style)],
        [Paragraph("Swelling / Compliance", text_style), Paragraph(_safe(f"{week_data.get('Swelling', '-')} / {week_data.get('Compliance_pct', '-')}%"), text_style), Paragraph("Target <= Mild / >= 80%", text_style)],
    ]
    t_metrics = Table(metrics_data, colWidths=[180, 180, 180])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 12))

    # Pressure Points & Flags
    if pressure_points:
        story.append(Paragraph("Flagged Recovery Pressure Points", heading_style))
        pp_data = [[Paragraph("<b>Category</b>", bold_text_style), Paragraph("<b>Observation</b>", bold_text_style), Paragraph("<b>Clinical Risk Flag</b>", bold_text_style)]]
        for pp in pressure_points:
            pp_data.append([Paragraph(_safe(pp.get("Category", "-")), text_style), Paragraph(_safe(pp.get("Actual", "-")), text_style), Paragraph(_safe(pp.get("Flag", "-")), text_style)])
        t_pp = Table(pp_data, colWidths=[140, 140, 260])
        t_pp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FEF2F2')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#FCA5A5')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FCA5A5')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_pp)
        story.append(Spacer(1, 12))

    # Treatment Clones Comparison Table
    if treatment_clones:
        story.append(Paragraph("Simulated Treatment Clones & Physiological Reactions", heading_style))
        cl_data = [[
            Paragraph("<b>Treatment Protocol</b>", bold_text_style),
            Paragraph("<b>Est. Duration</b>", bold_text_style),
            Paragraph("<b>Final ROM</b>", bold_text_style),
            Paragraph("<b>Final Strength</b>", bold_text_style),
            Paragraph("<b>Recovery %</b>", bold_text_style)
        ]]
        for cl in treatment_clones:
            cl_data.append([
                Paragraph(_safe(cl.get("name", "-")), text_style),
                Paragraph(f"{cl.get('weeks', 0):.1f} wks", text_style),
                Paragraph(f"{cl.get('final_rom', 0):.1f}°", text_style),
                Paragraph(f"{cl.get('final_strength', 0):.1f}%", text_style),
                Paragraph(f"<b>{cl.get('recovery_pct', 0):.1f}%</b>", bold_text_style),
            ])
        t_cl = Table(cl_data, colWidths=[170, 90, 90, 90, 100])
        t_cl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_cl)

    doc.build(story)
    return buffer.getvalue()

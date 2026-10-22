"""
SVG Quality Badge and PDF Report generator for CodeGuard.
"""
import io
import os
import json
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SVG Badge
# ---------------------------------------------------------------------------

def _badge_color(score: int) -> str:
    if score >= 80:
        return "#44cc11"   # green
    elif score >= 60:
        return "#f0a030"   # orange
    elif score >= 40:
        return "#e05d44"   # red
    return "#9f9f9f"       # grey


def generate_badge_svg(score: int, label: str = "Code Health") -> str:
    """
    Generate a Shields.io-style SVG badge.

    Returns the SVG string; write to a .svg file or serve as image/svg+xml.
    """
    color = _badge_color(score)
    value_text = f"{score}/100"
    label_width = 90
    value_width = 65
    total_width = label_width + value_width

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110">
    <text x="{label_width // 2 * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(label_width - 10) * 10}" lengthAdjust="spacing">{label}</text>
    <text x="{label_width // 2 * 10}" y="140" transform="scale(.1)" textLength="{(label_width - 10) * 10}" lengthAdjust="spacing">{label}</text>
    <text x="{(label_width + value_width // 2) * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(value_width - 10) * 10}" lengthAdjust="spacing">{value_text}</text>
    <text x="{(label_width + value_width // 2) * 10}" y="140" transform="scale(.1)" textLength="{(value_width - 10) * 10}" lengthAdjust="spacing">{value_text}</text>
  </g>
</svg>"""
    return svg


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------

def generate_pdf_report(
    project_name: str,
    health_score: int,
    predictions: list[dict],
    model_metrics: dict,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF quality report using ReportLab.

    Returns raw PDF bytes. Optionally writes to output_path.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.units import cm
    except ImportError:
        log.error("reportlab not installed — PDF generation unavailable.")
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#1a1a2e"),
                                  spaceAfter=6)
    story.append(Paragraph(f"CodeGuard Quality Report — {project_name}", title_style))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                           styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 0.4*cm))

    # Summary stats
    health_color = "#44cc11" if health_score >= 80 else "#f0a030" if health_score >= 60 else "#e05d44"
    summary_data = [
        ["Metric", "Value"],
        ["Project Health Score", f"{health_score}/100"],
        ["Model F1-Score", f"{model_metrics.get('f1_score', 0):.1%}"],
        ["Transfer Learning F1 Improvement",
         f"+{model_metrics.get('f1_improvement', 0):.1%}"],
        ["Statistical Test p-value", f"{model_metrics.get('p_value', 1.0):.4f}"],
        ["Total Files Analysed", str(len(predictions))],
        ["High-Risk Files", str(sum(1 for p in predictions if p.get("risk_tier") == "High"))],
        ["Medium-Risk Files", str(sum(1 for p in predictions if p.get("risk_tier") == "Medium"))],
        ["Low-Risk Files", str(sum(1 for p in predictions if p.get("risk_tier") == "Low"))],
    ]
    t = Table(summary_data, colWidths=[10*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6*cm))

    # Per-file table (top 30)
    story.append(Paragraph("File Risk Analysis (Top 30 by Risk Score)", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))

    tier_colors = {"High": "#ffcccc", "Medium": "#fff3cc", "Low": "#ccffcc"}
    file_rows = [["File", "LOC", "CC", "Coupling", "Risk", "Debt (h)", "Explanation"]]

    top_preds = sorted(predictions, key=lambda p: p.get("defect_probability", 0), reverse=True)[:30]
    for p in top_preds:
        metrics = p.get("metrics", {})
        file_rows.append([
            os.path.basename(p.get("file_path", "—")),
            str(metrics.get("loc", "—")),
            str(metrics.get("cyclomatic_complexity", "—")),
            str(metrics.get("coupling", "—")),
            p.get("risk_tier", "—"),
            str(p.get("tech_debt_hours", "—")),
            (p.get("nlp_explanation", "") or "")[:80] + "…",
        ])

    ft = Table(file_rows, colWidths=[4*cm, 1.2*cm, 1*cm, 1.5*cm, 1.5*cm, 1.5*cm, 5.8*cm])
    file_table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3436")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 7),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#dee2e6")),
        ("PADDING",    (0, 0), (-1, -1), 4),
        ("WORDWRAP",   (6, 1), (6, -1), True),
    ]
    for i, row in enumerate(file_rows[1:], start=1):
        tier = row[4]
        bg = tier_colors.get(tier, "#ffffff")
        file_table_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(bg)))
    ft.setStyle(TableStyle(file_table_style))
    story.append(ft)

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(
        "<i>CodeGuard informs; a human decides. Override decisions are logged with justification.</i>",
        styles["Italic"]
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes

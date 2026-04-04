"""
report_generator.py
-------------------
Generate a PDF alert report for a Streamlit dashboard session.

Dependency: reportlab
    pip install reportlab
"""
import os
import datetime
from typing import List, Dict, Any


def generate_pdf_report(
    alerts: List[Dict[str, Any]],
    output_path: str = "reports/xai_ids_report.pdf",
    model_name: str = "Random Forest",
) -> str:
    """
    Generate a PDF report summarising detected alerts from a session.

    Parameters
    ----------
    alerts      : list of dicts, each with keys:
                    'flow_index', 'prediction', 'confidence', 'top_features'
    output_path : where to write the PDF file
    model_name  : name of the model used (shown in header)

    Returns the output_path on success.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF export. "
            "Install with: pip install reportlab"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Header ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "title", parent=styles["Heading1"],
        fontSize=16, spaceAfter=4
    )
    story.append(Paragraph("XAI-Based NIDS — Alert Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Model: {model_name}  |  Total Alerts: {len(alerts)}",
        styles["Normal"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, spaceAfter=12))

    if not alerts:
        story.append(Paragraph("No alerts were detected in this session.",
                                styles["Normal"]))
    else:
        # ── Alert table ─────────────────────────────────────────────────
        table_data = [
            ["#", "Flow", "Prediction", "Confidence", "Top Feature"]
        ]
        for i, a in enumerate(alerts, 1):
            top_feat = a.get("top_features", [{}])[0]
            feat_str = (
                f"{top_feat.get('feature', 'N/A')} "
                f"(+{top_feat.get('shap_value', 0):.3f})"
                if top_feat else "N/A"
            )
            table_data.append([
                str(i),
                str(a.get("flow_index", i)),
                a.get("prediction", "UNKNOWN"),
                f"{a.get('confidence', 0)*100:.1f}%",
                feat_str,
            ])

        col_widths = [1*cm, 2*cm, 4*cm, 3*cm, 7*cm]
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#01696f")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f9f8f5"), colors.HexColor("#f3f0ec")]),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#d4d1ca")),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)

    doc.build(story)
    print(f"  [report] PDF saved to {output_path}")
    return output_path

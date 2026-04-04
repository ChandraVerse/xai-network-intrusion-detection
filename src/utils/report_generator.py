"""PDF alert report generator using ReportLab.

Builds a formatted PDF from a list of alert dictionaries produced by
the Streamlit dashboard session.

Usage:
    from src.utils.report_generator import generate_report

    alerts = [
        {
            "flow_id": 1,
            "prediction": "DDoS",
            "confidence": 0.973,
            "top_features": [
                {"name": "Flow Duration", "shap": 0.43},
                {"name": "Fwd Packet Length Max", "shap": 0.29},
            ],
            "timestamp": "2026-04-04T15:00:00",
        },
    ]
    generate_report(alerts, output_path="report.pdf")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("ReportLab is required: pip install reportlab") from exc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Severity colour map matching MITRE ATT&CK severity levels
_SEVERITY_COLOURS: dict[str, tuple[float, float, float]] = {
    "DDoS": (0.85, 0.15, 0.15),
    "DoS Hulk": (0.85, 0.15, 0.15),
    "DoS GoldenEye": (0.85, 0.15, 0.15),
    "DoS Slowloris": (0.85, 0.35, 0.10),
    "DoS Slowhttptest": (0.85, 0.35, 0.10),
    "FTP-Patator": (0.90, 0.55, 0.10),
    "SSH-Patator": (0.90, 0.55, 0.10),
    "PortScan": (0.95, 0.75, 0.10),
    "Web Attack -- Brute Force": (0.85, 0.20, 0.10),
    "Web Attack -- XSS": (0.85, 0.20, 0.10),
    "Web Attack -- Sql Injection": (0.85, 0.15, 0.15),
    "Infiltration": (0.70, 0.10, 0.10),
    "Bot": (0.55, 0.10, 0.55),
    "BENIGN": (0.20, 0.65, 0.20),
}


def _severity_color(label: str) -> colors.Color:
    rgb = _SEVERITY_COLOURS.get(label, (0.5, 0.5, 0.5))
    return colors.Color(*rgb)


def generate_report(
    alerts: list[dict[str, Any]],
    output_path: str | Path = "alert_report.pdf",
    title: str = "XAI-NIDS Alert Report",
) -> Path:
    """Generate a PDF report from a list of alert dicts.

    Args:
        alerts:       List of alert dictionaries. Each must contain:
                        'flow_id', 'prediction', 'confidence', 'top_features',
                        'timestamp'.
        output_path:  Path where the PDF will be saved.
        title:        Report title string.

    Returns:
        Path to the generated PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=16, spaceAfter=6, textColor=colors.HexColor("#1a202c"),
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#718096"),
    )
    heading_style = ParagraphStyle(
        "AlertHeading", parent=styles["Heading2"],
        fontSize=12, spaceBefore=12, spaceAfter=4,
    )
    body_style = styles["Normal"]
    body_style.fontSize = 9

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []

    # ---- Cover header ----
    story.append(Paragraph(title, title_style))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"Generated: {generated_at}  |  Total alerts: {len(alerts)}", sub_style
    ))
    story.append(Spacer(1, 0.5 * cm))

    # ---- Summary table ----
    summary_data = [["Flow ID", "Prediction", "Confidence", "Timestamp"]]
    for a in alerts:
        summary_data.append([
            str(a.get("flow_id", "\u2014")),
            str(a.get("prediction", "\u2014")),
            f"{a.get('confidence', 0) * 100:.1f}%",
            str(a.get("timestamp", "\u2014")),
        ])

    summary_table = Table(summary_data, colWidths=[2.5 * cm, 5 * cm, 3.5 * cm, 5.5 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f8fa"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5 * cm))

    # ---- Per-alert detail ----
    for alert in alerts:
        fid = alert.get("flow_id", "?")
        pred = alert.get("prediction", "Unknown")
        conf = alert.get("confidence", 0.0)
        feats = alert.get("top_features", [])
        ts = alert.get("timestamp", "")

        alert_color = _severity_color(pred)
        heading_para = Paragraph(
            f"Alert #{fid} \u2014 "
            f"<font color='{alert_color.hexval() if hasattr(alert_color, 'hexval') else 'red'}'>"
            f"{pred}</font>  ({conf * 100:.1f}% confidence)"
            f"  <font size=8 color='grey'>{ts}</font>",
            heading_style,
        )
        story.append(heading_para)

        if feats:
            feat_data = [["Feature", "SHAP Contribution", "Direction"]]
            for ft in feats:
                name = ft.get("name", "\u2014")
                sv = ft.get("shap", 0.0)
                direction = "\u2192 attack" if sv > 0 else "\u2190 benign"
                feat_data.append([name, f"{sv:+.4f}", direction])

            feat_table = Table(feat_data, colWidths=[8 * cm, 4 * cm, 4.5 * cm])
            feat_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7f8fa"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(feat_table)
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    log.info("PDF report saved \u2192 %s  (%d alerts)", output_path, len(alerts))
    return output_path

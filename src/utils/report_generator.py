"""Report generator for XAI-NIDS.

Provides two interfaces:

1. ``ReportGenerator`` class (used by tests and dashboard)
   -------------------------------------------------------
   rg = ReportGenerator()
   rg.save_json(metrics_dict, path)   -> writes JSON file
   rg.save_pdf(alerts, path)          -> writes PDF (requires reportlab)
   rg.generate(alerts, output_path)   -> alias for save_pdf

2. ``generate_report()`` function (legacy / CLI usage)
   ---------------------------------------------------
   generate_report(alerts, output_path, title)  -> Path to PDF

Test contract (test_integration.py)
------------------------------------
    from src.utils import ReportGenerator
    rg = ReportGenerator()
    rg.save_json(metrics_dict, path)   # must write a valid JSON file
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReportGenerator class — primary interface used by tests
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate JSON or PDF reports from XAI-NIDS metric / alert data."""

    # ------------------------------------------------------------------
    # JSON output (used by test_integration.py)
    # ------------------------------------------------------------------

    def save_json(
        self,
        data: dict,
        output_path: str | Path,
        indent: int = 2,
    ) -> Path:
        """Serialise *data* to a JSON file at *output_path*.

        Non-serialisable values (numpy arrays, DataFrames, etc.) are
        converted to Python native types automatically.

        Args:
            data:         Dict to serialise (e.g. output of compute_metrics).
            output_path:  Destination path for the .json file.
            indent:       JSON indentation level.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = self._make_serialisable(data)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=indent)

        log.info("JSON report saved -> %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # PDF output (requires reportlab — optional dependency)
    # ------------------------------------------------------------------

    def save_pdf(
        self,
        alerts: list[dict[str, Any]],
        output_path: str | Path = "alert_report.pdf",
        title: str = "XAI-NIDS Alert Report",
    ) -> Path:
        """Generate a PDF report from a list of alert dicts."""
        return generate_report(alerts, output_path=output_path, title=title)

    # Alias for backward compatibility
    def generate(self, alerts, output_path="alert_report.pdf", title="XAI-NIDS Alert Report"):
        return self.save_pdf(alerts, output_path, title)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_serialisable(obj: Any) -> Any:
        """Recursively convert numpy / pandas objects to JSON-safe types."""
        import numpy as np

        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.to_list()
        except ImportError:
            pass

        if isinstance(obj, dict):
            return {k: ReportGenerator._make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ReportGenerator._make_serialisable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj


# ---------------------------------------------------------------------------
# generate_report() — legacy function kept for CLI / existing code
# ---------------------------------------------------------------------------

def generate_report(
    alerts: list[dict[str, Any]],
    output_path: str | Path = "alert_report.pdf",
    title: str = "XAI-NIDS Alert Report",
) -> Path:
    """Generate a PDF report from a list of alert dicts.

    Requires ``reportlab``.  Falls back to JSON if reportlab is absent so
    the rest of the test suite can still run.

    Args:
        alerts:       List of alert dicts with keys: flow_id, prediction,
                      confidence, top_features, timestamp.
        output_path:  Path where the PDF (or fallback JSON) will be saved.
        title:        Report title string.

    Returns:
        Path to the generated file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        log.warning(
            "reportlab not installed — writing JSON fallback to %s", output_path
        )
        rg = ReportGenerator()
        json_path = output_path.with_suffix(".json")
        rg.save_json({"alerts": alerts, "title": title}, json_path)
        return json_path

    _SEVERITY_COLOURS: dict[str, tuple] = {
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
    story.append(Paragraph(title, title_style))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"Generated: {generated_at}  |  Total alerts: {len(alerts)}", sub_style
    ))
    story.append(Spacer(1, 0.5 * cm))

    summary_data = [["Flow ID", "Prediction", "Confidence", "Timestamp"]]
    for a in alerts:
        summary_data.append([
            str(a.get("flow_id", "\u2014")),
            str(a.get("prediction", "\u2014")),
            f"{a.get('confidence', 0) * 100:.1f}%",
            str(a.get("timestamp", "\u2014")),
        ])

    summary_table = Table(summary_data, colWidths=[2.5*cm, 5*cm, 3.5*cm, 5.5*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f7f8fa"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5 * cm))

    for alert in alerts:
        fid  = alert.get("flow_id", "?")
        pred = alert.get("prediction", "Unknown")
        conf = alert.get("confidence", 0.0)
        feats = alert.get("top_features", [])
        ts   = alert.get("timestamp", "")
        alert_color = _severity_color(pred)
        hex_color = (
            alert_color.hexval()
            if hasattr(alert_color, "hexval")
            else "red"
        )
        story.append(Paragraph(
            f"Alert #{fid} \u2014 "
            f"<font color='{hex_color}'>{pred}</font>"
            f"  ({conf * 100:.1f}% confidence)"
            f"  <font size=8 color='grey'>{ts}</font>",
            heading_style,
        ))
        if feats:
            feat_data = [["Feature", "SHAP Contribution", "Direction"]]
            for ft in feats:
                name = ft.get("name", "\u2014")
                sv   = ft.get("shap", 0.0)
                direction = "\u2192 attack" if sv > 0 else "\u2190 benign"
                feat_data.append([name, f"{sv:+.4f}", direction])
            feat_table = Table(feat_data, colWidths=[8*cm, 4*cm, 4.5*cm])
            feat_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTSIZE",   (0, 0), (-1, -1), 8),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f7f8fa"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(feat_table)
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    log.info("PDF report saved -> %s  (%d alerts)", output_path, len(alerts))
    return output_path

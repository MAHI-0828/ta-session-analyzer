"""
PDF report generator for TA session quality analysis results.
Mirrors the layout in the TA Session Analyzer PRD's sample report.
"""

from datetime import datetime

from fpdf import FPDF

from ta_core import WEIGHTS

RATING_COLORS = {
    "Excellent": (26, 127, 55),
    "Good":      (45, 164, 78),
    "Fair":      (210, 153, 34),
    "Poor":      (185, 28, 28),
    "Fully Resolved":     (26, 127, 55),
    "Partially Resolved": (210, 153, 34),
    "Not Resolved":       (185, 28, 28),
    "Concept-based": (26, 127, 55),
    "Mixed":         (210, 153, 34),
    "Answer-based":  (185, 28, 28),
    "Correct":           (26, 127, 55),
    "Incorrect Guidance": (185, 28, 28),
}

DIRECT_SOLUTION_COLORS = {"Good": (26, 127, 55), "Warning": (210, 153, 34), "Violation": (185, 28, 28)}


def score_color(pct):
    if pct is None:
        return (139, 148, 158)
    if pct >= 85:
        return (26, 127, 55)
    if pct >= 70:
        return (45, 164, 78)
    if pct >= 50:
        return (210, 153, 34)
    return (185, 28, 28)


_UNICODE_REPLACEMENTS = {
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "•": "-",
}


def pdf_safe(text) -> str:
    """The core Helvetica font only supports Latin-1. LLM-generated summaries/
    recommendations (or session metadata) can contain Hindi script, smart
    quotes, or other characters outside that range — normalize common cases
    and replace anything left over so PDF generation never crashes."""
    text = str(text)
    for u, a in _UNICODE_REPLACEMENTS.items():
        text = text.replace(u, a)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def write_indented(pdf, text, indent=12, line_height=5):
    pdf.set_x(indent)
    pdf.multi_cell(190 - indent, line_height, pdf_safe(text), new_x="LMARGIN", new_y="NEXT")


class TAReportPDF(FPDF):
    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"Page {self.page_no()}  |  TA Session Analyzer", align="C")


def _stat_row(pdf, label, value, value_color=None):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(60, 7, pdf_safe(label))
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*(value_color or (30, 30, 30)))
    pdf.cell(0, 7, pdf_safe(value), new_x="LMARGIN", new_y="NEXT")


def generate_ta_pdf(session_meta: dict, report: dict) -> bytes:
    """
    session_meta: {"session_id": str, "ta_name": str, "student_name": str}
    report: the dict returned by ta_core.analyze_ta_session()
    """
    analysis = report["analysis"]
    breakdown = report["score_breakdown"]
    participation = report["participation"]

    pdf = TAReportPDF()
    pdf.set_margins(10, 10, 10)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Header banner ────────────────────────────────────────────────────
    pdf.set_fill_color(22, 40, 80)
    pdf.rect(0, 0, 210, 36, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "TA Session Quality Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 200, 255)
    pdf.set_x(10)
    pdf.cell(0, 6, pdf_safe(f"Session: {session_meta.get('session_id', '-')}   |   Date: {datetime.now().strftime('%d %b %Y')}"),
             new_x="LMARGIN", new_y="NEXT")

    # ── Overall score ─────────────────────────────────────────────────────
    pdf.set_y(42)
    overall = breakdown["overall"]
    r, g, b = score_color(overall)
    pdf.set_fill_color(r, g, b)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(80, 11, f"  Overall Score: {overall:.1f} / 100", fill=True)
    pdf.ln(16)

    # ── Session stats ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Session Details", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    doubt = analysis["doubt_resolution"]["status"]
    direct = analysis["direct_solution"]["classification"]
    tech = analysis["technical_accuracy"]["status"]
    concept = analysis["concept_vs_answer"]["classification"]
    prof = analysis["professionalism"]["rating"]

    _stat_row(pdf, "TA Name", session_meta.get("ta_name", "-"))
    _stat_row(pdf, "Student", session_meta.get("student_name", "-"))
    _stat_row(pdf, "Duration", f"{report['duration_minutes']} min")
    _stat_row(pdf, "Doubt Resolution", doubt, RATING_COLORS.get(doubt))
    _stat_row(pdf, "Teaching Quality", f"{analysis['teaching_quality']['score']} / 5")
    _stat_row(pdf, "TA Participation", f"{participation['ta_pct']}%")
    _stat_row(pdf, "Student Participation", f"{participation['student_pct']}%")
    _stat_row(pdf, "Concept vs Answer", concept, RATING_COLORS.get(concept))
    _stat_row(pdf, "Direct Answers Given", "Yes" if direct != "Good" else "No", DIRECT_SOLUTION_COLORS.get(direct))
    _stat_row(pdf, "Incorrect Guidance", "Yes" if tech == "Incorrect Guidance" else "No", RATING_COLORS.get(tech))
    _stat_row(pdf, "Professionalism", prof, RATING_COLORS.get(prof))
    pdf.ln(4)

    # ── Weighted score breakdown ──────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Score Breakdown", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(50, 60, 100)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(90, 7, "  Metric", fill=True)
    pdf.cell(45, 7, "Points", fill=True, align="C")
    pdf.cell(45, 7, "Weight", fill=True, align="C")
    pdf.ln(7)

    labels = {
        "doubt_resolution": "Doubt Resolution",
        "teaching_quality": "Teaching Quality",
        "student_engagement": "Student Engagement",
        "no_direct_answers": "No Direct Answers",
        "technical_accuracy": "Technical Accuracy",
        "professionalism": "Professionalism",
        "communication": "Communication",
        "session_structure": "Session Structure",
    }
    pdf.set_font("Helvetica", "", 9)
    for idx, (key, label) in enumerate(labels.items()):
        row = breakdown[key]
        row_bg = (248, 249, 252) if idx % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*row_bg)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(90, 7, f"  {label}", fill=True)
        pdf.cell(45, 7, f"{row['points']} / {row['max']}", fill=True, align="C")
        pdf.cell(45, 7, f"{WEIGHTS[key]}%", fill=True, align="C")
        pdf.ln(7)
    pdf.ln(6)

    # ── Summary ────────────────────────────────────────────────────────────
    pdf.set_fill_color(230, 236, 248)
    pdf.set_text_color(22, 40, 80)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "   Summary", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    write_indented(pdf, analysis.get("summary", ""))
    pdf.ln(4)

    # ── Recommendations ─────────────────────────────────────────────────────
    recs = analysis.get("recommendations", [])
    if recs:
        pdf.set_fill_color(230, 245, 233)
        pdf.set_text_color(20, 100, 40)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "   Recommendations", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        for rec in recs:
            write_indented(pdf, f"-  {rec}")
        pdf.ln(4)

    # ── AI Flags ─────────────────────────────────────────────────────────────
    flags = report.get("flags", [])
    if flags:
        pdf.set_fill_color(255, 233, 220)
        pdf.set_text_color(140, 45, 20)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "   AI Flags - Recommended for Manual Review", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(140, 45, 20)
        for flag in flags:
            write_indented(pdf, f"!  {flag}")
    else:
        pdf.set_fill_color(230, 245, 233)
        pdf.set_text_color(20, 100, 40)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "   No flags - session looks healthy", fill=True, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

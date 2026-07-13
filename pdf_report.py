"""
PDF report generator for lecture quality analysis results.
"""

from datetime import datetime
from fpdf import FPDF
from core import PARAMETERS, MAJOR_PARAMETERS, MINOR_PARAMETERS, PARAMETER_LABELS, aggregate_scores

# Chatbox pop-out tip is always shown regardless of score
CHATBOX_SUGGESTION = (
    "The chat panel was not clearly visible on screen. Use the platform's pop-out feature "
    "to float the chat window on a second monitor or a separate device (phone/tablet), so you "
    "can monitor messages without looking away from the camera."
)

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def score_color(score):
    if score is None:
        return (139, 148, 158)
    s = max(1, min(5, round(score)))
    return {5: (26, 127, 55), 4: (45, 164, 78), 3: (180, 135, 20), 2: (207, 81, 38), 1: (185, 28, 28)}[s]


def rating_label(score):
    if score is None: return "N/A"
    if score >= 4.5:  return "Excellent"
    if score >= 3.5:  return "Good"
    if score >= 2.5:  return "Fair"
    return "Needs Work"


def best_texts(results, param):
    """Return (observation from best screenshot, improvement from worst screenshot)."""
    entries = []
    for r in results:
        data = r.get("scores", {}).get(param, {})
        if not isinstance(data, dict):
            continue
        s = data.get("score")
        o = data.get("observation", "")
        i = data.get("improvement", "")
        if isinstance(s, (int, float)) and o and o != "Not applicable in this screenshot.":
            entries.append((s, o, i))
    if not entries:
        return "", ""
    best_obs  = max(entries, key=lambda x: x[0])[1]
    worst_imp = min(entries, key=lambda x: x[0])[2]
    return best_obs, worst_imp


def write_indented(pdf, text, indent=12, line_height=5):
    """Write a multi-line text block with left indent, staying within right margin."""
    pdf.set_x(indent)
    pdf.multi_cell(190 - indent, line_height, text, new_x="LMARGIN", new_y="NEXT")


# ─── PDF class ───────────────────────────────────────────────────────────────────

class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"Page {self.page_no()}  |  Lecture Quality Analyzer", align="C")


# ─── Main generator ──────────────────────────────────────────────────────────────

def generate_pdf(batch: str, module: str, results: list) -> bytes:
    averages = aggregate_scores(results)
    overall_list = [r["overall_score"] for r in results if isinstance(r.get("overall_score"), (int, float))]
    overall = sum(overall_list) / len(overall_list) if overall_list else 0

    # Good = avg >= 4.0 | Improve = avg < 4.0 or null
    good_params    = sorted([(p, v) for p, v in averages.items() if v is not None and v >= 4.0], key=lambda x: -x[1])
    improve_params = sorted([(p, v) for p, v in averages.items() if v is None or v < 4.0],       key=lambda x: (x[1] or 0))

    # Always include chat_panel_visible in improve if not already there (null = not detected)
    chat_avg = averages.get("chat_panel_visible")
    if chat_avg is None and not any(p == "chat_panel_visible" for p, _ in improve_params):
        improve_params = [("chat_panel_visible", None)] + improve_params

    pdf = ReportPDF()
    pdf.set_margins(10, 10, 10)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Header banner ────────────────────────────────────────────────────────────
    pdf.set_fill_color(22, 40, 80)
    pdf.rect(0, 0, 210, 42, "F")

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "Lecture Quality Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 200, 255)
    pdf.set_x(10)
    pdf.cell(0, 6, f"Batch: {batch}   |   Module: {module}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(10)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%d %b %Y')}   |   {len(results)} screenshot(s) analyzed",
             new_x="LMARGIN", new_y="NEXT")

    # ── Overall score ────────────────────────────────────────────────────────────
    pdf.set_y(48)
    r, g, b = score_color(overall)
    pdf.set_fill_color(r, g, b)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(75, 10, f"  Overall Score: {overall:.1f} / 5.0  ({rating_label(overall)})", fill=True)
    pdf.ln(14)

    # ── Score table ──────────────────────────────────────────────────────────────
    def render_score_section(section_label, section_color, params):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(*section_color)
        pdf.cell(190, 7, f"  {section_label}", fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(50, 60, 100)
        pdf.cell(100, 7, "  Parameter",  fill=True)
        pdf.cell(30,  7, "Avg Score",    fill=True, align="C")
        pdf.cell(30,  7, "Rating",       fill=True, align="C")
        pdf.cell(30,  7, "Visual",       fill=True, align="C")
        pdf.ln(7)

        pdf.set_font("Helvetica", "", 9)
        for idx, param in enumerate(params):
            avg = averages.get(param)
            if avg is None:
                continue
            label = PARAMETER_LABELS[param]
            r, g, b = score_color(avg)
            row_bg  = (248, 249, 252) if idx % 2 == 0 else (255, 255, 255)

            pdf.set_fill_color(*row_bg)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(100, 7, f"  {label}", fill=True)
            pdf.cell(30,  7, f"{avg:.1f} / 5.0", fill=True, align="C")

            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(30, 7, rating_label(avg), fill=True, align="C")

            bar_x = pdf.get_x()
            bar_y = pdf.get_y()
            pdf.set_fill_color(*row_bg)
            pdf.cell(30, 7, "", fill=True)
            pdf.set_fill_color(r, g, b)
            pdf.rect(bar_x + 2, bar_y + 2, int((avg / 5) * 24), 3, "F")
            pdf.ln(7)
        pdf.ln(4)

    render_score_section("Major Checks", (160, 30, 30), MAJOR_PARAMETERS)
    render_score_section("Minor Checks", (30, 80, 160), MINOR_PARAMETERS)
    pdf.ln(6)

    # ── What Went Well ───────────────────────────────────────────────────────────
    def render_good_subgroup(subheader, params_set, items):
        filtered = [(p, v) for p, v in items if p in params_set]
        if not filtered:
            return
        pdf.set_font("Helvetica", "BI", 10)
        pdf.set_text_color(20, 100, 40)
        pdf.cell(0, 6, f"  {subheader}", new_x="LMARGIN", new_y="NEXT")
        for param, avg in filtered:
            obs, _ = best_texts(results, param)
            if not obs:
                continue
            label   = PARAMETER_LABELS[param]
            r, g, b = score_color(avg)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(r, g, b)
            pdf.cell(0, 6, f"  [+]  {label}  -  {avg:.1f}/5", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(55, 75, 55)
            write_indented(pdf, obs)
            pdf.ln(2)

    if good_params:
        pdf.set_fill_color(220, 245, 228)
        pdf.set_text_color(20, 100, 40)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 9, "   What Went Well", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        render_good_subgroup("Major Checks", set(MAJOR_PARAMETERS), good_params)
        render_good_subgroup("Minor Checks", set(MINOR_PARAMETERS), good_params)

    pdf.ln(6)

    # ── What Can Be Improved ─────────────────────────────────────────────────────
    def render_improve_subgroup(subheader, params_set, items):
        filtered = [(p, v) for p, v in items if p in params_set]
        if not filtered:
            return
        pdf.set_font("Helvetica", "BI", 10)
        pdf.set_text_color(140, 45, 20)
        pdf.cell(0, 6, f"  {subheader}", new_x="LMARGIN", new_y="NEXT")
        for param, avg in filtered:
            obs, imp = best_texts(results, param)
            label    = PARAMETER_LABELS[param]
            r, g, b  = score_color(avg) if avg else (139, 148, 158)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(r, g, b)
            score_str = f"{avg:.1f}/5" if avg else "Not detected"
            pdf.cell(0, 6, f"  [!]  {label}  -  {score_str}", new_x="LMARGIN", new_y="NEXT")
            if obs:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(80, 80, 80)
                write_indented(pdf, f"Observed: {obs}")
            suggestion = CHATBOX_SUGGESTION if param == "chat_panel_visible" else imp
            if suggestion:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(140, 60, 20)
                write_indented(pdf, f"Suggestion: {suggestion}")
            pdf.ln(3)

    if improve_params:
        pdf.set_fill_color(255, 233, 220)
        pdf.set_text_color(140, 45, 20)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 9, "   What Can Be Improved", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        render_improve_subgroup("Major Checks", set(MAJOR_PARAMETERS), improve_params)
        render_improve_subgroup("Minor Checks", set(MINOR_PARAMETERS), improve_params)

    return bytes(pdf.output())

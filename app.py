import io
import csv
import json
from datetime import datetime

import streamlit as st

from core import (
    PARAMETERS, MAJOR_PARAMETERS, MINOR_PARAMETERS, PARAMETER_LABELS, PARAMETER_DESCRIPTIONS,
    load_image_bytes, load_image_file, analyze_image,
    build_csv_rows, aggregate_scores,
)
from pdf_report import generate_pdf

# ─── Page config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lecture Quality Analyzer",
    page_icon="🎓",
    layout="wide",
)

# ─── Styles ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.score-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
    color: white;
}
.score-5 { background: #1a7f37; }
.score-4 { background: #2da44e; }
.score-3 { background: #d29922; }
.score-2 { background: #cf5126; }
.score-1 { background: #b91c1c; }
.score-na { background: #8b949e; }
.flag-row { background: #fff3cd; border-left: 3px solid #f0ad4e; padding: 4px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def score_pill(score):
    if score is None:
        return '<span class="score-pill score-na">N/A</span>'
    cls = f"score-{max(1, min(5, int(score)))}"
    return f'<span class="score-pill {cls}">{score}/5</span>'


def score_color(score):
    if score is None:
        return "#8b949e"
    if score >= 4.5:
        return "#1a7f37"
    if score >= 3.5:
        return "#2da44e"
    if score >= 2.5:
        return "#d29922"
    if score >= 1.5:
        return "#cf5126"
    return "#b91c1c"


def build_csv_bytes(batch, module, results):
    rows = build_csv_rows(batch, module, results)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ─── Header ──────────────────────────────────────────────────────────────────────

st.title("🎓 Lecture Quality Analyzer")
st.caption("Upload screenshots from a recorded lecture and get a scored report across 16 visual quality parameters.")

st.divider()

# ─── Inputs ──────────────────────────────────────────────────────────────────────

col_l, col_r = st.columns([1, 2])

with col_l:
    st.subheader("Session Details")
    batch_name = st.text_input("Batch Name", placeholder="e.g. DS-Batch-12")
    lecture_module = st.text_input("Lecture Module", placeholder="e.g. Module-3-SQL-Joins")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_... (or set GROQ_API_KEY env var)",
    )
    st.caption("Free key from console.groq.com — never stored.")

with col_r:
    st.subheader("Upload Screenshots")
    uploaded_files = st.file_uploader(
        "Drop up to 10 screenshots (PNG / JPG / WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
        preview_cols = st.columns(min(len(uploaded_files), 5))
        for i, f in enumerate(uploaded_files[:5]):
            with preview_cols[i]:
                st.image(f, use_container_width=True, caption=f.name)
        if len(uploaded_files) > 5:
            st.caption(f"+ {len(uploaded_files) - 5} more not shown in preview")

st.divider()

# ─── Analyze button ──────────────────────────────────────────────────────────────

ready = batch_name and lecture_module and uploaded_files
analyze_btn = st.button("Analyze Screenshots", type="primary", disabled=not ready)

if not ready and not analyze_btn:
    missing = []
    if not batch_name:       missing.append("Batch Name")
    if not lecture_module:   missing.append("Lecture Module")
    if not uploaded_files:   missing.append("at least one screenshot")
    if missing:
        st.info(f"Fill in: {', '.join(missing)}")

# ─── Analysis ────────────────────────────────────────────────────────────────────

if analyze_btn and ready:
    st.session_state.pop("report", None)  # clear previous report on new analysis

if analyze_btn and ready:
    import os
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        st.error("No API key provided. Enter it above or set the GROQ_API_KEY environment variable.")
        st.stop()

    results = []

    progress_bar = st.progress(0, text="Starting analysis...")
    status_area  = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_area.markdown(f"Analyzing **{uploaded_file.name}** ({i+1}/{len(uploaded_files)})…")
        try:
            file_bytes = uploaded_file.read()
            image_data, media_type = load_image_bytes(file_bytes, uploaded_file.name)
            result = analyze_image(key, image_data, media_type)
            result["screenshot"]  = uploaded_file.name
            result["analyzed_at"] = datetime.now().isoformat()
            results.append(result)
        except Exception as e:
            st.warning(f"Skipped {uploaded_file.name}: {e}")
        progress_bar.progress((i + 1) / len(uploaded_files),
                               text=f"Analyzed {i+1}/{len(uploaded_files)}")

    status_area.empty()
    progress_bar.empty()

    if not results:
        st.error("Analysis failed for all screenshots.")
        st.stop()

    # Persist results so downloading a file doesn't reset the page
    st.session_state["report"] = {
        "results":  results,
        "batch":    batch_name,
        "module":   lecture_module,
    }

# Show report if results exist (survives download button reruns)
if "report" in st.session_state:
    results     = st.session_state["report"]["results"]
    batch_name  = st.session_state["report"]["batch"]
    lecture_module = st.session_state["report"]["module"]

    st.success(f"Done! Analyzed {len(results)} screenshot(s).")
    st.divider()

    # ── Report card ──────────────────────────────────────────────────────────────

    st.subheader("📊 Report Card")
    averages = aggregate_scores(results)
    overall_scores = [r["overall_score"] for r in results
                      if isinstance(r.get("overall_score"), (int, float))]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else None

    if overall_avg is not None:
        col_ov, col_batch, col_mod = st.columns(3)
        col_ov.metric("Overall Score", f"{overall_avg:.1f} / 5")
        col_batch.metric("Batch", batch_name)
        col_mod.metric("Module", lecture_module)

    st.markdown("---")

    def render_param_group(title, params, averages):
        st.markdown(f"**{title}**")
        cols = st.columns(len(params))
        for col, param in zip(cols, params):
            avg = averages.get(param)
            label = PARAMETER_LABELS[param]
            short = label.replace("Content Type on Screen", "Live Coding")
            description = PARAMETER_DESCRIPTIONS.get(param, label)
            if avg is not None:
                col.metric(short, f"{avg:.1f}", help=description)
                col.markdown(
                    f'<div style="height:6px;border-radius:3px;background:{score_color(avg)};'
                    f'width:{int(avg/5*100)}%"></div>',
                    unsafe_allow_html=True,
                )
            else:
                col.metric(short, "N/A", help=description)
        st.markdown("")

    st.markdown("#### 🔴 Major Checks")
    render_param_group("", MAJOR_PARAMETERS, averages)

    st.markdown("#### 🔵 Minor Checks")
    render_param_group("", MINOR_PARAMETERS, averages)

    # Flags — major first
    flagged_major = [(PARAMETER_LABELS[p], averages[p]) for p in MAJOR_PARAMETERS
                     if averages.get(p) is not None and averages[p] < 3.5]
    flagged_minor = [(PARAMETER_LABELS[p], averages[p]) for p in MINOR_PARAMETERS
                     if averages.get(p) is not None and averages[p] < 3.5]

    if flagged_major or flagged_minor:
        st.markdown("---")
        st.markdown("**🚩 Parameters needing attention (avg < 3.5)**")
        if flagged_major:
            st.markdown("*Major:*")
            for label, avg in sorted(flagged_major, key=lambda x: x[1]):
                st.markdown(f'<div class="flag-row">⚠️ <b>{label}</b> — {avg:.1f}/5</div>', unsafe_allow_html=True)
        if flagged_minor:
            st.markdown("*Minor:*")
            for label, avg in sorted(flagged_minor, key=lambda x: x[1]):
                st.markdown(f'<div class="flag-row">⚠️ <b>{label}</b> — {avg:.1f}/5</div>', unsafe_allow_html=True)

    # ── Per-screenshot breakdown ─────────────────────────────────────────────────

    st.divider()
    st.subheader("🖼 Per-Screenshot Breakdown")

    for r in results:
        with st.expander(f"📸 {r['screenshot']}  —  overall {r.get('overall_score', '?')}/5"):
            if r.get("priority_fix"):
                st.info(f"**Priority fix:** {r['priority_fix']}")

            for param in PARAMETERS:
                data = r.get("scores", {}).get(param, {})
                if not isinstance(data, dict):
                    continue
                score = data.get("score")
                obs   = data.get("observation", "")
                imp   = data.get("improvement", "")

                if obs == "Not applicable in this screenshot.":
                    continue

                label = PARAMETER_LABELS[param]
                pill  = score_pill(score)

                st.markdown(
                    f"{pill} &nbsp; **{label}**",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                c1.markdown(f"*What I see:* {obs}")
                c2.markdown(f"*Improve:* {imp}")
                st.markdown('<hr style="margin:6px 0;border-color:#30363d">', unsafe_allow_html=True)

    # ── Downloads ────────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("⬇️ Download Reports")

    slug      = f"{batch_name}_{lecture_module}".replace(" ", "-").replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{slug}_{timestamp}"

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # PDF
    try:
        pdf_bytes = generate_pdf(batch_name, lecture_module, results)
        dl_col1.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"{base_name}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        dl_col1.warning(f"PDF error: {e}")

    # CSV
    csv_bytes = build_csv_bytes(batch_name, lecture_module, results)
    dl_col2.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{base_name}.csv",
        mime="text/csv",
    )

    # JSON
    json_payload = json.dumps({
        "batch_name":     batch_name,
        "lecture_module": lecture_module,
        "analyzed_at":    timestamp,
        "results":        results,
    }, indent=2, ensure_ascii=False).encode("utf-8")
    dl_col3.download_button(
        label="Download JSON",
        data=json_payload,
        file_name=f"{base_name}.json",
        mime="application/json",
    )

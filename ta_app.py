"""
Streamlit UI for the TA Session Analyzer — internal team tool.

Two modes:
  - Single Session: paste a recording URL or upload a video, run one
    analysis, view the scorecard, download the PDF/JSON.
  - Batch (CSV): upload a CSV in the same format ta_session_analyzer.py's
    daily batch runner expects, run the whole day's sessions, watch
    progress, download the rollup + PDFs.

Secrets (Streamlit Cloud: app settings -> Secrets. Locally: create
.streamlit/secrets.toml — it's gitignored, never commit it):
    GEMINI_API_KEY = "..."   # shared team key, used if no key is entered
    APP_PASSWORD   = "..."   # optional — gates the whole app if set

Run locally:
    streamlit run ta_app.py
"""

import io
import json
import os
import tempfile
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st

from recording_utils import extract_video_url, download_video
from ta_core import analyze_ta_session
from ta_pdf_report import generate_ta_pdf
from ta_session_analyzer import process_session, OUTPUT_DIR

st.set_page_config(page_title="TA Session Analyzer", page_icon="📋", layout="wide")


def _get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def _require_password() -> bool:
    app_password = _get_secret("APP_PASSWORD")
    if not app_password:
        return True
    if st.session_state.get("authed"):
        return True
    st.title("📋 TA Session Analyzer")
    pw = st.text_input("Team password", type="password")
    if st.button("Enter") and pw == app_password:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    return False


if not _require_password():
    st.stop()

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

st.title("📋 TA Session Analyzer")
st.caption("Internal tool — scores TA doubt-clearing session recordings against the quality rubric.")

with st.sidebar:
    st.subheader("Settings")
    if GEMINI_API_KEY:
        st.success("Gemini API key loaded from server config.")
    else:
        GEMINI_API_KEY = st.text_input("Gemini API key", type="password",
                                        help="Free key: aistudio.google.com")
        st.caption("Key is used for this session only — never stored or logged.")


def _render_report(meta: dict, report: dict):
    breakdown = report["score_breakdown"]
    analysis = report["analysis"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Score", f"{breakdown['overall']:.1f} / 100")
    c2.metric("Doubt Resolution", analysis["doubt_resolution"]["status"])
    c3.metric("Duration", f"{report['duration_minutes']} min")

    if report["flags"]:
        st.warning("**AI Flags — recommended for manual review**\n\n" +
                   "\n".join(f"- {f}" for f in report["flags"]))
    else:
        st.success("No flags — session looks healthy.")

    st.markdown("**Summary**")
    st.write(analysis.get("summary", ""))

    if analysis.get("recommendations"):
        st.markdown("**Recommendations**")
        for r in analysis["recommendations"]:
            st.write(f"- {r}")

    st.markdown("**Score Breakdown**")
    rows = [
        {"Metric": k.replace("_", " ").title(), "Points": v["points"], "Max": v["max"]}
        for k, v in breakdown.items() if k != "overall"
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    p = report["participation"]
    st.markdown(f"**Participation** — TA {p['ta_pct']}% | Student {p['student_pct']}%")

    if report.get("screen_share"):
        st.markdown("**Screen Share**")
        st.write(report["screen_share"].get("summary", ""))

    with st.expander("Full transcript"):
        st.text(report["transcript_text"])

    pdf_bytes = generate_ta_pdf(meta, report)
    json_bytes = json.dumps({"session_meta": meta, "report": report},
                             indent=2, ensure_ascii=False).encode("utf-8")
    dcol1, dcol2 = st.columns(2)
    dcol1.download_button("Download PDF report", pdf_bytes,
                           file_name=f"{meta['session_id']}.pdf", mime="application/pdf")
    dcol2.download_button("Download JSON", json_bytes,
                           file_name=f"{meta['session_id']}.json", mime="application/json")


tab_single, tab_batch = st.tabs(["Single Session", "Batch (CSV)"])

# ─── Single Session ───────────────────────────────────────────────────────────

with tab_single:
    st.subheader("Analyze one session")
    input_mode = st.radio("Input", ["Recording URL", "Upload video file"], horizontal=True)

    url_value, uploaded_video = "", None
    if input_mode == "Recording URL":
        url_value = st.text_input("Recording URL", placeholder="https://my.newtonschool.co/play-video/?url=...")
    else:
        uploaded_video = st.file_uploader("Upload recording", type=["mp4", "mov", "mkv", "webm"])

    col1, col2 = st.columns(2)
    with col1:
        ta_name = st.text_input("TA name", value="")
        session_id = st.text_input("Session ID", value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with col2:
        student_name = st.text_input("Student name", value="")
        analyze_screen = st.checkbox("Analyze shared screen", value=True)

    chat_text = None
    chat_file = st.file_uploader("Optional: chat log (.txt)", type=["txt"], key="single_chat")
    if chat_file:
        chat_text = chat_file.read().decode("utf-8", errors="ignore")

    if st.button("Analyze session", type="primary", disabled=not GEMINI_API_KEY):
        if not session_id:
            st.error("Session ID is required.")
        elif input_mode == "Recording URL" and not url_value:
            st.error("Enter a recording URL.")
        elif input_mode == "Upload video file" and not uploaded_video:
            st.error("Upload a video file.")
        else:
            try:
                with st.spinner("Downloading and analyzing — this can take a minute or two..."):
                    with tempfile.TemporaryDirectory() as tmp:
                        video_path = os.path.join(tmp, "recording.mp4")
                        if input_mode == "Recording URL":
                            resolved = extract_video_url(url_value)
                            download_video(resolved, video_path)
                        else:
                            with open(video_path, "wb") as f:
                                f.write(uploaded_video.read())

                        report = analyze_ta_session(
                            GEMINI_API_KEY, video_path,
                            analyze_screen=analyze_screen, chat_text=chat_text,
                        )
                st.session_state["last_report"] = report
                st.session_state["last_meta"] = {
                    "session_id": session_id, "ta_name": ta_name, "student_name": student_name,
                }
                st.success("Done.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if "last_report" in st.session_state:
        st.divider()
        _render_report(st.session_state["last_meta"], st.session_state["last_report"])

# ─── Batch (CSV) ───────────────────────────────────────────────────────────────

with tab_batch:
    st.subheader("Batch run from CSV")
    st.caption("Columns: recording_url, ta_name, student_name, session_id, analyze_screen (yes/no), chat_log_path (leave blank)")
    csv_file = st.file_uploader("Upload sessions CSV", type=["csv"], key="batch_csv")

    if csv_file and st.button("Run batch", type="primary", disabled=not GEMINI_API_KEY):
        df_in = pd.read_csv(csv_file, dtype=str).fillna("")
        rows = df_in.to_dict("records")
        run_date = datetime.now().strftime("%Y-%m-%d")

        progress = st.progress(0.0)
        status = st.empty()
        results, errors = [], []
        for i, row in enumerate(rows):
            status.write(f"Processing `{row.get('session_id', i)}` ({i + 1}/{len(rows)})...")
            try:
                results.append(process_session(row, run_date, api_key=GEMINI_API_KEY))
            except Exception as e:
                errors.append({"session_id": row.get("session_id", "?"), "error": str(e)})
            progress.progress((i + 1) / len(rows))
        status.write("Done.")

        st.session_state["batch_results"] = results
        st.session_state["batch_errors"] = errors

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        errors = st.session_state["batch_errors"]
        st.success(f"{len(results)} succeeded, {len(errors)} failed.")

        if results:
            display_df = pd.DataFrame(results).drop(columns=["pdf_path"])
            st.dataframe(display_df, use_container_width=True)

            rollup_csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download rollup CSV", rollup_csv,
                                file_name=f"ta_rollup_{datetime.now().strftime('%Y%m%d')}.csv")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for r in results:
                    if r.get("pdf_path") and os.path.exists(r["pdf_path"]):
                        zf.write(r["pdf_path"], arcname=os.path.basename(r["pdf_path"]))
            st.download_button("Download all PDFs (zip)", buf.getvalue(),
                                file_name="ta_reports.zip", mime="application/zip")

        if errors:
            st.error("Failures:")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

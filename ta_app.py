"""
Streamlit UI for the TA Session Analyzer — internal team tool.

Transcript-only is the default everywhere: a run transcribes the recording
and saves the transcript, with no Gemini call. Tick "Also run AI quality
analysis" to score the session against the rubric as well — that reuses a
saved transcript, so transcribing first and analyzing later never pays for
transcription twice.

Three tabs:
  - Single Session: paste a recording URL or upload a video, get the
    transcript (and optionally the scorecard), download TXT/JSON/PDF.
  - Batch (CSV): upload a CSV in the same format ta_session_analyzer.py's
    daily batch runner expects, run the whole day's sessions, watch
    progress, download all transcripts / the rollup / PDFs.
  - Saved results: every session ever processed on this machine, plus
    backup/restore of the whole results store.

Everything is saved to disk as it finishes (ta_reports/sessions/, see
session_store.py), so a page refresh or crash mid-batch loses nothing:
upload the same CSV and press Run again — finished sessions are skipped
with no API calls, and a half-done one resumes from its last stage.

Secrets (Streamlit Cloud: app settings -> Secrets. Locally: create
.streamlit/secrets.toml — it's gitignored, never commit it):
    DEEPGRAM_API_KEY = "..."   # transcription/diarization — console.deepgram.com
                               # (not needed if you pick local Whisper)
    GEMINI_API_KEY   = "..."   # only needed for the optional AI analysis
    APP_PASSWORD     = "..."   # optional — gates the whole app if set

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

import session_store
from local_transcribe import available_backend
from ta_pdf_report import generate_ta_pdf
from ta_session_analyzer import is_done, process_session, result_mode, row_key, summary_row

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
DEEPGRAM_API_KEY = _get_secret("DEEPGRAM_API_KEY")
LOCAL_BACKEND = available_backend()

st.title("📋 TA Session Analyzer")
st.caption("Internal tool — transcribes TA doubt-clearing session recordings, "
           "and optionally scores them against the quality rubric.")

with st.sidebar:
    st.subheader("Settings")
    transcriber_options = ["deepgram"] + (["local"] if LOCAL_BACKEND else [])
    TRANSCRIBER = st.radio(
        "Transcription",
        transcriber_options,
        index=transcriber_options.index("local") if LOCAL_BACKEND and not DEEPGRAM_API_KEY else 0,
        format_func=lambda t: ("Deepgram API (diarized)" if t == "deepgram"
                               else f"Local Whisper — free ({LOCAL_BACKEND})"),
    )
    if not LOCAL_BACKEND:
        st.caption("Local Whisper not installed. On your Mac: "
                   "`pip install -r requirements-local.txt` to transcribe for free.")

    if TRANSCRIBER == "deepgram":
        if DEEPGRAM_API_KEY:
            st.success("Deepgram API key loaded from server config.")
        else:
            DEEPGRAM_API_KEY = st.text_input("Deepgram API key", type="password",
                                              help="Free key: console.deepgram.com — used for transcription/diarization.")
            st.caption("Key is used for this session only — never stored or logged.")

    if GEMINI_API_KEY:
        st.success("Gemini API key loaded from server config.")
    else:
        GEMINI_API_KEY = st.text_input("Gemini API key (only for AI analysis)", type="password",
                                        help="Free key: aistudio.google.com — not needed for transcripts.")
        st.caption("Key is used for this session only — never stored or logged.")

    saved_count = len(session_store.list_results())
    st.divider()
    st.metric("Sessions saved on disk", saved_count)
    st.caption("Results survive page refreshes. On Streamlit Cloud, download a backup "
               "from the **Saved results** tab — its disk resets when the app restarts.")

TRANSCRIBE_READY = TRANSCRIBER == "local" or bool(DEEPGRAM_API_KEY)


def _keys_ready(analyze: bool) -> bool:
    return TRANSCRIBE_READY and (bool(GEMINI_API_KEY) or not analyze)


def _missing_key_hint(analyze: bool):
    if not TRANSCRIBE_READY:
        st.caption("Enter a Deepgram API key in the sidebar, or install local Whisper.")
    elif analyze and not GEMINI_API_KEY:
        st.caption("AI analysis needs a Gemini API key (sidebar) — or untick it to just get the transcript.")


def _run_session(row: dict, log=lambda m: None, **kwargs) -> dict:
    return process_session(row, datetime.now().strftime("%Y-%m-%d"),
                           api_key=GEMINI_API_KEY, deepgram_api_key=DEEPGRAM_API_KEY,
                           transcriber=TRANSCRIBER, log=log, **kwargs)


def _render_downloads(meta: dict, report: dict, key_prefix: str):
    json_bytes = json.dumps({"session_meta": meta, "report": report},
                             indent=2, ensure_ascii=False).encode("utf-8")
    cols = st.columns(3)
    cols[0].download_button("Download transcript (.txt)", report["transcript_text"].encode("utf-8"),
                            key=f"{key_prefix}txt", file_name=f"{meta['session_id']}_transcript.txt",
                            mime="text/plain", type="primary")
    cols[1].download_button("Download JSON", json_bytes, key=f"{key_prefix}json",
                            file_name=f"{meta['session_id']}.json", mime="application/json")
    if report.get("mode", "analysis") == "analysis":
        cols[2].download_button("Download PDF report", generate_ta_pdf(meta, report), key=f"{key_prefix}pdf",
                                file_name=f"{meta['session_id']}.pdf", mime="application/pdf")


def _render_transcript_only(meta: dict, report: dict, key_prefix: str):
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{report['duration_minutes']} min")
    c2.metric("Transcribed with", report.get("transcriber", "deepgram"))
    c3.metric("Confidence", f"{report.get('transcription_confidence_pct', '?')}%")
    st.markdown("**Transcript**")
    st.text_area("Transcript", report["transcript_text"], height=420,
                 # session id in the key so switching sessions shows the new text
                 key=f"{key_prefix}transcript_view_{meta['session_id']}", label_visibility="collapsed")
    _render_downloads(meta, report, key_prefix)
    st.caption("Want a scorecard? Run this session again with **Also run AI quality analysis** "
               "ticked — it reuses this transcript, so only the Gemini call is made.")


def _render_report(meta: dict, report: dict, key_prefix: str = ""):
    if report.get("mode", "analysis") == "transcript":
        _render_transcript_only(meta, report, key_prefix)
        return

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

    st.caption(f"Transcribed with: {report.get('transcriber', 'deepgram')} · "
               f"confidence {report.get('transcription_confidence_pct', '?')}%")
    with st.expander("Full transcript"):
        st.text(report["transcript_text"])

    _render_downloads(meta, report, key_prefix)


def _zip_pdfs(summaries: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for r in summaries:
            if r.get("pdf_path") and os.path.exists(r["pdf_path"]):
                zf.write(r["pdf_path"], arcname=f"{r['ta_name']}_{r['session_id']}.pdf".replace(" ", "-"))
    return buf.getvalue()


def _zip_transcripts(results: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            meta = r["session_meta"]
            name = f"{meta.get('ta_name', '')}_{meta['session_id']}".strip("_").replace(" ", "-")
            zf.writestr(f"{name}_transcript.txt", r["report"]["transcript_text"])
    return buf.getvalue()


def _summary_df(summaries: list, with_transcript: bool = False) -> pd.DataFrame:
    """Table of sessions. The full transcript column is kept only for the
    CSV download (with_transcript) — it's too long for the on-page table."""
    drop = ["pdf_path", "transcript_path"] + ([] if with_transcript else ["transcript"])
    df = pd.DataFrame(summaries).drop(columns=drop, errors="ignore")
    if "flags" in df:
        df["flags"] = [("; ".join(f) or "none") if mode == "analysis" else ""
                       for f, mode in zip(df["flags"], df["mode"])]
    if (df.get("mode") == "transcript").all():
        # Transcript-only table: drop the empty score columns.
        df = df.drop(columns=["overall_score", "doubt_resolution", "ta_speaking_pct",
                              "student_speaking_pct", "flags"], errors="ignore")
    return df


def _batch_downloads(results: list, key_prefix: str):
    summaries = [summary_row(r) for r in results]
    cols = st.columns(3)
    cols[0].download_button(f"Download all transcripts ({len(results)}, zip)", _zip_transcripts(results),
                            file_name=f"ta_transcripts_{datetime.now().strftime('%Y%m%d')}.zip",
                            mime="application/zip", type="primary", key=f"{key_prefix}txts")
    # utf-8-sig (BOM) so Excel shows Hinglish/Devanagari text correctly.
    rollup_csv = _summary_df(summaries, with_transcript=True).to_csv(index=False).encode("utf-8-sig")
    cols[1].download_button("Download results CSV (with transcripts)", rollup_csv,
                            file_name=f"ta_rollup_{datetime.now().strftime('%Y%m%d')}.csv",
                            key=f"{key_prefix}csv")
    if any(s.get("pdf_path") for s in summaries):
        cols[2].download_button("Download all PDFs (zip)", _zip_pdfs(summaries),
                                file_name="ta_reports.zip", mime="application/zip", key=f"{key_prefix}pdfs")


tab_single, tab_batch, tab_saved = st.tabs(["Single Session", "Batch (CSV)", "Saved results"])

# ─── Single Session ───────────────────────────────────────────────────────────

with tab_single:
    st.subheader("Transcribe one session")
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

    single_analyze = st.checkbox("Also run AI quality analysis (Gemini)", value=False, key="single_analyze",
                                 help="Off = transcript only (no Gemini call). On = transcript + rubric "
                                      "scorecard + PDF.")
    analyze_screen, chat_file = False, None
    if single_analyze:
        analyze_screen = st.checkbox("Analyze shared screen", value=True)
        chat_file = st.file_uploader("Optional: chat log (.txt)", type=["txt"], key="single_chat")

    run_label = "Transcribe + analyze" if single_analyze else "Get transcript"
    if st.button(run_label, type="primary", disabled=not _keys_ready(single_analyze)):
        if not session_id:
            st.error("Session ID is required.")
        elif input_mode == "Recording URL" and not url_value:
            st.error("Enter a recording URL.")
        elif input_mode == "Upload video file" and not uploaded_video:
            st.error("Upload a video file.")
        else:
            row = {"recording_url": url_value, "ta_name": ta_name, "student_name": student_name,
                   "session_id": session_id, "analyze_screen": "yes" if analyze_screen else "no"}
            try:
                with st.status("Working — this can take a few minutes...", expanded=True) as status:
                    with tempfile.TemporaryDirectory() as tmp:
                        local_video = None
                        if uploaded_video:
                            local_video = os.path.join(tmp, "recording.mp4")
                            with open(local_video, "wb") as f:
                                f.write(uploaded_video.getbuffer())
                        if chat_file:
                            row["chat_log_path"] = os.path.join(tmp, "chat.txt")
                            with open(row["chat_log_path"], "wb") as f:
                                f.write(chat_file.getbuffer())
                        _run_session(row, log=status.write, local_video_path=local_video,
                                     analyze=single_analyze)
                    status.update(label="Done — saved to disk.", state="complete")
                st.session_state["last_key"] = row_key(row)
            except Exception as e:
                st.error(f"Failed: {e}. Anything already finished (e.g. the transcript) "
                         f"was saved — press **{run_label}** again with the same Session ID to resume.")
    _missing_key_hint(single_analyze)

    last = session_store.load_result(st.session_state["last_key"]) if "last_key" in st.session_state else None
    if last:
        st.divider()
        _render_report(last["session_meta"], last["report"], key_prefix="single_")

# ─── Batch (CSV) ───────────────────────────────────────────────────────────────

with tab_batch:
    st.subheader("Batch run from CSV")
    st.caption("Columns: recording_url, ta_name, student_name, session_id, analyze_screen (yes/no), "
               "chat_log_path (leave blank). Only recording_url and session_id matter for transcripts.")
    csv_file = st.file_uploader("Upload sessions CSV", type=["csv"], key="batch_csv")

    if csv_file:
        rows = pd.read_csv(io.BytesIO(csv_file.getvalue()), dtype=str).fillna("").to_dict("records")

        batch_analyze = st.checkbox("Also run AI quality analysis (Gemini)", value=False, key="batch_analyze",
                                    help="Off = transcripts only (no Gemini calls). On = transcript + rubric "
                                         "scorecard + PDF per session; already-saved transcripts are reused.")

        saved_results = {row_key(r): session_store.load_result(row_key(r)) for r in rows}
        n_done = sum(is_done(saved_results[row_key(r)], batch_analyze) for r in rows)
        n_reuse = sum(not is_done(saved_results[row_key(r)], batch_analyze)
                      and session_store.has_transcript(row_key(r)) for r in rows)
        done_word = "analyzed" if batch_analyze else "transcribed"
        st.info(f"**{len(rows)} sessions in CSV** — {n_done} already {done_word} (skipped, no API calls), "
                + (f"{n_reuse} transcribed but not analyzed (transcript reused, only Gemini runs), "
                   if batch_analyze else
                   f"{n_reuse} partly done (will resume), ")
                + f"{len(rows) - n_done - n_reuse} new.")

        opt1, opt2 = st.columns(2)
        reanalyze = False
        if batch_analyze:
            reanalyze = opt1.checkbox("Re-score finished sessions with Gemini",
                                      help="Reuses saved transcripts + screenshots — only the Gemini call "
                                           "is repeated. Use after changing the prompt/rubric.")
        force = opt2.checkbox("Redo everything from scratch",
                              help="Ignores all saved work, including transcripts. Costs API calls again.")

        batch_label = "Run batch: transcripts + analysis" if batch_analyze else "Run batch: transcripts"
        if st.button(batch_label, type="primary", disabled=not _keys_ready(batch_analyze)):
            progress = st.progress(0.0)
            status = st.empty()
            log_box = st.empty()
            live_table = st.empty()
            results, errors, log_lines = [], [], []

            def log(msg):
                log_lines.append(msg)
                log_box.code("\n".join(log_lines[-8:]), language=None)

            for i, row in enumerate(rows):
                sid = row.get("session_id") or row_key(row)
                status.write(f"Processing `{sid}` ({i + 1}/{len(rows)})...")
                try:
                    results.append(_run_session(row, log=log, force=force, force_reanalysis=reanalyze,
                                                analyze=batch_analyze))
                except Exception as e:
                    errors.append({"session_id": sid, "error": f"{type(e).__name__}: {e}"})
                    log(f"[{sid}] FAILED — {e}")
                progress.progress((i + 1) / len(rows))
                if results:
                    live_table.dataframe(_summary_df(results), use_container_width=True)
            status.write("Done.")
            st.session_state["batch_errors"] = errors
            st.rerun()
        _missing_key_hint(batch_analyze)

        # Always rebuilt from disk, so the table survives refreshes and
        # shows everything finished so far — even from an interrupted run.
        finished = [res for res in (session_store.load_result(row_key(r)) for r in rows) if res]
        errors = st.session_state.get("batch_errors", [])
        if finished:
            n_analyzed = sum(result_mode(r) == "analysis" for r in finished)
            st.success(f"{len(finished)}/{len(rows)} sessions from this CSV have a saved transcript"
                       + (f" ({n_analyzed} also analyzed)." if n_analyzed else "."))
            st.dataframe(_summary_df([summary_row(r) for r in finished]), use_container_width=True)
            _batch_downloads(finished, key_prefix="batch_")
        if errors:
            st.error(f"{len(errors)} failed in the last run — press the run button again to retry just those.")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

# ─── Saved results ─────────────────────────────────────────────────────────────

with tab_saved:
    st.subheader("All saved results")
    saved = session_store.list_results()

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Backup**")
        st.download_button("Download full backup (.zip)", session_store.export_zip(),
                           file_name=f"ta_results_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                           mime="application/zip", disabled=not saved,
                           help="Transcripts, analyses, reports and PDFs for every saved session.")
    with b2:
        st.markdown("**Restore**")
        backup = st.file_uploader("Upload a backup .zip", type=["zip"], key="restore_zip")
        if backup and st.button("Restore backup"):
            try:
                n = session_store.import_zip(backup.getvalue())
                st.success(f"Restored {n} sessions.")
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")

    if not saved:
        st.info("Nothing saved yet.")
    else:
        st.dataframe(_summary_df([summary_row(r) for r in saved]), use_container_width=True)
        _batch_downloads(saved, key_prefix="saved_")

        def _label(r):
            meta = r["session_meta"]
            tag = (f"{r['report']['score_breakdown']['overall']:.0f}/100"
                   if result_mode(r) == "analysis" else "transcript")
            return f"{meta['session_id']} — {meta.get('ta_name', '')} ({tag})"

        labels = {r["_key"]: _label(r) for r in saved}
        pick = st.selectbox("View a session", list(labels), format_func=labels.get)
        chosen = next(r for r in saved if r["_key"] == pick)
        _render_report(chosen["session_meta"], chosen["report"], key_prefix="saved_")

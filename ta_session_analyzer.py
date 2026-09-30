"""
TA Session Analyzer — daily batch runner
-----------------------------------------
Scores TA doubt-clearing session recordings against the quality rubric in the
TA Session Analyzer PRD (doubt resolution, teaching quality, direct-solution
detection, participation, communication, professionalism, technical accuracy,
session flow, dead air) and produces a per-session PDF/CSV/JSON report plus a
daily rollup — the same pattern as auto_lecture_analyzer.py.

Flow per row in the day's CSV:
  1. Resolve the direct video URL from the portal link (recording_utils.py —
     same `?url=` unwrapping used for lecture recordings)
  2. Download the recording to a temp file
  3. Transcribe + diarize with Deepgram (generic speaker labels), sample a
     few shared-screen frames, then let Gemini map speakers to TA/Student
     and run the full quality analysis (ta_core.py)
  4. Optionally sample screen-share frames from the same recording to detect
     directly-shared final solutions on screen
  5. Compute the weighted 0-100 score and AI flags
  6. Write per-session PDF (ta_pdf_report.py), CSV, and JSON
  7. Roll all of the day's sessions up into one combined CSV + JSON
  8. Delete the temp video (transcript, screenshots and analysis are KEPT
     in ta_reports/sessions/<session_id>/ — see session_store.py)

Resuming: every stage is saved as it finishes, so if a run dies at session
15, just run the same command again — the 15 finished sessions are skipped
(no API calls), and a half-done session resumes from its last stage
(e.g. transcript already saved -> only the Gemini call is made).

CSV input format (ta_sessions_today.csv):
    recording_url,ta_name,student_name,session_id,analyze_screen,chat_log_path
    https://my.newtonschool.co/play-video/?url=...,Rahul,Anita,sess_2026_07_09_01,yes,
    ...

    - analyze_screen: "yes" (default) or "no" — set "no" to skip screen-share
      sampling (e.g. audio-only calls) and save a few vision API calls.
    - chat_log_path: optional path to a plain-text chat export for this
      session. Leave blank if you don't have chat logs yet — chat analysis
      is skipped and simply won't appear in the report.

Requirements (on top of requirements.txt):
    google-genai, pydantic, requests, plus ffmpeg/ffprobe on PATH (duration,
    audio quality heuristic, and screen-share frame sampling — Deepgram reads
    the recording directly, no local audio extraction needed).

Run manually:
    GEMINI_API_KEY=your_gemini_key DEEPGRAM_API_KEY=your_deepgram_key \\
        python ta_session_analyzer.py ta_sessions_today.csv

    # Free local transcription (no Deepgram key needed) — see local_transcribe.py
    GEMINI_API_KEY=your_gemini_key \\
        python ta_session_analyzer.py ta_sessions_today.csv --transcriber local

    # Re-score already-transcribed sessions (e.g. after a prompt change) —
    # reuses saved transcripts + screenshots, only Gemini is called again
    python ta_session_analyzer.py ta_sessions_today.csv --reanalyze

Provider note: transcription/diarization runs on Deepgram (see ta_core.py)
so its free credits absorb the most expensive part of the pipeline, while
Gemini is only spent on the analysis call (rubric scoring + a handful of
screenshots) plus an optional small chat-analysis call — both comfortably
within Gemini's free tier for a 20-30 session test batch. Sessions are
processed one at a time with retries/backoff below to ride out transient
rate limits.
"""

import argparse
import csv
import json
import os
import time
import traceback
from datetime import datetime

import session_store
from recording_utils import extract_video_url, download_video
from ta_core import analyze_ta_session
from ta_pdf_report import generate_ta_pdf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # never hardcode this
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")  # never hardcode this
OUTPUT_DIR = "ta_reports"
MAX_RETRIES = 3
# Free-tier Gemini rate limits are per-minute, so back off in tens of
# seconds rather than 2-4s. Only the failed stage is retried — finished
# stages are loaded from disk.
RETRY_BASE_WAIT_SECONDS = 15


def row_key(row: dict) -> str:
    return session_store.session_key(row.get("session_id", ""), row.get("recording_url", ""))


def summary_row(result: dict) -> dict:
    """Flatten a saved result.json into one rollup/table row."""
    meta, report = result["session_meta"], result["report"]
    return {
        "session_id": meta["session_id"],
        "ta_name": meta.get("ta_name", ""),
        "student_name": meta.get("student_name", ""),
        "date": result.get("date", ""),
        "duration_minutes": report["duration_minutes"],
        "overall_score": report["score_breakdown"]["overall"],
        "doubt_resolution": report["analysis"]["doubt_resolution"]["status"],
        "ta_speaking_pct": report["participation"]["ta_pct"],
        "student_speaking_pct": report["participation"]["student_pct"],
        "transcriber": report.get("transcriber", "deepgram"),
        "flags": report["flags"],
        "pdf_path": result.get("pdf_path"),
    }


# ---------------------------------------------------------------------------
# Process one session end-to-end
# ---------------------------------------------------------------------------

def process_session(row: dict, run_date: str, api_key: str = None, deepgram_api_key: str = None,
                    transcriber: str = "deepgram", force: bool = False,
                    force_reanalysis: bool = False, log=print, local_video_path: str = None) -> dict:
    """Analyze one CSV row, resuming from whatever is already saved for it.
    Returns the rollup summary row. A session whose result.json exists is
    returned straight from disk with no API calls, unless force (redo
    everything) or force_reanalysis (reuse transcript, re-run Gemini).
    local_video_path skips the download (e.g. a file uploaded in the UI)."""
    api_key = api_key or GEMINI_API_KEY
    deepgram_api_key = deepgram_api_key or DEEPGRAM_API_KEY
    key = row_key(row)
    session_id = (row.get("session_id") or "").strip() or key
    ta_name = row.get("ta_name", "")
    student_name = row.get("student_name", "")
    analyze_screen = (row.get("analyze_screen") or "yes").strip().lower() != "no"
    chat_log_path = (row.get("chat_log_path") or "").strip()

    def _log(msg):
        log(f"[{session_id}] {msg}")

    folder = session_store.session_dir(key)
    if force:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                os.remove(path)
        frames_marker = os.path.join(folder, session_store.FRAMES_DIR, "frames.json")
        if os.path.exists(frames_marker):
            os.remove(frames_marker)

    existing = session_store.load_result(key)
    if existing and not force_reanalysis:
        _log("already analyzed — loaded from disk (no API calls)")
        return summary_row(existing)

    _log("starting...")

    chat_text = None
    if chat_log_path and os.path.exists(chat_log_path):
        with open(chat_log_path, encoding="utf-8") as f:
            chat_text = f.read()

    video_url = local_video_path or extract_video_url(row.get("recording_url", ""))
    if not video_url:
        raise ValueError(f"Could not resolve a video URL from: {row.get('recording_url', '')!r}")

    tmp_video = os.path.join(OUTPUT_DIR, ".tmp", f"{key}.mp4")
    os.makedirs(os.path.dirname(tmp_video), exist_ok=True)

    def fetch_video() -> str:
        if local_video_path:
            return local_video_path
        if not os.path.exists(tmp_video):
            _log("downloading...")
            download_video(video_url, tmp_video + ".part")
            os.replace(tmp_video + ".part", tmp_video)
        return tmp_video

    try:
        report = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                _log(f"analyzing (attempt {attempt})...")
                report = analyze_ta_session(
                    api_key, fetch_video, deepgram_api_key,
                    analyze_screen=analyze_screen, chat_text=chat_text,
                    cache_dir=folder, transcriber=transcriber,
                    force_reanalysis=force_reanalysis, log=_log,
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_BASE_WAIT_SECONDS * attempt
                _log(f"analysis failed ({type(e).__name__}: {e}), retrying in {wait}s...")
                time.sleep(wait)
    finally:
        for leftover in (tmp_video, tmp_video + ".part"):
            if os.path.exists(leftover):
                os.remove(leftover)

    session_meta = {"session_id": session_id, "ta_name": ta_name, "student_name": student_name}

    pdf_path = os.path.join(folder, session_store.PDF_FILE)
    try:
        session_store.save_bytes(folder, session_store.PDF_FILE, generate_ta_pdf(session_meta, report))
    except Exception as e:
        _log(f"PDF generation failed: {e}")
        pdf_path = None

    result = {"session_meta": session_meta, "date": run_date, "report": report, "pdf_path": pdf_path}
    # result.json is written last — its existence is what marks the session done.
    session_store.save_json(folder, session_store.RESULT_FILE, result)

    overall = report["score_breakdown"]["overall"]
    _log(f"done — overall score {overall}/100, flags: {report['flags'] or 'none'}")
    return summary_row(result)


# ---------------------------------------------------------------------------
# Run the whole day's CSV
# ---------------------------------------------------------------------------

def write_rollup(reports: list, errors: list, run_date: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"report_{run_date}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"reports": reports, "errors": errors}, f, indent=2, ensure_ascii=False)

    out_csv = os.path.join(OUTPUT_DIR, f"report_{run_date}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_id", "ta_name", "student_name", "duration_minutes",
            "overall_score", "doubt_resolution", "ta_speaking_pct",
            "student_speaking_pct", "transcriber", "flags",
        ])
        for r in reports:
            writer.writerow([
                r["session_id"], r["ta_name"], r["student_name"], r["duration_minutes"],
                r["overall_score"], r["doubt_resolution"], r["ta_speaking_pct"],
                r["student_speaking_pct"], r["transcriber"], "; ".join(r["flags"]) or "none",
            ])
    return out_csv


def run_daily_batch(csv_path: str, transcriber: str = "deepgram", force: bool = False,
                    force_reanalysis: bool = False):
    if not GEMINI_API_KEY:
        raise EnvironmentError("Set GEMINI_API_KEY as an environment variable before running.")
    if transcriber == "deepgram" and not DEEPGRAM_API_KEY:
        raise EnvironmentError("Set DEEPGRAM_API_KEY, or pass --transcriber local for free local Whisper.")

    run_date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reports = []
    errors = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, 1):
        print(f"\n── {i}/{len(rows)} ──")
        try:
            reports.append(process_session(row, run_date, transcriber=transcriber, force=force,
                                           force_reanalysis=force_reanalysis))
        except Exception as e:
            err = f"{row.get('session_id', '?')}: {type(e).__name__}: {e}"
            print(f"  FAILED — {err}\n{traceback.format_exc()}")
            errors.append(err)
        # Refresh the rollup after every session so it's always current,
        # even if the run is killed partway through.
        write_rollup(reports, errors, run_date)

    flagged = [r for r in reports if r["flags"]]
    print(f"\nDone. {len(reports)} succeeded, {len(errors)} failed, {len(flagged)} flagged for manual review.")
    if errors:
        print("Re-run the same command to retry the failed ones — finished sessions are skipped.")
    print(f"Per-session results in {session_store.STORE_DIR}/, rollup in {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a CSV of TA session recordings (resumable).")
    parser.add_argument("csv", nargs="?", default="ta_sessions_today.csv")
    parser.add_argument("--transcriber", choices=["deepgram", "local"], default="deepgram",
                        help="deepgram (API) or local (free Whisper on this machine)")
    parser.add_argument("--reanalyze", action="store_true",
                        help="re-run Gemini scoring, reusing saved transcripts/screenshots")
    parser.add_argument("--force", action="store_true",
                        help="ignore everything saved and redo each session from scratch")
    args = parser.parse_args()
    run_daily_batch(args.csv, transcriber=args.transcriber, force=args.force,
                    force_reanalysis=args.reanalyze)


# ---------------------------------------------------------------------------
# Scheduling — run this automatically every day, for free
# ---------------------------------------------------------------------------
#
# Linux/Mac (cron) — runs every day at 8 PM:
#   crontab -e
#   0 20 * * * cd "/path/to/Lecture analyzer" && GEMINI_API_KEY=xxx DEEPGRAM_API_KEY=yyy /usr/bin/python3 ta_session_analyzer.py ta_sessions_today.csv >> logs/ta_run.log 2>&1
#
# Windows (Task Scheduler):
#   1. Task Scheduler > Create Basic Task > Daily, pick a time
#   2. Action: Start a program
#      Program: python.exe
#      Arguments: ta_session_analyzer.py ta_sessions_today.csv
#      Start in: the lecture-analyzer folder
#   3. Set GEMINI_API_KEY and DEEPGRAM_API_KEY as permanent environment
#      variables so the scheduled task can see them.
#
# Either way, ta_sessions_today.csv needs to be updated with that day's TA
# session recording links before the scheduled time runs.

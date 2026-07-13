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
  3. Extract audio -> transcribe with Groq Whisper -> label TA vs Student
     turns with an LLM -> run the full quality analysis (ta_core.py)
  4. Optionally sample screen-share frames from the same recording to detect
     directly-shared final solutions on screen
  5. Compute the weighted 0-100 score and AI flags
  6. Write per-session PDF (ta_pdf_report.py), CSV, and JSON
  7. Roll all of the day's sessions up into one combined CSV + JSON
  8. Delete all temp video/audio/frame files

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
    Same as auto_lecture_analyzer.py, plus ffmpeg on PATH for audio extraction.

Run manually:
    GROQ_API_KEY=your_key python ta_session_analyzer.py ta_sessions_today.csv

Free-tier note: this whole pipeline runs entirely on Groq (Whisper for
transcription, Llama for labeling/analysis/vision) so a 20-30 session test
batch stays within the free tier's request limits. Sessions are processed
one at a time with retries/backoff below to ride out transient rate limits.
"""

import os
import csv
import json
import time
import traceback
from datetime import datetime

from recording_utils import extract_video_url, download_video
from ta_core import analyze_ta_session
from ta_pdf_report import generate_ta_pdf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # never hardcode this
OUTPUT_DIR = "ta_reports"
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Process one session end-to-end
# ---------------------------------------------------------------------------

def process_session(row: dict, run_date: str) -> dict:
    session_id = row["session_id"]
    ta_name = row.get("ta_name", "")
    student_name = row.get("student_name", "")
    analyze_screen = row.get("analyze_screen", "yes").strip().lower() != "no"
    chat_log_path = (row.get("chat_log_path") or "").strip()

    print(f"\n[{session_id}] starting...")

    chat_text = None
    if chat_log_path and os.path.exists(chat_log_path):
        with open(chat_log_path, encoding="utf-8") as f:
            chat_text = f.read()

    video_url = extract_video_url(row["recording_url"])
    if not video_url:
        raise ValueError(f"Could not resolve a video URL from: {row['recording_url']}")

    tmp_video = os.path.join(OUTPUT_DIR, ".tmp", f"{session_id}.mp4")
    os.makedirs(os.path.dirname(tmp_video), exist_ok=True)
    try:
        print(f"[{session_id}] downloading...")
        download_video(video_url, tmp_video)

        last_error = None
        report = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[{session_id}] analyzing (attempt {attempt})...")
                report = analyze_ta_session(
                    GROQ_API_KEY, tmp_video,
                    analyze_screen=analyze_screen, chat_text=chat_text,
                )
                break
            except Exception as e:
                last_error = e
                if attempt == MAX_RETRIES:
                    raise
                wait = 2 ** attempt
                print(f"[{session_id}] analysis failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)

    session_meta = {"session_id": session_id, "ta_name": ta_name, "student_name": student_name}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = f"{ta_name}_{session_id}".replace(" ", "-").replace("/", "-")

    try:
        pdf_bytes = generate_ta_pdf(session_meta, report)
        with open(os.path.join(OUTPUT_DIR, f"{slug}.pdf"), "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        print(f"[{session_id}] PDF generation failed: {e}")

    with open(os.path.join(OUTPUT_DIR, f"{slug}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "session_meta": session_meta,
            "date": run_date,
            "report": report,
        }, f, indent=2, ensure_ascii=False)

    overall = report["score_breakdown"]["overall"]
    print(f"[{session_id}] done — overall score {overall}/100, flags: {report['flags'] or 'none'}")

    return {
        "session_id": session_id,
        "ta_name": ta_name,
        "student_name": student_name,
        "date": run_date,
        "duration_minutes": report["duration_minutes"],
        "overall_score": overall,
        "doubt_resolution": report["analysis"]["doubt_resolution"]["status"],
        "ta_speaking_pct": report["participation"]["ta_pct"],
        "student_speaking_pct": report["participation"]["student_pct"],
        "flags": report["flags"],
    }


# ---------------------------------------------------------------------------
# Run the whole day's CSV
# ---------------------------------------------------------------------------

def run_daily_batch(csv_path: str):
    if not GROQ_API_KEY:
        raise EnvironmentError("Set GROQ_API_KEY as an environment variable before running.")

    run_date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reports = []
    errors = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        try:
            reports.append(process_session(row, run_date))
        except Exception as e:
            err = f"{row.get('session_id', '?')}: {type(e).__name__}: {e}"
            print(f"  FAILED — {err}\n{traceback.format_exc()}")
            errors.append(err)

    out_json = os.path.join(OUTPUT_DIR, f"report_{run_date}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"reports": reports, "errors": errors}, f, indent=2, ensure_ascii=False)

    out_csv = os.path.join(OUTPUT_DIR, f"report_{run_date}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_id", "ta_name", "student_name", "duration_minutes",
            "overall_score", "doubt_resolution", "ta_speaking_pct",
            "student_speaking_pct", "flags",
        ])
        for r in reports:
            writer.writerow([
                r["session_id"], r["ta_name"], r["student_name"], r["duration_minutes"],
                r["overall_score"], r["doubt_resolution"], r["ta_speaking_pct"],
                r["student_speaking_pct"], "; ".join(r["flags"]) or "none",
            ])

    flagged = [r for r in reports if r["flags"]]
    print(f"\nDone. {len(reports)} succeeded, {len(errors)} failed, {len(flagged)} flagged for manual review.")
    print(f"Per-session PDF/JSON + rollup written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "ta_sessions_today.csv"
    run_daily_batch(csv_arg)


# ---------------------------------------------------------------------------
# Scheduling — run this automatically every day, for free
# ---------------------------------------------------------------------------
#
# Linux/Mac (cron) — runs every day at 8 PM:
#   crontab -e
#   0 20 * * * cd "/path/to/Lecture analyzer" && GROQ_API_KEY=xxx /usr/bin/python3 ta_session_analyzer.py ta_sessions_today.csv >> logs/ta_run.log 2>&1
#
# Windows (Task Scheduler):
#   1. Task Scheduler > Create Basic Task > Daily, pick a time
#   2. Action: Start a program
#      Program: python.exe
#      Arguments: ta_session_analyzer.py ta_sessions_today.csv
#      Start in: the lecture-analyzer folder
#   3. Set GROQ_API_KEY as a permanent environment variable so the scheduled
#      task can see it.
#
# Either way, ta_sessions_today.csv needs to be updated with that day's TA
# session recording links before the scheduled time runs.

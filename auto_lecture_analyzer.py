"""
Auto Lecture Analyzer — daily batch runner
--------------------------------------------
Extends the lecture-analyzer app so you don't have to take screenshots manually.

Flow per row in the day's CSV:
  1. Resolve the direct video URL from the portal link
  2. Download the recording to a temp file
  3. Extract N evenly-spaced frames with OpenCV (free, local, no API cost)
  4. Score each frame using core.analyze_image() — the same function app.py and
     analyze.py use, so results are identical in shape (scores / observation /
     improvement / overall_score / priority_fix per frame)
  5. Aggregate into one report per session — PDF (via pdf_report.generate_pdf,
     the exact same "Report Card" the Streamlit app produces), CSV, and JSON
  6. Roll all of the day's sessions up into one combined CSV + JSON
  7. Delete all temp video/frame files

CSV input format (recordings_today.csv):
    recording_url,batch,module,session_id
    https://portal.example/watch?url=...,B12,DSA,sess_2026_07_09_01
    ...

Requirements (on top of requirements.txt):
    pip install opencv-python-headless requests

Run manually:
    GEMINI_API_KEY=your_key python auto_lecture_analyzer.py recordings_today.csv

Automate daily (see bottom of file for scheduling notes).
"""

import os
import csv
import json
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import cv2

from core import (
    PARAMETERS, PARAMETER_LABELS,
    load_image_file, analyze_image,
    build_csv_rows, aggregate_scores,
)
from pdf_report import generate_pdf
from recording_utils import extract_video_url, download_video

# ---------------------------------------------------------------------------
# CONFIG — adjust these to match your setup
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # never hardcode this
FRAMES_PER_VIDEO = 8          # app caps uploads at 10 screenshots; 8 stays under that
SKIP_INTRO_OUTRO_PCT = 0.03   # skip first/last 3% of the video (title/goodbye slides)
OUTPUT_DIR = "reports"
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# 1. Resolve + download the recording (see recording_utils.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. Extract evenly-spaced frames with OpenCV (free — no ffmpeg dependency)
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, out_dir: str, n_frames: int = FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no readable frames (corrupt download?).")

    start = int(total_frames * SKIP_INTRO_OUTRO_PCT)
    end = int(total_frames * (1 - SKIP_INTRO_OUTRO_PCT))
    span = max(end - start, 1)

    frame_paths = []
    for i in range(n_frames):
        target_frame = start + int(span * (i / max(n_frames - 1, 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()
    if not frame_paths:
        raise RuntimeError("Frame extraction produced nothing usable.")
    return frame_paths


# ---------------------------------------------------------------------------
# 3. Score frames — reuses core.analyze_image() exactly as app.py/analyze.py do
# ---------------------------------------------------------------------------

def score_frame(image_path: str, session_id: str, frame_index: int, api_key: str) -> dict:
    image_data, media_type = load_image_file(Path(image_path))

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = analyze_image(api_key, image_data, media_type)
            result["screenshot"] = f"{session_id}_frame_{frame_index:02d}.jpg"
            result["analyzed_at"] = datetime.now().isoformat()
            return result
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait = 2 ** attempt
            print(f"  scoring failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise last_error


# ---------------------------------------------------------------------------
# 4. Process one session end-to-end
# ---------------------------------------------------------------------------

def process_session(row: dict, run_date: str, api_key: str = None) -> dict:
    api_key = api_key or GEMINI_API_KEY
    session_id = row["session_id"]
    batch = row.get("batch", "")
    module = row.get("module", "")
    print(f"\n[{session_id}] starting...")

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, f"{session_id}.mp4")
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        video_url = extract_video_url(row["recording_url"])
        if not video_url:
            raise ValueError(f"Could not resolve a video URL from: {row['recording_url']}")

        print(f"[{session_id}] downloading...")
        download_video(video_url, video_path)

        print(f"[{session_id}] extracting frames...")
        frame_paths = extract_frames(video_path, frames_dir)

        print(f"[{session_id}] scoring {len(frame_paths)} frames...")
        frame_results = [
            score_frame(fp, session_id, i, api_key) for i, fp in enumerate(frame_paths)
        ]

        # Session-level aggregation, same logic app.py uses for the Report Card
        averages = aggregate_scores(frame_results)
        overall_scores = [
            r["overall_score"] for r in frame_results
            if isinstance(r.get("overall_score"), (int, float))
        ]
        overall = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else None
        flagged = {
            PARAMETER_LABELS[p]: round(v, 2)
            for p, v in averages.items() if v is not None and v < 3.5
        }

        # Per-session PDF / CSV / JSON, written straight to OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        slug = f"{batch}_{module}_{session_id}".replace(" ", "-").replace("/", "-")

        pdf_path = os.path.join(OUTPUT_DIR, f"{slug}.pdf")
        try:
            pdf_bytes = generate_pdf(batch, module, frame_results)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
        except Exception as e:
            print(f"[{session_id}] PDF generation failed: {e}")
            pdf_path = None

        csv_rows = build_csv_rows(batch, module, frame_results)
        with open(os.path.join(OUTPUT_DIR, f"{slug}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        with open(os.path.join(OUTPUT_DIR, f"{slug}.json"), "w") as f:
            json.dump({
                "batch_name": batch, "lecture_module": module,
                "session_id": session_id, "analyzed_at": run_date,
                "results": frame_results,
            }, f, indent=2, ensure_ascii=False)

        print(f"[{session_id}] done — overall score {overall}")
        return {
            "session_id": session_id,
            "batch": batch,
            "module": module,
            "date": run_date,
            "frame_count": len(frame_paths),
            "overall_score": overall,
            "flagged_below_3_5": flagged,
            "pdf_path": pdf_path,
        }
        # tmp dir (video + frames) is auto-deleted on exit


# ---------------------------------------------------------------------------
# 5. Run the whole day's CSV
# ---------------------------------------------------------------------------

def run_daily_batch(csv_path: str):
    if not GEMINI_API_KEY:
        raise EnvironmentError("Set GEMINI_API_KEY as an environment variable before running.")

    run_date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reports = []
    errors = []

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        try:
            reports.append(process_session(row, run_date))
        except Exception as e:
            err = f"{row.get('session_id', '?')}: {type(e).__name__}: {e}"
            print(f"  FAILED — {err}\n{traceback.format_exc()}")
            errors.append(err)

    # write day's combined rollup
    out_json = os.path.join(OUTPUT_DIR, f"report_{run_date}.json")
    with open(out_json, "w") as f:
        json.dump({"reports": reports, "errors": errors}, f, indent=2)

    out_csv = os.path.join(OUTPUT_DIR, f"report_{run_date}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "batch", "module", "overall_score", "flagged_params"])
        for r in reports:
            writer.writerow([
                r["session_id"], r["batch"], r["module"], r["overall_score"],
                "; ".join(r["flagged_below_3_5"].keys()) or "none",
            ])

    print(f"\nDone. {len(reports)} succeeded, {len(errors)} failed.")
    print(f"Per-session PDF/CSV/JSON + rollup written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "recordings_today.csv"
    run_daily_batch(csv_arg)


# ---------------------------------------------------------------------------
# Scheduling — run this automatically every day, for free
# ---------------------------------------------------------------------------
#
# Linux/Mac (cron) — runs every day at 8 PM:
#   crontab -e
#   0 20 * * * cd "/path/to/Lecture analyzer" && GEMINI_API_KEY=xxx /usr/bin/python3 auto_lecture_analyzer.py recordings_today.csv >> logs/run.log 2>&1
#
# Windows (Task Scheduler):
#   1. Task Scheduler > Create Basic Task > Daily, pick a time
#   2. Action: Start a program
#      Program: python.exe
#      Arguments: auto_lecture_analyzer.py recordings_today.csv
#      Start in: the lecture-analyzer folder
#   3. Set GEMINI_API_KEY as a permanent environment variable (System Properties >
#      Environment Variables) so the scheduled task can see it.
#
# Whatever you automate this with, remember why the old cron pipeline in the
# separate lecture-analyzer repo got retired: unattended runs with no
# per-session checkpointing can lose a whole batch to one stuck call. Prefer
# something that watches progress (like the Batch tab in app.py) over a truly
# unattended job, or add incremental writeback if you do automate it.
#
# Either way, recordings_today.csv needs to be updated with that day's
# recording links before the scheduled time runs (e.g. export it from your
# portal, or write a small script that appends to it after each class).

"""
On-disk store that makes TA session analysis resumable.

Every session gets its own folder, and every pipeline stage writes its
output there the moment it finishes:

    ta_reports/sessions/<session_key>/
        media.json           duration + audio-volume heuristic
        frames/              sampled shared-screen screenshots (+ frames.json)
        transcript.json      Deepgram or local-Whisper transcript  <- the costly bit
        transcript.txt       readable transcript (always written)
        analysis.json        raw Gemini rubric output (+ the inputs it used) — analysis mode only
        chat_analysis.json   optional chat-log analysis — analysis mode only
        result.json          final report + session meta  <- "this one is done"
                             (report.mode is "transcript" or "analysis")
        report.pdf           analysis mode only

So if a batch dies after 15 sessions (a page refresh, a crash, a rate
limit), re-running the same CSV skips those 15 entirely, and any session
that died halfway resumes from its last finished stage.

Streamlit Cloud's disk is wiped when the app reboots or goes to sleep, so
the UI also offers export_zip()/import_zip(): download a backup, and upload
it again to restore everything.

Writes are atomic (temp file + rename), so a crash mid-write can never leave
a half-written JSON behind that later looks "done".
"""

import hashlib
import io
import json
import os
import re
import zipfile
from typing import List, Optional

STORE_DIR = os.path.join("ta_reports", "sessions")
RESULT_FILE = "result.json"
PDF_FILE = "report.pdf"
TRANSCRIPT_TXT_FILE = "transcript.txt"
FRAMES_DIR = "frames"


def session_key(session_id: str = "", recording_url: str = "") -> str:
    """Filesystem-safe folder name for a session. Falls back to a hash of the
    recording URL when the CSV row has no session_id, so the same recording
    still maps to the same folder on every run."""
    session_id = (session_id or "").strip()
    if session_id:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", session_id).strip("-") or "session"
    return "url-" + hashlib.sha1((recording_url or "").encode("utf-8")).hexdigest()[:12]


def session_dir(key: str, root: str = STORE_DIR) -> str:
    path = os.path.join(root, key)
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_write(path: str, data: bytes):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def save_json(folder: str, name: str, data) -> None:
    os.makedirs(folder, exist_ok=True)
    _atomic_write(os.path.join(folder, name),
                  json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))


def load_json(folder: str, name: str):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupt/partial file — treat the stage as not done


def save_bytes(folder: str, name: str, data: bytes) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    _atomic_write(path, data)
    return path


def save_frames(folder: str, frames: List[bytes]) -> None:
    frames_dir = os.path.join(folder, FRAMES_DIR)
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        _atomic_write(os.path.join(frames_dir, f"frame_{i:02d}.jpg"), frame)
    # Written last: its presence is what marks the frames stage as complete.
    save_json(frames_dir, "frames.json", {"count": len(frames)})


def load_frames(folder: str) -> Optional[List[bytes]]:
    frames_dir = os.path.join(folder, FRAMES_DIR)
    marker = load_json(frames_dir, "frames.json")
    if marker is None:
        return None
    frames = []
    for i in range(marker["count"]):
        path = os.path.join(frames_dir, f"frame_{i:02d}.jpg")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            frames.append(f.read())
    return frames


def load_result(key: str, root: str = STORE_DIR) -> Optional[dict]:
    folder = os.path.join(root, key)
    return load_json(folder, RESULT_FILE) if os.path.isdir(folder) else None


def has_transcript(key: str, root: str = STORE_DIR) -> bool:
    return load_json(os.path.join(root, key), "transcript.json") is not None


def list_results(root: str = STORE_DIR) -> List[dict]:
    """Every finished session in the store, newest first."""
    if not os.path.isdir(root):
        return []
    out = []
    for key in os.listdir(root):
        result = load_result(key, root)
        if result is not None:
            result["_key"] = key
            result["_mtime"] = os.path.getmtime(os.path.join(root, key, RESULT_FILE))
            out.append(result)
    return sorted(out, key=lambda r: r["_mtime"], reverse=True)


def export_zip(root: str = STORE_DIR) -> bytes:
    """Zip the whole store — transcripts, analyses, reports, PDFs — for backup."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(root):
            for dirpath, _dirs, files in os.walk(root):
                for name in files:
                    if name.endswith(".tmp"):
                        continue
                    full = os.path.join(dirpath, name)
                    zf.write(full, arcname=os.path.relpath(full, root))
    return buf.getvalue()


def import_zip(data: bytes, root: str = STORE_DIR) -> int:
    """Restore a backup made by export_zip. Existing files are overwritten
    with the backup's copy. Returns the number of sessions in the backup."""
    os.makedirs(root, exist_ok=True)
    root_abs = os.path.abspath(root)
    keys = set()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = os.path.abspath(os.path.join(root_abs, member.filename))
            if not target.startswith(root_abs + os.sep):
                continue  # refuse path traversal ("../") entries
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member) as src:
                _atomic_write(target, src.read())
            keys.add(member.filename.replace("\\", "/").split("/")[0])
    return len(keys)

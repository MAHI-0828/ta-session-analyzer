"""
Shared analysis logic — used by both the CLI (analyze.py) and the Streamlit app (app.py).
"""

import base64
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image

GEMINI_MODEL    = "gemini-2.5-flash-lite"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Free tier is nominally 15 requests/minute, but a burst of test traffic can
# apparently get throttled harder than that — pace conservatively and give
# retries a much longer runway before giving up on a frame.
_MIN_GEMINI_INTERVAL = 8.0
_last_gemini_call = 0.0
_MAX_RETRIES = 8

# ─── Parameter registry ─────────────────────────────────────────────────────────

MAJOR_PARAMETERS = [
    "camera_angle",
    "face_lighting",
    "background_quality",
    "appearance",
    "content_type",
    "chat_panel_visible",
]

MINOR_PARAMETERS = [
    "instructor_energy",
    "screen_share_active",
    "slide_text_density",
    "visual_first_design",
    "code_readability",
    "annotation_activity",
    "desktop_notifications",
    "dual_screen_evidence",
]

PARAMETERS = MAJOR_PARAMETERS + MINOR_PARAMETERS

PARAMETER_LABELS = {
    "camera_angle":          "Camera Angle",
    "face_lighting":         "Face Lighting",
    "background_quality":    "Background Quality",
    "appearance":            "Appearance & Clothing",
    "instructor_energy":     "Instructor Energy & Posture",
    "screen_share_active":   "Screen Share Active",
    "content_type":          "Content Type on Screen",
    "slide_text_density":    "Slide Text Density",
    "visual_first_design":   "Visual-First Slide Design",
    "code_readability":      "Code Readability",
    "annotation_activity":   "Whiteboard Detected",
    "chat_panel_visible":    "Chat Panel Visible",
    "desktop_notifications": "Desktop Notifications",
    "dual_screen_evidence":  "Dual Screen Evidence",
}

PARAMETER_DESCRIPTIONS = {
    "camera_angle": (
        "Checks if the camera is at eye level and the instructor is centered in frame. "
        "A good angle looks natural and professional. "
        "Extreme angles (too high, too low, off to the side) appear unprofessional and distracting."
    ),
    "face_lighting": (
        "Evaluates how evenly the instructor's face is lit. "
        "Front-facing, even light makes the instructor clearly visible. "
        "Heavy shadows, backlighting from a window, or harsh overhead light reduce visibility and feel."
    ),
    "background_quality": (
        "Assesses what appears behind the instructor. "
        "A plain, neutral wall is ideal and keeps focus on the instructor. "
        "Cluttered shelves, open doors, or bright windows behind are distracting."
    ),
    "appearance": (
        "Reviews the instructor's clothing and presentation. "
        "Solid-color, professional attire photographs well on camera. "
        "Busy patterns like checks or stripes can cause visual flickering (moiré effect) on screen."
    ),
    "content_type": (
        "Identifies what is actively shown — live coding, diagrams, or static slides. "
        "Live coding and interactive content signal active teaching and score higher. "
        "A black screen or irrelevant desktop means students have nothing to follow."
    ),
    "chat_panel_visible": (
        "Checks whether the instructor can see and monitor student chat. "
        "Chat should be visible on screen or on a separate device during the session. "
        "Missing chat access means student questions may go unnoticed."
    ),
    "instructor_energy": (
        "Reads posture, expression, and engagement level from the instructor's body language. "
        "An upright, smiling, and expressive instructor holds learner attention better. "
        "Slouching or a blank expression signals disengagement and affects class energy."
    ),
    "screen_share_active": (
        "Confirms the instructor is actively sharing relevant content on screen. "
        "A correctly shared screen with teaching material is the baseline expectation. "
        "A blank, black, or generic desktop means the share is missing or inactive."
    ),
    "slide_text_density": (
        "Measures how much text is on a single slide. "
        "Under 30 words per slide keeps content digestible and lets the instructor explain. "
        "Walls of text compete with the spoken explanation and are hard to read from small screens."
    ),
    "visual_first_design": (
        "Checks if slides use visuals — diagrams, charts, icons — rather than just text. "
        "Visual content aids retention and makes abstract concepts concrete. "
        "Text-only slides are harder to follow during a live session."
    ),
    "code_readability": (
        "Evaluates whether code on screen uses a large monospace font with syntax highlighting. "
        "Well-formatted code is readable even on smaller student screens. "
        "Small or unformatted code forces students to strain and miss details."
    ),
    "annotation_activity": (
        "Detects whether a whiteboard or annotation tool is active in the screenshot. "
        "Drawing, underlining, or circling key concepts makes explanations clearer. "
        "No annotation during conceptual explanations is a missed teaching opportunity."
    ),
    "desktop_notifications": (
        "Checks if notification banners are visible on the shared screen. "
        "Notifications are distracting and may expose private messages to the class. "
        "All alerts should be silenced before going live."
    ),
    "dual_screen_evidence": (
        "Looks for signs the instructor uses a second monitor — visible in the webcam or suggested by eye movement. "
        "A second screen lets the instructor view notes or chat without covering the main display. "
        "Single-screen setups force awkward window juggling during the session."
    ),
}

# ─── System prompt ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert instructional quality analyst reviewing screenshots from live online lectures.

For each screenshot, evaluate the following 14 parameters. For each parameter return:
- score: integer 1–5, or null if the parameter is not applicable to this screenshot
- observation: exactly 2 sentences describing what you can literally see in the screenshot (be specific)
- improvement: exactly 2 sentences of actionable advice for the instructor

If a parameter is not applicable (e.g., code_readability when no code is on screen), set score to null and write "Not applicable in this screenshot." for both observation and improvement.

Return ONLY valid JSON with no markdown, no code fences, no extra text:
{
  "scores": {
    "camera_angle":          { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "face_lighting":         { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "background_quality":    { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "appearance":            { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "instructor_energy":     { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "screen_share_active":   { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "content_type":          { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "slide_text_density":    { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "visual_first_design":   { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "code_readability":      { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "annotation_activity":   { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "chat_panel_visible":    { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "desktop_notifications": { "score": <1-5|null>, "observation": "...", "improvement": "..." },
    "dual_screen_evidence":  { "score": <1-5|null>, "observation": "...", "improvement": "..." }
  },
  "overall_score": <average of all non-null scores, rounded to 1 decimal>,
  "priority_fix": "<single most impactful improvement the instructor should make, 1-2 sentences>"
}

Scoring rubrics:
- camera_angle:          5=perfect eye level, centered; 3=slightly off; 1=extreme angle (nostrils/ceiling/far off-axis)
- face_lighting:         5=evenly lit from front, no shadows; 3=mild one-side shadowing; 1=face mostly dark or washed out
- background_quality:    5=plain neutral wall, zero distractions; 3=minor clutter; 1=very cluttered or bright backlit window
- appearance:            5=professional solid-color attire; 3=casual but acceptable; 1=unprofessional or heavy moiré pattern
- instructor_energy:     5=upright, smiling, visibly engaged; 3=neutral posture/expression; 1=slouched, blank, clearly disengaged
- screen_share_active:   5=relevant content fully shared; 3=screen shared but generic desktop; 1=black/blank or no share
- content_type:          5=live coding/whiteboard/active annotation; 3=static slides; 1=black screen or irrelevant content
- slide_text_density:    5=under 30 words, concise bullets; 3=45-70 words, somewhat crowded; 1=wall of text over 100 words
- visual_first_design:   5=primarily diagrams/infographics; 3=mixed text and visuals; 1=entirely text-based
- code_readability:      5=monospace font, syntax highlighting, large text (18pt+); 3=code visible but small/unformatted; 1=unreadable
- annotation_activity:   5=whiteboard/annotation tool clearly active with drawings visible; 3=annotation surface open but sparse use; 1=no whiteboard or annotation detected
- chat_panel_visible:    5=chat clearly in layout or on separate device; 3=ambiguous; 1=no evidence of chat monitoring
- desktop_notifications: 5=no notifications visible; 3=minor badge only; 1=multiple notification banners on shared screen
- dual_screen_evidence:  5=second monitor visible in webcam frame; 3=natural eye-shift suggests side screen; 1=single screen, overlapping windows"""


# ─── Image helpers ───────────────────────────────────────────────────────────────

def load_image_file(image_path: Path) -> tuple:
    """Return (base64_string, media_type) for an image file."""
    suffix = image_path.suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def load_image_bytes(file_bytes: bytes, filename: str = "image.png") -> tuple:
    """Return (base64_string, media_type) for raw bytes."""
    suffix = Path(filename).suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    data = base64.standard_b64encode(file_bytes).decode("utf-8")
    return data, media_type


# ─── Analysis ────────────────────────────────────────────────────────────────────

def analyze_image(api_key: str, image_data: str, media_type: str) -> dict:
    """Send an image to Gemini (Google AI Studio) and return the structured JSON result."""
    global _last_gemini_call

    for attempt in range(_MAX_RETRIES):
        wait = _MIN_GEMINI_INTERVAL - (time.time() - _last_gemini_call)
        if wait > 0:
            time.sleep(wait)
        _last_gemini_call = time.time()

        response = requests.post(
            GEMINI_ENDPOINT,
            params={"key": api_key},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": media_type, "data": image_data}},
                        {"text": "Analyze this lecture screenshot and return the JSON evaluation."},
                    ],
                }],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=60,
        )

        if response.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
            retry_after = response.headers.get("Retry-After")
            backoff = float(retry_after) if retry_after else min((2 ** attempt) * 3, 60)
            time.sleep(backoff)
            continue

        response.raise_for_status()
        raw = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)


# ─── CSV helpers ─────────────────────────────────────────────────────────────────

def build_csv_rows(batch: str, module: str, results: list) -> list:
    """Flatten results into one dict per screenshot for CSV output."""
    rows = []
    for r in results:
        row = {
            "batch_name":     batch,
            "lecture_module": module,
            "screenshot":     r["screenshot"],
            "analyzed_at":    r["analyzed_at"],
        }
        for param in PARAMETERS:
            data = r.get("scores", {}).get(param)
            if not isinstance(data, dict):
                data = {}
            row[f"{param}_score"]       = data.get("score", "")
            row[f"{param}_observation"] = data.get("observation", "")
            row[f"{param}_improvement"] = data.get("improvement", "")
        row["overall_score"] = r.get("overall_score", "")
        row["priority_fix"]  = r.get("priority_fix", "")
        rows.append(row)
    return rows


def aggregate_scores(results: list) -> dict:
    """Return {param: avg_score} across all results, skipping nulls."""
    param_scores = {p: [] for p in PARAMETERS}
    for r in results:
        scores_data = r.get("scores", {})
        if not isinstance(scores_data, dict):
            continue
        for param in PARAMETERS:
            param_data = scores_data.get(param, {})
            if not isinstance(param_data, dict):
                continue
            score = param_data.get("score")
            if isinstance(score, (int, float)):
                param_scores[param].append(score)
    return {p: (sum(v) / len(v) if v else None) for p, v in param_scores.items()}

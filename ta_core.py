"""
Core analysis engine for the TA Session Analyzer.

Pipeline (per the PRD):
    Recording -> Extract Audio -> Speech-to-Text -> Speaker Diarization
              -> LLM Analysis -> Scorecard + Report

Diarization note: true audio diarization (e.g. pyannote.audio) needs torch +
a Hugging Face auth token, which is a heavy install for a free-tier testing
phase. Instead, speakers are separated by transcribing with Groq Whisper
(timestamped segments) and then asking an LLM to label each utterance as
TA or Student from conversational cues (who's guiding vs. who's asking).
This works well for the single-TA / single-student case this tool targets,
including English/Hindi code-switched speech, but is not real audio-based
diarization — it will be less reliable with overlapping speech or more than
one student.
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
from groq import Groq

from core import load_image_file
from recording_utils import get_duration_seconds

# ─── Scoring config (per PRD "Final Score" table) ────────────────────────────

WEIGHTS = {
    "doubt_resolution":   30,
    "teaching_quality":   20,
    "student_engagement": 10,
    "no_direct_answers":  15,
    "technical_accuracy": 10,
    "professionalism":    5,
    "communication":      5,
    "session_structure":  5,
}

SESSION_STAGES = [
    "Greeting",
    "Student explains issue",
    "TA understands",
    "Explanation",
    "Interactive discussion",
    "Student tries solution",
    "Final confirmation",
    "Closing",
]

TRANSCRIBE_MODEL   = "whisper-large-v3"
LABEL_MODEL        = "llama-3.1-8b-instant"
ANALYSIS_MODEL     = "llama-3.3-70b-versatile"
VISION_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"

GROQ_AUDIO_SIZE_LIMIT = 24 * 1024 * 1024  # Groq free tier caps uploads at 25MB


# ─── Small shared helpers ────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ─── 1. Audio extraction ─────────────────────────────────────────────────────

def extract_audio(video_path: str, out_path: str):
    """Mono 16kHz, low bitrate — small enough to stay under free-tier upload limits."""
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "64k", out_path, "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")


def estimate_audio_quality(audio_path: str) -> dict:
    """Lightweight heuristic proxy for audio quality using ffmpeg's volumedetect
    filter. NOT true noise/echo detection — just flags very low average volume,
    which usually means a poor/distant microphone."""
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    mean_vol = None
    for line in result.stderr.splitlines():
        if "mean_volume" in line:
            try:
                mean_vol = float(line.split(":")[1].strip().replace(" dB", ""))
            except (ValueError, IndexError):
                pass
    quality_flag = None
    if mean_vol is not None and mean_vol < -35:
        quality_flag = "Low average audio volume detected — possible poor microphone or distance from mic."
    return {"mean_volume_db": mean_vol, "quality_flag": quality_flag}


# ─── 2. Speech-to-text (Groq Whisper) ────────────────────────────────────────

def _transcribe_single(api_key: str, audio_path: str, time_offset: float = 0.0) -> list:
    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=(os.path.basename(audio_path), f.read()),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    segments = getattr(response, "segments", None)
    if segments is None and hasattr(response, "model_extra"):
        segments = (response.model_extra or {}).get("segments", [])
    segments = segments or []

    out = []
    for seg in segments:
        get = seg.get if isinstance(seg, dict) else (lambda k, d=None: getattr(seg, k, d))
        text = (get("text", "") or "").strip()
        if not text:
            continue
        out.append({
            "start": float(get("start", 0.0)) + time_offset,
            "end":   float(get("end", 0.0)) + time_offset,
            "text":  text,
            "avg_logprob": get("avg_logprob"),
        })
    return out


def transcribe_audio(api_key: str, audio_path: str, chunk_seconds: int = 1200) -> list:
    """Returns a flat list of {start, end, text, avg_logprob} segments across
    the whole file, splitting into ~20-minute chunks first if the file is too
    large for a single Groq upload."""
    if os.path.getsize(audio_path) <= GROQ_AUDIO_SIZE_LIMIT:
        return _transcribe_single(api_key, audio_path)

    with tempfile.TemporaryDirectory() as tmp:
        pattern = os.path.join(tmp, "chunk_%03d.mp3")
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-f", "segment",
             "-segment_time", str(chunk_seconds), "-c", "copy", pattern, "-y"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg chunk split failed: {result.stderr}")

        chunk_files = sorted(Path(tmp).glob("chunk_*.mp3"))
        all_segments = []
        for i, cf in enumerate(chunk_files):
            all_segments.extend(_transcribe_single(api_key, str(cf), time_offset=i * chunk_seconds))
        return all_segments


def estimate_transcription_confidence(raw_segments: list) -> float:
    """Derives a 0-100 confidence score from Whisper's own avg_logprob per segment."""
    logprobs = [s["avg_logprob"] for s in raw_segments if s.get("avg_logprob") is not None]
    if not logprobs:
        return 75.0  # unknown — moderate default
    avg = sum(logprobs) / len(logprobs)
    pct = max(0.0, min(100.0, (avg + 1.5) / 1.5 * 100))
    return round(pct, 1)


# ─── 3. Speaker diarization (LLM-based, see module docstring) ───────────────

def merge_segments(raw_segments: list, gap_threshold: float = 1.2) -> list:
    """Merge consecutive Whisper segments into larger utterances when the gap
    between them is small, so the labeling call has fewer, more meaningful
    chunks to classify."""
    merged = []
    for seg in raw_segments:
        if merged and seg["start"] - merged[-1]["end"] <= gap_threshold:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
    return merged


LABEL_SYSTEM_PROMPT = """You are labeling speaker turns from a TA (teaching assistant) doubt-clearing session recording.

Given a numbered list of transcript utterances with timestamps, label each one as either "TA" or "Student".
- The TA is the one guiding, explaining concepts, asking clarifying/verification questions, and concluding the discussion.
- The student is the one describing their problem/doubt, asking questions, and responding to explanations.
- If more than one non-TA voice appears, label all of them "Student".
- The session may mix English and Hindi (Hinglish) — this is normal; language switching does not indicate a speaker change by itself. Judge by role and content.

Return ONLY valid JSON, no markdown, no extra text:
{"labels": ["TA"|"Student", ...]}
The "labels" array must have exactly as many entries as there are utterances, in the same order."""


def label_speakers(api_key: str, merged_segments: list, batch_size: int = 100) -> list:
    if not merged_segments:
        return []
    client = Groq(api_key=api_key)
    labels = []
    for i in range(0, len(merged_segments), batch_size):
        batch = merged_segments[i:i + batch_size]
        lines = [
            f"[{j}] ({fmt_ts(seg['start'])}-{fmt_ts(seg['end'])}): {seg['text']}"
            for j, seg in enumerate(batch)
        ]
        user_content = f"Utterances (count={len(batch)}):\n" + "\n".join(lines)
        response = client.chat.completions.create(
            model=LABEL_MODEL,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        parsed = json.loads(_strip_fences(response.choices[0].message.content))
        batch_labels = parsed.get("labels", [])
        if len(batch_labels) < len(batch):
            batch_labels = batch_labels + ["Student"] * (len(batch) - len(batch_labels))
        labels.extend(batch_labels[:len(batch)])

    return [
        {**seg, "speaker": lab if lab in ("TA", "Student") else "Student"}
        for seg, lab in zip(merged_segments, labels)
    ]


def compute_participation(labeled_segments: list) -> dict:
    ta_time = sum(s["end"] - s["start"] for s in labeled_segments if s["speaker"] == "TA")
    student_time = sum(s["end"] - s["start"] for s in labeled_segments if s["speaker"] == "Student")
    spoken = ta_time + student_time
    ta_pct = round(ta_time / spoken * 100, 1) if spoken else 0.0
    student_pct = round(100 - ta_pct, 1) if spoken else 0.0
    return {
        "ta_seconds": round(ta_time, 1),
        "student_seconds": round(student_time, 1),
        "ta_pct": ta_pct,
        "student_pct": student_pct,
    }


def compute_dead_air(labeled_segments: list, total_duration: float, flag_threshold: float = 120.0) -> dict:
    gaps = []
    prev_end = 0.0
    for seg in labeled_segments:
        gap = seg["start"] - prev_end
        if gap > 0:
            gaps.append(gap)
        prev_end = seg["end"]
    tail_gap = total_duration - prev_end
    if tail_gap > 0:
        gaps.append(tail_gap)

    max_gap = max(gaps) if gaps else 0.0
    total_dead_air = sum(g for g in gaps if g > 3)  # ignore natural micro-pauses under 3s
    return {
        "max_gap_seconds": round(max_gap, 1),
        "total_dead_air_seconds": round(total_dead_air, 1),
        "flag": max_gap > flag_threshold,
    }


def build_transcript_text(labeled_segments: list) -> str:
    return "\n".join(
        f"[{fmt_ts(seg['start'])}] {seg['speaker']}: {seg['text']}"
        for seg in labeled_segments
    )


# ─── 4. Main LLM analysis ────────────────────────────────────────────────────

MAIN_SYSTEM_PROMPT = f"""You are an expert instructional-quality reviewer evaluating a TA (teaching assistant) doubt-clearing session, from its speaker-labeled transcript (TA vs Student). The transcript may mix English and Hindi (Hinglish) — treat this as normal, not a quality issue.

Evaluate these dimensions:

1. Doubt Resolution (highest priority): did the TA understand the question, ask clarifying questions, explain the concept, verify the student's understanding, and conclude the discussion? Confirmations from the student ("Got it", "Makes sense", "Thank you", "Understood") increase confidence it was resolved. If the session ends abruptly with no confirmation, lean toward "Not Resolved". Classify as "Fully Resolved", "Partially Resolved", or "Not Resolved".

2. Teaching Quality (1-5): does the TA explain concepts, break the problem into steps, give examples, use debugging, encourage the student's own thinking, and avoid spoon-feeding?

3. Direct Solution Detection (critical): did the TA give hints and guidance ("Good"), give too much help/almost-complete steps ("Warning"), or directly hand over the final code/query/assignment answer ("Warning" if borderline, "Violation" if a complete final solution was given outright)?

4. Communication (1-5): clarity, confidence, pace, friendliness, professionalism of delivery. Note any interruptions, long confusing tangents, or unclear explanations as issues.

5. Concept vs Answer: did the TA mostly teach concepts/logic/debugging ("Concept-based"), mostly give final answers ("Answer-based"), or a mix ("Mixed")?

6. Session Flow: which of these stages are clearly present in the transcript? {', '.join(SESSION_STAGES)}. List only the ones actually evidenced.

7. Professionalism: rate "Excellent", "Good", "Fair", or "Poor" based on tone — flag rude language, sarcasm, impatience, arguments, or negative tone as issues.

8. Technical Accuracy / Hallucination check: verify the technical correctness of anything the TA taught (code, SQL, DSA, concepts). Status is "Correct" or "Incorrect Guidance". If incorrect, give a confidence percentage (0-100) that the guidance was actually wrong, and explain what was wrong.

9. Student sentiment at the end of the session: "Satisfied", "Neutral", or "Dissatisfied".

Return ONLY valid JSON, no markdown, no extra text, in exactly this shape:
{{
  "doubt_resolution":  {{ "status": "Fully Resolved"|"Partially Resolved"|"Not Resolved", "reasoning": "2-3 sentences" }},
  "teaching_quality":  {{ "score": 1-5, "reasoning": "2-3 sentences" }},
  "direct_solution":   {{ "classification": "Good"|"Warning"|"Violation", "evidence": "quote or description of what happened" }},
  "communication":     {{ "score": 1-5, "issues": ["...", ...] }},
  "concept_vs_answer": {{ "classification": "Concept-based"|"Mixed"|"Answer-based", "reasoning": "1-2 sentences" }},
  "session_flow":      {{ "stages_present": ["subset of the stage list above"], "stages_missing": ["..."] }},
  "professionalism":   {{ "rating": "Excellent"|"Good"|"Fair"|"Poor", "issues": ["...", ...] }},
  "technical_accuracy":{{ "status": "Correct"|"Incorrect Guidance", "confidence_pct": 0-100, "details": "1-2 sentences" }},
  "student_sentiment": "Satisfied"|"Neutral"|"Dissatisfied",
  "summary": "3-5 sentence plain-English summary of what happened in the session",
  "recommendations": ["actionable recommendation 1", "actionable recommendation 2"]
}}"""


def analyze_session(api_key: str, transcript_text: str, stats: dict, screen_share_summary: str = None) -> dict:
    client = Groq(api_key=api_key)
    screen_line = (
        f"Screen share summary: {screen_share_summary}"
        if screen_share_summary else
        "Screen share: not available for this session."
    )
    user_content = f"""Session duration: {stats['duration_minutes']} minutes
TA speaking: {stats['ta_pct']}% | Student speaking: {stats['student_pct']}%
Longest silence gap: {stats['max_gap_seconds']}s
{screen_line}

Transcript (timestamps in mm:ss):
{transcript_text}

Analyze this TA doubt-clearing session and return the JSON evaluation."""

    response = client.chat.completions.create(
        model=ANALYSIS_MODEL,
        max_tokens=3000,
        messages=[
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return json.loads(_strip_fences(response.choices[0].message.content))


# ─── 5. Screen-share analysis (optional) ─────────────────────────────────────

SCREEN_SHARE_SYSTEM_PROMPT = """You are reviewing a single screenshot sampled from the shared screen of a TA (teaching assistant) doubt-clearing session.

Identify:
- content_type: one of "coding_ide", "terminal", "browser", "leetcode_or_judge", "jupyter_notebook", "presentation_slides", "file_explorer", "other"
- activity: one of "ta_writing_code", "student_writing_code", "explaining_debugging", "idle_or_static", "navigating_docs"
- visible_code_or_query: OCR any code, SQL query, or written solution visible verbatim (empty string if none)
- looks_like_final_solution: true if the visible content appears to be a complete, ready-to-submit final answer rather than a partial hint or in-progress work

Return ONLY valid JSON, no markdown, no extra text:
{"content_type": "...", "activity": "...", "visible_code_or_query": "...", "looks_like_final_solution": true|false}"""


def _extract_sample_frames(video_path: str, out_dir: str, n_frames: int = 6) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for screen-share sampling: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no readable frames.")

    frame_paths = []
    for i in range(n_frames):
        target = int(total_frames * (i / max(n_frames - 1, 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            continue
        path = os.path.join(out_dir, f"screen_{i:02d}.jpg")
        cv2.imwrite(path, frame)
        frame_paths.append(path)
    cap.release()
    return frame_paths


def analyze_screen_share(api_key: str, video_path: str, n_frames: int = 6) -> dict:
    client = Groq(api_key=api_key)
    with tempfile.TemporaryDirectory() as tmp:
        frame_paths = _extract_sample_frames(video_path, tmp, n_frames)
        frame_results = []
        for fp in frame_paths:
            image_data, media_type = load_image_file(Path(fp))
            response = client.chat.completions.create(
                model=VISION_MODEL,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SCREEN_SHARE_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                        {"type": "text", "text": "Classify this screen-share frame and return the JSON."},
                    ]},
                ],
            )
            try:
                frame_results.append(json.loads(_strip_fences(response.choices[0].message.content)))
            except (json.JSONDecodeError, IndexError):
                continue

    if not frame_results:
        return None

    content_counts = {}
    for r in frame_results:
        ct = r.get("content_type", "other")
        content_counts[ct] = content_counts.get(ct, 0) + 1

    any_final_solution = any(r.get("looks_like_final_solution") for r in frame_results)
    code_evidence = [r["visible_code_or_query"] for r in frame_results if r.get("visible_code_or_query")]

    classification = "Violation" if any_final_solution else ("Warning" if code_evidence else "Good")

    return {
        "frames_analyzed": len(frame_results),
        "content_type_breakdown": content_counts,
        "direct_solution_detected": any_final_solution,
        "code_evidence": code_evidence[:5],
        "classification": classification,
    }


# ─── 6. Chat analysis (optional — wire in once a chat-log source exists) ────

CHAT_SYSTEM_PROMPT = """You are reviewing the chat log of a TA (teaching assistant) doubt-clearing session, separate from the voice transcript.

Detect:
- direct_code_or_sql: verbatim code, SQL queries, or complete assignment answers pasted by the TA
- external_links: any links shared
- hint_examples: good hint-style messages (e.g. "Try using GROUP BY", "Check your loop condition")
- violation_examples: messages that directly hand over the solution (e.g. "SELECT * FROM Employees;", "Here is the entire solution")

Return ONLY valid JSON, no markdown, no extra text:
{"classification": "Good"|"Warning"|"Violation", "direct_code_or_sql": ["..."], "external_links": ["..."], "hint_examples": ["..."], "violation_examples": ["..."]}"""


def analyze_chat(api_key: str, chat_text: str) -> dict:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LABEL_MODEL,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Chat log:\n{chat_text}\n\nAnalyze it and return the JSON."},
        ],
    )
    return json.loads(_strip_fences(response.choices[0].message.content))


# ─── 7. Weighted scoring (per PRD "Final Score" table) ──────────────────────

def _engagement_points(ta_pct: float) -> float:
    """Full 10 points inside the 50-70% TA-speaking band; falls off linearly outside it."""
    if 50 <= ta_pct <= 70:
        return 10.0
    dist = (50 - ta_pct) if ta_pct < 50 else (ta_pct - 70)
    penalty = min(dist / 40 * 10, 10)
    return round(10 - penalty, 1)


def compute_final_score(analysis: dict, participation: dict) -> dict:
    doubt_pts = {"Fully Resolved": 30, "Partially Resolved": 15, "Not Resolved": 0}.get(
        analysis["doubt_resolution"]["status"], 0)
    teaching_pts = round((analysis["teaching_quality"]["score"] / 5) * 20, 1)
    engagement_pts = _engagement_points(participation["ta_pct"])
    no_direct_pts = {"Good": 15, "Warning": 8, "Violation": 0}.get(
        analysis["direct_solution"]["classification"], 8)

    tech = analysis["technical_accuracy"]
    tech_pts = 10.0 if tech["status"] == "Correct" else max(0.0, round(10 - tech.get("confidence_pct", 50) / 10, 1))

    prof_pts = {"Excellent": 5, "Good": 4, "Fair": 2.5, "Poor": 0}.get(
        analysis["professionalism"]["rating"], 2.5)
    comm_pts = round((analysis["communication"]["score"] / 5) * 5, 1)
    stages_present = analysis["session_flow"].get("stages_present", [])
    structure_pts = round((len(stages_present) / len(SESSION_STAGES)) * 5, 1)

    breakdown = {
        "doubt_resolution":   {"points": doubt_pts,      "max": WEIGHTS["doubt_resolution"]},
        "teaching_quality":   {"points": teaching_pts,   "max": WEIGHTS["teaching_quality"]},
        "student_engagement": {"points": engagement_pts, "max": WEIGHTS["student_engagement"]},
        "no_direct_answers":  {"points": no_direct_pts,  "max": WEIGHTS["no_direct_answers"]},
        "technical_accuracy": {"points": tech_pts,       "max": WEIGHTS["technical_accuracy"]},
        "professionalism":    {"points": prof_pts,       "max": WEIGHTS["professionalism"]},
        "communication":      {"points": comm_pts,       "max": WEIGHTS["communication"]},
        "session_structure":  {"points": structure_pts,  "max": WEIGHTS["session_structure"]},
    }
    breakdown["overall"] = round(sum(v["points"] for v in breakdown.values()), 1)
    return breakdown


# ─── 8. AI flags (per PRD "AI Flags" section) ────────────────────────────────

def build_flags(analysis: dict, participation: dict, dead_air: dict, duration_minutes: float,
                 ai_confidence_pct: float, screen_share: dict = None,
                 confidence_threshold: float = 60.0) -> list:
    flags = []
    if analysis["doubt_resolution"]["status"] != "Fully Resolved":
        flags.append("Doubt not resolved")
    if analysis["direct_solution"]["classification"] == "Violation" or (screen_share and screen_share["classification"] == "Violation"):
        flags.append("Direct solution shared")
    if analysis["technical_accuracy"]["status"] == "Incorrect Guidance":
        flags.append("Technical explanation incorrect")
    if analysis.get("student_sentiment") == "Dissatisfied":
        flags.append("Student dissatisfied")
    if dead_air["flag"]:
        flags.append("Excessive silence")
    if duration_minutes < 5:
        flags.append("Session duration under 5 minutes")
    if analysis["professionalism"]["rating"] == "Poor":
        flags.append("Unprofessional language")
    if participation["ta_pct"] > 90:
        flags.append("TA dominated the conversation (>90% speaking time)")
    if ai_confidence_pct < confidence_threshold:
        flags.append("AI confidence below threshold")
    return flags


# ─── 9. End-to-end orchestration ─────────────────────────────────────────────

def analyze_ta_session(api_key: str, video_path: str, analyze_screen: bool = True,
                        chat_text: str = None) -> dict:
    """Runs the full pipeline on one recording and returns a report dict ready
    for scoring output / PDF / CSV / JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.mp3")
        extract_audio(video_path, audio_path)
        duration = get_duration_seconds(audio_path)
        raw_segments = transcribe_audio(api_key, audio_path)
        transcription_confidence = estimate_transcription_confidence(raw_segments)
        audio_quality = estimate_audio_quality(audio_path)

    if not raw_segments:
        raise RuntimeError("No speech detected in recording.")

    merged = merge_segments(raw_segments)
    labeled = label_speakers(api_key, merged)
    participation = compute_participation(labeled)
    dead_air = compute_dead_air(labeled, duration)
    transcript_text = build_transcript_text(labeled)

    screen_share = None
    if analyze_screen:
        try:
            screen_share = analyze_screen_share(api_key, video_path)
        except Exception:
            screen_share = None  # screen-share analysis is best-effort/optional

    screen_summary_text = None
    if screen_share:
        screen_summary_text = (
            f"{screen_share['frames_analyzed']} frames sampled; "
            f"content types seen: {screen_share['content_type_breakdown']}; "
            f"possible final solution visible on screen: {screen_share['direct_solution_detected']}"
        )

    stats = {
        "duration_minutes": round(duration / 60, 1),
        "ta_pct": participation["ta_pct"],
        "student_pct": participation["student_pct"],
        "max_gap_seconds": dead_air["max_gap_seconds"],
    }

    analysis = analyze_session(api_key, transcript_text, stats, screen_summary_text)

    chat_analysis = None
    if chat_text:
        try:
            chat_analysis = analyze_chat(api_key, chat_text)
        except Exception:
            chat_analysis = None

    score_breakdown = compute_final_score(analysis, participation)
    flags = build_flags(analysis, participation, dead_air, stats["duration_minutes"],
                         transcription_confidence, screen_share)
    if audio_quality.get("quality_flag"):
        flags.append(audio_quality["quality_flag"])
    if chat_analysis and chat_analysis.get("classification") == "Violation":
        if "Direct solution shared" not in flags:
            flags.append("Direct solution shared")

    return {
        "duration_minutes": stats["duration_minutes"],
        "participation": participation,
        "dead_air": dead_air,
        "audio_quality": audio_quality,
        "transcription_confidence_pct": transcription_confidence,
        "screen_share": screen_share,
        "chat_analysis": chat_analysis,
        "analysis": analysis,
        "score_breakdown": score_breakdown,
        "flags": flags,
        "transcript_text": transcript_text,
    }

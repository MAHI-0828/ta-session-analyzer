"""
Core analysis engine for the TA Session Analyzer.

Pipeline:
    Recording -> Deepgram prerecorded API transcribes + diarizes it (speaker
    labels come back generic: "0", "1", ...) -> a handful of evenly-spaced
    screenshots are pulled from the shared screen -> one Gemini call maps the
    generic speakers to TA/Student and scores the session against the rubric
    using the transcript + screenshots -> local participation/dead-air math
    -> weighted scorecard + report.

Provider note: transcription/diarization runs on Deepgram (nova-2,
prerecorded API) instead of Gemini. Gemini is only asked to do the judgment
calls — who's the TA, is the shared screen a violation, rubric scoring —
from the transcript text plus a few sampled screenshots, instead of
ingesting the whole video. That's both cheaper (no full-video multimodal
call) and lets Deepgram's free transcription credits absorb the most
expensive part of the pipeline while Gemini quota is spent only on analysis.
Trade-off: TA/Student attribution now comes from conversational content
alone (who explains vs. who asks) rather than voice + visual cues together,
so it's worth spot-checking on calls with unusual dynamics.
The rest of the codebase (app.py, analyze.py, core.py, auto_lecture_analyzer.py)
is untouched and still runs on Gemini per-frame image scoring.
"""

import json
import mimetypes
import os
import re
import subprocess
import tempfile
from typing import List, Literal, Optional

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

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

# If this model name 404s on your account, try "gemini-2.5-flash" instead —
# both are free-tier eligible as of mid-2026.
GEMINI_MODEL = "gemini-3.5-flash"

DEEPGRAM_MODEL = "nova-2"
DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"

# How many evenly-spaced screenshots to pull from the shared screen for
# Gemini to look at — enough to catch a solution left on screen without
# paying full-video multimodal token costs.
SCREEN_FRAME_COUNT = 8


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ─── 1. Local, provider-agnostic signal (no LLM needed) ─────────────────────

def estimate_audio_quality(media_path: str) -> dict:
    """Lightweight heuristic proxy for audio quality using ffmpeg's volumedetect
    filter, run straight on the video (ffmpeg pulls out the audio stream
    itself, so there's no need to pre-extract audio to a separate file). NOT
    true noise/echo detection — just flags very low average volume, which
    usually means a poor/distant microphone."""
    result = subprocess.run(
        ["ffmpeg", "-i", media_path, "-af", "volumedetect", "-f", "null", "-"],
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


def sample_screen_frames(video_path: str, duration: float, count: int = SCREEN_FRAME_COUNT) -> List[bytes]:
    """Pull `count` evenly-spaced JPEG frames from the recording so Gemini can
    look at the shared screen without ingesting the full video."""
    if duration <= 0 or count <= 0:
        return []
    frames = []
    interval = duration / (count + 1)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, count + 1):
            ts = interval * i
            out_path = os.path.join(tmp, f"frame_{i}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", video_path,
                 "-frames:v", "1", "-q:v", "3", out_path],
                capture_output=True,
            )
            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    frames.append(f.read())
    return frames


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


def build_generic_transcript_text(segments: list) -> str:
    """Same formatting as build_transcript_text, but before TA/Student roles
    have been assigned — used for the transcript we hand to Gemini."""
    return "\n".join(
        f"[{fmt_ts(seg['start'])}] Speaker {seg['speaker']}: {seg['text']}"
        for seg in segments
    )


# ─── 2. Deepgram transcription + diarization ─────────────────────────────────

def transcribe_with_deepgram(api_key: str, media_path: str) -> dict:
    """Sends the recording straight to Deepgram's prerecorded API — it reads
    the audio out of the video container itself, no local extraction needed
    — and returns diarized utterances plus an overall confidence score.
    Speakers come back as generic integers (0, 1, ...); Gemini maps them to
    TA/Student afterwards from the transcript content alone."""
    mime = mimetypes.guess_type(media_path)[0] or "video/mp4"
    with open(media_path, "rb") as f:
        response = requests.post(
            DEEPGRAM_ENDPOINT,
            params={
                "model": DEEPGRAM_MODEL,
                "diarize": "true",
                "punctuate": "true",
                "smart_format": "true",
                "utterances": "true",
            },
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": mime,
            },
            data=f,
            timeout=600,
        )
    response.raise_for_status()
    payload = response.json()
    utterances = payload.get("results", {}).get("utterances") or []
    segments = [
        {
            "start": u["start"],
            "end": u["end"],
            "speaker": str(u.get("speaker", 0)),
            "text": u["transcript"].strip(),
            "confidence": u.get("confidence", 0.0),
        }
        for u in utterances if u.get("transcript", "").strip()
    ]
    if not segments:
        raise RuntimeError("No speech detected in recording.")
    avg_confidence = sum(s["confidence"] for s in segments) / len(segments)
    return {"segments": segments, "confidence_pct": round(avg_confidence * 100, 1)}


def _normalize_speaker_label(label: str) -> str:
    match = re.search(r"\d+", str(label))
    return match.group(0) if match else str(label).strip()


# ─── 3. Gemini response schemas ──────────────────────────────────────────────

class SpeakerRoles(BaseModel):
    ta_speaker_label: str
    reasoning: str


class ScreenShare(BaseModel):
    summary: str
    content_types_observed: List[str]
    direct_solution_detected: bool
    code_or_query_evidence: List[str]
    classification: Literal["Good", "Warning", "Violation"]


class DoubtResolution(BaseModel):
    status: Literal["Fully Resolved", "Partially Resolved", "Not Resolved"]
    reasoning: str


class TeachingQuality(BaseModel):
    score: int
    reasoning: str


class DirectSolution(BaseModel):
    classification: Literal["Good", "Warning", "Violation"]
    evidence: str


class Communication(BaseModel):
    score: int
    issues: List[str]


class ConceptVsAnswer(BaseModel):
    classification: Literal["Concept-based", "Mixed", "Answer-based"]
    reasoning: str


class SessionFlow(BaseModel):
    stages_present: List[str]
    stages_missing: List[str]


class Professionalism(BaseModel):
    rating: Literal["Excellent", "Good", "Fair", "Poor"]
    issues: List[str]


class TechnicalAccuracy(BaseModel):
    status: Literal["Correct", "Incorrect Guidance"]
    confidence_pct: int
    details: str


class SessionAnalysisResult(BaseModel):
    speaker_roles: SpeakerRoles
    screen_share: ScreenShare
    doubt_resolution: DoubtResolution
    teaching_quality: TeachingQuality
    direct_solution: DirectSolution
    communication: Communication
    concept_vs_answer: ConceptVsAnswer
    session_flow: SessionFlow
    professionalism: Professionalism
    technical_accuracy: TechnicalAccuracy
    student_sentiment: Literal["Satisfied", "Neutral", "Dissatisfied"]
    summary: str
    recommendations: List[str]


class ChatAnalysisResult(BaseModel):
    classification: Literal["Good", "Warning", "Violation"]
    direct_code_or_sql: List[str]
    external_links: List[str]
    hint_examples: List[str]
    violation_examples: List[str]


# ─── 4. Prompts ───────────────────────────────────────────────────────────────

COMBINED_SYSTEM_PROMPT = f"""You are an expert instructional-quality reviewer analyzing a TA (teaching assistant) doubt-clearing session. You are given (a) an already-transcribed, speaker-diarized transcript produced by a separate speech-to-text service, using generic speaker labels like "Speaker 0", "Speaker 1", and (b) up to {SCREEN_FRAME_COUNT} evenly-spaced screenshots sampled from the shared screen during the call, if screen analysis was requested. The transcript's wording and timestamps are already final — do not alter them, just use them. The session may mix English and Hindi (Hinglish) — treat this as normal, not a quality issue.

Step 1 — Identify who is the TA:
Exactly one speaker label is the TA — the one guiding, explaining concepts, asking clarifying/verification questions, and concluding the discussion. The student is the one describing their doubt, asking questions, and responding to explanations. If more than one non-TA label appears, they are still collectively "the student". Return ta_speaker_label as exactly the label text as it appears in the transcript (e.g. "0"), with a short reasoning.

Step 2 — Read the shared-screen screenshots (if provided):
Note what kind of content was shown (coding IDE, terminal, browser, LeetCode/judge, notebook, slides, file explorer, other), and whether any code/SQL/query visible on screen was a complete, ready-to-submit final solution rather than a partial hint. classification is "Good" if only hints/guidance were visible, "Warning" if borderline/near-complete help was shown, "Violation" if a complete final solution was shown on screen. If screen analysis was not requested for this session (see the user message), set classification to "Good", leave content_types_observed/code_or_query_evidence empty, and note in summary that screen analysis was skipped by request.

Step 3 — Evaluate these dimensions using the transcript and the shared screen together:

1. Doubt Resolution (highest priority): did the TA understand the question, ask clarifying questions, explain the concept, verify the student's understanding, and conclude the discussion? Confirmations from the student ("Got it", "Makes sense", "Thank you", "Understood") increase confidence it was resolved. If the session ends abruptly with no confirmation, lean toward "Not Resolved".

2. Teaching Quality (1-5): does the TA explain concepts, break the problem into steps, give examples, use debugging, encourage the student's own thinking, and avoid spoon-feeding?

3. Direct Solution Detection (critical): did the TA give hints and guidance ("Good"), give too much help/almost-complete steps ("Warning"), or directly hand over the final code/query/assignment answer ("Warning" if borderline, "Violation" if a complete final solution was given outright, spoken or shown on screen)?

4. Communication (1-5): clarity, confidence, pace, friendliness, professionalism of delivery. Note any interruptions, long confusing tangents, or unclear explanations as issues.

5. Concept vs Answer: did the TA mostly teach concepts/logic/debugging ("Concept-based"), mostly give final answers ("Answer-based"), or a mix ("Mixed")?

6. Session Flow: which of these stages are clearly present in the recording? {', '.join(SESSION_STAGES)}. List only the ones actually evidenced, in stages_present; list the rest in stages_missing.

7. Professionalism: rate "Excellent", "Good", "Fair", or "Poor" based on tone — flag rude language, sarcasm, impatience, arguments, or negative tone as issues.

8. Technical Accuracy / Hallucination check: verify the technical correctness of anything the TA taught (code, SQL, DSA, concepts). Status is "Correct" or "Incorrect Guidance". If incorrect, give a confidence percentage (0-100) that the guidance was actually wrong, and explain what was wrong in details.

9. Student sentiment at the end of the session: "Satisfied", "Neutral", or "Dissatisfied".

Also include a 3-5 sentence plain-English summary of what happened, and a short list of actionable recommendations for the TA."""


CHAT_SYSTEM_PROMPT = """You are reviewing the chat log of a TA (teaching assistant) doubt-clearing session, separate from the voice transcript.

Detect:
- direct_code_or_sql: verbatim code, SQL queries, or complete assignment answers pasted by the TA
- external_links: any links shared
- hint_examples: good hint-style messages (e.g. "Try using GROUP BY", "Check your loop condition")
- violation_examples: messages that directly hand over the solution (e.g. "SELECT * FROM Employees;", "Here is the entire solution")

classification is "Good" if only hints were shared, "Warning" if borderline, "Violation" if a full solution was pasted."""


# ─── 5. Gemini plumbing ───────────────────────────────────────────────────────

def analyze_session_with_gemini(client: "genai.Client", transcript_text: str,
                                 frames: List[bytes], analyze_screen: bool) -> SessionAnalysisResult:
    screen_requested = analyze_screen and bool(frames)
    user_prompt = (
        f"screen_share_analysis_requested: {'yes' if screen_requested else 'no'}\n\n"
        f"Diarized transcript:\n{transcript_text}\n\n"
        "Analyze this TA doubt-clearing session and return the JSON."
    )
    contents = [user_prompt]
    if screen_requested:
        contents += [types.Part.from_bytes(data=frame, mime_type="image/jpeg") for frame in frames]

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=COMBINED_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=SessionAnalysisResult,
        ),
    )
    return SessionAnalysisResult.model_validate_json(response.text)


def analyze_chat(client: "genai.Client", chat_text: str) -> dict:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[f"Chat log:\n{chat_text}\n\nAnalyze it and return the JSON."],
        config=types.GenerateContentConfig(
            system_instruction=CHAT_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ChatAnalysisResult,
        ),
    )
    return ChatAnalysisResult.model_validate_json(response.text).model_dump()


# ─── 6. Weighted scoring (per PRD "Final Score" table) ──────────────────────

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


# ─── 7. AI flags (per PRD "AI Flags" section) ────────────────────────────────

def build_flags(analysis: dict, participation: dict, dead_air: dict, duration_minutes: float,
                 ai_confidence_pct: float, screen_share: Optional[dict] = None,
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


# ─── 8. End-to-end orchestration ─────────────────────────────────────────────

def analyze_ta_session(gemini_api_key: str, video_path: str, deepgram_api_key: str,
                        analyze_screen: bool = True, chat_text: str = None) -> dict:
    """Runs the full pipeline on one recording and returns a report dict ready
    for scoring output / PDF / CSV / JSON. Transcription + diarization run on
    Deepgram; Gemini only maps speakers to TA/Student and scores the rubric."""
    duration = get_duration_seconds(video_path)
    audio_quality = estimate_audio_quality(video_path)

    transcript = transcribe_with_deepgram(deepgram_api_key, video_path)
    segments = transcript["segments"]
    transcription_confidence = transcript["confidence_pct"]
    generic_transcript_text = build_generic_transcript_text(segments)

    frames = sample_screen_frames(video_path, duration) if analyze_screen else []

    client = genai.Client(api_key=gemini_api_key)
    result = analyze_session_with_gemini(client, generic_transcript_text, frames, analyze_screen)
    data = result.model_dump()

    ta_label = _normalize_speaker_label(data.pop("speaker_roles")["ta_speaker_label"])
    labeled = [
        {
            "start": s["start"], "end": s["end"],
            "speaker": "TA" if _normalize_speaker_label(s["speaker"]) == ta_label else "Student",
            "text": s["text"],
        }
        for s in segments
    ]
    participation = compute_participation(labeled)
    dead_air = compute_dead_air(labeled, duration)
    transcript_text = build_transcript_text(labeled)

    screen_share = data.pop("screen_share")
    if not analyze_screen:
        screen_share = None

    analysis = data  # remaining keys match the original `analysis` contract

    chat_analysis = None
    if chat_text:
        try:
            chat_analysis = analyze_chat(client, chat_text)
        except Exception:
            chat_analysis = None  # chat analysis is best-effort/optional

    duration_minutes = round(duration / 60, 1)
    score_breakdown = compute_final_score(analysis, participation)
    flags = build_flags(analysis, participation, dead_air, duration_minutes,
                         transcription_confidence, screen_share)
    if audio_quality.get("quality_flag"):
        flags.append(audio_quality["quality_flag"])
    if chat_analysis and chat_analysis.get("classification") == "Violation":
        if "Direct solution shared" not in flags:
            flags.append("Direct solution shared")

    return {
        "duration_minutes": duration_minutes,
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

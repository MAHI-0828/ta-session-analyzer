"""
Free, fully-local speech-to-text for the TA Session Analyzer — an
alternative to Deepgram so transcription costs nothing and the recording's
audio never leaves the machine.

Backends (picked automatically, first one installed wins):
  1. mlx-whisper    — Apple Silicon (M1/M2/M3) Macs. Runs on the Mac's GPU
                      via Apple's MLX; large-v3-turbo transcribes a 30-min
                      session in roughly 2-4 min on an 8GB M1.
  2. faster-whisper — everything else (Intel Mac, Linux, Windows). CPU int8;
                      noticeably slower, so prefer a smaller model there.

Install with:  pip install -r requirements-local.txt

Whisper does NOT diarize (no speaker labels), so segments come back with
speaker=None. ta_core.py then has Gemini assign each segment to TA/Student
from the conversation itself, in the same call that scores the rubric — so
local transcription adds zero extra API calls.

Config (env vars, all optional):
    WHISPER_MODEL     model name/repo. Defaults: mlx ->
                      "mlx-community/whisper-large-v3-turbo" (~1.6GB, fits
                      in 8GB RAM); faster-whisper -> "small".
                      On a Mac that runs out of memory, try
                      "mlx-community/whisper-small-mlx".
    WHISPER_LANGUAGE  "en" (default — keeps Hinglish in Latin script, same
                      as the Deepgram path), "hi", or "auto" to detect.
"""

import importlib.util
import math
import os

# Nudges Whisper toward Latin-script Hinglish + programming vocabulary
# instead of translating or switching scripts mid-session.
INITIAL_PROMPT = (
    "A Hinglish (Hindi + English) doubt-clearing session between a teaching "
    "assistant and a student about programming, SQL, and data structures."
)

MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
FASTER_WHISPER_DEFAULT_MODEL = "small"


def available_backend():
    """Name of the local backend that will be used, or None if none is installed."""
    if importlib.util.find_spec("mlx_whisper"):
        return "mlx-whisper"
    if importlib.util.find_spec("faster_whisper"):
        return "faster-whisper"
    return None


def _language():
    lang = os.environ.get("WHISPER_LANGUAGE", "en").strip().lower()
    return None if lang in ("", "auto") else lang


def _logprob_to_confidence(avg_logprob) -> float:
    if avg_logprob is None:
        return 0.0
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def _transcribe_mlx(audio_path: str) -> list:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=os.environ.get("WHISPER_MODEL", MLX_DEFAULT_MODEL),
        language=_language(),
        initial_prompt=INITIAL_PROMPT,
        # Stops Whisper looping on / hallucinating text over long silences
        # and gives tighter segment boundaries for the dead-air math.
        condition_on_previous_text=False,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
    )
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"],
         "confidence": _logprob_to_confidence(s.get("avg_logprob"))}
        for s in result.get("segments", [])
    ]


def _transcribe_faster_whisper(audio_path: str) -> list:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        os.environ.get("WHISPER_MODEL", FASTER_WHISPER_DEFAULT_MODEL),
        device="auto", compute_type="int8",
    )
    segments, _info = model.transcribe(
        audio_path,
        language=_language(),
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=False,
        vad_filter=True,  # skips silence -> better dead-air gaps, fewer hallucinations
    )
    return [
        {"start": s.start, "end": s.end, "text": s.text,
         "confidence": _logprob_to_confidence(s.avg_logprob)}
        for s in segments
    ]


def transcribe_locally(audio_path: str) -> dict:
    """Same return shape as ta_core.transcribe_with_deepgram, except each
    segment's speaker is None (Whisper can't tell voices apart)."""
    backend = available_backend()
    if backend == "mlx-whisper":
        raw = _transcribe_mlx(audio_path)
    elif backend == "faster-whisper":
        raw = _transcribe_faster_whisper(audio_path)
    else:
        raise RuntimeError(
            "Local transcription needs mlx-whisper (Apple Silicon) or faster-whisper — "
            "run: pip install -r requirements-local.txt"
        )

    segments = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "speaker": None,
         "text": s["text"].strip(), "confidence": s["confidence"]}
        for s in raw if s["text"].strip()
    ]
    if not segments:
        raise RuntimeError("No speech detected in recording.")
    avg_confidence = sum(s["confidence"] for s in segments) / len(segments)
    return {"segments": segments, "confidence_pct": round(avg_confidence * 100, 1),
            "backend": backend}

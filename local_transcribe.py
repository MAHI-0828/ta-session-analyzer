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
    WHISPER_LANGUAGE  "hi" (default), "en", or "auto" to detect. Sessions are
                      mostly Hinglish, and forcing "en" makes Whisper
                      TRANSLATE the Hindi parts into English instead of
                      transcribing them. "hi" keeps what was actually said;
                      the Romanized-Hinglish INITIAL_PROMPT below steers it
                      toward Latin script (Whisper imitates the prompt's
                      style), though some lines may still come out in
                      Devanagari — Gemini reads both fine.

Compare settings on one real recording before a big batch:
    python local_transcribe.py some_session.mp4 --language hi en auto
Transcripts are cached per session, so after changing WHISPER_LANGUAGE,
re-run a batch with --force to re-transcribe sessions done earlier.
"""

import importlib.util
import math
import os

# Whisper continues in the style of its prompt, so this is written the way
# the sessions sound: Romanized Hinglish with English tech terms. That nudges
# it toward Latin-script Hinglish instead of Devanagari or an English
# translation, and primes programming vocabulary (GROUP BY, array, loop...).
INITIAL_PROMPT = (
    "Haan, toh aapka doubt kya hai? Sir, mera SQL query mein GROUP BY ka error "
    "aa raha hai. Achha, ek baar apna code screen pe share karo. Dekho, yeh "
    "column aggregate nahi hua hai, isliye error aa raha hai. Array, loop, "
    "function, recursion, JOIN, WHERE clause. Okay sir, samajh aa gaya, thank you."
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
    lang = os.environ.get("WHISPER_LANGUAGE", "hi").strip().lower()
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


if __name__ == "__main__":
    # A/B helper: transcribe one recording with each language setting and
    # print the start of each transcript, to pick WHISPER_LANGUAGE by eye.
    import argparse
    import tempfile
    import time

    from ta_core import extract_audio, fmt_ts

    parser = argparse.ArgumentParser(description="Compare local Whisper language settings on one recording.")
    parser.add_argument("recording", help="video or audio file")
    parser.add_argument("--language", nargs="+", default=["hi", "en", "auto"],
                        help="settings to try (default: hi en auto)")
    parser.add_argument("--lines", type=int, default=25, help="transcript lines to print per setting")
    args = parser.parse_args()

    print(f"backend: {available_backend()}  model: "
          f"{os.environ.get('WHISPER_MODEL', MLX_DEFAULT_MODEL if available_backend() == 'mlx-whisper' else FASTER_WHISPER_DEFAULT_MODEL)}")
    with tempfile.TemporaryDirectory() as tmp:
        audio = extract_audio(args.recording, os.path.join(tmp, "audio.ogg"))
        for lang in args.language:
            os.environ["WHISPER_LANGUAGE"] = lang
            started = time.time()
            result = transcribe_locally(audio)
            print(f"\n===== WHISPER_LANGUAGE={lang}  ({time.time() - started:.0f}s, "
                  f"confidence {result['confidence_pct']}%) =====")
            for seg in result["segments"][:args.lines]:
                print(f"[{fmt_ts(seg['start'])}] {seg['text']}")

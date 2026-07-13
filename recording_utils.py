"""
Shared helpers for resolving + downloading session recordings from the
portal link format used across this project:

    https://my.newtonschool.co/play-video/?url=<direct-cloudfront-mp4-url>

Used by both auto_lecture_analyzer.py and ta_session_analyzer.py so the
link-resolution logic only lives in one place.
"""

import subprocess
import urllib.parse

import requests

REQUEST_TIMEOUT = 60


def extract_video_url(main_url: str) -> str:
    """If the portal wraps the real video URL in a query param (e.g. ?url=...),
    pull it out. If the portal link IS already the direct video URL, fall back
    to returning it unchanged."""
    parsed = urllib.parse.urlparse(main_url)
    query_params = urllib.parse.parse_qs(parsed.query)
    video_url = query_params.get("url", [None])[0]
    return video_url or main_url


def download_video(video_url: str, save_path: str):
    response = requests.get(video_url, stream=True, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def get_duration_seconds(media_path: str) -> float:
    """ffprobe wrapper — works for both video and audio files."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")
    return float(probe.stdout.strip())

"""OpenAI Whisper API client. Audio file -> text transcript.

Used by the conversation pipeline for the server-side STT path (works on
iOS Safari, where Web Speech API is unreliable).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from .costing import whisper_cost


log = logging.getLogger("conference-ai.transcribe")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
MAX_BYTES = 25 * 1024 * 1024  # OpenAI Whisper hard cap on file uploads.


def _say(msg: str) -> None:
    """Print to stderr so it shows up in the uvicorn terminal alongside Claude streams."""
    sys.stderr.write(f"[transcribe] {msg}\n")
    sys.stderr.flush()


def transcribe_audio(audio_path: str):
    """Transcribe an audio file via OpenAI Whisper.

    Returns ``(transcript_text, cost_usd)``. Raises on hard failures so
    the pipeline can mark the conversation as errored.
    """
    p = Path(audio_path)
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    size = p.stat().st_size
    size_mb = size / 1_000_000
    if size > MAX_BYTES:
        raise ValueError(
            f"Audio file is {size_mb:.1f} MB; OpenAI Whisper's per-file "
            "limit is 25 MB. For longer calls, split before uploading "
            "(or switch to a streaming-friendly STT vendor)."
        )

    _say(f"sending {p.name} ({size_mb:.2f} MB) to Whisper model={WHISPER_MODEL}")
    t0 = time.monotonic()

    client = OpenAI()  # picks up OPENAI_API_KEY from env
    with open(p, "rb") as f:
        # verbose_json gives us a `duration` field (seconds), used for cost.
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
        )

    text = (getattr(result, "text", "") or "").strip()
    duration_sec = float(getattr(result, "duration", 0) or 0)
    cost = whisper_cost(duration_sec)
    elapsed = time.monotonic() - t0
    _say(
        f"received transcript: {len(text)} chars, audio "
        f"{duration_sec:.1f}s, cost ${cost:.4f} in {elapsed:.1f}s wall"
    )
    if text:
        first = text.split("\n", 1)[0]
        _say(f'preview: "{first[:120]}{"…" if len(first) > 120 else ""}"')
    return text, cost

"""Transcribe the downloaded video locally with faster-whisper.

faster-whisper is a CTranslate2 reimplementation of Whisper -- same accuracy,
much faster, and comfortably fits a 6GB GPU (e.g. RTX 4050) at "medium" size
in float16, or "large-v3" in int8_float16 if you want max accuracy.
"""
from dataclasses import dataclass, field
from pathlib import Path
from faster_whisper import WhisperModel


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def transcribe(
    video_path: Path,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
) -> list[Segment]:
    """
    model_size: tiny / base / small / medium / large-v3.
      - "medium" is the sweet spot for a 6GB GPU (float16) -- good accuracy,
        several-x realtime speed.
      - Drop to "small" if you hit an out-of-memory error.
      - Use device="cpu", compute_type="int8" if you have no GPU at all
        (slower, but still completely free).
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, _info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,  # drops long silences, tightens segment boundaries
        vad_parameters={"min_silence_duration_ms": 400},
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words = [Word(w.start, w.end, w.word.strip()) for w in (seg.words or []) if w.word.strip()]
        segments.append(Segment(seg.start, seg.end, seg.text.strip(), words))
    return segments

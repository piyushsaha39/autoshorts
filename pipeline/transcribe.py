"""Local audio transcription using faster-whisper with content-hashed caching."""

from dataclasses import dataclass, field
from pathlib import Path
from faster_whisper import WhisperModel

try:
    from pipeline.cache import get_cache_dir, load_cached, save_cached
except ImportError:
    get_cache_dir = load_cached = save_cached = None


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
    cache_enabled: bool = True,
) -> list[Segment]:
    cache_dir = None
    if cache_enabled and get_cache_dir is not None:
        cache_dir = get_cache_dir(video_path)
        cached_data = load_cached(cache_dir, "transcript.json")
        if cached_data is not None:
            print("      -> Using cached transcription result.")
            return [
                Segment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    words=[Word(w["start"], w["end"], w["text"]) for w in s.get("words", [])],
                )
                for s in cached_data
            ]

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, _info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words = [Word(w.start, w.end, w.word.strip()) for w in (seg.words or []) if w.word.strip()]
        segments.append(Segment(seg.start, seg.end, seg.text.strip(), words))

    if cache_enabled and cache_dir is not None and save_cached is not None:
        serialized = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [{"start": w.start, "end": w.end, "text": w.text} for w in s.words],
            }
            for s in segments
        ]
        save_cached(cache_dir, "transcript.json", serialized)

    return segments
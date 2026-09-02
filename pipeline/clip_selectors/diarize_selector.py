"""
Speaker diarization selector.

Scores a candidate window by turn-taking density (speaker changes per
second): dialogue with frequent back-and-forth exchanges scores higher
than long monologue, which tends to convert better for podcast/interview
style short-form content.

Requires: pyannote.audio, torch, and an HF_TOKEN with access to the gated
pyannote/speaker-diarization-3.1 model (see requirements-extra.txt).
The heavy pipeline is loaded lazily and cached at module level so
is_available() and score() stay cheap after the first call, and so this
file can be imported even when pyannote isn't installed - is_available()
just returns False.
"""
from typing import Optional
from .base import BaseSelector, Candidate

_pipeline_cache = None
_load_failed = False


def _get_pipeline():
    global _pipeline_cache, _load_failed
    if _pipeline_cache is not None or _load_failed:
        return _pipeline_cache
    try:
        import os
        from pyannote.audio import Pipeline
        _pipeline_cache = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN"),
        )
    except Exception:
        _load_failed = True
        _pipeline_cache = None
    return _pipeline_cache


class DiarizationSelector(BaseSelector):
    name = "diarization"

    def __init__(self):
        self._turns_by_audio_path = {}

    def is_available(self) -> bool:
        return _get_pipeline() is not None

    def _get_turns(self, audio_path: str):
        if audio_path in self._turns_by_audio_path:
            return self._turns_by_audio_path[audio_path]

        pipeline = _get_pipeline()
        diarization = pipeline(audio_path)
        turns = [
            {"speaker": speaker, "start": turn.start, "end": turn.end}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        self._turns_by_audio_path[audio_path] = turns
        return turns

    def score(self, candidate: Candidate, context: dict) -> Optional[float]:
        audio_path = context.get("full_audio_path")
        if not audio_path:
            return None

        turns = self._get_turns(audio_path)
        window_turns = [t for t in turns if t["end"] > candidate.start and t["start"] < candidate.end]
        if not window_turns:
            return None

        speaker_changes = 0
        for i in range(1, len(window_turns)):
            if window_turns[i]["speaker"] != window_turns[i - 1]["speaker"]:
                speaker_changes += 1

        duration = max(0.1, candidate.end - candidate.start)
        changes_per_second = speaker_changes / duration
        # Saturate: 1 change every ~4s (0.25/s) is already a lively exchange
        return min(1.0, changes_per_second / 0.25)

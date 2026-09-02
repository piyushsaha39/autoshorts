"""
Auto-thumbnail selection. Reuses per-timestamp expression-intensity scores
from pipeline/clip_selectors/emotion_selector.py (Phase 1.2) when available -
picks the most "expressive face" frame in the clip's range. Falls back to
the clip's midpoint when no expression data exists, so this never fails
outright even if emotion scoring is disabled.
"""
import subprocess


def pick_best_thumbnail_timestamp(clip_start: float, clip_end: float, expression_scores=None) -> float:
    """
    `expression_scores`: optional list of (timestamp, score) tuples, e.g.
    from EmotionSelector.get_timestamp_scores().
    """
    if expression_scores:
        in_range = [(t, s) for t, s in expression_scores if clip_start <= t <= clip_end]
        if in_range:
            return max(in_range, key=lambda pair: pair[1])[0]
    return (clip_start + clip_end) / 2


def extract_thumbnail(video_path: str, timestamp: float, out_path: str) -> str:
    subprocess.run([
        "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path,
        "-frames:v", "1", "-q:v", "2", out_path,
    ], check=True, capture_output=True)
    return out_path

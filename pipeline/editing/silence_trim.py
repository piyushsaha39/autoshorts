"""
Silence/dead-air trimming. Reuses word-level timestamps you already get
from faster-whisper — no new transcription work, just a cut-list built
from gaps between words, then a concat-demuxer render.

This is a NEW rendering path, invoked only when
config.editing.silence_trim.enabled is true, and only AFTER your existing
crop/vertical-format render.py step (feed it the already-cropped clip).
When the flag is off, render.py's current single-pass output is untouched.
"""
import os
import subprocess
import tempfile


def find_keep_segments(words: list, clip_start: float, clip_end: float,
                        min_gap_ms: int = 700, padding_ms: int = 120) -> list:
    """
    `words`: list of {"start": float, "end": float, ...} dicts from
    faster-whisper, already filtered to those inside [clip_start, clip_end].

    Returns a list of (start, end) ranges to KEEP: gaps between
    consecutive words larger than `min_gap_ms` are cut, with `padding_ms`
    of breathing room preserved around each remaining group so cuts don't
    clip the start/end of speech.
    """
    if not words:
        return [(clip_start, clip_end)]

    pad = padding_ms / 1000.0
    min_gap = min_gap_ms / 1000.0
    segments = []
    seg_start = max(clip_start, words[0]["start"] - pad)
    prev_end = words[0]["end"]

    for w in words[1:]:
        gap = w["start"] - prev_end
        if gap > min_gap:
            segments.append((seg_start, min(prev_end + pad, clip_end)))
            seg_start = max(clip_start, w["start"] - pad)
        prev_end = w["end"]

    segments.append((seg_start, min(prev_end + pad, clip_end)))
    return segments


def render_trimmed_clip(source_path: str, keep_segments: list, out_path: str,
                         video_codec: str = "libx264", crf: int = 18) -> str:
    """
    Cuts `keep_segments` out of `source_path` and concatenates them back to
    back via ffmpeg's concat demuxer (more reliable across builds than a
    single complex filter_complex trim/concat expression).
    """
    with tempfile.TemporaryDirectory() as tmp:
        part_paths = []
        for i, (s, e) in enumerate(keep_segments):
            part_path = os.path.join(tmp, f"part_{i:03d}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                "-i", source_path,
                "-c:v", video_codec, "-crf", str(crf),
                "-c:a", "aac",
                part_path,
            ], check=True, capture_output=True)
            part_paths.append(part_path)

        concat_list = os.path.join(tmp, "concat.txt")
        with open(concat_list, "w") as f:
            for p in part_paths:
                f.write(f"file '{p}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c", "copy", out_path,
        ], check=True, capture_output=True)

    return out_path

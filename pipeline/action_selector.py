"""Action highlight selection via YAMNet audio event detection and sliding window analysis."""

import os
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
import tensorflow_hub as hub

try:
    from pipeline.cache import get_cache_dir, load_cached, save_cached
except ImportError:
    get_cache_dir = load_cached = save_cached = None


class ActionClip:
    def __init__(self, start: float, end: float, title: str, score: float = 1.0, hook: str = "Action Highlight"):
        self.start = float(start)
        self.end = float(end)
        self.title = title
        self.score = float(score)
        self.hook = hook


def extract_audio(video_path: Path, out_wav: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def select_action_clips(
    video_path: Path,
    work_dir: Path,
    num_clips: int = 6,
    min_len: int = 10,
    max_len: int = 60,
    cache_enabled: bool = True,
) -> list[ActionClip]:
    cache_dir = None
    if cache_enabled and get_cache_dir is not None:
        cache_dir = get_cache_dir(video_path)
        cached_clips = load_cached(cache_dir, "yamnet.json")
        if cached_clips is not None:
            print("      -> Using cached YAMNet action selections.")
            return [
                ActionClip(
                    start=c["start"],
                    end=c["end"],
                    title=c["title"],
                    score=c.get("score", 1.0),
                    hook=c.get("hook", "Action Highlight"),
                )
                for c in cached_clips
            ][:num_clips]

    print("      -> Extracting 16kHz mono audio for YAMNet...")
    wav_path = work_dir / "temp_audio.wav"
    extract_audio(video_path, wav_path)

    print("      -> Loading YAMNet model from Kaggle...")
    model = hub.load("https://kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1")

    wav_data, _ = sf.read(wav_path)
    waveform = wav_data.astype(np.float32)

    print("      -> Analyzing audio events (Sliding Window Algorithm)...")
    scores, _, _ = model(waveform)
    scores_np = scores.numpy()

    action_intensity = np.sum(scores_np[:, 137:], axis=1)

    frame_duration = 0.48
    total_frames = len(action_intensity)
    clip_duration = min(max_len, max(min_len, 25))
    window_frames = int(clip_duration / frame_duration)

    if total_frames < window_frames:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return [ActionClip(0.0, total_frames * frame_duration, "Action Event 1")]

    window_scores = []
    stride_frames = int(2.0 / frame_duration)

    for start_f in range(0, total_frames - window_frames + 1, max(1, stride_frames)):
        end_f = start_f + window_frames
        avg_score = float(np.mean(action_intensity[start_f:end_f]))
        window_scores.append((start_f, end_f, avg_score))

    window_scores.sort(key=lambda x: x[2], reverse=True)

    selected_intervals = []
    clips: list[ActionClip] = []

    for start_f, end_f, score in window_scores:
        start_sec = round(start_f * frame_duration, 2)
        end_sec = round(end_f * frame_duration, 2)

        overlap = any(max(start_sec, s) < min(end_sec, e) for s, e in selected_intervals)
        if not overlap:
            selected_intervals.append((start_sec, end_sec))
            clips.append(ActionClip(start_sec, end_sec, f"Action Highlight {len(clips)+1}", score=score))
            if len(clips) >= num_clips:
                break

    if os.path.exists(wav_path):
        os.remove(wav_path)

    clips.sort(key=lambda x: x.start)

    if cache_enabled and cache_dir is not None and save_cached is not None:
        serialized = [
            {"start": c.start, "end": c.end, "title": c.title, "score": c.score, "hook": c.hook}
            for c in clips
        ]
        save_cached(cache_dir, "yamnet.json", serialized)

    return clips
import os
import subprocess
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

class ActionClip:
    def __init__(self, start, end, title, score=1.0, hook="Action Highlight"):
        self.start = float(start)
        self.end = float(end)
        self.title = title
        self.score = float(score)
        self.hook = hook

def extract_audio(video_path, out_wav):
    """Extracts 16kHz mono audio required by YAMNet."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def select_action_clips(video_path, work_dir, num_clips=6, min_len=10, max_len=60):
    print("      -> Extracting 16kHz mono audio for YAMNet...")
    wav_path = work_dir / "temp_audio.wav"
    extract_audio(video_path, wav_path)

    print("      -> Loading YAMNet model from Kaggle...")
    model = hub.load("https://kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1")

    wav_data, sample_rate = sf.read(wav_path)
    waveform = wav_data.astype(np.float32)

    print("      -> Analyzing audio events (Sliding Window Algorithm)...")
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()

    # Classes 137 to 521 contain intense sounds (Music, Impacts, Weapons, Explosions, Roars)
    action_intensity = np.sum(scores_np[:, 137:], axis=1)

    frame_duration = 0.48
    total_frames = len(action_intensity)
    
    # Target clip duration (Defaulting to ~25s for standard YouTube Shorts)
    clip_duration = min(max_len, max(min_len, 25)) 
    window_frames = int(clip_duration / frame_duration)

    if total_frames < window_frames:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return [ActionClip(0, total_frames * frame_duration, "Action Event 1")]

    # 1. Calculate a rolling average of action intensity for every possible sliding window
    window_scores = []
    stride_frames = int(2.0 / frame_duration) # Slide the window forward by 2 seconds at a time

    for start_f in range(0, total_frames - window_frames + 1, max(1, stride_frames)):
        end_f = start_f + window_frames
        avg_score = np.mean(action_intensity[start_f:end_f])
        window_scores.append((start_f, end_f, avg_score))

    # 2. Sort the windows by the highest average energy
    window_scores.sort(key=lambda x: x[2], reverse=True)

    # 3. Non-Maximum Suppression (Filter out overlapping clips)
    selected_intervals = []
    clips = []

    for start_f, end_f, score in window_scores:
        start_sec = round(start_f * frame_duration, 2)
        end_sec = round(end_f * frame_duration, 2)

        # Check if this high-energy window overlaps with any clip we've already selected
        overlap = False
        for s, e in selected_intervals:
            if max(start_sec, s) < min(end_sec, e): 
                overlap = True
                break
        
        if not overlap:
            selected_intervals.append((start_sec, end_sec))
            clips.append(ActionClip(start_sec, end_sec, f"Action Highlight {len(clips)+1}", score=score))
            if len(clips) >= num_clips:
                break

    if os.path.exists(wav_path):
        os.remove(wav_path)

    # Sort clips chronologically so they render in order
    clips.sort(key=lambda x: x.start)
    return clips
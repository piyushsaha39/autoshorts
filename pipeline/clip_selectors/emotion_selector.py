"""
Facial reaction selector — a signal independent of audio, useful for
gaming/reaction content where the payoff is a face, not a line of dialogue.

Reuses simple, explainable heuristics on MediaPipe face landmarks rather
than a separate emotion-classification model, to avoid adding a second
heavy dependency on top of the MediaPipe you already use for cropping:
  - mouth_open_ratio: vertical lip distance / face height  -> surprise/laughter
  - smile_width_ratio: mouth corner distance / face width  -> smiling/laughing

Frames are sampled sparsely (every `sample_interval_s`) rather than every
frame, since this is a coarse per-window signal, not frame-accurate crop
tracking (that's still MediaPipe's existing job in crop.py/tracker.py).
"""
from typing import Optional
import subprocess
import tempfile
import os
from .base import BaseSelector, Candidate

_face_mesh_available = None


def _mediapipe_ready() -> bool:
    global _face_mesh_available
    if _face_mesh_available is not None:
        return _face_mesh_available
    try:
        import mediapipe  # noqa: F401
        _face_mesh_available = True
    except ImportError:
        _face_mesh_available = False
    return _face_mesh_available


def _extract_sample_frames(video_path: str, start: float, end: float, interval_s: float):
    """Yields (timestamp, frame_bgr) tuples via ffmpeg + numpy, sampled at interval_s."""
    import numpy as np
    duration = end - start
    n_samples = max(1, int(duration / interval_s))
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_samples):
            t = start + i * interval_s
            frame_path = os.path.join(tmp, f"f_{i:04d}.png")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path,
                 "-frames:v", "1", frame_path],
                check=True, capture_output=True,
            )
            if os.path.exists(frame_path):
                import cv2
                frame = cv2.imread(frame_path)
                if frame is not None:
                    yield t, frame


def _expression_intensity(landmarks) -> float:
    """
    landmarks: MediaPipe FaceLandmarker result for one face.
    Returns a 0..1 heuristic "expressiveness" score (mouth-open + smile-width).
    """
    pts = landmarks.landmark
    # Landmark indices for MediaPipe FaceMesh (468-point model)
    top_lip, bottom_lip = pts[13], pts[14]
    left_mouth, right_mouth = pts[61], pts[291]
    top_face, bottom_face = pts[10], pts[152]

    face_height = max(1e-4, abs(bottom_face.y - top_face.y))
    mouth_open_ratio = abs(bottom_lip.y - top_lip.y) / face_height
    smile_width_ratio = abs(right_mouth.x - left_mouth.x) / face_height

    # Empirically reasonable saturation points, not physiologically exact -
    # tune against your own footage before relying on absolute values.
    score = min(1.0, mouth_open_ratio / 0.15) * 0.6 + min(1.0, smile_width_ratio / 0.6) * 0.4
    return score


class EmotionSelector(BaseSelector):
    name = "emotion"

    def __init__(self, sample_interval_s: float = 0.5):
        self.sample_interval_s = sample_interval_s
        self._timestamp_scores = []   # populated as (t, score) pairs; reusable by thumbnail.py

    def is_available(self) -> bool:
        return _mediapipe_ready()

    def get_timestamp_scores(self):
        """Exposed for Phase 5.3 (auto-thumbnail) to reuse without recomputation."""
        return self._timestamp_scores

    def score(self, candidate: Candidate, context: dict) -> Optional[float]:
        video_path = context.get("full_video_path")
        if not video_path:
            return None

        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        intensities = []
        for t, frame_bgr in _extract_sample_frames(video_path, candidate.start, candidate.end, self.sample_interval_s):
            import cv2
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue
            intensity = _expression_intensity(result.multi_face_landmarks[0])
            intensities.append(intensity)
            self._timestamp_scores.append((t, intensity))

        face_mesh.close()

        if not intensities:
            return None   # no face detected in this window - not this selector's call to make
        return max(intensities)   # peak reaction, not average - one big laugh matters more than a flat baseline

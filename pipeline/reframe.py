"""Decide how to reframe a 16:9 (or any wide) clip into 9:16 vertical.

Strategy: sample a handful of frames across the clip and run on-device face
detection (mediapipe, free/local/CPU). If a face is reliably found, crop the
clip horizontally so the face stays centered ("smart crop" -- the same idea
tools like Opus Clip use). If no face is reliably found (slides, gameplay,
screen recordings, wide group shots), fall back to a blurred-background pad
so nothing gets cropped out.
"""
import cv2
import mediapipe as mp
import numpy as np


def analyze_crop(video_path: str, start: float, end: float, samples: int = 6):
    """Returns ("crop", x_center_ratio) or ("pad", None).

    x_center_ratio is the median horizontal face-center position (0..1,
    fraction of frame width) across the sampled frames. A single static
    crop position per clip is used rather than a moving one -- simpler,
    robust, and plenty for typical 10-60s talking-head footage where the
    speaker doesn't walk across frame.
    """
    cap = cv2.VideoCapture(video_path)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    duration = max(end - start, 0.1)
    centers: list[float] = []
    try:
        for i in range(samples):
            t = start + (i + 0.5) * duration / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.detections:
                box = result.detections[0].location_data.relative_bounding_box
                centers.append(box.xmin + box.width / 2)
    finally:
        cap.release()
        detector.close()

    if len(centers) < max(2, samples * 0.4):
        return "pad", None
    return "crop", float(np.median(centers))

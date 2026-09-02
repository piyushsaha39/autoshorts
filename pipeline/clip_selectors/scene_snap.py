"""
Scene-boundary snapping: nudges selected candidate start/end timestamps to
land on actual cut points instead of arbitrary mid-shot moments.

This is NOT a BaseSelector — it doesn't score candidates, it adjusts the
timestamps of whichever candidates were already chosen (by legacy mode or
ensemble mode, either way). It's a pure post-processing step, which is
why it's safe to enable independently of everything else in Phase 1.
"""
import json
import os


def detect_scenes(video_path: str, cache_dir: str = None) -> list:
    """
    Returns a sorted list of scene-boundary timestamps (seconds) using
    PySceneDetect's ContentDetector. Cached to <cache_dir>/scenes.json
    when cache_dir is given (see pipeline/cache.py) since this runs once
    over the full source video regardless of how many clips you extract.
    """
    if cache_dir:
        cache_file = os.path.join(cache_dir, "scenes.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)

    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    boundaries = sorted(set(
        [round(start.get_seconds(), 3) for start, _ in scene_list] +
        [round(end.get_seconds(), 3) for _, end in scene_list]
    ))

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "scenes.json"), "w") as f:
            json.dump(boundaries, f)

    return boundaries


def snap_to_scene(start: float, end: float, boundaries: list, tolerance: float = 1.5):
    """
    Nudges `start`/`end` to the nearest boundary within `tolerance` seconds.
    Leaves the original timestamp untouched if nothing is close enough -
    this fallback is what keeps the feature non-destructive by construction:
    worst case, clips look exactly like they do today.
    """
    def nearest(t):
        if not boundaries:
            return t
        closest = min(boundaries, key=lambda b: abs(b - t))
        return closest if abs(closest - t) <= tolerance else t

    return nearest(start), nearest(end)

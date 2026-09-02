"""
Active-speaker crop switching for multi-person footage.

This assumes your existing pipeline/crop.py or tracker.py exposes (or can
be made to expose with a small change) a per-frame list of tracked face
boxes, e.g.:

    face_tracks_by_frame = {
        123: [{"track_id": 0, "x": .., "y": .., "w": .., "h": ..,
               "mouth_open_ratio": ..}, ...],
        ...
    }

and that pipeline/clip_selectors/diarize_selector.py (Phase 1.1) gives you
speaker turns:

    speaker_turns = [{"speaker": "SPEAKER_00", "start": 12.4, "end": 15.1}, ...]

Because face-track IDs and diarization speaker labels come from two
different systems with no shared identity, a mapping step is needed. The
heuristic below links them via mouth-movement correlation: whichever
tracked face moves its mouth the most during a speaker's turn is that
speaker's face. It's not perfect, but it degrades gracefully — see
`fallback_box` in get_crop_target_for_frame below.
"""
from collections import defaultdict
import numpy as np


def build_speaker_to_track_map(face_tracks_by_frame: dict, speaker_turns: list, fps: float) -> dict:
    """Returns {"SPEAKER_00": track_id, ...} via majority vote across all of a speaker's turns."""
    votes = defaultdict(list)

    for turn in speaker_turns:
        start_frame = int(turn["start"] * fps)
        end_frame = int(turn["end"] * fps)
        motion_by_track = defaultdict(list)

        for f in range(start_frame, end_frame):
            for face in face_tracks_by_frame.get(f, []):
                motion_by_track[face["track_id"]].append(face.get("mouth_open_ratio", 0.0))

        variances = {tid: float(np.var(v)) for tid, v in motion_by_track.items() if len(v) > 2}
        if variances:
            best_track = max(variances, key=variances.get)
            votes[turn["speaker"]].append(best_track)

    return {
        speaker: max(set(tracks), key=tracks.count)
        for speaker, tracks in votes.items()
        if tracks
    }


def get_crop_target_for_frame(frame_idx: int, fps: float, speaker_turns: list,
                               speaker_to_track: dict, face_tracks_by_frame: dict,
                               fallback_box):
    """
    Returns the (x, y, w, h) box to center the vertical crop on for this
    frame: the active speaker's face if resolvable, else `fallback_box`
    (your existing MediaPipe primary-subject box). Falling back to the
    current behavior whenever speaker resolution fails is what keeps this
    feature non-destructive.
    """
    t = frame_idx / fps
    active_speaker = next(
        (s["speaker"] for s in speaker_turns if s["start"] <= t <= s["end"]), None
    )
    if active_speaker is None:
        return fallback_box

    track_id = speaker_to_track.get(active_speaker)
    if track_id is None:
        return fallback_box

    for face in face_tracks_by_frame.get(frame_idx, []):
        if face["track_id"] == track_id:
            return (face["x"], face["y"], face["w"], face["h"])

    return fallback_box

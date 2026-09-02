"""
Emphasis punch-ins: subtle, brief zoom keyframes at punchline/loud moments.

RISK NOTE: zoompan expression syntax and available time variables differ
across ffmpeg builds. This is written against `on` (frame count), which
is universally supported, rather than a `t`/`time` variable, whose
support is less consistent. Test on one clip with your ffmpeg version
before enabling pipeline-wide - this is the single riskiest piece of
ffmpeg in this whole upgrade set, more so than anything else here.
"""
from typing import List


def build_zoompan_filter(punch_times: List[float], clip_duration: float,
                          fps: int = 30, zoom_amount: float = 1.08,
                          ramp_seconds: float = 0.15):
    """
    Returns an ffmpeg filter string implementing a triangular zoom ramp
    (1.0 -> zoom_amount -> 1.0) centered on each timestamp in
    `punch_times` (seconds, relative to clip start). Returns None if
    `punch_times` is empty, so callers can skip filter insertion entirely.
    """
    if not punch_times:
        return None

    total_frames = int(clip_duration * fps)
    ramp_frames = max(1, int(ramp_seconds * fps))

    conditions = []
    for t in punch_times:
        center_frame = int(t * fps)
        start_f = max(0, center_frame - ramp_frames)
        end_f = min(total_frames, center_frame + ramp_frames)
        conditions.append(
            f"if(between(on,{start_f},{end_f}),"
            f"1+({zoom_amount}-1)*(1-abs(on-{center_frame})/{ramp_frames}),1)"
        )

    # Fold multiple punches into one expression, keeping whichever gives
    # the larger zoom at each frame (punches should be spaced apart in
    # practice, so overlap is rare, but this keeps behavior sane if not).
    z_expr = conditions[0]
    for c in conditions[1:]:
        z_expr = f"max({c},{z_expr})"

    return (
        f"zoompan=z='{z_expr}':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s=1080x1920:fps={fps}"
    )

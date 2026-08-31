"""Cut a highlight out of the source video, reframe it to 9:16, burn in the
karaoke captions, and encode the final Short -- all with ffmpeg (free)."""
import subprocess
from pathlib import Path

from .reframe import analyze_crop

TARGET_W, TARGET_H = 1080, 1920


def _probe_dimensions(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def _build_filter(mode: str, crop_x_ratio: float, src_w: int, src_h: int) -> str:
    out_w = round(src_h * TARGET_W / TARGET_H)  # 9:16-wide slice at full source height

    if mode == "crop" and out_w <= src_w:
        x = crop_x_ratio * src_w - out_w / 2
        x = max(0, min(x, src_w - out_w))
        return f"crop={out_w}:{src_h}:{int(x)}:0,scale={TARGET_W}:{TARGET_H}"

    # Pad mode: whole frame fits inside a blurred, zoomed copy of itself
    # (the standard "blurred bars" look for content that can't be cropped).
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},gblur=sigma=20[bg2];"
        f"[fg]scale={TARGET_W}:-2[fg2];"
        f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
    )


def render_clip(
    source_video: Path,
    start: float,
    end: float,
    ass_path: Path,
    out_path: Path,
    use_nvenc: bool = False,
) -> None:
    src_w, src_h = _probe_dimensions(source_video)
    mode, crop_x_ratio = analyze_crop(str(source_video), start, end)
    vf = _build_filter(mode, crop_x_ratio or 0.5, src_w, src_h)
    # escape the path for ffmpeg's filter-graph string (colons need escaping on all OSes)
    escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
    vf += f",subtitles='{escaped_ass}'"

    codec = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
        if use_nvenc
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(source_video), "-t", str(end - start),
        "-vf", vf,
        *codec,
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

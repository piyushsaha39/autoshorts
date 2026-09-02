"""Render vertical 9:16 clips using FFmpeg with black background framing and hardware acceleration."""

from pathlib import Path
import subprocess

try:
    from pipeline.hw_detect import get_video_encoder
except ImportError:
    def get_video_encoder(prefer_hw: bool = False) -> str:
        return "h264_nvenc" if prefer_hw else "libx264"

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


def _build_filter(
    src_w: int,
    src_h: int,
    crop_to_speaker: bool = False,
    crop_x_ratio: float = 0.5,
    zoom_landscape_200: bool = True,
) -> str:
    """Build FFmpeg video filter chain to produce strict 1080x1920 with black background."""
    is_landscape = src_w > src_h

    # Mode 1: Center speaker crop (if enabled by user)
    if crop_to_speaker:
        out_w = round(src_h * TARGET_W / TARGET_H)
        if out_w <= src_w:
            x = crop_x_ratio * src_w - out_w / 2
            x = max(0, min(x, src_w - out_w))
            return f"crop={out_w}:{src_h}:{int(x)}:0,scale={TARGET_W}:{TARGET_H}"

    # Mode 2: Landscape video zoomed 200% on black background (User Default)
    if is_landscape and zoom_landscape_200:
        # Scale to 200% of target width, crop horizontally to 1080, pad height to 1920 with black
        return (
            f"scale={TARGET_W * 2}:-2,"
            f"crop={TARGET_W}:min(ih\\,{TARGET_H}):(iw-{TARGET_W})/2:(ih-min(ih\\,{TARGET_H}))/2,"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    # Mode 3: Fit whole frame inside 1080x1920 with black bars (no blur)
    return (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def render_clip(
    source_video: Path,
    start: float,
    end: float,
    ass_path: Path,
    out_path: Path,
    crop_to_speaker: bool = False,
    zoom_landscape_200: bool = True,
    prefer_hw: bool = False,
) -> None:
    src_w, src_h = _probe_dimensions(source_video)

    crop_x_ratio = 0.5
    if crop_to_speaker:
        _, crop_x_ratio_found = analyze_crop(str(source_video), start, end)
        crop_x_ratio = crop_x_ratio_found or 0.5

    vf = _build_filter(
        src_w,
        src_h,
        crop_to_speaker=crop_to_speaker,
        crop_x_ratio=crop_x_ratio,
        zoom_landscape_200=zoom_landscape_200,
    )

    # Escape path for FFmpeg filtergraph
    escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
    vf += f",subtitles='{escaped_ass}'"

    encoder = get_video_encoder(prefer_hw=prefer_hw)
    if encoder == "h264_nvenc":
        codec_opts = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    else:
        codec_opts = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source_video),
        "-t", str(end - start),
        "-vf", vf,
        *codec_opts,
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
"""
Hardware acceleration detection.

Every function here is designed to NEVER raise and to ALWAYS fall back to
your current, known-good CPU path (torch absent -> "cpu", ffmpeg encoder
query fails -> "libx264"). This is what makes hardware_accel.enabled: true
safe to flip on any machine, including ones with no GPU at all.
"""
import subprocess


def get_device() -> str:
    """Returns 'cuda' if a GPU is available to torch, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _encoder_actually_works(codec: str) -> bool:
    """
    `ffmpeg -encoders` only reports encoders ffmpeg was COMPILED with - it
    says nothing about whether the hardware is physically present. On a
    machine with no NVIDIA GPU, h264_nvenc can still show up in that list
    and then fail at encode time. This runs a real 1-frame encode to
    confirm the codec is actually usable before trusting it.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
             "-c:v", codec, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_video_encoder(prefer_hw: bool = True, fallback: str = "libx264") -> str:
    """
    Returns the best available ffmpeg video encoder.

    Always returns `fallback` (your current encoder) if `prefer_hw` is False,
    if ffmpeg can't be queried, or if no hardware encoder both (a) reports
    as compiled-in AND (b) passes a real functional test. This keeps
    existing output reproducible whenever hardware_accel.enabled is false
    in config/pipeline.yaml, and keeps a render from ever failing outright
    on a machine where the "available" hardware encoder doesn't actually work.
    """
    if not prefer_hw:
        return fallback

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        available = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return fallback

    # Ordered by how commonly they're available (NVIDIA -> Intel -> Apple)
    for candidate in ("h264_nvenc", "h264_qsv", "h264_videotoolbox"):
        if candidate in available and _encoder_actually_works(candidate):
            return candidate

    return fallback


if __name__ == "__main__":
    print(f"device: {get_device()}")
    print(f"encoder (hw preferred): {get_video_encoder(prefer_hw=True)}")
    print(f"encoder (hw disabled):  {get_video_encoder(prefer_hw=False)}")

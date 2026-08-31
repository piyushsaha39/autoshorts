"""Download a YouTube video with yt-dlp (free, open-source, no API key)."""
from pathlib import Path
import yt_dlp


def download_video(url: str, out_dir: Path) -> tuple[Path, str]:
    """Downloads the best available <=1080p mp4 and returns (path, title)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'extractor_args': {'youtube': {'player_client': ['android,ios,web']}},
        'cookiefile': 'cookies.txt',
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    if not filepath.exists():
        raise FileNotFoundError(
            f"yt-dlp reported success but {filepath} is missing -- "
            "check that ffmpeg is installed (needed to merge video+audio)."
        )
    return filepath, info.get("title", "video")

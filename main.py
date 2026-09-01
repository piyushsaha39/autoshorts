"""AutoShorts: turn one long YouTube video into several ready-to-post
vertical clips -- fully free, runs on your own machine.

Example:
    python main.py "https://youtube.com/watch?v=XXXXXXXXXXX" --num-clips 6
    python main.py "my_video.mp4" --num-clips 6

See README.md for setup (Python deps, ffmpeg, and picking a free LLM
provider for the highlight-selection step).
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from pipeline.llm_selector import select_clips
from pipeline.action_selector import select_action_clips  # ADD THIS LINE
from pipeline.captions import build_captions
from pipeline.render import render_clip

from dotenv import load_dotenv

from pipeline.download import download_video
from pipeline.transcribe import transcribe
from pipeline.llm_selector import select_clips
from pipeline.captions import build_captions
from pipeline.render import render_clip


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="YouTube video URL or path to local video file")
    ap.add_argument("--num-clips", type=int, default=6)
    ap.add_argument("--min-len", type=int, default=10, help="minimum clip length, seconds")
    ap.add_argument("--max-len", type=int, default=60, help="maximum clip length, seconds")
    ap.add_argument("--whisper-model", default="medium", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--compute-type", default="float16", help="float16 (GPU) or int8 (CPU)")
    ap.add_argument(
        "--llm-model",
        default="groq/llama-3.3-70b-versatile",
        help="any LiteLLM model string, e.g. gemini/gemini-2.5-flash-lite or ollama/qwen2.5:7b-instruct",
    )
    ap.add_argument("--nvenc", action="store_true", help="use NVIDIA hardware encoding (much faster on an Nvidia GPU)")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    work_dir = out_dir / "_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

   # Check if the input is a local file path
    if os.path.exists(args.url):
        print("[1/4] Local video file detected. Copying to workspace...")
        title = Path(args.url).stem
        video_path = work_dir / Path(args.url).name
        
        # Safely copy the file only if the source and destination are different
        if str(Path(args.url).resolve()) != str(video_path.resolve()):
            shutil.copy2(args.url, video_path)
    else:
        print("[1/4] Downloading video...")
        video_path, title = download_video(args.url, work_dir)
        print(f"      -> {title}")

    print("[2/4] Transcribing locally (faster-whisper)...")
    segments = transcribe(
        video_path, model_size=args.whisper_model, device=args.device, compute_type=args.compute_type
    )
    print(f"[3/4] Selecting highlights with {args.llm_model}...")
    candidates = select_clips(
        segments, model=args.llm_model, num_clips=args.num_clips, min_len=args.min_len, max_len=args.max_len
    )
    
    # --- NEW FALLBACK LOGIC ---
    if not candidates:
        print(f"[3.5/4] No dialogue detected. Triggering Audio Event Detection fallback...")
        candidates = select_action_clips(
            video_path, work_dir, num_clips=args.num_clips, min_len=args.min_len, max_len=args.max_len
        )
    # --------------------------

    candidates = candidates[: args.num_clips]
    print(f"      -> {len(candidates)} candidates kept")

    print(f"[4/4] Rendering {len(candidates)} clip(s)...")
    metadata = []
    for i, c in enumerate(candidates, start=1):
        clip_out = out_dir / f"clip_{i:02d}.mp4"
        ass_path = work_dir / f"clip_{i:02d}.ass"
        build_captions(segments, c.start, c.end, str(ass_path))
        render_clip(video_path, c.start, c.end, ass_path, clip_out, use_nvenc=args.nvenc)
        metadata.append(
            {
                "file": clip_out.name,
                "start": round(c.start, 1),
                "end": round(c.end, 1),
                "duration": round(c.end - c.start, 1),
                "title": c.title,
                "hook": c.hook,
                "score": c.score,
            }
        )
        print(f'      + {clip_out.name}  ({c.end - c.start:.0f}s)  "{c.title}"')

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. Clips + metadata.json are in {out_dir}/")


if __name__ == "__main__":
    main()
"""AutoShorts: Automated long-form to 9:16 vertical short-form video pipeline."""

import argparse
from datetime import datetime
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow C++ info and warnings
from pathlib import Path
import re
import shutil
import sys
import yaml
from dotenv import load_dotenv
import litellm

from pipeline.download import download_video
from pipeline.transcribe import transcribe
from pipeline.llm_selector import select_clips
from pipeline.action_selector import select_action_clips
from pipeline.captions import build_captions
from pipeline.render import render_clip

# --- NEW IMPORTS ---
try:
    from pipeline.clip_selectors import scene_snap
    from pipeline.clip_selectors.hook_scorer import score_hook_strength
    from pipeline.distribution.metadata_gen import generate_clip_metadata
    from pipeline.cache import get_cache_dir
except ImportError:
    pass


def load_config(path: str = "config/pipeline.yaml") -> dict:
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "performance": {"cache": {"enabled": True}, "hardware_accel": {"enabled": False}},
        "captions": {"style": "default"},
        "scene_snap": {"enabled": True, "tolerance_seconds": 1.5},
        "distribution": {"metadata_gen": {"enabled": True}, "platform_profile": "youtube_shorts"},
    }

def load_profiles(path: str = "config/platform_profiles.yaml") -> dict:
    profile_path = Path(path)
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {"youtube_shorts": {"resolution": [1080, 1920], "caption_safe_zone_bottom_px": 480}}

def ask_user(prompt: str, default: str) -> str:
    try:
        val = input(f"{prompt} [Default: {default}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default

def sanitize_filename(name: str) -> str:
    clean = re.sub(r'[\\/*?:"<>| ]+', "_", name)
    return clean.strip("_")[:40] or "video_run"

def main() -> None:
    load_dotenv()
    config = load_config()
    profiles = load_profiles()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", nargs="?", default=None)
    ap.add_argument("--num-clips", type=int, default=None)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--whisper-model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--llm-model", default="ollama/qwen2.5:7b-instruct") # Ollama Default
    ap.add_argument("--nvenc", action="store_true")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    video_input = args.url
    if not video_input:
        video_input = ask_user("Enter video path or YouTube URL", "").strip('"\'')
        if not video_input:
            print("Error: No video source provided.")
            sys.exit(1)

    if args.num_clips is None:
        try:
            num_clips = int(ask_user("At max how many clips do you need?", "6"))
        except ValueError:
            num_clips = 6
    else:
        num_clips = args.num_clips

    crop_to_speaker = ask_user("Should it crop to a person speaking? (y/N)", "N").lower() in ("y", "yes", "true", "1")
    zoom_landscape = ask_user("Zoom in 200% with black background for landscape video to fit 9:16? (Y/n)", "Y").lower() not in ("n", "no", "false", "0")

    use_hw = args.nvenc or config.get("performance", {}).get("hardware_accel", {}).get("enabled", False)
    cache_enabled = config.get("performance", {}).get("cache", {}).get("enabled", True)
    caption_style = config.get("captions", {}).get("style", "default")
    
    active_profile = profiles.get(config.get("distribution", {}).get("platform_profile", "youtube_shorts"), {})

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    temp_stage = base_out_dir / "_temp_stage"
    temp_stage.mkdir(parents=True, exist_ok=True)

    if os.path.exists(video_input):
        print("[1/4] Local video file detected.")
        raw_title = Path(video_input).stem
        source_video_path = Path(video_input).resolve()
    else:
        print("[1/4] Downloading video...")
        source_video_path, raw_title = download_video(video_input, temp_stage)
        print(f"      -> {raw_title}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{sanitize_filename(raw_title)}_{timestamp_str}"
    run_out_dir = base_out_dir / folder_name
    work_dir = run_out_dir / "_work"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / source_video_path.name
    if source_video_path != video_path.resolve():
        shutil.copy2(source_video_path, video_path)

    if temp_stage.exists() and temp_stage != work_dir:
        shutil.rmtree(temp_stage, ignore_errors=True)

    meta_path = run_out_dir / "metadata.json"
    metadata = {
        "title": raw_title,
        "source": str(video_input),
        "target_platform": config.get("distribution", {}).get("platform_profile", "youtube_shorts"),
        "last_completed_stage": "initialized",
        "clips": [],
    }

    print("[2/4] Transcribing locally (faster-whisper)...")
    segments = transcribe(
        video_path,
        model_size=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        cache_enabled=cache_enabled,
    )
    metadata["last_completed_stage"] = "transcription"

    print(f"[3/4] Selecting dialogue highlights with {args.llm_model}...")
    llm_candidates = select_clips(
        segments,
        model=args.llm_model,
        num_clips=num_clips,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    print("[3.5/4] Scanning for action-heavy highlights (YAMNet)...")
    action_candidates = select_action_clips(
        video_path,
        work_dir,
        num_clips=num_clips,
        min_len=args.min_len,
        max_len=args.max_len,
        cache_enabled=cache_enabled,
    )

    # Combine candidates from both the LLM (dialogue) and YAMNet (action)
    all_candidates = llm_candidates + action_candidates

    # --- UPGRADE: Scene Snapping ---
    if config.get("scene_snap", {}).get("enabled", False) and 'scene_snap' in sys.modules:
        print("      -> Snapping clip boundaries to physical scene cuts...")
        boundaries = scene_snap.detect_scenes(video_path, cache_dir=get_cache_dir(video_path) if cache_enabled else None)
        for c in all_candidates:
            c.start, c.end = scene_snap.snap_to_scene(
                c.start, c.end, boundaries,
                tolerance=config["scene_snap"].get("tolerance_seconds", 1.5)
            )

    # --- UPGRADE: Hook Scoring ---
    if 'score_hook_strength' in sys.modules:
        print("      -> Scoring hook strength via LLM...")
        for c in all_candidates:
            hook_score = score_hook_strength(
                litellm, args.llm_model,
                first_seconds_text=getattr(c, 'hook', "Opening seconds"),
                full_candidate_text=getattr(c, 'title', "")
            )
            if hook_score:
                c.score += (hook_score * 0.5)

    # Sort all candidates from highest to lowest score
    all_candidates.sort(key=lambda x: getattr(x, 'score', 1.0), reverse=True)

    # Filter out overlapping clips (so a dialogue clip and action clip don't cover the same timeframe)
    final_candidates = []
    for c in all_candidates:
        overlap = any(max(c.start, fc.start) < min(c.end, fc.end) for fc in final_candidates)
        if not overlap:
            final_candidates.append(c)
        if len(final_candidates) >= num_clips:
            break

    candidates = final_candidates
    print(f"      -> {len(candidates)} best non-overlapping candidates kept from both engines")
    metadata["last_completed_stage"] = "selection"

    # --- MISSING RENDERING BLOCK RESTORED ---
    print(f"[4/4] Rendering {len(candidates)} clip(s)...")
    for i, c in enumerate(candidates, start=1):
        clip_out = run_out_dir / f"clip_{i:02d}.mp4"
        ass_path = work_dir / f"clip_{i:02d}.ass"

        build_captions(segments, c.start, c.end, str(ass_path), style_name=caption_style)
        render_clip(
            video_path,
            c.start,
            c.end,
            ass_path,
            clip_out,
            crop_to_speaker=crop_to_speaker,
            zoom_landscape_200=zoom_landscape,
            prefer_hw=use_hw,
        )

        clip_data = {
            "file": clip_out.name,
            "start": round(c.start, 2),
            "end": round(c.end, 2),
            "duration": round(c.end - c.start, 2),
            "title": getattr(c, "title", f"Clip {i}"),
            "score": getattr(c, "score", 1.0),
        }

        # --- UPGRADE: Metadata Generation ---
        if config.get("distribution", {}).get("metadata_gen", {}).get("enabled", False) and 'generate_clip_metadata' in sys.modules:
            print(f"      -> Generating titles and hashtags for {clip_out.name}...")
            clip_meta = generate_clip_metadata(
                litellm, args.llm_model, str(getattr(c, 'title', '')),
                platform=config["distribution"]["platform_profile"]
            )
            clip_data.update(clip_meta)

        metadata["clips"].append(clip_data)
        print(f'      + {clip_out.name} ({clip_data["duration"]}s) "{clip_data["title"]}"')

    metadata["last_completed_stage"] = "render"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. All clips and metadata are saved in: {run_out_dir}")

if __name__ == "__main__":
    main()
"""
Content-hash based caching for expensive, deterministic pipeline stages
(transcription, YAMNet event detection, scene detection).

Design goal: re-running main.py on the same source file with different
--num-clips / --min-len should NOT re-run faster-whisper or YAMNet from
scratch. This has zero effect on output correctness - it only skips
recomputation of stages whose input (the source file) hasn't changed.
"""
import hashlib
import json
import os


def compute_file_signature(path: str, sample_bytes: int = 1_000_000) -> str:
    """
    Fast signature for a local video file: size + hash of the first and
    last `sample_bytes`. Deliberately avoids hashing multi-GB files in
    full on every run.

    If your workflow frequently overwrites a file in place with new
    content under the same name and size, swap this for a full sha256
    of the entire file instead - the partial hash trades a small
    collision risk for speed.
    """
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes:
            f.seek(max(0, size - sample_bytes))
            h.update(f.read(sample_bytes))
    return h.hexdigest()[:16]


def get_cache_dir(source_path: str, work_dir: str = "_work/cache") -> str:
    sig = compute_file_signature(source_path)
    cache_dir = os.path.join(work_dir, sig)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_cached(cache_dir: str, name: str):
    """Returns the cached object, or None if it doesn't exist yet."""
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_cached(cache_dir: str, name: str, data) -> str:
    path = os.path.join(cache_dir, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path

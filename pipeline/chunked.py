"""
chunked.py — Chunked R2V generation with last-frame chaining

Generates long videos at full "None" quality by splitting into short chunks,
using the last frame of each chunk as the input portrait for the next.
Chunks are stitched together with a crossfade transition.

Usage (called automatically by run_reel.py when duration > 5s and vram_mode == "none"):
    generate_chunked_r2v(portrait, prompt, total_duration, out_path)
"""

import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.skyreels import generate as skyreels_generate

CHUNK_DURATION = 5   # seconds per chunk (safe limit for "None" vram_mode)
CROSSFADE = 0.5      # seconds of crossfade overlap between chunks


def _get_duration(path: str) -> float:
    """Get video/audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _extract_last_frame(video_path: str, out_png: str) -> str:
    """Extract the last frame of a video as a PNG image."""
    cmd = ["ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
           "-update", "1", "-q:v", "1", out_png]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract last frame from {video_path}")
    return out_png


def _crossfade_videos(video_paths: list, crossfade_dur: float, target_duration: int, out_path: str) -> None:
    """
    Stitch N silent video clips together with crossfade transitions in a single ffmpeg pass.
    Trims the final output to target_duration seconds.
    """
    n = len(video_paths)

    if n == 1:
        import shutil
        shutil.copy(video_paths[0], out_path)
        return

    # Probe actual durations (SkyReels may not produce exactly chunk_duration seconds)
    durations = [_get_duration(v) for v in video_paths]

    # Build -i inputs
    inputs = []
    for v in video_paths:
        inputs.extend(["-i", v])

    # Build chained xfade filter
    # [0:v][1:v]xfade=...[v1];[v1][2:v]xfade=...[v2];...;[vN-2][N-1:v]xfade=...[vout]
    filter_parts = []
    running_duration = durations[0]

    for i in range(1, n):
        in_label = "[0:v]" if i == 1 else f"[v{i-1}]"
        out_label = "[vout]" if i == n - 1 else f"[v{i}]"
        offset = max(0.0, running_duration - crossfade_dur)
        filter_parts.append(
            f"{in_label}[{i}:v]xfade=transition=fade"
            f":duration={crossfade_dur}"
            f":offset={offset:.3f}{out_label}"
        )
        running_duration += durations[i] - crossfade_dur

    filter_complex = ";".join(filter_parts)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-t", str(target_duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-movflags", "+faststart",
            "-an",   # no audio — music is added later by polish.py
            out_path,
        ]
    )

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg crossfade failed")


def generate_chunked_r2v(
    portrait: str,
    prompt: str,
    total_duration: int,
    out_path: str,
    chunk_duration: int = CHUNK_DURATION,
    crossfade: float = CROSSFADE,
    vram_mode: str = "none",
) -> None:
    """
    Generate a long R2V video by chaining short chunks at full quality.

    Each chunk uses the last frame of the previous chunk as its input portrait,
    maintaining visual continuity. Chunks are stitched with crossfade transitions.

    portrait:       path to the character reference PNG
    prompt:         scene/mood prompt passed to SkyReels
    total_duration: desired total video length in seconds
    out_path:       where to write the final stitched MP4
    chunk_duration: seconds per chunk (default 5 — safe for "None" vram_mode)
    crossfade:      crossfade overlap between chunks in seconds (default 0.5)
    vram_mode:      passed through to skyreels.generate (default "none")
    """
    ts = int(time.time())
    chunk_paths = []
    frame_paths = []

    # Number of chunks needed to cover total_duration accounting for crossfade overlap.
    # Effective unique content per chunk (except last) = chunk_duration - crossfade.
    # total = N * chunk_duration - (N-1) * crossfade  →  N = ceil((total - crossfade) / (chunk_duration - crossfade))
    n_chunks = math.ceil((total_duration - crossfade) / (chunk_duration - crossfade))
    n_chunks = max(n_chunks, 1)

    print(f"[chunked] {total_duration}s total → {n_chunks} chunks of {chunk_duration}s "
          f"with {crossfade}s crossfade")

    try:
        current_portrait = portrait

        for i in range(n_chunks):
            chunk_path = f"/tmp/chunk_{ts}_{i}.mp4"
            chunk_paths.append(chunk_path)

            print(f"\n[chunked] chunk {i+1}/{n_chunks}")
            try:
                skyreels_generate(
                    current_portrait, None, prompt, chunk_path,
                    duration=chunk_duration, vram_mode=vram_mode,
                )
            except SystemExit:
                raise RuntimeError(f"SkyReels failed on chunk {i+1}/{n_chunks}")

            # Extract last frame to use as portrait for the next chunk
            if i < n_chunks - 1:
                frame_path = f"/tmp/chunk_{ts}_lastframe_{i}.png"
                frame_paths.append(frame_path)
                _extract_last_frame(chunk_path, frame_path)
                current_portrait = frame_path

        # Stitch all chunks together with crossfade, trimmed to exact target duration
        print(f"\n[chunked] stitching {n_chunks} chunks → {out_path}")
        _crossfade_videos(chunk_paths, crossfade, total_duration, out_path)
        print(f"[chunked] done → {out_path} ({total_duration}s)")

    finally:
        for f in chunk_paths + frame_paths:
            if os.path.exists(f):
                os.remove(f)

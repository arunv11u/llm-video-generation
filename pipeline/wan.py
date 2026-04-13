"""
wan.py — Wan 2.2 I2V (Image-to-Video) wrapper

Takes a starting scene image + text prompt and generates a video with
realistic motion. Designed for cinematic full-body/environment scenes,
not talking heads.

Usage (called by app.py Scene Video tab):
    generate(image, prompt, out_path, duration=15, vram_mode="none")
"""

import glob
import math
import os
import shutil
import subprocess
import sys
import time

WAN_DIR   = os.environ.get("WAN_DIR",   "/workspace/Wan2.2")
WAN_MODEL = os.environ.get("WAN_MODEL", "/workspace/Wan2.2-I2V-A14B")

WAN_FPS = int(os.environ.get("WAN_FPS", "16"))          # frames per second Wan generates at
WAN_SIZE = os.environ.get("WAN_SIZE", "832*480")         # width*height for 9:16 vertical


def generate(image: str, prompt: str, out_path: str,
             duration: int = 15, vram_mode: str = "none") -> None:
    """
    Run Wan 2.2 I2V to produce a scene video from a starting image.

    image:      path to starting scene image (PNG or JPG)
    prompt:     motion/scene prompt (e.g. "woman walks out of pool, slow motion")
    out_path:   where to write the final MP4
    duration:   video length in seconds
    vram_mode:  "none" (default), "offload", or "low_vram"
    """
    script = os.path.join(WAN_DIR, "generate.py")
    if not os.path.exists(script):
        print(f"ERROR: Wan not found at {WAN_DIR}", file=sys.stderr)
        sys.exit(1)

    # Wan requires frame_num to be 4n+1 (e.g. 81, 97, 121...)
    raw_frames = duration * WAN_FPS
    frame_num = round((raw_frames - 1) / 4) * 4 + 1

    cmd = [
        sys.executable, script,
        "--task",      "i2v-A14B",
        "--image",     image,
        "--prompt",    prompt,
        "--size",      WAN_SIZE,
        "--frame_num", str(frame_num),
        "--ckpt_dir",  WAN_MODEL,
        "--save_file", out_path,
    ]

    if vram_mode == "offload":
        cmd.extend(["--offload_model", "True"])
    elif vram_mode == "low_vram":
        cmd.extend(["--offload_model", "True"])   # Wan uses --offload_model for both low-memory modes

    print(f"[wan] running: {' '.join(cmd)}")

    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    result = subprocess.run(cmd, cwd=WAN_DIR, env=env)

    if result.returncode != 0:
        print("ERROR: Wan I2V generation failed.", file=sys.stderr)
        sys.exit(1)

    # If Wan wrote directly to --save_file, we're done
    if os.path.exists(out_path):
        print(f"[wan] saved {out_path}")
        return

    # Fallback: scan output dir for newest MP4 (in case --save_file is not supported)
    out_dir = os.path.join(WAN_DIR, "outputs")
    candidates = glob.glob(os.path.join(out_dir, "**", "*.mp4"), recursive=True)
    if not candidates:
        print("ERROR: Wan produced no output file.", file=sys.stderr)
        sys.exit(1)

    newest = max(candidates, key=os.path.getmtime)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.move(newest, out_path)
    print(f"[wan] saved {out_path}")


CHUNK_DURATION = 5   # seconds per chunk (safe limit for 80GB VRAM)
CROSSFADE = 0.5      # seconds of crossfade overlap between chunks


def _get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _extract_last_frame(video_path: str, out_png: str) -> None:
    cmd = ["ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
           "-update", "1", "-q:v", "1", out_png]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract last frame from {video_path}")


def _crossfade_videos(video_paths: list, crossfade_dur: float, target_duration: int, out_path: str) -> None:
    n = len(video_paths)
    if n == 1:
        shutil.copy(video_paths[0], out_path)
        return

    durations = [_get_duration(v) for v in video_paths]
    inputs = []
    for v in video_paths:
        inputs.extend(["-i", v])

    filter_parts = []
    running_duration = durations[0]
    for i in range(1, n):
        in_label = "[0:v]" if i == 1 else f"[v{i-1}]"
        out_label = "[vout]" if i == n - 1 else f"[v{i}]"
        offset = max(0.0, running_duration - crossfade_dur)
        filter_parts.append(
            f"{in_label}[{i}:v]xfade=transition=fade"
            f":duration={crossfade_dur}:offset={offset:.3f}{out_label}"
        )
        running_duration += durations[i] - crossfade_dur

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", ";".join(filter_parts),
           "-map", "[vout]",
           "-t", str(target_duration),
           "-c:v", "libx264", "-crf", "18", "-preset", "fast",
           "-movflags", "+faststart", "-an",
           out_path]
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg crossfade failed")


def generate_chunked(
    image: str, prompt: str, out_path: str,
    total_duration: int = 15,
    chunk_duration: int = CHUNK_DURATION,
    crossfade: float = CROSSFADE,
    vram_mode: str = "none",
) -> None:
    """
    Generate a long scene video by chaining 5s Wan I2V chunks.
    Each chunk uses the last frame of the previous chunk as its starting image.
    """
    ts = int(time.time())
    chunk_paths = []
    frame_paths = []

    n_chunks = math.ceil((total_duration - crossfade) / (chunk_duration - crossfade))
    n_chunks = max(n_chunks, 1)

    print(f"[wan-chunked] {total_duration}s total → {n_chunks} chunks of {chunk_duration}s "
          f"with {crossfade}s crossfade")

    try:
        current_image = image

        for i in range(n_chunks):
            chunk_path = f"/tmp/wan_chunk_{ts}_{i}.mp4"
            chunk_paths.append(chunk_path)

            print(f"\n[wan-chunked] chunk {i+1}/{n_chunks}")
            generate(current_image, prompt, chunk_path,
                     duration=chunk_duration, vram_mode=vram_mode)

            if i < n_chunks - 1:
                frame_path = f"/tmp/wan_chunk_{ts}_lastframe_{i}.png"
                frame_paths.append(frame_path)
                _extract_last_frame(chunk_path, frame_path)
                current_image = frame_path

        print(f"\n[wan-chunked] stitching {n_chunks} chunks → {out_path}")
        _crossfade_videos(chunk_paths, crossfade, total_duration, out_path)
        print(f"[wan-chunked] done → {out_path} ({total_duration}s)")

    finally:
        for f in chunk_paths + frame_paths:
            if os.path.exists(f):
                os.remove(f)

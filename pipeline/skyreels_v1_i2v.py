"""
skyreels_v1_i2v.py — SkyReels V1 I2V (Image-to-Video) wrapper

Takes a starting scene image + text prompt and generates a human-centric
cinematic video. Trained on 10M+ film/TV clips — better facial expressions
and natural body movement than general-purpose models.

Usage (called by app.py Scene Video tab when model = "SkyReels V1"):
    generate(image, prompt, out_path, duration=4, vram_mode="none")
"""

import glob
import math
import os
import shutil
import subprocess
import sys
import time

SKYREELS_V1_DIR   = os.environ.get("SKYREELS_V1_DIR",   "/workspace/SkyReels-V1")
SKYREELS_V1_MODEL = os.environ.get("SKYREELS_V1_MODEL", "Skywork/SkyReels-V1-Hunyuan-I2V")

FPS        = 24    # SkyReels V1 generates at 24fps
NUM_FRAMES = 97    # fixed frame count = ~4s at 24fps

CHUNK_DURATION = 4    # seconds per chunk
CROSSFADE      = 1.0  # seconds of crossfade overlap between chunks


def generate(image: str, prompt: str, out_path: str,
             duration: int = 4, vram_mode: str = "none") -> None:
    """
    Run SkyReels V1 I2V to produce a scene video from a starting image.

    image:      path to starting scene image (PNG or JPG)
    prompt:     motion/scene prompt
    out_path:   where to write the final MP4
    duration:   ignored for single-shot (always 97 frames = ~4s); used by generate_chunked
    vram_mode:  "none" (default), "offload", or "low_vram"
    """
    script = os.path.join(SKYREELS_V1_DIR, "video_generate.py")
    if not os.path.exists(script):
        print(f"ERROR: SkyReels V1 not found at {SKYREELS_V1_DIR}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, script,
        "--model_id",              SKYREELS_V1_MODEL,
        "--task_type",             "i2v",
        "--image",                 image,
        "--prompt",                prompt,
        "--height",                "544",
        "--width",                 "960",
        "--num_frames",            str(NUM_FRAMES),
        "--guidance_scale",        "6.0",
        "--embedded_guidance_scale", "1.0",
    ]

    if vram_mode in ("offload", "low_vram"):
        cmd.extend(["--offload", "--quant", "--high_cpu_memory"])

    print(f"[skyreels-v1] running: {' '.join(cmd)}")

    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    result = subprocess.run(cmd, cwd=SKYREELS_V1_DIR, env=env)

    if result.returncode != 0:
        print("ERROR: SkyReels V1 generation failed.", file=sys.stderr)
        sys.exit(1)

    # If out_path already exists (e.g. via --output flag), we're done
    if os.path.exists(out_path):
        print(f"[skyreels-v1] saved {out_path}")
        return

    # Fallback: find newest MP4 written by the script and move it
    candidates = glob.glob(os.path.join(SKYREELS_V1_DIR, "**", "*.mp4"), recursive=True)
    if not candidates:
        print("ERROR: SkyReels V1 produced no output file.", file=sys.stderr)
        sys.exit(1)

    newest = max(candidates, key=os.path.getmtime)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.move(newest, out_path)
    print(f"[skyreels-v1] saved {out_path}")


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
    image: str, prompts: list, out_path: str,
    total_duration: int = 15,
    chunk_duration: int = CHUNK_DURATION,
    crossfade: float = CROSSFADE,
    vram_mode: str = "none",
) -> None:
    """
    Generate a long scene video by chaining 4s SkyReels V1 I2V chunks.
    Each chunk uses the last frame of the previous chunk as its starting image.

    prompts: list of prompts, one per chunk. If fewer than n_chunks, the last
             prompt repeats for remaining chunks.
    """
    from pipeline.face_swap import swap as face_swap

    # Check if Deep-Live-Cam is installed
    dlc_script = os.path.join(
        os.environ.get("DEEP_LIVE_CAM_DIR", "/workspace/Deep-Live-Cam"), "run.py"
    )
    has_face_swap = os.path.exists(dlc_script)
    if not has_face_swap:
        print("[skyreels-v1-chunked] Deep-Live-Cam not found — face correction skipped")

    ts = int(time.time())
    chunk_paths = []
    frame_paths = []
    swap_paths  = []

    n_chunks = math.ceil((total_duration - crossfade) / (chunk_duration - crossfade))
    n_chunks = max(n_chunks, 1)

    print(f"[skyreels-v1-chunked] {total_duration}s total → {n_chunks} chunks of {chunk_duration}s "
          f"with {crossfade}s crossfade")

    try:
        current_image = image

        for i in range(n_chunks):
            chunk_path = f"/tmp/sv1_chunk_{ts}_{i}.mp4"
            chunk_paths.append(chunk_path)

            chunk_prompt = prompts[i] if i < len(prompts) else prompts[-1]
            print(f"\n[skyreels-v1-chunked] chunk {i+1}/{n_chunks} | prompt: {chunk_prompt[:60]}...")
            generate(current_image, chunk_prompt, chunk_path,
                     duration=chunk_duration, vram_mode=vram_mode)

            # Restore original face identity (chunks 1+ may drift from original)
            if has_face_swap and i > 0:
                swapped_path = f"/tmp/sv1_chunk_{ts}_{i}_swapped.mp4"
                swap_paths.append(swapped_path)
                try:
                    face_swap(chunk_path, image, swapped_path)
                    shutil.move(swapped_path, chunk_path)
                except SystemExit:
                    print(f"[skyreels-v1-chunked] face swap skipped for chunk {i+1} (face not detected)")

            if i < n_chunks - 1:
                frame_path = f"/tmp/sv1_chunk_{ts}_lastframe_{i}.png"
                frame_paths.append(frame_path)
                _extract_last_frame(chunk_path, frame_path)
                current_image = frame_path

        print(f"\n[skyreels-v1-chunked] stitching {n_chunks} chunks → {out_path}")
        _crossfade_videos(chunk_paths, crossfade, total_duration, out_path)
        print(f"[skyreels-v1-chunked] done → {out_path} ({total_duration}s)")

    finally:
        for f in chunk_paths + frame_paths + swap_paths:
            if os.path.exists(f):
                os.remove(f)

"""
describe_video.py — Extract motion/scene description from a video using GPT-4o vision

Uses OpenAI GPT-4o to analyze frames from an input video and generate
a detailed prompt suitable for SkyReels R2V/A2V video generation.

Fallback: if OPENAI_API_KEY is not set, returns the user's prompt as-is.
"""

import base64
import glob
import os
import subprocess
import sys
import time


def _extract_frames(video_path: str, num_frames: int = 6) -> list:
    """Extract evenly-spaced frames from video, return list of JPEG paths."""
    ts = int(time.time())
    out_pattern = f"/tmp/describe_frame_{ts}_%03d.jpg"

    # Get total frame count
    probe = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    total = int(probe.stdout.strip()) if probe.stdout.strip() else 100
    step = max(1, total // num_frames)

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"select=not(mod(n\\,{step}))",
         "-vsync", "vfr", "-frames:v", str(num_frames),
         out_pattern],
        capture_output=True,
    )

    frames = sorted(glob.glob(f"/tmp/describe_frame_{ts}_*.jpg"))
    return frames


def _cleanup_frames(frames: list) -> None:
    for f in frames:
        if os.path.exists(f):
            os.remove(f)


def describe(video_path: str, user_prompt: str = "") -> str:
    """
    Analyze a video and return an enriched prompt for video generation.

    If OPENAI_API_KEY is not set, returns user_prompt as-is (graceful fallback).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[describe_video] OPENAI_API_KEY not set, using user prompt only", file=sys.stderr)
        return user_prompt

    try:
        from openai import OpenAI
    except ImportError:
        print("[describe_video] openai not installed, using user prompt only", file=sys.stderr)
        return user_prompt

    frames = _extract_frames(video_path)
    if not frames:
        print("[describe_video] no frames extracted, using user prompt only", file=sys.stderr)
        return user_prompt

    # Encode frames as base64
    image_contents = []
    for frame_path in frames:
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    _cleanup_frames(frames)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a video description assistant. Given frames from a video, "
                    "write a concise prompt for an AI video generator. Focus on: "
                    "body posture, movement direction and speed, gestures, facial expression, "
                    "camera angle, background/setting, and lighting. "
                    "Write in present tense, as a single paragraph, under 100 words."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the motion and scene in this video:"},
                    *image_contents,
                ],
            },
        ],
        max_tokens=200,
    )

    description = response.choices[0].message.content.strip()
    print(f"[describe_video] GPT-4o description: {description}")

    # Combine: LLM description first, user prompt appended for style/mood
    if user_prompt:
        return f"{description}. {user_prompt}"
    return description

"""
skyreels.py — SkyReels V3 wrapper
Supports two modes:
  - talking_avatar: portrait + audio + prompt → talking-head video (lip sync)
  - reference_to_video: portrait + prompt → natural movement video (no speech)

Usage (pod):
    python pipeline/skyreels.py \
        --portrait character/reference.png \
        --audio /tmp/tts_voice.wav \
        --prompt "selfie vlog, golden hour" \
        --out /tmp/raw_video.mp4
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

# Path where SkyReels V3 is cloned on the pod
SKYREELS_DIR = os.environ.get("SKYREELS_DIR", "/workspace/SkyReels-V3")
SKYREELS_MODEL = os.environ.get("SKYREELS_MODEL", "/workspace/SkyReels-V3-A2V-19B")


def generate(portrait: str, audio: str, prompt: str, out_path: str) -> None:
    """
    Run SkyReels V3 to produce a video.
    - If audio is provided: uses talking_avatar mode (lip sync)
    - If audio is None: uses reference_to_video mode (natural movement, no speech)
    """
    script = os.path.join(SKYREELS_DIR, "generate_video.py")
    if not os.path.exists(script):
        print(f"ERROR: SkyReels not found at {SKYREELS_DIR}", file=sys.stderr)
        sys.exit(1)

    if audio:
        task_type = "talking_avatar"
        cmd = [
            sys.executable, script,
            "--task_type", task_type,
            "--model_id", SKYREELS_MODEL,
            "--input_image", portrait,
            "--input_audio", audio,
            "--prompt", prompt,
        ]
        out_subdir = "talking_avatar"
    else:
        task_type = "reference_to_video"
        cmd = [
            sys.executable, script,
            "--task_type", task_type,
            "--model_id", SKYREELS_MODEL,
            "--ref_imgs", portrait,   # R2V uses --ref_imgs, not --input_image
            "--prompt", prompt,
        ]
        out_subdir = "reference_to_video"

    print(f"[skyreels] mode: {task_type}")
    print(f"[skyreels] running: {' '.join(cmd)}")

    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    result = subprocess.run(cmd, cwd=SKYREELS_DIR, env=env)

    if result.returncode != 0:
        print("ERROR: SkyReels V3 generation failed.", file=sys.stderr)
        sys.exit(1)

    # Find the newest MP4 produced in result/<task_type>/
    out_dir = os.path.join(SKYREELS_DIR, "result", out_subdir)
    candidates = glob.glob(os.path.join(out_dir, "*.mp4"))
    if not candidates:
        print("ERROR: SkyReels produced no output file.", file=sys.stderr)
        sys.exit(1)

    newest = max(candidates, key=os.path.getmtime)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.move(newest, out_path)
    print(f"[skyreels] saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", required=True, help="Path to portrait PNG")
    parser.add_argument("--audio", default=None, help="Path to TTS WAV (omit for natural movement mode)")
    parser.add_argument("--prompt", required=True, help="Scene/mood prompt")
    parser.add_argument("--out", default="/tmp/raw_video.mp4", help="Output MP4 path")
    args = parser.parse_args()

    generate(args.portrait, args.audio, args.prompt, args.out)

"""
wan.py — Wan 2.2 I2V (Image-to-Video) wrapper

Takes a starting scene image + text prompt and generates a video with
realistic motion. Designed for cinematic full-body/environment scenes,
not talking heads.

Usage (called by app.py Scene Video tab):
    generate(image, prompt, out_path, duration=15, vram_mode="none")
"""

import glob
import os
import shutil
import subprocess
import sys

WAN_DIR   = os.environ.get("WAN_DIR",   "/workspace/Wan2.2")
WAN_MODEL = os.environ.get("WAN_MODEL", "/workspace/Wan2.2-I2V-14B")

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

    frame_num = duration * WAN_FPS

    cmd = [
        sys.executable, script,
        "--task",      "i2v-14B",
        "--image",     image,
        "--prompt",    prompt,
        "--size",      WAN_SIZE,
        "--frame_num", str(frame_num),
        "--ckpt_dir",  WAN_MODEL,
        "--save_file", out_path,
    ]

    if vram_mode == "offload":
        cmd.append("--offload")
    elif vram_mode == "low_vram":
        cmd.append("--offload")   # Wan uses --offload for both low-memory modes

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

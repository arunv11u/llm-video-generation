"""
skyreels_v2_i2v.py — SkyReels V2 I2V (Image-to-Video) wrapper

Uses Diffusion Forcing (generate_video_df.py) to generate long videos natively
with internal 17-frame overlap — no manual chunking, no visible transitions.

Usage (called by app.py Scene Video tab when model = "SkyReels V2"):
    generate(image, prompt, out_path, duration=15, vram_mode="none")
"""

import glob
import os
import shutil
import subprocess
import sys
import time

SKYREELS_V2_DIR   = os.environ.get("SKYREELS_V2_DIR",   "/workspace/SkyReels-V2")
SKYREELS_V2_MODEL = os.environ.get("SKYREELS_V2_MODEL", "Skywork/SkyReels-V2-DF-14B-540P")

FPS = 24

# Recommended frame counts from SkyReels V2 docs (all follow 4n+1 pattern)
# Each additional 5s beyond base (97 frames) adds ~120 frames
FRAME_MAP = {5: 97, 10: 257, 15: 377, 20: 497, 25: 617, 30: 737}


def generate(image: str, prompt: str, out_path: str,
             duration: int = 15, vram_mode: str = "none",
             addnoise_condition: int = 0,
             overlap_history: int = 33,
             base_num_frames: int = 97,
             guidance_scale: float = 5.0) -> None:
    """
    Run SkyReels V2 Diffusion Forcing I2V to produce a long scene video.

    image:              path to starting scene image (PNG or JPG)
    prompt:             motion/scene prompt (all chunk prompts pre-joined by caller)
    out_path:           where to write the final MP4
    duration:           video length in seconds (5–30)
    vram_mode:          "none" (default), "offload", or "low_vram"
    addnoise_condition: noise re-injected into overlap frames (0 = max identity
                        preservation, 20 = smoother seams but more drift)
    overlap_history:    frames of context carried between chunks (higher = less
                        face drift, slightly more VRAM/time)
    base_num_frames:    frames per chunk before extension (higher = fewer chunks
                        = less drift, more VRAM)
    """
    script = os.path.join(SKYREELS_V2_DIR, "generate_video_df.py")
    if not os.path.exists(script):
        print(f"ERROR: SkyReels V2 not found at {SKYREELS_V2_DIR}", file=sys.stderr)
        sys.exit(1)

    # Look up recommended frame count, fallback to formula
    frame_count = FRAME_MAP.get(duration, round((duration * FPS - 1) / 4) * 4 + 1)

    cmd = [
        sys.executable, script,
        "--model_id",           SKYREELS_V2_MODEL,
        "--resolution",         "540P",
        "--num_frames",         str(frame_count),
        "--base_num_frames",    str(base_num_frames),
        "--overlap_history",    str(overlap_history),
        "--ar_step",            "0",
        "--addnoise_condition", str(addnoise_condition),
        "--image",              image,
        "--prompt",             prompt,
        "--guidance_scale",     str(guidance_scale),
        "--shift",              "3.0",
        "--fps",                str(FPS),
        "--use_ret_steps",
    ]

    if vram_mode in ("offload", "low_vram"):
        cmd.extend(["--offload", "--teacache", "--teacache_thresh", "0.3"])

    print(f"[skyreels-v2] running: {' '.join(cmd)}")

    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    result = subprocess.run(cmd, cwd=SKYREELS_V2_DIR, env=env)

    if result.returncode != 0:
        print("ERROR: SkyReels V2 generation failed.", file=sys.stderr)
        sys.exit(1)

    # If out_path already exists (e.g. script wrote there directly), we're done
    if os.path.exists(out_path):
        print(f"[skyreels-v2] saved {out_path}")
        return

    # Fallback: V2 writes to result/diffusion_forcing/ with auto-generated names
    # Find the newest MP4 produced and move it to out_path
    result_dir = os.path.join(SKYREELS_V2_DIR, "result")
    candidates = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
    if not candidates:
        print("ERROR: SkyReels V2 produced no output file.", file=sys.stderr)
        sys.exit(1)

    newest = max(candidates, key=os.path.getmtime)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.move(newest, out_path)
    print(f"[skyreels-v2] saved {out_path}")

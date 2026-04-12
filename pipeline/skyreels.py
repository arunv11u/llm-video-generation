"""
skyreels.py — SkyReels V3 A2V wrapper
Takes a portrait image + audio file + text prompt → produces a talking-head video.
Calls SkyReels V3's generate_video.py CLI as a subprocess.

Usage (pod):
    python pipeline/skyreels.py \
        --portrait character/reference.png \
        --audio /tmp/tts_voice.wav \
        --prompt "selfie vlog, golden hour, Parisian cafe" \
        --out /tmp/raw_video.mp4
"""

import argparse
import os
import subprocess
import sys

# Path where SkyReels V3 is cloned on the pod
SKYREELS_DIR = os.environ.get("SKYREELS_DIR", "/workspace/SkyReels-V3")
SKYREELS_MODEL = os.environ.get("SKYREELS_MODEL", "Skywork/SkyReels-V3-A2V-19B")


def generate(portrait: str, audio: str, prompt: str, out_path: str) -> None:
    """
    Run SkyReels V3 A2V to produce a talking-head video.
    """
    script = os.path.join(SKYREELS_DIR, "generate_video.py")
    if not os.path.exists(script):
        print(f"ERROR: SkyReels not found at {SKYREELS_DIR}", file=sys.stderr)
        print("Run docs/runpod_setup.md setup first.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, script,
        "--task_type", "talking_avatar",
        "--model_id", SKYREELS_MODEL,
        "--image", portrait,
        "--audio", audio,
        "--prompt", prompt,
        "--output_path", out_path,
    ]

    print(f"[skyreels] running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    result = subprocess.run(cmd, cwd=SKYREELS_DIR, env=env)

    if result.returncode != 0:
        print("ERROR: SkyReels V3 generation failed.", file=sys.stderr)
        sys.exit(1)

    print(f"[skyreels] saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", required=True, help="Path to portrait PNG")
    parser.add_argument("--audio", required=True, help="Path to TTS WAV")
    parser.add_argument("--prompt", required=True, help="Scene/mood prompt")
    parser.add_argument("--out", default="/tmp/raw_video.mp4", help="Output MP4 path")
    args = parser.parse_args()

    generate(args.portrait, args.audio, args.prompt, args.out)

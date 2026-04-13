"""
face_swap.py — Deep-Live-Cam wrapper for face replacement

Swaps the face in a video with a reference face image using Deep-Live-Cam
in headless CLI mode. The original video's audio is preserved.

Usage (pod):
    python pipeline/face_swap.py \
        --video input.mp4 \
        --face character/reference.png \
        --out /tmp/swapped.mp4
"""

import argparse
import os
import subprocess
import sys

DEEP_LIVE_CAM_DIR = os.environ.get("DEEP_LIVE_CAM_DIR", "/workspace/Deep-Live-Cam")


def swap(video_path: str, face_image: str, out_path: str) -> None:
    """
    Swap faces in video_path with face_image, save result to out_path.
    Uses Deep-Live-Cam in headless CLI mode with CUDA acceleration.
    """
    script = os.path.join(DEEP_LIVE_CAM_DIR, "run.py")
    if not os.path.exists(script):
        print(f"ERROR: Deep-Live-Cam not found at {DEEP_LIVE_CAM_DIR}", file=sys.stderr)
        print("Install: cd /workspace && git clone https://github.com/hacksider/Deep-Live-Cam.git", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, script,
        "-s", face_image,
        "-t", video_path,
        "-o", out_path,
        "--frame-processor", "face_swapper",
        "--execution-provider", "cuda",
        "--keep-audio",
    ]

    print(f"[face_swap] running: {' '.join(cmd)}")

    env = os.environ.copy()
    result = subprocess.run(cmd, cwd=DEEP_LIVE_CAM_DIR, env=env)

    if result.returncode != 0:
        print("ERROR: Deep-Live-Cam face swap failed.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(out_path):
        print(f"ERROR: face swap produced no output at {out_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[face_swap] saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--face", required=True, help="Reference face image path")
    parser.add_argument("--out", default="/tmp/swapped.mp4", help="Output video path")
    args = parser.parse_args()

    swap(args.video, args.face, args.out)

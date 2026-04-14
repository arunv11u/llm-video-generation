"""
face_fusion.py — FaceFusion wrapper for face identity correction

Swaps the reference face back onto a generated video using FaceFusion CLI
in headless mode. Preserves original expressions and pose from the target
video while restoring source face identity across all frames.

Usage (pod):
    python pipeline/face_fusion.py \
        --video input.mp4 \
        --face character/reference.png \
        --out /tmp/corrected.mp4
"""

import argparse
import os
import subprocess
import sys

FACEFUSION_DIR = os.environ.get("FACEFUSION_DIR", "/workspace/facefusion")


def swap(video_path: str, face_image: str, out_path: str) -> None:
    """
    Swap faces in video_path with face_image, save result to out_path.
    Uses FaceFusion in headless mode with CUDA acceleration.
    """
    script = os.path.join(FACEFUSION_DIR, "facefusion.py")
    if not os.path.exists(script):
        print(f"ERROR: FaceFusion not found at {FACEFUSION_DIR}", file=sys.stderr)
        print("Install: cd /workspace && git clone https://github.com/facefusion/facefusion.git && cd facefusion && python install.py --onnxruntime cuda", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, script,
        "headless-run",
        "-s", face_image,
        "-t", video_path,
        "-o", out_path,
        "--processors", "face_swapper",
        "--execution-providers", "cuda",
    ]

    print(f"[face_fusion] running: {' '.join(cmd)}")

    env = os.environ.copy()
    result = subprocess.run(cmd, cwd=FACEFUSION_DIR, env=env)

    if result.returncode != 0:
        print("ERROR: FaceFusion face swap failed.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(out_path):
        print(f"ERROR: FaceFusion produced no output at {out_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[face_fusion] saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--face", required=True, help="Reference face image path")
    parser.add_argument("--out", default="/tmp/ff_corrected.mp4", help="Output video path")
    args = parser.parse_args()

    swap(args.video, args.face, args.out)

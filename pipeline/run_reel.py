"""
run_reel.py — M1 main orchestrator CLI
Full pipeline: transcript → TTS → SkyReels V3 A2V → ffmpeg polish → final MP4

Usage (pod):
    python pipeline/run_reel.py \
        --transcript "Hey guys, welcome back to my channel..." \
        --music path/to/bg.mp3 \
        --prompt "selfie vlog, golden hour, Parisian cafe"

Output: outputs/reel_<timestamp>.mp4
"""

import argparse
import os
import sys
import time

# Add project root to path so pipeline modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.tts import generate as tts_generate
from pipeline.skyreels import generate as skyreels_generate
from pipeline.polish import polish

PORTRAIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "character", "reference.png")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def run(transcript: str, music: str, prompt: str) -> str:
    """
    Run the full reel pipeline. Returns path to the final MP4.
    """
    if not os.path.exists(PORTRAIT):
        print("ERROR: character/reference.png not found.", file=sys.stderr)
        print("Run M0 first: python pipeline/pick_portrait.py", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(music):
        print(f"ERROR: music file not found: {music}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = int(time.time())
    tts_path = f"/tmp/tts_voice_{ts}.wav"
    raw_path = f"/tmp/raw_video_{ts}.mp4"
    out_path = os.path.join(OUTPUTS_DIR, f"reel_{ts}.mp4")

    print(f"\n=== Step 1/3: TTS ===")
    tts_generate(transcript, tts_path)

    print(f"\n=== Step 2/3: SkyReels V3 A2V ===")
    skyreels_generate(PORTRAIT, tts_path, prompt, raw_path)

    print(f"\n=== Step 3/3: Polish (audio swap + captions) ===")
    polish(raw_path, music, transcript, out_path)

    # Clean up temp files
    for f in [tts_path, raw_path]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\n=== Done ===")
    print(f"Output: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an Instagram reel")
    parser.add_argument("--transcript", required=True, help="Words she will mouth")
    parser.add_argument("--music", required=True, help="Background music file (MP3/WAV)")
    parser.add_argument("--prompt", default="glamorous selfie vlog, soft studio lighting, confident expression",
                        help="Scene/mood prompt for SkyReels V3")
    args = parser.parse_args()

    run(args.transcript, args.music, args.prompt)

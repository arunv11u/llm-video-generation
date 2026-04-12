"""
run_reel.py — M1 main orchestrator
Full pipeline: inputs → TTS (if needed) → SkyReels → ffmpeg polish → final MP4

Audio modes (auto-selected based on inputs):
  transcript only           → tts_only      (voice as final audio)
  music only                → music_only    (natural movement + music)
  transcript + music        → voice_and_music OR lipsync_only (user choice)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.tts import generate as tts_generate
from pipeline.skyreels import generate as skyreels_generate
from pipeline.polish import polish

PORTRAIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "character", "reference.png")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def run(transcript: str, music: str, prompt: str, audio_mode: str = None, duration: int = None) -> str:
    """
    Run the full reel pipeline. Returns path to the final MP4.

    audio_mode is auto-determined if not provided:
      - transcript only  → tts_only
      - music only       → music_only
      - both             → voice_and_music
    When both are provided, caller can override with 'lipsync_only'.
    """
    if not os.path.exists(PORTRAIT):
        print("ERROR: character/reference.png not found.", file=sys.stderr)
        sys.exit(1)

    has_transcript = bool(transcript and transcript.strip())
    has_music = bool(music and os.path.exists(music))

    if not has_transcript and not has_music:
        print("ERROR: provide at least a transcript or background music.", file=sys.stderr)
        sys.exit(1)

    # Auto-select audio mode if not provided
    if audio_mode is None:
        if has_transcript and has_music:
            audio_mode = "voice_and_music"
        elif has_transcript:
            audio_mode = "tts_only"
        else:
            audio_mode = "music_only"

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = int(time.time())
    tts_path = f"/tmp/tts_voice_{ts}.wav"
    raw_path = f"/tmp/raw_video_{ts}.mp4"
    out_path = os.path.join(OUTPUTS_DIR, f"reel_{ts}.mp4")

    # Step 1: TTS — needed when transcript exists
    needs_tts = has_transcript  # always run TTS if transcript provided (for lip sync or voice)
    if needs_tts:
        print(f"\n=== Step 1/3: TTS ===")
        tts_generate(transcript.strip(), tts_path)
    else:
        tts_path = None
        print(f"\n=== Step 1/3: TTS — skipped (no transcript) ===")

    # Step 2: SkyReels
    print(f"\n=== Step 2/3: SkyReels V3 ===")
    if needs_tts:
        # Use talking_avatar mode (lip sync)
        skyreels_generate(PORTRAIT, tts_path, prompt.strip(), raw_path)
    else:
        # Use reference_to_video mode (natural movement, no speech)
        skyreels_generate(PORTRAIT, None, prompt.strip(), raw_path, duration=duration)

    # Step 3: Polish
    print(f"\n=== Step 3/3: Polish ===")
    polish(
        video=raw_path,
        tts=tts_path,
        music=music if has_music else None,
        transcript=transcript.strip() if has_transcript else "",
        audio_mode=audio_mode,
        out_path=out_path,
    )

    # Cleanup
    for f in [tts_path, raw_path]:
        if f and os.path.exists(f):
            os.remove(f)

    print(f"\n=== Done ===")
    print(f"Output: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an Instagram reel")
    parser.add_argument("--transcript", default="", help="Words she will mouth")
    parser.add_argument("--music", default=None, help="Background music file (MP3/WAV)")
    parser.add_argument("--prompt", default="glamorous selfie vlog, soft studio lighting, confident expression")
    parser.add_argument("--audio_mode", default=None,
                        choices=["tts_only", "music_only", "voice_and_music", "lipsync_only"],
                        help="Override audio mode (auto-detected if omitted)")
    parser.add_argument("--duration", default=None, type=int,
                        help="Video duration in seconds 5-30 (only for music-only/dance mode)")
    args = parser.parse_args()

    run(args.transcript, args.music, args.prompt, args.audio_mode, args.duration)

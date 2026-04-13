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
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.tts import generate as tts_generate
from pipeline.skyreels import generate as skyreels_generate
from pipeline.polish import polish
from pipeline.describe_video import describe as describe_video
from pipeline.face_swap import swap as face_swap

PORTRAIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "character", "reference.png")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _get_video_duration(path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def run(transcript: str, music: str, prompt: str, audio_mode: str = None,
        duration: int = None, vram_mode: str = "none",
        input_video: str = None, video_face_mode: str = None) -> str:
    """
    Run the full reel pipeline. Returns path to the final MP4.

    audio_mode is auto-determined if not provided:
      - transcript only  → tts_only
      - music only       → music_only
      - both             → voice_and_music
    When both are provided, caller can override with 'lipsync_only'.

    input_video: optional reference video for Video + Face mode
    video_face_mode: "approximate" (AI-generated) or "exact" (face swap)
    """
    if not os.path.exists(PORTRAIT):
        print("ERROR: character/reference.png not found.", file=sys.stderr)
        sys.exit(1)

    has_transcript = bool(transcript and transcript.strip())
    has_music = bool(music and os.path.exists(music))

    if not has_transcript and not has_music and not input_video:
        print("ERROR: provide at least a transcript or background music.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = int(time.time())
    tts_path = f"/tmp/tts_voice_{ts}.wav"
    raw_path = f"/tmp/raw_video_{ts}.mp4"
    out_path = os.path.join(OUTPUTS_DIR, f"reel_{ts}.mp4")

    # ── Video + Face: Exact (face swap) ──────────────────────────────────
    if input_video and video_face_mode == "exact":
        print(f"\n=== Face Swap Mode ===")
        face_swap(input_video, PORTRAIT, raw_path)

        if not has_transcript and not has_music:
            # No post-processing needed, return face-swapped video as-is
            import shutil
            shutil.move(raw_path, out_path)
        else:
            # Determine audio mode for polish
            if has_transcript and not has_music:
                audio_mode = "keep_audio"
            elif has_music and not has_transcript:
                audio_mode = "music_only"
            else:  # both transcript and music
                audio_mode = audio_mode or "lipsync_only"

            print(f"\n=== Polish ===")
            polish(
                video=raw_path,
                tts=None,
                music=music if has_music else None,
                transcript=transcript.strip() if has_transcript else "",
                audio_mode=audio_mode,
                out_path=out_path,
            )

        # Cleanup
        if os.path.exists(raw_path):
            os.remove(raw_path)

        print(f"\n=== Done ===")
        print(f"Output: {out_path}")
        return out_path

    # ── Video + Face: Approximate (describe → existing pipeline) ─────────
    if input_video and video_face_mode == "approximate":
        print(f"\n=== Approximate Mode: describing input video ===")
        prompt = describe_video(input_video, prompt)
        if not has_transcript:
            duration = int(_get_video_duration(input_video))

    # Auto-select audio mode if not provided
    if audio_mode is None:
        if has_transcript and has_music:
            audio_mode = "voice_and_music"
        elif has_transcript:
            audio_mode = "tts_only"
        else:
            audio_mode = "music_only"

    # Step 1: TTS — needed when transcript exists
    needs_tts = has_transcript
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
        skyreels_generate(PORTRAIT, None, prompt.strip(), raw_path, duration=duration, vram_mode=vram_mode)

    # Step 3: Polish (skip if no audio sources)
    if has_transcript or has_music:
        print(f"\n=== Step 3/3: Polish ===")
        polish(
            video=raw_path,
            tts=tts_path,
            music=music if has_music else None,
            transcript=transcript.strip() if has_transcript else "",
            audio_mode=audio_mode,
            out_path=out_path,
        )
    else:
        # No transcript, no music — output silent video as-is
        print(f"\n=== Step 3/3: Polish — skipped (no audio sources) ===")
        import shutil
        shutil.move(raw_path, out_path)

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
    parser.add_argument("--vram_mode", default="none",
                        choices=["none", "offload", "low_vram"],
                        help="Memory mode: none (default) | offload (fixes OOM) | low_vram (extreme cases)")
    parser.add_argument("--input_video", default=None,
                        help="Reference video for Video + Face mode")
    parser.add_argument("--video_face_mode", default=None,
                        choices=["approximate", "exact"],
                        help="Video + Face mode: approximate (AI-generated) | exact (face swap)")
    args = parser.parse_args()

    run(args.transcript, args.music, args.prompt, args.audio_mode, args.duration, args.vram_mode,
        args.input_video, args.video_face_mode)

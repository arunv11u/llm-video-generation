"""
tts.py — ElevenLabs TTS wrapper
Converts a transcript string to a WAV file using ElevenLabs API.
The output audio drives lip-sync and may also be used as final audio.

Usage (pod):
    python pipeline/tts.py --transcript "Hey guys..." --out /tmp/tts_voice.wav
"""

import argparse
import os
import sys


DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah (American female)

# Good female voice IDs:
# EXAVITQu4vr4xnSDxMaL — Sarah (warm, natural American female)
# 21m00Tcm4TlvDq8ikWAM — Rachel (calm, American)
# AZnzlk1XvdvUeBnXmlld — Domi (strong, American)
# MF3mGyEYCl7XYWbV9V6O — Elli (young, American)


def generate(transcript: str, out_path: str) -> float:
    """
    Run ElevenLabs TTS on the transcript, save to out_path.
    Returns duration in seconds.
    """
    # Read at call time so UI overrides via os.environ take effect
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set.", file=sys.stderr)
        print("Run: export ELEVENLABS_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import save
    except ImportError:
        print("ERROR: elevenlabs not installed. Run: pip install elevenlabs", file=sys.stderr)
        sys.exit(1)

    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
    print(f"[tts] generating with ElevenLabs voice: {voice_id}")

    client = ElevenLabs(api_key=api_key)
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=transcript,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # Save as MP3 first then convert to WAV for compatibility
    mp3_path = out_path.replace(".wav", ".mp3")
    save(audio, mp3_path)

    # Convert MP3 to WAV using ffmpeg
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, out_path],
        capture_output=True
    )
    os.remove(mp3_path)

    if result.returncode != 0:
        print("ERROR: ffmpeg mp3→wav conversion failed.", file=sys.stderr)
        sys.exit(1)

    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out_path],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip())
    print(f"[tts] saved {out_path} ({duration:.1f}s)")
    return duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True, help="Text to synthesize")
    parser.add_argument("--out", default="/tmp/tts_voice.wav", help="Output WAV path")
    args = parser.parse_args()

    generate(args.transcript, args.out)

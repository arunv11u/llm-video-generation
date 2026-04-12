"""
tts.py — F5-TTS wrapper
Converts a transcript string to a WAV file using F5-TTS.
The output audio is throwaway — it drives lip-sync only, never heard by the viewer.

Usage (pod):
    python pipeline/tts.py --transcript "Hey guys..." --out /tmp/tts_voice.wav
"""

import argparse
import os
import sys


def generate(transcript: str, out_path: str) -> float:
    """
    Run F5-TTS on the transcript, save to out_path.
    Returns duration in seconds.
    """
    try:
        from f5_tts.api import F5TTS
    except ImportError:
        print("ERROR: f5_tts not installed. Run: pip install f5-tts", file=sys.stderr)
        sys.exit(1)

    tts = F5TTS()

    # F5-TTS requires a reference audio file. Use the known bundled example path.
    ref_file = "/usr/local/lib/python3.12/dist-packages/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    print(f"[tts] using reference: {ref_file}")

    wav, sr, _ = tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=transcript,
        seed=-1,             # random seed, prosody variety
    )

    import soundfile as sf
    sf.write(out_path, wav, sr)

    duration = len(wav) / sr
    print(f"[tts] saved {out_path} ({duration:.1f}s)")
    return duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True, help="Text to synthesize")
    parser.add_argument("--out", default="/tmp/tts_voice.wav", help="Output WAV path")
    args = parser.parse_args()

    generate(args.transcript, args.out)

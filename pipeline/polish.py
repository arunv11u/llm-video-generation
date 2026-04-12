"""
polish.py — ffmpeg audio mix + caption burn + final export

Audio modes:
  - tts_only:      TTS voice as final audio (no music)
  - music_only:    Background music only (no voice)
  - voice_and_music: TTS voice + background music mixed
  - lipsync_only:  Music plays, TTS was only for lip sync (discarded)

Usage (pod):
    python pipeline/polish.py \
        --video /tmp/raw_video.mp4 \
        --tts /tmp/tts_voice.wav \
        --music path/to/bg.mp3 \
        --transcript "Hey guys..." \
        --audio_mode voice_and_music \
        --out outputs/reel_1234.mp4
"""

import argparse
import os
import subprocess
import sys


def _get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _build_caption_filter(transcript: str, duration: float) -> str:
    words = transcript.strip().split()
    if not words:
        return ""

    line_size = 6
    lines = [" ".join(words[i:i+line_size]) for i in range(0, len(words), line_size)]
    n = len(lines)
    time_per_line = duration / n

    filters = []
    for i, line in enumerate(lines):
        start = i * time_per_line
        end = start + time_per_line
        safe = line.replace("'", "\\'").replace(":", "\\:")
        filters.append(
            f"drawtext=text='{safe}'"
            f":fontcolor=white:fontsize=48:borderw=3:bordercolor=black"
            f":x=(w-text_w)/2:y=h-120"
            f":enable='between(t,{start:.2f},{end:.2f})'"
        )

    return ",".join(filters)


def polish(video: str, tts: str, music: str, transcript: str, audio_mode: str, out_path: str) -> None:
    """
    audio_mode options:
      tts_only        — TTS voice is final audio (no music)
      music_only      — music is final audio (no voice)
      voice_and_music — TTS voice + music mixed together
      lipsync_only    — music is final audio (TTS was only for lip sync)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    video_duration = _get_duration(video)
    caption_filter = _build_caption_filter(transcript, video_duration)

    fade = f"afade=t=out:st={max(0, video_duration - 0.5):.2f}:d=0.5"

    if audio_mode == "tts_only":
        # TTS voice as final audio, trim to video length
        tts_filter = f"[1:a]atrim=0:{video_duration:.2f},{fade}[aout]"
        if caption_filter:
            fc = f"[0:v]{caption_filter}[vout];{tts_filter}"
            cmd = ["ffmpeg", "-y", "-i", video, "-i", tts,
                   "-filter_complex", fc,
                   "-map", "[vout]", "-map", "[aout]",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", video, "-i", tts,
                   "-filter_complex", tts_filter,
                   "-map", "0:v", "-map", "[aout]",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]

    elif audio_mode in ("music_only", "lipsync_only"):
        # Music as final audio
        music_filter = f"[1:a]atrim=0:{video_duration:.2f},{fade}[aout]"
        if caption_filter:
            fc = f"[0:v]{caption_filter}[vout];{music_filter}"
            cmd = ["ffmpeg", "-y", "-i", video, "-i", music,
                   "-filter_complex", fc,
                   "-map", "[vout]", "-map", "[aout]",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", video, "-i", music,
                   "-filter_complex", music_filter,
                   "-map", "0:v", "-map", "[aout]",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]

    elif audio_mode == "voice_and_music":
        # Mix TTS voice + music together
        mix_filter = (
            f"[1:a]atrim=0:{video_duration:.2f},asetpts=PTS-STARTPTS,volume=1.5[voice];"
            f"[2:a]atrim=0:{video_duration:.2f},asetpts=PTS-STARTPTS,volume=0.4[bg];"
            f"[voice][bg]amix=inputs=2:duration=first[mixed];"
            f"[mixed]{fade}[aout]"
        )
        if caption_filter:
            fc = f"[0:v]{caption_filter}[vout];{mix_filter}"
            cmd = ["ffmpeg", "-y", "-i", video, "-i", tts, "-i", music,
                   "-filter_complex", fc,
                   "-map", "[vout]", "-map", "[aout]",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", video, "-i", tts, "-i", music,
                   "-filter_complex", mix_filter,
                   "-map", "0:v", "-map", "[aout]",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    else:
        print(f"ERROR: unknown audio_mode '{audio_mode}'", file=sys.stderr)
        sys.exit(1)

    print(f"[polish] audio_mode={audio_mode}, exporting {out_path}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: ffmpeg polish failed.", file=sys.stderr)
        sys.exit(1)

    print(f"[polish] done → {out_path} ({video_duration:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tts", default=None)
    parser.add_argument("--music", default=None)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--audio_mode", required=True,
                        choices=["tts_only", "music_only", "voice_and_music", "lipsync_only"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    polish(args.video, args.tts, args.music, args.transcript, args.audio_mode, args.out)

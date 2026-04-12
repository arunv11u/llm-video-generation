"""
polish.py — ffmpeg audio swap + caption burn + final export
Takes raw_video.mp4 (from SkyReels) + user's music + transcript
→ final MP4 with music audio and captions burned in.

Usage (pod):
    python pipeline/polish.py \
        --video /tmp/raw_video.mp4 \
        --music path/to/bg.mp3 \
        --transcript "Hey guys, welcome back..." \
        --out outputs/reel_1234.mp4
"""

import argparse
import os
import subprocess
import sys


def _get_duration(path: str) -> float:
    """Return duration of a media file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _build_caption_filter(transcript: str, duration: float) -> str:
    """
    Evenly distribute transcript words across the video duration.
    Returns an ffmpeg drawtext filter string.
    """
    words = transcript.strip().split()
    if not words:
        return ""

    # Split into lines of ~6 words for readability
    line_size = 6
    lines = [" ".join(words[i:i+line_size]) for i in range(0, len(words), line_size)]
    n = len(lines)
    time_per_line = duration / n

    filters = []
    for i, line in enumerate(lines):
        start = i * time_per_line
        end = start + time_per_line
        # Escape special ffmpeg characters
        safe = line.replace("'", "\\'").replace(":", "\\:")
        filters.append(
            f"drawtext=text='{safe}'"
            f":fontcolor=white:fontsize=48:borderw=3:bordercolor=black"
            f":x=(w-text_w)/2:y=h-120"
            f":enable='between(t,{start:.2f},{end:.2f})'"
        )

    return ",".join(filters)


def polish(video: str, music: str, transcript: str, out_path: str) -> None:
    """
    Swap audio, burn captions, export final MP4.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    video_duration = _get_duration(video)
    caption_filter = _build_caption_filter(transcript, video_duration)

    # Build ffmpeg filter chain
    # - amix: trim music to video duration, fade out last 0.5s
    audio_filter = (
        f"[1:a]atrim=0:{video_duration:.2f},"
        f"afade=t=out:st={max(0, video_duration - 0.5):.2f}:d=0.5[aout]"
    )

    if caption_filter:
        # Combine audio + video filters in one filter_complex to avoid -vf/-filter_complex conflict
        combined = f"[0:v]{caption_filter}[vout];{audio_filter}"
        cmd = [
            "ffmpeg", "-y",
            "-i", video,
            "-i", music,
            "-filter_complex", combined,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video,
            "-i", music,
            "-filter_complex", audio_filter,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path,
        ]

    print(f"[polish] exporting {out_path}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: ffmpeg polish failed.", file=sys.stderr)
        sys.exit(1)

    print(f"[polish] done → {out_path} ({video_duration:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Raw video from SkyReels")
    parser.add_argument("--music", required=True, help="Background music MP3/WAV")
    parser.add_argument("--transcript", required=True, help="Transcript text for captions")
    parser.add_argument("--out", required=True, help="Output MP4 path")
    args = parser.parse_args()

    polish(args.video, args.music, args.transcript, args.out)

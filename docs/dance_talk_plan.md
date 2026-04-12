# Plan: Two-Stage Dance + Talk Pipeline

## Why This Exists
No single AI model can make the woman dance AND lip-sync simultaneously.
- **A2V (current)** = great lip-sync, weak dancing
- **R2V** = great dancing, no lip-sync

Solution: run two models back-to-back, combine their outputs.

---

## How It Works

```
Stage 1: portrait + dance prompt  →  SkyReels R2V-14B  →  dance_video.mp4  (body dances, mouth closed)
Stage 2: dance_video + tts_audio  →  MuseTalk           →  lipsync_dance.mp4 (mouth inpainted with lip-sync)
Stage 3: lipsync_dance + music    →  polish.py           →  outputs/reel_{ts}.mp4 (final)
```

MuseTalk only replaces the **mouth region** — the dancing body, head motion, and everything else stays from Stage 1.

---

## Known Quality Tradeoffs

| Issue | Severity | Notes |
|---|---|---|
| Mouth seam / patch artifact around lips | Medium-High | Two models with different training = visible join |
| Head pose drift in MuseTalk | Medium | Fast head motion during dancing causes mouth overlay to slip |
| No beat-synced dancing | Medium | R2V dances from text prompt, doesn't "hear" the music |
| Two-model quality ceiling | Medium | Final quality limited by the weaker of the two models |
| Frame rate mismatch (R2V=24fps, MuseTalk=25fps) | Low | Fixed by ffmpeg re-encode before MuseTalk |
| Longer generation time | Low | ~15-25 min vs ~6-12 min today |

**Sweet spot:** Works best for slow, light dance moves. Fast/energetic dancing increases artifacts.

---

## New Environment Variables (Pod)

```bash
export SKYREELS_R2V_MODEL=/workspace/SkyReels-V3-R2V-14B
export MUSETALK_DIR=/workspace/MuseTalk
export MUSETALK_VERSION=v15   # optional, defaults to v15
```

---

## Pod Setup (one-time)

```bash
# Download SkyReels R2V model (~53GB)
huggingface-cli download Skywork/SkyReels-V3-R2V-14B \
  --local-dir /workspace/SkyReels-V3-R2V-14B

# Clone and install MuseTalk
git clone https://github.com/TMElyralab/MuseTalk /workspace/MuseTalk
cd /workspace/MuseTalk
pip install -r requirements.txt

# Download MuseTalk v1.5 model weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MuseTalk', local_dir='models/musetalkV15')
"
```

---

## Code Changes

### 1. `pipeline/musetalk.py` — NEW FILE

**Function:** `apply_lipsync(video_path, audio_path, out_path)`

Steps:
1. Check `MUSETALK_DIR/scripts/inference.py` exists
2. Re-encode dance video to 25fps via ffmpeg (MuseTalk training fps)
3. Write per-job YAML config:
   ```yaml
   - video_path: "/tmp/dance_{ts}_25fps.mp4"
     audio_path: "/tmp/tts_voice_{ts}.wav"
   ```
4. Run MuseTalk subprocess:
   ```bash
   python -m scripts.inference \
     --inference_config /tmp/musetalk_cfg_{ts}.yaml \
     --result_dir /tmp/musetalk_out_{ts} \
     --unet_model_path {MUSETALK_DIR}/models/musetalkV15/unet.pth \
     --unet_config {MUSETALK_DIR}/models/musetalkV15/musetalk.json \
     --version v15 \
     --output_vid_name result.mp4 \
     --fps 25
   ```
   **Important:** `cwd=MUSETALK_DIR` required for module path resolution
5. Move `{result_dir}/result.mp4` → `out_path`
6. Cleanup: yaml, result_dir, 25fps temp file

Model path logic (from `MUSETALK_VERSION`):
- `v15` → `models/musetalkV15/unet.pth`
- `v1`  → `models/musetalk/unet.pth`

> **License note:** MuseTalk uses Stability AI SVD base. Free for commercial use under 1M MAU.

---

### 2. `pipeline/skyreels.py` — ADD ~40 lines

Add module-level constant:
```python
SKYREELS_R2V_MODEL = os.environ.get("SKYREELS_R2V_MODEL", "/workspace/SkyReels-V3-R2V-14B")
```

Add new function `generate_dance(portrait, prompt, out_path)`:
- `task_type = "reference_to_video"`
- Uses `SKYREELS_R2V_MODEL` (not `SKYREELS_MODEL`)
- Uses `--ref_imgs` instead of `--input_image` (R2V arg difference)
- No `--input_audio` argument
- Same subprocess/output scanning pattern as existing `generate()`
- Scans `result/reference_to_video/*.mp4` for output

Update `__main__` to add `--dance_mode` flag → dispatches to `generate_dance()`.

---

### 3. `pipeline/run_reel.py` — ADD ~30 lines

Add imports:
```python
from pipeline.musetalk import apply_lipsync as musetalk_lipsync
from pipeline.skyreels import generate_dance as skyreels_dance
```

Add temp paths inside `run()`:
```python
dance_path   = f"/tmp/dance_video_{ts}.mp4"
lipsync_path = f"/tmp/lipsync_dance_{ts}.mp4"
```

Add new mode branch (before existing Step 2, returns early):
```python
if audio_mode == "dance_and_talk":
    # Step 1: TTS — already done above
    # Step 2: R2V dancing body
    skyreels_dance(PORTRAIT, prompt.strip(), dance_path)
    # Step 3: MuseTalk mouth inpainting
    musetalk_lipsync(dance_path, tts_path, lipsync_path)
    # Step 4: Polish — music as final audio (TTS was for lip-sync only)
    polish_mode = "lipsync_only" if has_music else "tts_only"
    polish(video=lipsync_path, tts=tts_path, music=music,
           transcript=transcript.strip(), audio_mode=polish_mode, out_path=out_path)
    # Cleanup
    for f in [tts_path, dance_path, lipsync_path]:
        if f and os.path.exists(f): os.remove(f)
    return out_path
```

Update argparse `choices` to include `"dance_and_talk"`.

Note: `dance_and_talk` is never auto-detected — only triggered when explicitly selected in UI or CLI.

---

### 4. `app.py` — CHANGE ~8 lines

Expand radio choices:
```python
choices=[
    "Voice + Music (speak over music)",
    "Lip sync only (music plays)",
    "Dance + Talk (dancing body + lip sync)",  # NEW
]
```

Update `generate_reel()` mapping (add before existing elif chain):
```python
if audio_mode_choice == "Dance + Talk (dancing body + lip sync)":
    if not has_transcript:
        return None, "Dance + Talk mode requires a transcript."
    audio_mode = "dance_and_talk"
elif has_transcript and has_music:
    ...  # unchanged
```

---

## Temp File Flow

```
/tmp/tts_voice_{ts}.wav        ← Step 1 TTS output
/tmp/dance_video_{ts}.mp4      ← Step 2 R2V output
/tmp/dance_{ts}_25fps.mp4      ← fps-normalized (inside musetalk.py, cleaned up internally)
/tmp/musetalk_cfg_{ts}.yaml    ← MuseTalk config (cleaned up internally)
/tmp/musetalk_out_{ts}/        ← MuseTalk result dir (cleaned up internally)
/tmp/lipsync_dance_{ts}.mp4    ← Step 3 MuseTalk output
outputs/reel_{ts}.mp4          ← Final output (kept)
```

---

## Testing

```bash
# 1. Unit test MuseTalk wrapper alone
python pipeline/musetalk.py \
  --video /tmp/test_dance.mp4 \
  --audio /tmp/test_voice.wav \
  --out /tmp/musetalk_out.mp4

# 2. Unit test R2V dance generation alone
python pipeline/skyreels.py \
  --portrait character/reference.png \
  --prompt "woman dancing, light sway, confident" \
  --dance_mode \
  --out /tmp/dance_out.mp4

# 3. Full pipeline integration test
python pipeline/run_reel.py \
  --transcript "Hey everyone, today I am dancing for you." \
  --music samples/test_transcript.txt \
  --prompt "nightclub dance floor, neon lights, confident expression" \
  --audio_mode dance_and_talk

# 4. UI test
# Select "Dance + Talk" in Gradio, enter transcript + music + prompt, click Generate
```

**Success criteria:**
- Output is 9:16 MP4
- Body shows visible dance movement
- Mouth visibly syncs to transcript words
- Music plays in background
- Captions appear at bottom of screen
- No leftover temp files in /tmp/

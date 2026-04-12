"""
app.py — Gradio UI for the reel generator
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.run_reel import run as run_reel
from pipeline.pick_portrait import pick as pick_portraits

CHARACTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character")
REFERENCE_PNG = os.path.join(CHARACTER_DIR, "reference.png")


# ── Reel tab ──────────────────────────────────────────────────────────────────

def generate_reel(transcript: str, music_path: str, prompt: str, audio_mode_choice: str, voice_id: str, duration: int = 15, vram_mode: str = "None"):
    transcript = transcript.strip() if transcript else ""
    music_path = music_path if music_path else None
    prompt = prompt.strip() if prompt else ""
    voice_id = voice_id.strip() if voice_id else ""

    has_transcript = bool(transcript)
    has_music = bool(music_path)

    if not has_transcript and not has_music:
        return None, "Please enter a transcript or upload background music."

    if not os.path.exists(REFERENCE_PNG):
        return None, "No reference portrait found. Go to the 'Pick a Face' tab first."

    # Set voice ID for this request
    if voice_id:
        os.environ["ELEVENLABS_VOICE_ID"] = voice_id

    # Determine audio mode
    if has_transcript and has_music:
        audio_mode = "lipsync_only" if audio_mode_choice == "Lip sync only (music plays)" else "voice_and_music"
    elif has_transcript:
        audio_mode = "tts_only"
    else:
        audio_mode = "music_only"

    try:
        out = run_reel(transcript, music_path, prompt, audio_mode, duration=duration if not has_transcript else None, vram_mode=vram_mode.lower())
        return out, f"Done! Saved to {out}"
    except SystemExit:
        return None, "Pipeline failed. Check the pod terminal for details."
    except Exception as e:
        return None, f"Error: {e}"


# ── Pick a Face tab ───────────────────────────────────────────────────────────

def generate_candidates(prompt: str, count: int):
    if not prompt.strip():
        return [], "Please enter a description."
    try:
        paths = pick_portraits(prompt.strip(), int(count))
        return paths, f"Generated {len(paths)} candidates. Pick one below."
    except Exception as e:
        return [], f"Error: {e}"


def set_reference(gallery, evt: gr.SelectData):
    selected = gallery[evt.index][0]
    os.makedirs(CHARACTER_DIR, exist_ok=True)
    import shutil
    shutil.copy(selected, REFERENCE_PNG)
    return f"Reference set to: {selected}"


def upload_reference(image_path: str):
    if not image_path:
        return "No image uploaded."
    os.makedirs(CHARACTER_DIR, exist_ok=True)
    import shutil
    shutil.copy(image_path, REFERENCE_PNG)
    return f"Reference set from upload: {image_path}"


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Reel Generator") as demo:
    gr.Markdown("# Reel Generator\nGenerate Instagram-ready reels of your AI virtual influencer.")

    with gr.Tab("Generate Reel"):
        gr.Markdown("Paste your transcript and/or upload background music, then hit Generate.")

        with gr.Row():
            with gr.Column():
                transcript_box = gr.Textbox(
                    label="Transcript (optional)",
                    placeholder="Hey guys, welcome back to my channel. Today I am in Paris...",
                    lines=4,
                )
                music_upload = gr.Audio(
                    label="Background Music (optional)",
                    type="filepath",
                    sources=["upload"],
                )
                prompt_box = gr.Textbox(
                    label="Scene / Mood Prompt",
                    value="glamorous selfie vlog, golden hour, confident expression",
                    lines=2,
                )
                audio_mode_radio = gr.Radio(
                    choices=["Voice + Music (speak over music)", "Lip sync only (music plays)"],
                    value="Voice + Music (speak over music)",
                    label="Audio mode (only applies when both transcript and music are provided)",
                )
                voice_id_box = gr.Textbox(
                    label="ElevenLabs Voice ID",
                    value="EXAVITQu4vr4xnSDxMaL",
                    placeholder="e.g. EXAVITQu4vr4xnSDxMaL",
                    info="Find voice IDs at elevenlabs.io/voice-library. Leave default for Sarah (American female).",
                )
                duration_slider = gr.Slider(
                    minimum=5, maximum=30, step=5, value=15,
                    label="Video Duration (seconds)",
                    info="Only applies to music-only / dance reels (no transcript). Talking reels use TTS audio length.",
                )
                vram_mode_radio = gr.Radio(
                    choices=["None", "Offload", "Low VRAM"],
                    value="None",
                    label="Memory Mode (dance/music-only reels)",
                    info="None: fastest, use for ≤10s  |  Offload: fixes OOM, ~20% slower, no quality loss  |  Low VRAM: slowest, extreme cases only",
                )
                generate_btn = gr.Button("Generate Reel", variant="primary")

            with gr.Column():
                video_out = gr.Video(label="Output Reel")
                status_out = gr.Textbox(label="Status", interactive=False)

        generate_btn.click(
            fn=generate_reel,
            inputs=[transcript_box, music_upload, prompt_box, audio_mode_radio, voice_id_box, duration_slider, vram_mode_radio],
            outputs=[video_out, status_out],
        )

    with gr.Tab("Pick a Face (M0)"):
        gr.Markdown("Upload your own image **or** generate AI candidates. Either way, click to set as reference.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload your own photo")
                upload_image = gr.Image(
                    label="Upload Image",
                    type="filepath",
                    sources=["upload"],
                )
                upload_btn = gr.Button("Use This Image as Reference", variant="primary")
                upload_status = gr.Textbox(label="Upload status", interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Or generate AI candidates")
                face_prompt = gr.Textbox(
                    label="Describe the woman",
                    value="photorealistic portrait of a 25yo woman, warm skin, long dark hair, glamorous vlog aesthetic, soft studio lighting, shoulders up, close-up",
                    lines=3,
                )
                face_count = gr.Slider(minimum=1, maximum=12, value=4, step=1, label="Number of candidates")
                face_btn = gr.Button("Generate Candidates", variant="secondary")
                face_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                face_gallery = gr.Gallery(label="Candidates — click one to set as reference", columns=2)
                reference_status = gr.Textbox(label="Reference status", interactive=False)

        upload_btn.click(
            fn=upload_reference,
            inputs=[upload_image],
            outputs=[upload_status],
        )
        face_btn.click(
            fn=generate_candidates,
            inputs=[face_prompt, face_count],
            outputs=[face_gallery, face_status],
        )
        face_gallery.select(
            fn=set_reference,
            inputs=[face_gallery],
            outputs=[reference_status],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

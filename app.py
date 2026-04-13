"""
app.py — Gradio UI for the reel generator
"""

import os
import shutil
import sys
import time

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.run_reel import run as run_reel
from pipeline.pick_portrait import pick as pick_portraits

CHARACTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character")
REFERENCE_PNG = os.path.join(CHARACTER_DIR, "reference.png")
OUTPUTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


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
        out = run_reel(transcript, music_path, prompt, audio_mode, duration=duration if not has_transcript else None, vram_mode=vram_mode.lower().replace(" ", "_"))
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


# ── Scene Video tab ───────────────────────────────────────────────────────────

def generate_scene_video(image_path: str, prompt: str, music_path: str, duration: int, vram_mode: str):
    if not image_path:
        return None, "Please upload a starting scene image."

    from pipeline.wan import generate as wan_generate, generate_chunked as wan_generate_chunked
    from pipeline.polish import polish

    # Parse prompts — one per line, one per 5s chunk
    prompts = [p.strip() for p in (prompt or "").split("\n") if p.strip()]
    if not prompts:
        return None, "Please enter at least one prompt."
    music_path = music_path if music_path else None
    vram = vram_mode.lower().replace(" ", "_")

    ts = int(time.time())
    raw_path = f"/tmp/sv_raw_{ts}.mp4"
    out_path = os.path.join(OUTPUTS_DIR, f"scene_{ts}.mp4")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    try:
        if duration > 5:
            wan_generate_chunked(image_path, prompts, raw_path,
                                 total_duration=duration, vram_mode=vram)
        else:
            wan_generate(image_path, prompts[0], raw_path,
                         duration=duration, vram_mode=vram)

        if music_path:
            polish(video=raw_path, tts=None, music=music_path, transcript="",
                   audio_mode="music_only", out_path=out_path)
            if os.path.exists(raw_path):
                os.remove(raw_path)
        else:
            shutil.move(raw_path, out_path)

        return out_path, f"Done! Saved to {out_path}"
    except SystemExit:
        return None, "Pipeline failed. Check the pod terminal for details."
    except Exception as e:
        return None, f"Error: {e}"


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
                    info="Only for music-only / dance reels. Long durations auto-chunk at full quality in None mode.",
                )
                vram_mode_radio = gr.Radio(
                    choices=["None", "Offload", "Low VRAM"],
                    value="None",
                    label="Memory Mode (dance/music-only reels)",
                    info="None: best quality, auto-chunks long videos  |  Offload: single-pass, ~20% slower  |  Low VRAM: single-pass, slowest",
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

    with gr.Tab("Video + Face"):
        gr.Markdown(
            "Upload a video and your character's face will be applied.\n\n"
            "**Approximate:** AI recreates similar motion (prompt-guided, not frame-exact).\n\n"
            "**Exact:** Swaps face directly onto the video (frame-exact, may have artifacts)."
        )

        with gr.Row():
            with gr.Column():
                vf_video = gr.Video(
                    label="Reference Video",
                    sources=["upload"],
                )
                vf_mode_radio = gr.Radio(
                    choices=["Approximate (AI-generated)", "Exact (face swap)"],
                    value="Approximate (AI-generated)",
                    label="Mode",
                )
                vf_prompt_box = gr.Textbox(
                    label="Scene / Mood Prompt (optional)",
                    placeholder="glamorous selfie vlog, golden hour, confident expression",
                    lines=2,
                )
                vf_transcript_box = gr.Textbox(
                    label="Transcript (optional — for captions)",
                    placeholder="Hey guys, welcome back...",
                    lines=3,
                )
                vf_music_upload = gr.Audio(
                    label="Background Music (optional)",
                    type="filepath",
                    sources=["upload"],
                )
                vf_voice_id_box = gr.Textbox(
                    label="ElevenLabs Voice ID",
                    value="EXAVITQu4vr4xnSDxMaL",
                    info="Only used in Approximate mode with transcript (for lip-sync TTS).",
                )
                vf_vram_mode_radio = gr.Radio(
                    choices=["None", "Offload", "Low VRAM"],
                    value="None",
                    label="Memory Mode (Approximate mode only)",
                    info="None: fastest  |  Offload: fixes OOM  |  Low VRAM: slowest",
                )
                vf_generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                vf_video_out = gr.Video(label="Output")
                vf_status_out = gr.Textbox(label="Status", interactive=False)

        def generate_video_face(input_video, mode, prompt, transcript, music_path, voice_id, vram_mode):
            if not input_video:
                return None, "Please upload a reference video."
            if not os.path.exists(REFERENCE_PNG):
                return None, "No reference portrait found. Go to the 'Pick a Face' tab first."

            transcript = transcript.strip() if transcript else ""
            music_path = music_path if music_path else None
            prompt = prompt.strip() if prompt else ""
            has_transcript = bool(transcript)
            has_music = bool(music_path)

            video_face_mode = "approximate" if "Approximate" in mode else "exact"

            # Set voice ID for approximate + transcript (TTS lip-sync)
            if voice_id and voice_id.strip():
                os.environ["ELEVENLABS_VOICE_ID"] = voice_id.strip()

            # Determine audio mode
            if has_transcript and has_music:
                audio_mode = "lipsync_only" if video_face_mode == "exact" else "voice_and_music"
            elif has_transcript:
                audio_mode = "keep_audio" if video_face_mode == "exact" else "tts_only"
            elif has_music:
                audio_mode = "music_only"
            else:
                audio_mode = None

            try:
                out = run_reel(
                    transcript, music_path, prompt, audio_mode,
                    vram_mode=vram_mode.lower().replace(" ", "_"),
                    input_video=input_video,
                    video_face_mode=video_face_mode,
                )
                return out, f"Done! Saved to {out}"
            except SystemExit:
                return None, "Pipeline failed. Check the pod terminal for details."
            except Exception as e:
                return None, f"Error: {e}"

        vf_generate_btn.click(
            fn=generate_video_face,
            inputs=[vf_video, vf_mode_radio, vf_prompt_box, vf_transcript_box,
                    vf_music_upload, vf_voice_id_box, vf_vram_mode_radio],
            outputs=[vf_video_out, vf_status_out],
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


    with gr.Tab("Scene Video (Wan I2V)"):
        gr.Markdown(
            "Upload a scene photo and describe the motion — Wan 2.2 animates it.\n\n"
            "Great for full-body cinematic shots: walking, environment scenes, slow-motion action."
        )

        with gr.Row():
            with gr.Column():
                sv_image = gr.Image(
                    label="Starting Scene Image",
                    type="filepath",
                    sources=["upload"],
                )
                sv_prompt = gr.Textbox(
                    label="Motion Prompts (one per line = one per 5s chunk)",
                    placeholder="woman steps out of luxury infinity pool, wet hair, golden hour\nwalks confidently toward camera, slow motion\nsits at poolside, looks into distance, cinematic",
                    lines=6,
                    info="Each line drives one 5s chunk. If fewer lines than chunks, the last line repeats.",
                )
                sv_music = gr.Audio(
                    label="Background Music (optional)",
                    type="filepath",
                    sources=["upload"],
                )
                sv_duration = gr.Slider(
                    minimum=5, maximum=30, step=5, value=15,
                    label="Duration (seconds)",
                )
                sv_vram_mode = gr.Radio(
                    choices=["None", "Offload", "Low VRAM"],
                    value="None",
                    label="Memory Mode",
                    info="None: fastest, full quality  |  Offload: lower VRAM  |  Low VRAM: slowest",
                )
                sv_generate_btn = gr.Button("Generate Scene Video", variant="primary")

            with gr.Column():
                sv_video_out = gr.Video(label="Output")
                sv_status_out = gr.Textbox(label="Status", interactive=False)

        sv_generate_btn.click(
            fn=generate_scene_video,
            inputs=[sv_image, sv_prompt, sv_music, sv_duration, sv_vram_mode],
            outputs=[sv_video_out, sv_status_out],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

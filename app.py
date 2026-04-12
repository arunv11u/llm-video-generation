"""
app.py — Gradio UI for the reel generator
Runs on the pod. Access via RunPod proxy URL.

Start:
    python app.py

Then open the URL shown in the terminal (e.g. https://<pod-id>-7860.proxy.runpod.net)
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

def generate_reel(transcript: str, music_path: str, prompt: str):
    if not transcript.strip():
        return None, "Please enter a transcript."
    if not music_path:
        return None, "Please upload a background music file."
    if not os.path.exists(REFERENCE_PNG):
        return None, "No reference portrait found. Go to the 'Pick a Face' tab first."

    try:
        out = run_reel(transcript.strip(), music_path, prompt.strip())
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
    selected = gallery[evt.index][0]  # tuple is (filepath, caption)
    os.makedirs(CHARACTER_DIR, exist_ok=True)
    import shutil
    shutil.copy(selected, REFERENCE_PNG)
    return f"Reference set to: {selected}"


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Reel Generator") as demo:
    gr.Markdown("# Reel Generator\nGenerate Instagram-ready reels of your AI virtual influencer.")

    with gr.Tab("Generate Reel"):
        gr.Markdown("Paste your transcript, upload background music, and hit Generate.")

        with gr.Row():
            with gr.Column():
                transcript_box = gr.Textbox(
                    label="Transcript",
                    placeholder="Hey guys, welcome back to my channel. Today I am in Paris...",
                    lines=4,
                )
                music_upload = gr.Audio(
                    label="Background Music",
                    type="filepath",
                    sources=["upload"],
                )
                prompt_box = gr.Textbox(
                    label="Scene / Mood Prompt",
                    value="glamorous selfie vlog, golden hour, confident expression",
                    lines=2,
                )
                generate_btn = gr.Button("Generate Reel", variant="primary")

            with gr.Column():
                video_out = gr.Video(label="Output Reel")
                status_out = gr.Textbox(label="Status", interactive=False)

        generate_btn.click(
            fn=generate_reel,
            inputs=[transcript_box, music_upload, prompt_box],
            outputs=[video_out, status_out],
        )

    with gr.Tab("Pick a Face (M0)"):
        gr.Markdown("Generate candidate portraits. Click one to set it as your influencer's reference face.")

        with gr.Row():
            with gr.Column():
                face_prompt = gr.Textbox(
                    label="Describe the woman",
                    value="photorealistic portrait of a 25yo woman, warm skin, long dark hair, glamorous vlog aesthetic, soft studio lighting, shoulders up, close-up",
                    lines=3,
                )
                face_count = gr.Slider(minimum=1, maximum=12, value=4, step=1, label="Number of candidates")
                face_btn = gr.Button("Generate Candidates", variant="primary")
                face_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                face_gallery = gr.Gallery(label="Candidates — click one to set as reference", columns=2)
                reference_status = gr.Textbox(label="Reference status", interactive=False)

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

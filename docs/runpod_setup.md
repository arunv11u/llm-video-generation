# RunPod H100 Pod Setup

Complete these steps once when you first start the pod. After this, every reel costs ~$0.30–0.80.

---

## 1. Start the pod

- Template: **PyTorch 2.8.0** (CUDA 12.8)
- GPU: **H100 PCIe 80GB**
- Disk: **200 GB** (models are large)
- Expose ports: `8188` (ComfyUI), `7860` (Gradio UI)

---

## 2. Open a terminal on the pod

In the RunPod dashboard → click your pod → **Connect** → **Start Web Terminal**

---

## 3. Clone this project

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/LLM-Video.git
cd LLM-Video
```

Or upload the folder via the RunPod file manager.

---

## 4. Install system dependencies

```bash
apt-get update && apt-get install -y ffmpeg git-lfs
```

---

## 5. Install ComfyUI (for portrait generation)

```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

---

## 6. Download model weights

```bash
# Flux.1-schnell
cd /workspace/ComfyUI/models/checkpoints
wget -q "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"

# F5-TTS (downloaded automatically on first run via pip)
pip install f5-tts soundfile

# SkyReels V3 A2V
cd /workspace
git clone https://github.com/SkyworkAI/SkyReels-V3.git
cd SkyReels-V3
pip install -r requirements.txt

# Download SkyReels V3 weights (~38GB — takes ~10 min)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Skywork/SkyReels-V3-A2V-19B', local_dir='/workspace/SkyReels-V3-A2V-19B')
"
```

---

## 7. Install project dependencies

```bash
cd /workspace/LLM-Video
pip install gradio
```

---

## 8. Start ComfyUI (background)

```bash
cd /workspace/ComfyUI
nohup python main.py --listen 0.0.0.0 --port 8188 > /tmp/comfy.log 2>&1 &
echo "ComfyUI started. Log: /tmp/comfy.log"
```

Wait ~30 seconds, then verify: open `https://<pod-id>-8188.proxy.runpod.net` in your browser.

---

## 9. Set environment variables

```bash
export COMFY_HOST="http://localhost:8188"
export SKYREELS_DIR="/workspace/SkyReels-V3"
export SKYREELS_MODEL="/workspace/SkyReels-V3-A2V-19B"
```

Add these to `~/.bashrc` so they persist across terminal sessions:
```bash
echo 'export COMFY_HOST="http://localhost:8188"' >> ~/.bashrc
echo 'export SKYREELS_DIR="/workspace/SkyReels-V3"' >> ~/.bashrc
echo 'export SKYREELS_MODEL="/workspace/SkyReels-V3-A2V-19B"' >> ~/.bashrc
source ~/.bashrc
```

---

## 10. Start the Gradio UI

```bash
cd /workspace/LLM-Video
python app.py
```

Open the URL shown in the terminal: `https://<pod-id>-7860.proxy.runpod.net`

---

## Usage

### M0 — Pick a face (one-time)
1. Go to the **"Pick a Face"** tab in the UI
2. Enter a description of the woman you want
3. Click **Generate Candidates**
4. Click your favourite portrait to set it as the reference

### M1 — Generate a reel
1. Go to the **"Generate Reel"** tab
2. Paste your transcript
3. Upload background music (MP3 or WAV)
4. Enter a scene/mood prompt
5. Click **Generate Reel** — wait ~6–12 min
6. Download the output video

---

## Cost management

- **Stop the pod** when not generating — H100 charges by the minute
- RunPod dashboard → your pod → **Stop**
- Restart it next time from the same dashboard — all files in `/workspace` persist

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ComfyUI not accessible | `tail -f /tmp/comfy.log` — check for errors |
| SkyReels not found | Verify `SKYREELS_DIR` env var points to the cloned repo |
| Out of VRAM | Shouldn't happen on H100 80GB. If it does, restart the pod |
| F5-TTS import error | `pip install f5-tts soundfile` |
| ffmpeg not found | `apt-get install -y ffmpeg` |

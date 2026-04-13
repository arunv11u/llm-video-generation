# Pod Startup Guide

## Every time you start the pod

### Step 1 — Install dependencies (required after every restart)

```bash
pip install gradio elevenlabs imageio imageio-ffmpeg wget easydict diffusers==0.34.0 transformers==4.44.2 accelerate==1.8.1 moviepy==2.2.1 omegaconf pyloudnorm librosa kornia ftfy torchao==0.10.0 opencv-python av sqlalchemy openai decord peft dashscope && grep -v "flash_attn" /workspace/SkyReels-V3/requirements.txt | pip install -r /dev/stdin 2>&1 | tail -5 && grep -v "flash_attn" /workspace/Wan2.2/requirements.txt | pip install -r /dev/stdin 2>&1 | tail -5
pip install flash_attn --no-build-isolation
```

### Step 2 — Verify the SkyReels attention patch is in place

This patch is needed because flash_attn is not compatible with torch 2.8. It only needs to be applied once (it persists in /workspace), but verify it's there:

```bash
grep -c "Fallback to PyTorch native SDPA" /workspace/SkyReels-V3/skyreels_v3/modules/attention.py
```

If it prints `1`, you're good. If it prints `0`, re-apply the patch:

```bash
python3 << 'EOF'
content = open('/workspace/SkyReels-V3/skyreels_v3/modules/attention.py').read()
old = """    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))"""
new = """    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        # Fallback to PyTorch native SDPA when flash_attn is unavailable
        import warnings
        warnings.warn("flash_attn not available, falling back to PyTorch SDPA.")
        q_t = q.unflatten(0, (b, lq)).transpose(1, 2)
        k_t = k.unflatten(0, (b, lk)).transpose(1, 2)
        v_t = v.unflatten(0, (b, lk)).transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        ).transpose(1, 2).flatten(0, 1).unflatten(0, (b, lq))"""
if old in content:
    open('/workspace/SkyReels-V3/skyreels_v3/modules/attention.py', 'w').write(content.replace(old, new))
    print("Patched successfully")
else:
    print("Already patched or pattern not found")
EOF
```

### Step 3 — Set ElevenLabs API key

```bash
export ELEVENLABS_API_KEY=your_api_key_here
```

> Get your API key from: elevenlabs.io → Developers → API Keys
> Get female voice IDs from: elevenlabs.io/voice-library

### Step 3b — Set R2V model path (for music-only / dance reels)

```bash
export SKYREELS_R2V_MODEL=/workspace/SkyReels-V3-R2V-14B
```

> Required when generating reels with music only (no transcript).
> R2V model must be downloaded first — see runpod_setup.md if not yet done.

### Step 3c — Set OpenAI API key (for Video + Face → Approximate mode)

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

> Optional. Used by the "Video + Face" tab's Approximate mode to auto-describe input video motion.
> If not set, Approximate mode still works but uses only your manual prompt (no auto-description).

### Step 3d — Set Wan model paths (for Scene Video tab)

```bash
export WAN_DIR=/workspace/Wan2.2
export WAN_MODEL=/workspace/Wan2.2-I2V-A14B
```

> Required for the Scene Video tab (Wan 2.2 option). Wan must be downloaded first — see runpod_setup.md if not yet done.

### Step 3e — Set SkyReels V1 I2V paths (optional — for Scene Video → SkyReels V1 option)

```bash
export SKYREELS_V1_DIR=/workspace/SkyReels-V1
export SKYREELS_V1_MODEL=Skywork/SkyReels-V1-Hunyuan-I2V
```

> Only needed if you want to use the "SkyReels V1" option in the Scene Video tab. Model auto-downloads from HuggingFace on first run (~24GB) if not pre-downloaded.

### Step 4 — Start the app

```bash
cd /workspace/LLM-Video && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python app.py
```

Open the **public Gradio URL** printed in the terminal (e.g. `https://xxxx.gradio.live`).

---

## If you want to generate new portraits (Pick a Face tab)

Start ComfyUI in a **second terminal**:

```bash
cd /workspace/ComfyUI && python main.py --listen 0.0.0.0 --port 8188
```

> **Important:** Kill ComfyUI before generating reels — it uses ~16GB VRAM that SkyReels needs.
> ```bash
> pkill -f "python main.py --listen"
> ```

---

## Generating a reel

1. Go to **Generate Reel** tab
2. Paste your transcript
3. Optionally upload background music
4. Enter a scene/mood prompt (e.g. `glamorous selfie vlog, golden hour, confident expression`)
5. Click **Generate Reel**
6. Wait ~8-12 minutes for SkyReels to finish

---

## Clearing disk space

```bash
# Delete all generated reels
rm -f /workspace/LLM-Video/outputs/*.mp4

# Delete all candidate portraits
rm -f /workspace/LLM-Video/outputs/candidates/*.png

# Delete everything in outputs at once
rm -rf /workspace/LLM-Video/outputs/*

# Check overall disk usage
df -h /workspace

# See what's taking the most space
du -sh /workspace/* | sort -rh | head -10
```

---

## Install Deep-Live-Cam (one-time, for Video + Face → Exact mode)

```bash
cd /workspace && git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam && pip install -r requirements.txt
```

Download the required model (inswapper_128.onnx) into the `models/` directory:

```bash
mkdir -p /workspace/Deep-Live-Cam/models
# Download inswapper_128.onnx from https://huggingface.co/hacksider/deep-live-cam/tree/main
# Place it at /workspace/Deep-Live-Cam/models/inswapper_128.onnx
```

> Only needed if you want to use the "Exact (face swap)" option in the Video + Face tab.

---

## Notes

- **Stop** the pod (not Terminate) to pause billing — all files in `/workspace` are preserved
- **Terminate** deletes everything permanently
- Outputs are saved to `/workspace/LLM-Video/outputs/`
- Reference portrait is at `/workspace/LLM-Video/character/reference.png`
- SkyReels A2V model is at `/workspace/SkyReels-V3-A2V-19B/` (53GB — do not delete)
- SkyReels R2V model is at `/workspace/SkyReels-V3-R2V-14B/` (53GB — do not delete, needed for dance/music-only reels)
- Flux model is at `/workspace/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors` (17GB)
- The SkyReels attention patch persists in `/workspace` across restarts — only pip packages are lost
- Wan I2V model is at `/workspace/Wan2.2-I2V-A14B/` (needed for Scene Video tab)

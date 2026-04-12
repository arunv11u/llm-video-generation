# Pod Startup Guide

## Every time you start the pod

### Step 1 — Install dependencies (required after every restart)

```bash
pip install gradio f5-tts soundfile imageio imageio-ffmpeg wget easydict diffusers==0.34.0 transformers==4.44.2 accelerate==1.8.1 moviepy==2.2.1 omegaconf pyloudnorm librosa kornia ftfy torchao==0.10.0 opencv-python av sqlalchemy && grep -v "flash_attn" /workspace/SkyReels-V3/requirements.txt | pip install -r /dev/stdin 2>&1 | tail -5
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

### Step 3 — Start the app

```bash
cd /workspace/LLM-Video && python app.py
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

## Notes

- **Stop** the pod (not Terminate) to pause billing — all files in `/workspace` are preserved
- **Terminate** deletes everything permanently
- Outputs are saved to `/workspace/LLM-Video/outputs/`
- Reference portrait is at `/workspace/LLM-Video/character/reference.png`
- SkyReels model is at `/workspace/SkyReels-V3-A2V-19B/` (53GB — do not delete)
- Flux model is at `/workspace/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors` (17GB)
- The SkyReels attention patch persists in `/workspace` across restarts — only pip packages are lost

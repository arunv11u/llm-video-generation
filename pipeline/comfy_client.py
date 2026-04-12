"""
comfy_client.py — ComfyUI HTTP client
Submits a workflow JSON to ComfyUI running on the pod, polls until done,
downloads output images.

Usage:
    from pipeline.comfy_client import run_workflow
    images = run_workflow(workflow_dict, output_dir="outputs/candidates")
"""

import json
import os
import time
import urllib.request
import urllib.parse
import uuid

COMFY_HOST = os.environ.get("COMFY_HOST", "http://localhost:8188")


def _post(endpoint: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{COMFY_HOST}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{COMFY_HOST}{endpoint}") as resp:
        return json.loads(resp.read())


def _download(filename: str, subfolder: str, out_dir: str) -> str:
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    url = f"{COMFY_HOST}/view?{params}"
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, filename)
    urllib.request.urlretrieve(url, dest)
    return dest


def run_workflow(workflow: dict, output_dir: str = "outputs/candidates") -> list[str]:
    """
    Submit workflow to ComfyUI, wait for completion, download and return image paths.
    """
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}

    resp = _post("/prompt", payload)
    prompt_id = resp["prompt_id"]
    print(f"[comfy] queued prompt_id={prompt_id}")

    # Poll until done
    while True:
        history = _get(f"/history/{prompt_id}")
        if prompt_id in history:
            break
        print("[comfy] waiting...")
        time.sleep(3)

    outputs = history[prompt_id]["outputs"]
    images = []
    for node_id, node_output in outputs.items():
        for img in node_output.get("images", []):
            path = _download(img["filename"], img.get("subfolder", ""), output_dir)
            images.append(path)
            print(f"[comfy] downloaded {path}")

    return images

"""
pick_portrait.py — M0 portrait generation CLI
Generates N candidate portraits using Flux.1-schnell via ComfyUI.
You inspect the results, pick your favourite, and copy it to character/reference.png.

Usage (pod):
    python pipeline/pick_portrait.py \
        --prompt "photorealistic portrait of a 25yo woman, warm skin, long dark hair, glamorous vlog aesthetic, soft studio lighting" \
        --count 4

Output: outputs/candidates/candidate_1.png ... candidate_N.png
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.comfy_client import run_workflow

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows")
CANDIDATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "candidates")
CHARACTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "character")


def load_workflow(prompt: str, count: int) -> dict:
    workflow_path = os.path.join(WORKFLOWS_DIR, "00_candidate_portraits.json")
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Inject the user prompt into the CLIP text encode node (node "6")
    workflow["6"]["inputs"]["text"] = prompt

    # Set batch size (node "5" is the EmptyLatentImage node)
    workflow["5"]["inputs"]["batch_size"] = count

    return workflow


def pick(prompt: str, count: int) -> list[str]:
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    print(f"[pick_portrait] generating {count} candidates...")
    workflow = load_workflow(prompt, count)
    images = run_workflow(workflow, output_dir=CANDIDATES_DIR)

    # Rename to candidate_1.png, candidate_2.png, ...
    renamed = []
    for i, src in enumerate(images, 1):
        dst = os.path.join(CANDIDATES_DIR, f"candidate_{i}.png")
        shutil.move(src, dst)
        renamed.append(dst)
        print(f"  → {dst}")

    print(f"\n[pick_portrait] Done. Inspect outputs/candidates/ and pick your favourite.")
    print(f"  cp outputs/candidates/candidate_X.png character/reference.png")
    return renamed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate candidate portraits (M0)")
    parser.add_argument(
        "--prompt",
        default="photorealistic portrait of a 25yo woman, warm skin, long dark hair, glamorous vlog aesthetic, soft studio lighting, shoulders up, close-up",
        help="Text description of the woman",
    )
    parser.add_argument("--count", type=int, default=4, help="Number of candidates to generate")
    args = parser.parse_args()

    pick(args.prompt, args.count)

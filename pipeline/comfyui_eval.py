"""
Evaluate LoRA via ComfyUI API — uses the full inference pipeline with
distilled LoRA, block swap, upscalers, chunked feedforward.

Copies the trained LoRA to ComfyUI's loras folder, queues a workflow,
and saves the output video to the batch's samples folder.
"""

import json
import logging
import os
import shutil
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

COMFYUI_URL = "http://127.0.0.1:8000"
COMFYUI_LORAS_DIR = Path("C:/Users/morte/Projects/playground/ComfyUI/models/loras/ltx2")
COMFYUI_INPUT_DIR = Path("C:/Users/morte/Projects/playground/ComfyUI/input")
COMFYUI_OUTPUT_DIR = Path("C:/Users/morte/Projects/playground/ComfyUI/output")
LORA_FILENAME = "autopilot_eval.safetensors"


COMFYUI_EXE = Path("C:/Users/morte/AppData/Local/Programs/ComfyUI/ComfyUI.exe")


def is_running() -> bool:
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def ensure_running() -> bool:
    """Start ComfyUI if not running. Returns True if ready."""
    if is_running():
        return True

    if not COMFYUI_EXE.exists():
        log.warning("ComfyUI exe not found at %s", COMFYUI_EXE)
        return False

    log.info("Starting ComfyUI Desktop...")
    import subprocess as _sp
    _sp.Popen([str(COMFYUI_EXE)], creationflags=getattr(_sp, "CREATE_NEW_PROCESS_GROUP", 0))

    # Wait for it to be ready
    for i in range(60):
        time.sleep(5)
        if is_running():
            log.info("ComfyUI ready (took %ds)", (i + 1) * 5)
            return True

    log.warning("ComfyUI failed to start within 5 minutes")
    return False


def copy_lora(checkpoint: Path) -> str:
    """Copy trained LoRA to ComfyUI's loras folder. Returns ComfyUI lora name."""
    dst = COMFYUI_LORAS_DIR / LORA_FILENAME
    shutil.copy2(checkpoint, dst)
    log.info("Copied LoRA to %s", dst)
    import os as _os
    return f"ltx2{_os.sep}{LORA_FILENAME}"


def queue_prompt(prompt: dict) -> str:
    """Queue a workflow prompt. Returns prompt_id."""
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        prompt_id = result.get("prompt_id", "")
        log.info("Queued prompt: %s", prompt_id)
        return prompt_id
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        log.error("ComfyUI API error %d: %s", e.code, error_body[:500])
        raise


def wait_for_completion(prompt_id: str, timeout: int = 900) -> dict | None:
    """Wait for prompt to complete. Returns output info or None on timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            history = json.loads(resp.read())
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed", False) or status.get("status_str") == "success":
                    return history[prompt_id]
        except Exception:
            pass
        time.sleep(5)
    return None


def build_aio_prompt(
    prompt_text: str,
    lora_name: str,
    lora_strength: float = 1.0,
    seed: int = 42,
    output_prefix: str = "autopilot_eval",
    condition_image: str | None = None,
) -> dict | None:
    """Load the full AIO workflow and modify prompt + LoRA."""
    aio_path = Path(__file__).parent.parent / "workflow.json"
    if not aio_path.exists():
        log.warning("workflow.json not found — using simple workflow")
        return None

    wf = json.loads(aio_path.read_text(encoding="utf-8"))

    # Update prompt text
    if "92:3" in wf:
        wf["92:3"]["inputs"]["text"] = prompt_text

    # Update our LoRA in Power Lora Loader
    if "523" in wf:
        wf["523"]["inputs"]["lora_2"] = {
            "on": True,
            "lora": lora_name,
            "strength": lora_strength,
        }

    # Update seed
    if "109" in wf:
        wf["109"]["inputs"]["seed"] = seed

    # Update output prefix
    if "141" in wf:
        wf["141"]["inputs"]["filename_prefix"] = output_prefix

    # Update conditioning image for i2v
    if condition_image and "98" in wf:
        img_src = Path(condition_image)
        img_dst = COMFYUI_INPUT_DIR / img_src.name
        shutil.copy2(img_src, img_dst)
        wf["98"]["inputs"]["image"] = img_src.name

    return wf


def build_t2v_prompt(
    prompt_text: str,
    lora_name: str,
    lora_strength: float = 1.0,
    width: int = 768,
    height: int = 448,
    num_frames: int = 89,
    seed: int = 42,
    output_prefix: str = "autopilot_eval",
) -> dict:
    """Build a ComfyUI API prompt — uses AIO workflow if available, else simple."""
    aio = build_aio_prompt(prompt_text, lora_name, lora_strength, seed, output_prefix)
    if aio:
        return aio

    # Fallback: simple workflow without upscaling
    return {
        "1": {
            "class_type": "DiffusionModelLoaderKJ",
            "inputs": {
                "model_name": os.sep.join(["ltx2", "ltx-2.3-22b-dev_transformer_only_bf16.safetensors"]),
                "weight_dtype": "default",
                "compute_dtype": "default",
                "patch_cublaslinear": False,
                "sage_attention": "disabled",
                "enable_fp16_accumulation": False,
            }
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "gemma-3-12b-abliterated-text-encoder.safetensors",
                "clip_name2": "ltx-2.3_text_projection_bf16.safetensors",
                "type": "ltxv",
                "device": "default",
            }
        },
        "3": {
            "class_type": "VAELoaderKJ",
            "inputs": {
                "vae_name": "LTX23_video_vae_bf16.safetensors",
                "device": "main_device",
                "dtype": "bf16",
                "weight_dtype": "bf16",
            }
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": os.sep.join(["ltx2", "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"]),
                "strength_model": 0.6,
            }
        },
        "5": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["4", 0],
                "lora_name": lora_name,
                "strength_model": lora_strength,
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt_text,
                "clip": ["2", 0],
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "clip": ["2", 0],
            }
        },
        "8": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            }
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "sgm_uniform",
                "denoise": 1.0,
            }
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0],
            }
        },
        "11": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["10", 0],
                "frame_rate": 25,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            }
        },
    }


def build_i2v_prompt(
    prompt_text: str,
    image_path: str,
    lora_name: str,
    lora_strength: float = 1.0,
    width: int = 768,
    height: int = 448,
    num_frames: int = 89,
    seed: int = 42,
    output_prefix: str = "autopilot_i2v",
) -> dict:
    """Build a ComfyUI API prompt for i2v generation."""
    p = build_t2v_prompt(prompt_text, lora_name, lora_strength, width, height, num_frames, seed, output_prefix)

    # Copy image to ComfyUI input dir
    img_src = Path(image_path)
    img_dst = COMFYUI_INPUT_DIR / img_src.name
    shutil.copy2(img_src, img_dst)

    # Add LoadImage node
    p["12"] = {
        "class_type": "LoadImage",
        "inputs": {
            "image": img_src.name,
        }
    }

    # Replace EmptyLatent with LTXVImgToVideo
    p["8"] = {
        "class_type": "LTXVImgToVideo",
        "inputs": {
            "positive": ["6", 0],
            "negative": ["7", 0],
            "vae": ["3", 0],
            "image": ["12", 0],
            "width": width,
            "height": height,
            "length": num_frames,
            "batch_size": 1,
        }
    }

    # Update KSampler to use i2v latent
    p["9"]["inputs"]["latent_image"] = ["8", 0]
    p["9"]["inputs"]["positive"] = ["8", 1]  # i2v conditioning
    p["9"]["inputs"]["negative"] = ["8", 2]

    return p


def run_eval(
    checkpoint: Path,
    step: int,
    output_dir: Path,
    prompts: list[str] | None = None,
    i2v_refs: list[dict] | None = None,
    width: int = 768,
    height: int = 448,
    num_frames: int = 89,
):
    """Run t2v + i2v evaluation via ComfyUI API."""
    if not ensure_running():
        log.warning("ComfyUI not available — skipping eval")
        return

    lora_name = copy_lora(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    # T2V
    if prompts:
        for i, prompt in enumerate(prompts):
            log.info("T2V %d: %.80s...", i, prompt)
            prefix = f"step_{step:06d}_t2v_{i}"
            p = build_aio_prompt(prompt, lora_name, seed=42+i, output_prefix=prefix)
            if p is None:
                p = build_t2v_prompt(prompt, lora_name, width=width, height=height,
                                    num_frames=num_frames, seed=42+i, output_prefix=prefix)
            prompt_id = queue_prompt(p)
            result = wait_for_completion(prompt_id)
            if result:
                log.info("  T2V %d complete", i)
                _copy_output(prefix, output_dir)
            else:
                log.error("  T2V %d timed out", i)

    # I2V
    if i2v_refs:
        for i, ref in enumerate(i2v_refs[:2]):
            log.info("I2V %d: %s", i, Path(ref["image"]).name)
            prefix = f"step_{step:06d}_i2v_{i}"
            # Try AIO workflow with conditioning image
            p = build_aio_prompt(ref["prompt"], lora_name, seed=42+i,
                                output_prefix=prefix, condition_image=ref["image"])
            if p is None:
                p = build_i2v_prompt(ref["prompt"], ref["image"], lora_name,
                                    width=width, height=height, num_frames=num_frames,
                                    seed=42+i, output_prefix=prefix)
            prompt_id = queue_prompt(p)
            result = wait_for_completion(prompt_id)
            if result:
                log.info("  I2V %d complete", i)
                _copy_output(prefix, output_dir)
            else:
                log.error("  I2V %d timed out", i)

    log.info("ComfyUI eval complete — results in %s", output_dir)


def _copy_output(prefix: str, output_dir: Path):
    """Copy generated video from ComfyUI output to our samples dir."""
    for f in sorted(COMFYUI_OUTPUT_DIR.glob(f"{prefix}*")):
        dst = output_dir / f.name
        shutil.copy2(f, dst)
        log.info("  Saved: %s (%.1f KB)", dst.name, dst.stat().st_size / 1024)

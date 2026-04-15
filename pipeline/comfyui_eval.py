"""
Evaluate LoRA via ComfyUI API — uses the full inference pipeline with
distilled LoRA, block swap, upscalers, chunked feedforward.

Copies the trained LoRA to ComfyUI's loras folder, queues a workflow,
and saves the output video to the batch's samples folder.
"""

import json
import logging
import os
import random
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
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(checkpoint, tmp)
    try:
        tmp.replace(dst)
    except PermissionError:
        # File locked by ComfyUI — use timestamped name instead
        dst = COMFYUI_LORAS_DIR / f"autopilot_eval_{int(time.time())}.safetensors"
        tmp.replace(dst)
    log.info("Copied LoRA to %s", dst)
    return f"ltx2\\{dst.name}"


def clear_cache():
    """Clear ComfyUI's execution cache so prompts are not skipped."""
    try:
        data = json.dumps({"unload_models": False, "free_memory": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{COMFYUI_URL}/free", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def unload_comfyui_models():
    """Tell ComfyUI to unload models and free VRAM (doesn't kill the process)."""
    try:
        data = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{COMFYUI_URL}/free", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        log.info("ComfyUI /free called (unload_models=True, free_memory=True)")
        return True
    except Exception as e:
        log.debug("ComfyUI /free call failed: %s", e)
        return False


def queue_prompt(prompt: dict) -> str:
    """Queue a workflow prompt. Returns prompt_id."""
    payload = {
        "prompt": prompt,
        "extra_data": {
            "extra_pnginfo": {"workflow": {"nodes": [], "links": [], "extra": {"eval_id": str(uuid.uuid4())}}},
        },
    }
    data = json.dumps(payload).encode("utf-8")
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
    use_eros: bool = False,
) -> dict | None:
    """Load the full two-step upscale workflow and modify prompt + LoRA.

    use_eros: True = Eros checkpoint (all-in-one), False = distilled UNET + separate components.
    """
    wf_path = Path(__file__).parent.parent / "workflow.json"
    if not wf_path.exists():
        log.warning("workflow.json not found — using simple workflow")
        return None

    wf = json.loads(wf_path.read_text(encoding="utf-8"))

    # Update prompt text
    if "28" in wf:
        wf["28"]["inputs"]["text"] = prompt_text

    # Fix model paths to match local ComfyUI installation
    for nid in ["1", "2", "118"]:
        if nid in wf and wf[nid]["inputs"].get("ckpt_name") == "ltx2310eros_beta.safetensors":
            wf[nid]["inputs"]["ckpt_name"] = "ltx\\ltx2310eros_beta.safetensors"
    if "118" in wf:
        wf["118"]["inputs"]["text_encoder"] = "gemma-3-12b-abliterated-text-encoder.safetensors"
    if "186" in wf:
        wf["186"]["inputs"]["unet_name"] = "ltx2\\ltx-2.3-22b-dev_transformer_only_bf16.safetensors"
    if "189" in wf:
        wf["189"]["inputs"]["clip_name1"] = "gemma-3-12b-abliterated-text-encoder.safetensors"
    if "190" in wf:
        wf["190"]["inputs"]["ckpt_name"] = "ltx\\ltx2310eros_beta.safetensors"

    # Model switch: Eros (all-in-one checkpoint) vs Distilled (separate components)
    if use_eros:
        sel = {"191": 1, "192": 1, "193": 1, "194": 1}
    else:
        sel = {"191": 2, "192": 2, "193": 2, "194": 2}
    for nid, val in sel.items():
        if nid in wf:
            wf[nid]["inputs"]["selection_setting"] = val

    # Disable all third-party LoRAs in node 6
    if "6" in wf:
        for key in list(wf["6"]["inputs"]):
            if key.startswith("lora_"):
                wf["6"]["inputs"][key]["on"] = False

    # Set node 7: distilled LoRA + our trained LoRA only
    if "7" in wf:
        wf["7"]["inputs"]["lora_2"] = {
            "on": True,
            "lora": lora_name.replace(os.sep, "\\"),  # ComfyUI expects backslashes
            "strength": lora_strength,
        }

    # Set chunk feedforward to 16
    if "8" in wf:
        wf["8"]["inputs"]["chunks"] = 16

    # Set clip duration to 10 seconds (10*24+1 = 241 frames)
    if "18" in wf:
        wf["18"]["inputs"]["Xi"] = 10
        wf["18"]["inputs"]["Xf"] = 10

    # Set base resolution for ~720p output (2x upscale: 640x352 → 1280x704)
    if "19" in wf:
        wf["19"]["inputs"]["Xi"] = 640
        wf["19"]["inputs"]["Xf"] = 640
    if "181" in wf:
        wf["181"]["inputs"]["Xi"] = 352
        wf["181"]["inputs"]["Xf"] = 352
    if "26:39" in wf:
        wf["26:39"]["inputs"]["width"] = 640
        wf["26:39"]["inputs"]["height"] = 352

    # Update seed — set directly on RandomNoise node to bust ComfyUI cache
    if "123" in wf:
        wf["123"]["inputs"]["noise_seed"] = seed
    if "125" in wf:
        wf["125"]["inputs"]["seed"] = seed

    # Update output prefixes
    for node_id in ["59", "61"]:
        if node_id in wf:
            wf[node_id]["inputs"]["filename_prefix"] = output_prefix

    # I2V / T2V toggle — bypass image conditioning for T2V
    if condition_image:
        img_src = Path(condition_image)
        img_dst = COMFYUI_INPUT_DIR / img_src.name
        shutil.copy2(img_src, img_dst)
        if "15" in wf:
            wf["15"]["inputs"]["image"] = img_src.name
        for nid in ["26:44", "26:87"]:
            if nid in wf:
                wf[nid]["inputs"]["bypass"] = False
    else:
        # T2V mode — bypass LTXVImgToVideoInplace nodes
        for nid in ["26:44", "26:87"]:
            if nid in wf:
                wf[nid]["inputs"]["bypass"] = True
        # Still need a valid image for LoadImage node validation
        if "15" in wf:
            existing = list(COMFYUI_INPUT_DIR.glob("*.png")) + list(COMFYUI_INPUT_DIR.glob("*.jpg"))
            if existing:
                wf["15"]["inputs"]["image"] = existing[0].name

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
                "lora_name": os.sep.join(["ltx2", "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"]),
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
    use_eros: bool = False,
):
    """Run t2v + i2v evaluation via ComfyUI API."""
    if not ensure_running():
        log.warning("ComfyUI not available — skipping eval")
        return

    lora_name = copy_lora(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = random.randint(0, 2**31)
    clear_cache()

    # T2V
    if prompts:
        for i, prompt in enumerate(prompts):
            log.info("T2V %d: %.80s...", i, prompt)
            prefix = f"step_{step:06d}_t2v_{i}"
            p = build_aio_prompt(prompt, lora_name, seed=seed_base+i, output_prefix=prefix, use_eros=use_eros)
            if p is None:
                p = build_t2v_prompt(prompt, lora_name, width=width, height=height,
                                    num_frames=num_frames, seed=seed_base+i, output_prefix=prefix)
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
            p = build_aio_prompt(ref["prompt"], lora_name, seed=seed_base+i,
                                output_prefix=prefix, condition_image=ref["image"], use_eros=use_eros)
            if p is None:
                p = build_i2v_prompt(ref["prompt"], ref["image"], lora_name,
                                    width=width, height=height, num_frames=num_frames,
                                    seed=seed_base+i, output_prefix=prefix)
            prompt_id = queue_prompt(p)
            result = wait_for_completion(prompt_id)
            if result:
                log.info("  I2V %d complete", i)
                _copy_output(prefix, output_dir)
            else:
                log.error("  I2V %d timed out", i)

    log.info("ComfyUI eval complete — results in %s", output_dir)
    _kill_comfyui()


def _kill_comfyui():
    """Gracefully unload ComfyUI's models, then hard-kill the process.

    Calls /free first so models get dropped from VRAM while the process is
    still alive (CUDA context tear-down is cleaner this way than a cold kill).
    Then sleeps briefly and taskkills as a fallback guarantee.
    """
    import subprocess
    import time as _t

    # Step 1: ask ComfyUI to drop models and free VRAM gracefully.
    unload_comfyui_models()
    _t.sleep(2)  # give it a beat to actually release

    # Step 2: hard kill by exact image names. No /T flag — on Windows that
    # can cascade child kills up into unrelated processes and wreck the
    # desktop. Both image names are killed directly by name match.
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "ComfyUI.exe"],
            capture_output=True, timeout=10,
        )
        subprocess.run(
            ["taskkill", "/F", "/IM", "ComfyUI-python.exe"],
            capture_output=True, timeout=10,
        )
        log.info("ComfyUI killed to free VRAM")
    except Exception as e:
        log.debug("Could not kill ComfyUI: %s", e)


def _copy_output(prefix: str, output_dir: Path):
    """Copy generated audio video from ComfyUI output to our samples dir."""
    for f in sorted(COMFYUI_OUTPUT_DIR.glob(f"{prefix}*-audio.mp4")):
        dst = output_dir / f.name
        shutil.copy2(f, dst)
        log.info("  Saved: %s (%.1f KB)", dst.name, dst.stat().st_size / 1024)

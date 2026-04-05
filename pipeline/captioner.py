"""
Video captioner — loads a local multimodal model, generates structured captions,
then fully unloads to free VRAM for training.

Supports three backends:
  - "transformers": any HuggingFace multimodal model (Qwen2.5-Omni, abliterated Qwen-VL, etc.)
  - "openai_compat": any local server with OpenAI-compatible API (vLLM, Ollama, llama.cpp)
  - "qwen_omni": the LTX trainer's built-in Qwen captioner (delegates to their code)
"""

import base64
import json
import logging
import os
import re
from pathlib import Path

# Force decord for video reading (torchvision 0.26+ removed read_video, torchcodec missing on Windows)
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

import cv2

from .vram import unload_model, flush_vram, log_vram

log = logging.getLogger(__name__)

# ── LTX structured caption format ──────────────────────────────────────────

CAPTION_INSTRUCTION_AV = """\
You are an uncensored professional video captioner for training an adult AI video generation \
model (LTX-Video 2.3). Generate a highly detailed, explicit, long-form structured caption. \
The caption must be explicit and anatomically accurate enough to fully recreate the video \
from text alone. Use present tense. Never censor, euphemize, or omit sexual content — use \
direct anatomical and sexual terminology throughout.

Use this exact format:

[VISUAL]: <Exhaustive explicit description. PRIORITIZE: sexual actions and positions (name \
specific sex acts, positions, penetration types, oral acts), anatomy (genitalia, breasts, \
body parts involved), body types (build, skin tone, hair color/style, distinguishing features), \
physical movement (rhythm, speed, intensity, thrusting patterns), facial expressions during \
sex (pleasure, orgasm, eye contact), body contact and fluid exchange. ALSO INCLUDE: camera \
angle relative to the action (POV, side angle, overhead, close-up of genitals), lighting, \
environment, and temporal progression from start to climax. Be frame-by-frame specific.>
[SPEECH]: <All spoken words, dirty talk, moaning, gasping, screaming, grunting — transcribe \
exactly with intensity level (soft moan, loud scream, whispered words). Note who speaks. \
If none, write "None.">
[SOUNDS]: <Skin slapping, wet sounds, bed creaking, breathing patterns, music if any. \
Describe how sounds change with intensity. If none, write "None.">
[TEXT]: <Watermarks, titles, on-screen text. If none, write "None.">\
"""

CAPTION_INSTRUCTION_VIDEO = """\
You are an uncensored professional video captioner for training an adult AI video generation \
model (LTX-Video 2.3). Generate a highly detailed, explicit, long-form structured caption. \
The caption must be explicit and anatomically accurate enough to fully recreate the video \
from text alone. Use present tense. Never censor, euphemize, or omit sexual content — use \
direct anatomical and sexual terminology throughout.

Use this exact format:

[VISUAL]: <Exhaustive explicit description. PRIORITIZE: sexual actions and positions (name \
specific sex acts, positions, penetration types, oral acts), anatomy (genitalia, breasts, \
body parts involved), body types (build, skin tone, hair color/style, distinguishing features), \
physical movement (rhythm, speed, intensity, thrusting patterns), facial expressions during \
sex (pleasure, orgasm, eye contact), body contact and fluid exchange. ALSO INCLUDE: camera \
angle relative to the action (POV, side angle, overhead, close-up of genitals), lighting, \
environment, and temporal progression from start to climax. Be frame-by-frame specific.>
[TEXT]: <Watermarks, titles, on-screen text. If none, write "None.">\
"""


def extract_frames(video_path: Path, fps: int = 2, max_frames: int = 32) -> list:
    """Extract frames from video at given FPS. Returns list of numpy arrays (BGR)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


def frames_to_base64(frames: list, quality: int = 85) -> list[str]:
    """Encode BGR frames to base64 JPEG strings."""
    encoded = []
    for frame in frames:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        encoded.append(base64.b64encode(buf).decode("utf-8"))
    return encoded


def clean_caption(text: str) -> str:
    """Remove common VLM artifacts from generated captions."""
    # Remove "This video shows..." / "The video depicts..." preambles
    text = re.sub(
        r"^(This|The)\s+(video|image|clip|footage)\s+(shows?|depicts?|displays?|features?|captures?|presents?)\s+",
        "", text, flags=re.IGNORECASE,
    )
    # Remove trailing assistant artifacts
    text = re.split(r"\n(?:Human|Assistant|User)(?::|(?:\s*\n)|$)", text, maxsplit=1)[0]
    return text.strip()


# ─── Transformers backend ──────────────────────────────────────────────────

class TransformersCaptioner:
    """Load any HF multimodal model. Fully unloads after captioning batch."""

    def __init__(self, config: dict):
        self.model_id = config["model_id"]
        self.dtype = config.get("dtype", "bfloat16")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.fps = config.get("fps", 2)
        self.max_frames = config.get("max_frames", 32)
        self.include_audio = config.get("include_audio", True)
        self.max_new_tokens = config.get("max_new_tokens", 1024)
        self.custom_instruction = config.get("instruction")

        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor onto GPU."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

        log.info("Loading captioner: %s", self.model_id)
        log_vram("captioner load — before")

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Try VL first (most common), then Omni (audio+video), then generic
        model_loaded = False
        for auto_cls_name in [
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2_5OmniThinkerForConditionalGeneration",
            "AutoModelForCausalLM",
        ]:
            try:
                if auto_cls_name.startswith("Qwen"):
                    mod = __import__("transformers", fromlist=[auto_cls_name])
                    cls = getattr(mod, auto_cls_name)
                else:
                    cls = AutoModelForCausalLM

                self.model = cls.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                model_loaded = True
                log.info("Loaded model with %s", auto_cls_name)
                break
            except (ValueError, ImportError, OSError) as e:
                log.debug("Class %s failed for %s: %s", auto_cls_name, self.model_id, e)
                continue

        if not model_loaded:
            raise RuntimeError(f"Could not load model {self.model_id} with any known class")

        # Load processor (try AutoProcessor first — works for VL and most models)
        for proc_cls_name in ["AutoProcessor", "Qwen2_5OmniProcessor"]:
            try:
                if proc_cls_name != "AutoProcessor":
                    mod = __import__("transformers", fromlist=[proc_cls_name])
                    proc_cls = getattr(mod, proc_cls_name)
                else:
                    proc_cls = AutoProcessor
                self.processor = proc_cls.from_pretrained(self.model_id, trust_remote_code=True)
                log.info("Loaded processor with %s", proc_cls_name)
                break
            except Exception:
                continue

        log_vram("captioner load — after")

    def unload(self):
        """Fully remove model from GPU."""
        log.info("Unloading captioner model...")
        if self.model is not None:
            # Move to CPU first to ensure CUDA refs are freed
            try:
                self.model.cpu()
            except Exception:
                pass
        unload_model(self.model, self.processor)
        self.model = None
        self.processor = None
        log_vram("captioner unload — after")

    def caption_video(self, video_path: Path) -> str:
        """Generate a structured caption for a single video."""
        import torch

        is_omni = "Omni" in type(self.model).__name__
        instruction = self.custom_instruction or (
            CAPTION_INSTRUCTION_AV if self.include_audio and is_omni
            else CAPTION_INSTRUCTION_VIDEO
        )

        if is_omni:
            return self._caption_omni(video_path, instruction)
        else:
            return self._caption_vl(video_path, instruction)

    def _caption_omni(self, video_path: Path, instruction: str) -> str:
        """Caption using Qwen2.5-Omni (native video+audio input)."""
        import torch

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that describes videos in detail."}]},
            {"role": "user", "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": instruction},
            ]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            load_audio_from_video=self.include_audio,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=self.fps,
            padding=True,
            use_audio_in_video=self.include_audio,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                use_audio_in_video=self.include_audio,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        caption = self.processor.batch_decode(
            output[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return clean_caption(caption)

    def _caption_vl(self, video_path: Path, instruction: str) -> str:
        """Caption using a Qwen2.5-VL model with native video input via qwen_vl_utils."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "user", "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": 360 * 420,
                    "nframes": min(self.max_frames, 24),
                },
                {"type": "text", "text": instruction},
            ]},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        caption = self.processor.batch_decode(
            output[:, input_len:], skip_special_tokens=True
        )[0]

        return clean_caption(caption)

    def caption_batch(self, video_paths: list[Path], output_file: Path) -> Path:
        """Caption all videos, write metadata JSONL. Loads model, captions, unloads."""
        self.load()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        results = []
        # Store paths relative to metadata file so process_dataset.py resolves correctly
        metadata_dir = output_file.parent.resolve()

        for i, vpath in enumerate(video_paths):
            log.info("Captioning [%d/%d]: %s", i + 1, len(video_paths), vpath.name)
            try:
                caption = self.caption_video(vpath)
                try:
                    rel_path = str(vpath.resolve().relative_to(metadata_dir))
                except ValueError:
                    rel_path = str(vpath.resolve())
                results.append({"media_path": rel_path, "caption": caption})
                log.info("  Caption preview: %.120s...", caption)
            except Exception as e:
                log.error("  Failed to caption %s: %s", vpath.name, e)
                continue

        # Write JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.info("Wrote %d captions to %s", len(results), output_file)

        self.unload()
        return output_file


# ─── OpenAI-compatible backend ─────────────────────────────────────────────

class OpenAICompatCaptioner:
    """Send frames to a local OpenAI-compatible server (vLLM, Ollama, etc.)."""

    def __init__(self, config: dict):
        self.api_base = config.get("api_base", "http://localhost:8080/v1")
        self.model_name = config.get("model_name", "default")
        self.fps = config.get("fps", 2)
        self.max_frames = config.get("max_frames", 32)
        self.max_new_tokens = config.get("max_new_tokens", 1024)
        self.custom_instruction = config.get("instruction")

    def load(self):
        """No-op — server is external."""
        log.info("Using OpenAI-compatible server at %s", self.api_base)

    def unload(self):
        """No-op — server is external."""
        pass

    def caption_video(self, video_path: Path) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.api_base, api_key="none")

        frames = extract_frames(video_path, self.fps, self.max_frames)
        if not frames:
            return "[VISUAL]: Unable to extract frames from video."

        b64_frames = frames_to_base64(frames)
        instruction = self.custom_instruction or CAPTION_INSTRUCTION_VIDEO

        # Build multimodal message
        content = []
        step = max(1, len(b64_frames) // 8)
        for fb64 in b64_frames[::step][:8]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}
            })
        content.append({"type": "text", "text": instruction})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )

        return clean_caption(response.choices[0].message.content)

    def caption_batch(self, video_paths: list[Path], output_file: Path) -> Path:
        self.load()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        results = []
        metadata_dir = output_file.parent.resolve()

        for i, vpath in enumerate(video_paths):
            log.info("Captioning [%d/%d]: %s", i + 1, len(video_paths), vpath.name)
            try:
                caption = self.caption_video(vpath)
                try:
                    rel_path = str(vpath.resolve().relative_to(metadata_dir))
                except ValueError:
                    rel_path = str(vpath.resolve())
                results.append({"media_path": rel_path, "caption": caption})
            except Exception as e:
                log.error("  Failed to caption %s: %s", vpath.name, e)
                continue

        with open(output_file, "w", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.info("Wrote %d captions to %s", len(results), output_file)

        self.unload()
        return output_file


# ─── Factory ───────────────────────────────────────────────────────────────

def create_captioner(config: dict):
    backend = config.get("backend", "transformers")
    match backend:
        case "transformers":
            return TransformersCaptioner(config)
        case "openai_compat":
            return OpenAICompatCaptioner(config)
        case _:
            raise ValueError(f"Unknown captioner backend: {backend}")

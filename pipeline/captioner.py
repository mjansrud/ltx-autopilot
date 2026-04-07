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
model (LTX-Video 2.3). Generate a highly detailed, explicit caption as a single flowing \
paragraph. The caption must be explicit and anatomically accurate enough to fully recreate \
the video from text alone. Use present tense. Never censor, euphemize, or omit sexual \
content — use direct anatomical and sexual terminology throughout.

If the clip is ONLY credits/logos/text with zero nudity, write only SKIP.

Write one flowing paragraph that includes: sexual actions and positions (name specific sex \
acts, positions, penetration types, oral acts), anatomy with sizes (breast size like small/medium/large/huge, \
breast shape like round/perky/saggy/teardrop, penis size like average/large/thick, nipple color \
and size, ass shape, how breasts move/bounce/sway during action), body types (build, \
skin tone, hair color/style, distinguishing features like tattoos, piercings), physical \
movement (rhythm, speed, intensity, thrusting patterns), facial expressions during sex \
(pleasure, orgasm, eye contact), body contact and fluid exchange, camera angle and any camera \
movements or cuts, lighting, environment, temporal progression from start to end. Also \
include any spoken words or dirty talk in quotes, moaning with intensity, and sounds like \
skin slapping, wet sounds, bed creaking, breathing patterns, background music.\
"""

CAPTION_INSTRUCTION_VIDEO = CAPTION_INSTRUCTION_AV  # Same prompt, audio section ignored by VL models


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
    text = re.split(r"\n(?:Human|Assistant|User)(?::|(?:\s*\n)|$)", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

        # Detect model type from config, then load the right class
        import json
        model_config = json.loads(Path(self.model_id, "config.json").read_text())
        model_type = model_config.get("model_type", "")

        if "omni" in model_type:
            class_names = ["Qwen2_5OmniForConditionalGeneration"]
        elif "qwen2_5_vl" in model_type or "qwen2_vl" in model_type:
            class_names = ["Qwen2_5_VLForConditionalGeneration"]
        else:
            class_names = ["AutoModelForCausalLM"]

        model_loaded = False
        for auto_cls_name in class_names:
            try:
                if auto_cls_name.startswith("Qwen"):
                    mod = __import__("transformers", fromlist=[auto_cls_name])
                    cls = getattr(mod, auto_cls_name)
                else:
                    cls = AutoModelForCausalLM

                load_kwargs = dict(
                    torch_dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                # Omni max_memory not needed with 4-bit (6.6GB vs 20GB bf16)

                self.model = cls.from_pretrained(self.model_id, **load_kwargs)
                model_loaded = True
                log.info("Loaded model with %s", auto_cls_name)
                break
            except (ValueError, ImportError, OSError) as e:
                log.debug("Class %s failed for %s: %s", auto_cls_name, self.model_id, e)
                continue

        if not model_loaded:
            raise RuntimeError(f"Could not load model {self.model_id} with any known class")

        # Load processor
        proc_order = ["Qwen2_5OmniProcessor", "AutoProcessor"] if "omni" in model_type else ["AutoProcessor"]
        for proc_cls_name in proc_order:
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
            try:
                self.model.cpu()
            except Exception:
                pass
        unload_model(self.model, self.processor)
        self.model = None
        self.processor = None
        flush_vram()
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
        """Caption using Qwen2.5-Omni via official qwen_omni_utils."""
        import torch
        from qwen_omni_utils import process_mm_info

        use_audio = self.include_audio

        # Must use Omni's default system prompt for audio understanding to work
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [
                {"type": "video", "video": str(video_path),
                 "max_pixels": 640 * 480, "nframes": 42},
                {"type": "text", "text": instruction},
            ]},
        ]

        # process_mm_info handles video/audio extraction (uses decord/librosa internally)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)

        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            # return_audio=False skips talker (avoids CUDA assert from
            # uninitialized talker weights in thinker-only checkpoint)
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                return_audio=False,
                do_sample=False,
                thinker_max_new_tokens=self.max_new_tokens,
            )

        # Slice to only new tokens
        caption = self.processor.batch_decode(text_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
        import torch as _torch
        self.load()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_dir = output_file.parent.resolve()
        count = 0

        # Clear file, then append each caption as it completes
        with open(output_file, "w", encoding="utf-8") as f:
            pass

        for i, vpath in enumerate(video_paths):
            log.info("Captioning [%d/%d]: %s", i + 1, len(video_paths), vpath.name)
            _torch.cuda.empty_cache()
            try:
                caption = self.caption_video(vpath)

                # Filter: model writes "SKIP" for non-sexual clips
                if caption.strip().upper().startswith("SKIP"):
                    log.info("  FILTERED (non-sexual): %s", vpath.name)
                    continue

                try:
                    rel_path = str(vpath.resolve().relative_to(metadata_dir))
                except ValueError:
                    rel_path = str(vpath.resolve())
                entry = {"media_path": rel_path, "caption": caption}
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                log.info("  Caption preview: %.120s...", caption)
            except Exception as e:
                log.error("  Failed to caption %s: %s", vpath.name, e)
                continue

        log.info("Wrote %d captions to %s", count, output_file)

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
            _torch.cuda.empty_cache()
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

        log.info("Wrote %d captions to %s", count, output_file)

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

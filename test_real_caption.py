#!/usr/bin/env python3
"""
Real end-to-end test — downloads Qwen2.5-VL-7B if needed, captions
the test videos with LTX 2.3 structured format, then unloads.
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
for name in ["urllib3", "httpx", "httpcore", "filelock", "transformers.configuration_utils"]:
    logging.getLogger(name).setLevel(logging.WARNING)

log = logging.getLogger("test_real")

import torch

from pipeline.crawler import LustpressServer, VideoCrawler
from pipeline import dashboard as dash
from pipeline.vram import flush_vram, get_vram_usage, log_vram


# ── LTX 2.3 caption prompt ─────────────────────────────────────────────────

CAPTION_PROMPT = """\
You are a professional video captioner for an AI video generation model (LTX-Video 2.3). \
Generate a highly detailed, long-form structured caption for this video. The caption must be \
thorough enough to recreate the video from text alone. Use present tense. \
Describe EVERY visual detail including camera angle, motion, lighting, colors, textures, \
facial expressions, body language, environment, and temporal changes frame by frame.

Use this exact format:

[VISUAL]: <Exhaustive description of all visual content. Include: subjects (appearance, clothing, \
features, actions, expressions), environment (setting, objects, background, foreground), \
cinematography (camera angle, movement, zoom, focus, depth of field), lighting (direction, color, \
intensity, shadows, highlights), color palette, composition, and how the scene evolves over time. \
Be specific about temporal progression - describe what happens at the beginning, middle, and end.>
[SPEECH]: <Exact word-for-word transcription of all spoken dialogue. Include speaker identification \
if multiple speakers. Note tone, emotion, and delivery style. If no speech, write "None.">
[SOUNDS]: <Detailed description of all audio: music (genre, instruments, tempo, mood), ambient \
sounds (nature, traffic, crowd), sound effects (impacts, transitions, foley). Describe how \
audio changes throughout the video. If no notable audio, write "None.">
[TEXT]: <All on-screen text: titles, subtitles, watermarks, signs, labels, UI elements, captions. \
Include position on screen and timing. If no text, write "None.">"""


def vram_bar(label: str):
    usage = get_vram_usage()
    dash.show_vram_status(label, usage)


def load_qwen_vl():
    """Load Qwen2.5-VL-7B-Instruct in float16 (fits in 32GB VRAM)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    dash.banner("LOADING CAPTIONER MODEL")
    print(f"  Model: {model_id}")
    print(f"  Precision: bfloat16 (no quant — 32GB VRAM is enough)")
    vram_bar("before load")

    t0 = time.time()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    vram_bar("after load")

    return model, processor


def caption_video(model, processor, video_path: Path) -> str:
    """Generate LTX 2.3 structured caption from a video file."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": 360 * 420,  # keep memory reasonable
                    "nframes": 16,  # sample 16 frames evenly from the video
                },
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1536,
            do_sample=False,
        )

    caption = processor.batch_decode(
        output[:, input_len:], skip_special_tokens=True
    )[0]

    return caption.strip()


def unload_model(model, processor):
    """Fully free GPU memory."""
    dash.banner("UNLOADING CAPTIONER")
    vram_bar("before unload")
    try:
        model.cpu()
    except Exception:
        pass
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    vram_bar("after unload")


def main():
    work_dir = Path("./test_workspace")
    raw_dir = work_dir / "raw"
    metadata_file = work_dir / "real_captions.jsonl"

    # ── Step 1: Ensure we have test videos ──────────────────────
    videos = sorted(raw_dir.glob("*.mp4")) if raw_dir.exists() else []

    if not videos:
        dash.banner("DOWNLOADING TEST VIDEOS VIA LUSTPRESS")
        server = LustpressServer("./lustpress", 3000)
        server.start()

        crawler = VideoCrawler(
            config={
                "search_queries": ["amateur couple"],
                "sources": ["xvideos"],
                "max_videos_per_batch": 2,
                "min_duration_sec": 30,
                "max_duration_sec": 180,
                "max_resolution": 480,
                "download_archive": str(work_dir / "test_seen.txt"),
                "include_random": False,
            },
            server=server,
        )
        videos = crawler.crawl(batch_num=0, output_dir=raw_dir)
        server.stop()

    if not videos:
        log.error("No videos available for testing")
        return

    # Limit to 2 videos for the test
    videos = videos[:2]

    print(f"\n  Test videos ({len(videos)}):")
    for v in videos:
        size_mb = v.stat().st_size / 1024 / 1024
        print(f"    {v.name}  ({size_mb:.1f} MB)")

    # ── Step 2: Load model and caption ──────────────────────────
    model, processor = load_qwen_vl()

    dash.banner("CAPTIONING VIDEOS (LTX 2.3 FORMAT)")
    results = []

    for i, video in enumerate(videos):
        print(f"\n  [{i+1}/{len(videos)}] {video.name}")
        vram_bar("before inference")

        t0 = time.time()
        caption = caption_video(model, processor, video)
        elapsed = time.time() - t0

        print(f"  Generated in {elapsed:.1f}s ({len(caption)} chars)")
        results.append({"media_path": str(video), "caption": caption})

        vram_bar("after inference")

    # ── Step 3: Unload model ────────────────────────────────────
    unload_model(model, processor)

    # ── Step 4: Save and display captions ───────────────────────
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    dash.show_captions(metadata_file)

    # Also print full captions for review
    dash.banner("FULL CAPTIONS (untruncated)")
    for i, entry in enumerate(results):
        print(f"\n--- Video {i+1}: {Path(entry['media_path']).name} ---")
        print(entry["caption"])
        print()

    dash.banner("TEST COMPLETE")
    print(f"  Captions saved to: {metadata_file}")
    print(f"  Videos: {len(videos)}")
    print(f"  Model: Qwen2.5-VL-7B-Instruct (4-bit NF4)")
    vram_bar("final")


if __name__ == "__main__":
    main()

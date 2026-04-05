"""
Live dashboard -- prints a rich, readable view of what's happening
at each pipeline stage. Shows crawled URLs, generated captions,
training progress, VRAM usage, etc.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# Terminal width for formatting
TERM_WIDTH = min(shutil.get_terminal_size().columns, 120)


def banner(text: str, char: str = "="):
    pad = max(0, TERM_WIDTH - len(text) - 4)
    print(f"\n{char * 2} {text} {char * pad}")


def section(text: str):
    banner(text, "-")


def kv(key: str, value, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def show_batch_header(batch_num: int, total_steps: int, query: str, sources: list[str]):
    print("\n" + "=" * TERM_WIDTH)
    print(f"  BATCH {batch_num}  |  Total steps: {total_steps}  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Query: \"{query}\"  |  Sources: {', '.join(sources)}")
    print("=" * TERM_WIDTH)


def show_crawl_results(candidates: list[dict], downloaded: list[Path]):
    section("CRAWL RESULTS")

    if not candidates:
        print("  (no candidates found)")
        return

    print(f"  Found {len(candidates)} candidates -> Downloaded {len(downloaded)}")
    print()

    for i, c in enumerate(candidates):
        status = "OK" if any(d.name.startswith(f"{c['source']}_{c.get('id', '')[:30]}") for d in downloaded) else "SKIP"
        dur = f"{c['duration']}s" if c.get("duration") else "?"
        title = c.get("title", "?")[:70]
        print(f"  [{status:4s}] [{c['source']:8s}] {dur:>6s}  {title}")
        print(f"         {c['url']}")

    print()


def show_scene_split(input_count: int, output_count: int, scene_dir: Path):
    section("SCENE SPLIT")
    print(f"  {input_count} videos -> {output_count} scene clips")
    if output_count > 0:
        clips = sorted(scene_dir.glob("*.mp4"))[:10]
        for clip in clips:
            size_mb = clip.stat().st_size / 1024 / 1024
            print(f"    {clip.name}  ({size_mb:.1f} MB)")
        if output_count > 10:
            print(f"    ... and {output_count - 10} more")
    print()


def show_captions(metadata_file: Path):
    section("GENERATED CAPTIONS (LTX 2.3 format)")

    if not metadata_file.exists():
        print("  (no captions generated)")
        return

    entries = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"  {len(entries)} videos captioned\n")

    for i, entry in enumerate(entries):
        media = Path(entry.get("media_path", "")).name
        caption = entry.get("caption", "")

        print(f"  +- Video {i+1}: {media}")

        # Parse and display structured caption sections
        sections_found = False
        for tag in ["[VISUAL]:", "[SPEECH]:", "[SOUNDS]:", "[TEXT]:"]:
            if tag in caption:
                sections_found = True
                start = caption.index(tag)
                # Find end (next tag or end of string)
                end = len(caption)
                for next_tag in ["[VISUAL]:", "[SPEECH]:", "[SOUNDS]:", "[TEXT]:"]:
                    if next_tag != tag and next_tag in caption[start + len(tag):]:
                        next_pos = caption.index(next_tag, start + len(tag))
                        end = min(end, next_pos)

                content = caption[start + len(tag):end].strip()
                label = tag.replace(":", "").replace("[", "").replace("]", "")
                # Truncate long content for display
                display = content[:200] + ("..." if len(content) > 200 else "")
                print(f"  |{label:7s}: {display}")

        if not sections_found:
            # Not structured -- show raw
            display = caption[:300] + ("..." if len(caption) > 300 else "")
            print(f"  |RAW: {display}")

        print(f"  +{'-' * 40}")
        print()


def show_preprocessing(precomputed_dir: Path):
    section("PREPROCESSING")

    if not precomputed_dir.exists():
        print("  (preprocessing output not found)")
        return

    for subdir in ["latents", "conditions", "audio_latents", "reference_latents"]:
        p = precomputed_dir / subdir
        if p.exists():
            count = len(list(p.iterdir()))
            print(f"  {subdir}/: {count} files")

    print()


def show_training_start(steps: int, checkpoint: str | None):
    section("TRAINING")
    print(f"  Steps this batch: {steps}")
    if checkpoint:
        print(f"  Resuming from: {checkpoint}")
    else:
        print("  Starting fresh (no prior checkpoint)")
    print()


def show_training_complete(checkpoint: str | None, total_steps: int):
    print(f"  Training complete!")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Total steps so far: {total_steps}")
    print()


def show_evaluation(eval_dir: Path, prompts: list[str]):
    section("EVALUATION")
    print(f"  Output: {eval_dir}")
    print(f"  Prompts:")
    for i, p in enumerate(prompts):
        print(f"    {i+1}. {p}")

    # Show generated files
    if eval_dir.exists():
        for mp4 in sorted(eval_dir.rglob("*.mp4")):
            rel = mp4.relative_to(eval_dir)
            size_mb = mp4.stat().st_size / 1024 / 1024
            print(f"  Generated: {rel}  ({size_mb:.1f} MB)")
    print()


def show_cleanup(kept_metadata: bool):
    section("CLEANUP")
    print(f"  Deleted raw videos and precomputed latents")
    if kept_metadata:
        print(f"  Metadata archived to ./metadata_archive/")
    print()


def show_batch_summary(batch_num: int, total_steps: int, checkpoint: str | None,
                       videos_count: int, captions_count: int):
    print("\n" + "=" * TERM_WIDTH)
    print(f"  BATCH {batch_num} COMPLETE")
    print(f"  Videos: {videos_count}  |  Captions: {captions_count}  |  Steps: {total_steps}  |  Checkpoint: {checkpoint or 'none'}")
    print("=" * TERM_WIDTH + "\n")


def show_vram_status(label: str, usage: dict):
    if not usage:
        return
    for gpu_id, info in usage.items():
        alloc = info["allocated_mb"]
        total = info["total_mb"]
        pct = (alloc / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"  GPU {gpu_id} [{bar}] {alloc:.0f}/{total:.0f} MB ({pct:.1f}%)  -- {label}")
